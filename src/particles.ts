// Smoke, dust and fire for the city's destruction, after Teardown: a blast is a fireball of hot
// gas that cools along the blackbody curve (white-yellow, orange, dull red) and rolls up into
// soot, a dark column of smoke that rises, spreads and drifts downwind, a ring of dust thrown
// out along the ground, and sparks; crumbling voxels shed dense dust that settles and lingers.
//
// It all lives on the GPU. Particles are a ring buffer of five vec4s each, advanced by a TSL
// compute kernel: curl noise (divergence free, so puffs swirl without bunching up), buoyancy
// from temperature, drag towards the wind, growth and cooling. The CPU only queues bursts: a
// 16-byte record per new particle plus a small table of burst parameters, and the spawn kernel
// rolls everything else (directions, sizes, lifetimes) from a hash.
//
// Self-shadowing: every frame the particles are splatted into a coarse 3D grid of optical
// depth near the camera, and each one marches a few cells towards the sun and upwards through
// it, so a column is lit on its sunny side and dark in its core and underneath. The billboards
// are camera-facing sprites with a procedural puff texture (tileable Worley billows and Perlin
// detail, built once), turned per particle, lit through a fake sphere normal bumped by the
// billows. Fire and sparks are emissive in HDR; nearby fireballs light the smoke from below.
//
// Drawing is unsorted, with straight alpha, so scene fog still applies correctly. Hot gas is
// optically thick, so fire is drawn near opaque with its own radiance rather than added up.
// Order is fixed per particle (no popping), and the puffs are soft enough that drawing out of
// order doesn't show much. Soft particles read the scene's depth, and puffs fade by height
// above the ground.
//
// Given the depth of an opaque pass (`sceneDepth`), the particles can be drawn in a pass of
// their own, at a lower resolution, and composited after the scene's post effects that should
// not touch them (ambient occlusion darkened smoke like a wall, and the sprites' normals
// smeared occlusion onto what stood behind them). Occlusion by the scene is then the soft
// particle fade alone: a puff behind a wall fades to nothing.

import {
  atan,
  atomicAdd,
  atomicLoad,
  atomicStore,
  cameraFar,
  cameraNear,
  cameraViewMatrix,
  clamp,
  cos,
  Discard,
  dot,
  exp,
  float,
  floor,
  Fn,
  hash,
  If,
  instancedArray,
  instanceIndex,
  length,
  max,
  min,
  mix,
  mx_noise_vec3,
  normalize,
  perspectiveDepthToViewZ,
  pow,
  Return,
  screenUV,
  select,
  sin,
  smoothstep,
  sqrt,
  texture,
  uint,
  uniform,
  uniformArray,
  uv,
  varying,
  vec2,
  vec3,
  vec4,
  viewportDepthTexture,
} from 'three/tsl';
import * as THREE from 'three/webgpu';

type Float = THREE.Node<'float'>;
type Vec3 = THREE.Node<'vec3'>;
type Vec4 = THREE.Node<'vec4'>;

export type ParticleQuality = 'low' | 'medium' | 'high';

/** What a particle is: how it moves and how it's drawn. Fire cools into soot, so it is smoke too. */
const Kind = { Fire: 0, Smoke: 1, Dust: 2, Spark: 3 } as const;

/** How a burst's particles are placed and launched (the spawn kernel). */
const Burst = { Fireball: 0, Smoke: 1, DustRing: 2, Crumble: 3, Impact: 4, Smoulder: 5, Flame: 6, Sparks: 7, Exhaust: 8, Jet: 9 } as const;

/** Each burst type's lifetime range (s) before its life scale; the live count mirrors it. */
const LIFE: [number, number][] = [
  [7, 12], // fireball, cooling into smoke
  [8, 14], // blast smoke
  [5, 10], // blast dust ring
  [4.5, 8.5], // crumbling dust
  [2, 4.5], // impact puff
  [7, 12], // smouldering smoke
  [0.6, 1.2], // smouldering flame
  [1.2, 3], // sparks
  [0.7, 1.4], // exhaust wisps
  [0.4, 0.7], // a suit jet's wisps
];

/**
 * The look is after a big game's collapse dust (The Finals, Battlefield): a brief white-hot
 * flash, then a few huge, soft, dense, pale masses that billow out, hang and roll slowly, not
 * many small busy puffs. Hence slow, broad flow, little buoyancy, big particles that grow a
 * lot early and then hang, and a low-frequency puff shape.
 */
/** Per kind (fire, smoke, dust, spark): how fast velocity relaxes to the flow (1/s): heavy dust stops soon. */
const DRAG = [1.8, 0.9, 1.7, 0.35];
/** Per kind: curl noise speed (m/s): slow, so clouds roll rather than swirl. */
const TURBULENCE = [1.1, 0.5, 0.3, 1.2];
/** Per kind: how quickly growth slows with age (radius grows as growth / (1 + decay·age)): billow out, then hang. */
const GROWTH_DECAY = [0.8, 0.45, 0.6, 0];

/** Quality: particles per burst, and their size to keep the cover about the same. */
const QUALITY: Record<ParticleQuality, { count: number; size: number }> = {
  low: { count: 0.45, size: 1.3 },
  medium: { count: 1, size: 1 },
  high: { count: 1.3, size: 0.9 },
};

/** The self-shadowing grid: cells (x, y, z) and their edge (m), centred ahead of the camera. */
const GRID = [40, 40, 32] as const;
const CELL = 4;
/** Fixed point for the grid's optical depths (WGSL atomics are integers). */
const FIXED = 1024;

/** Fire lights (blasts, smoulders) lighting the smoke from inside. */
const LIGHTS = 8;

/** Most new particles and bursts uploaded per frame; the rest wait for the next. */
const MAX_SPAWN = 16384;
const MAX_BURSTS = 256;
/** Most particles one dust() call makes at medium quality. */
const DUST_BUDGET = 160;
/**
 * Live particles, at most (times the quality's count), before new dust is thinned: fewer puffs,
 * each bigger, so a cloud covers about as much. At a collapse's peak crumbling and impacts kept
 * 16,000 dust puffs alive, and they cost frames at any resolution (each puff, however small on
 * screen, costs a few fragment quads). Fire and blast smoke are never thinned.
 */
const LIVE_BUDGET = 5000;

/** Planck's law at three wavelengths (µm) standing in for linear sRGB's primaries. */
const LAMBDA = [0.61, 0.55, 0.465];
const C2 = 14388; // second radiation constant (µm K)
/** Fire's radiance at 2500 K (HDR, before tone mapping). */
const FIRE_RADIANCE = 9;

/**
 * Blackbody colour at temperature `t` (K), normalised to red (the brightest channel below
 * about 6000 K): (λr / λ)^5 (e^xr − 1) / (e^x − 1), x = c2 / λT, written to stay finite.
 */
function blackbody(t: Float): Vec3 {
  const x = vec3(C2 / LAMBDA[0], C2 / LAMBDA[1], C2 / LAMBDA[2]).div(t);
  const ratio = vec3(1, (LAMBDA[0] / LAMBDA[1]) ** 5, (LAMBDA[0] / LAMBDA[2]) ** 5);
  return ratio
    .mul(exp(x.x.sub(x)))
    .mul(float(1).sub(exp(x.x.negate())))
    .div(vec3(1).sub(exp(x.negate()))) as unknown as Vec3;
}

/** Fire's radiance at `t`: Stefan–Boltzmann's T^4, gone below a dull red at about 750 K. */
const fireRadiance = (t: Float): Float => pow(t.div(2500), 4).mul(FIRE_RADIANCE).mul(smoothstep(700, 1100, t));

/** The same colour on the CPU, for the fire lights. */
function blackbodyCPU(t: number, out: THREE.Color): THREE.Color {
  const x = LAMBDA.map((l) => C2 / (l * t));
  const c = x.map((xi, i) => (LAMBDA[0] / LAMBDA[i]) ** 5 * Math.exp(x[0] - xi) * (1 - Math.exp(-x[0])) / (1 - Math.exp(-xi)));
  return out.setRGB(c[0], c[1], c[2]);
}

/**
 * A tileable puff texture, built once: R billows (Worley cells, inverted: cauliflower
 * lumps), G and B their slope in u and v (0.5 is flat; `slope` scales it back), A finer
 * Perlin detail for fire and erosion.
 */
function puffTexture(size = 256): { texture: THREE.DataTexture; slope: number } {
  const hash2 = (x: number, y: number, s: number) => {
    let h = Math.imul(x, 374761393) ^ Math.imul(y, 668265263) ^ Math.imul(s, 2246822519);
    h = Math.imul(h ^ (h >>> 13), 1274126177);
    return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
  };
  // Worley F1 with `n` cells across, wrapping
  const worley = (u: number, v: number, n: number, s: number) => {
    const x = u * n;
    const y = v * n;
    const cx = Math.floor(x);
    const cy = Math.floor(y);
    let best = 9;
    for (let j = -1; j <= 1; j++) {
      for (let i = -1; i <= 1; i++) {
        const gx = cx + i;
        const gy = cy + j;
        const wx = ((gx % n) + n) % n;
        const wy = ((gy % n) + n) % n;
        const dx = gx + hash2(wx, wy, s) - x;
        const dy = gy + hash2(wx, wy, s + 101) - y;
        best = Math.min(best, dx * dx + dy * dy);
      }
    }
    return Math.sqrt(best);
  };
  // Perlin gradient noise with `n` cells across, wrapping
  const perlin = (u: number, v: number, n: number, s: number) => {
    const x = u * n;
    const y = v * n;
    const x0 = Math.floor(x);
    const y0 = Math.floor(y);
    const fade = (t: number) => t * t * t * (t * (t * 6 - 15) + 10);
    const corner = (i: number, j: number) => {
      const a = hash2(((x0 + i) % n + n) % n, ((y0 + j) % n + n) % n, s) * 2 * Math.PI;
      return Math.cos(a) * (x - x0 - i) + Math.sin(a) * (y - y0 - j);
    };
    const fx = fade(x - x0);
    const fy = fade(y - y0);
    const a = corner(0, 0) + (corner(1, 0) - corner(0, 0)) * fx;
    const b = corner(0, 1) + (corner(1, 1) - corner(0, 1)) * fx;
    return a + (b - a) * fy;
  };
  const billow = new Float32Array(size * size);
  const detail = new Float32Array(size * size);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const u = x / size;
      const v = y / size;
      const i = y * size + x;
      billow[i] = 0.55 * (1 - worley(u, v, 4, 1)) + 0.3 * (1 - worley(u, v, 8, 2)) + 0.15 * (1 - worley(u, v, 16, 3));
      detail[i] = 0.5 * perlin(u, v, 8, 4) + 0.3 * perlin(u, v, 16, 5) + 0.2 * perlin(u, v, 32, 6);
    }
  }
  const normalise = (a: Float32Array) => {
    let lo = Infinity;
    let hi = -Infinity;
    for (const x of a) (lo = Math.min(lo, x)), (hi = Math.max(hi, x));
    for (let i = 0; i < a.length; i++) a[i] = (a[i] - lo) / (hi - lo);
  };
  normalise(billow);
  normalise(detail);
  const at = (x: number, y: number) => billow[((y + size) % size) * size + ((x + size) % size)];
  const gx = new Float32Array(size * size);
  const gy = new Float32Array(size * size);
  let slope = 0;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = y * size + x;
      gx[i] = ((at(x + 1, y) - at(x - 1, y)) * size) / 2;
      gy[i] = ((at(x, y + 1) - at(x, y - 1)) * size) / 2;
      slope = Math.max(slope, Math.abs(gx[i]), Math.abs(gy[i]));
    }
  }
  const data = new Uint8Array(size * size * 4);
  for (let i = 0; i < size * size; i++) {
    data[i * 4] = Math.round(billow[i] * 255);
    data[i * 4 + 1] = Math.round((gx[i] / slope) * 127.5 + 127.5);
    data[i * 4 + 2] = Math.round((gy[i] / slope) * 127.5 + 127.5);
    data[i * 4 + 3] = Math.round(detail[i] * 255);
  }
  const tex = new THREE.DataTexture(data, size, size, THREE.RGBAFormat);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.magFilter = THREE.LinearFilter;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.generateMipmaps = true;
  tex.needsUpdate = true;
  return { texture: tex, slope };
}

const _ahead = new THREE.Vector3();
const _centre = new THREE.Vector3();
const _colour = new THREE.Color();

/** A burst waiting to be uploaded: its parameters and the origins of its particles. */
interface Pending {
  /** type, size (m), speed (m/s), life scale | tint rgb, opacity | centre xyz, size scale. */
  params: number[];
  /** Particle origins (xyz), or one origin for all `count`. */
  origins: Float32Array;
  count: number;
  done: number;
}

/** Particles spawned together, for the live count: how many, when, and their life range. */
interface Batch {
  n: number;
  t0: number;
  min: number;
  max: number;
}

/** A light inside the smoke: a blast's fireball or a smoulder's flames. */
interface FireLight {
  at: THREE.Vector3;
  t0: number;
  /** Blast radius (m); 0 for a smoulder. */
  radius: number;
  until: number;
}

export class Particles {
  /** Add to the scene. */
  readonly object: THREE.Object3D;
  /**
   * Optional: a point light following the brightest fireball, for blasts to light their
   * surroundings. Not in `object`; add it to the scene to use it (it costs a light in every
   * lit material).
   */
  readonly flash = new THREE.PointLight(0xffa060, 0, 0, 2);
  /** Height of the ground (m): particles settle on it and the dust ring rolls along it. */
  groundHeight = 0;
  /** Light on the smoke: the sun's colour and strength, and the sky's and ground's ambient. */
  readonly sunColour = uniform(new THREE.Color(0xfff0d8).multiplyScalar(1.3));
  readonly skyAmbient = uniform(new THREE.Color(0xc4d8ec).multiplyScalar(0.7));
  readonly groundAmbient = uniform(new THREE.Color(0xb7a58a).multiplyScalar(0.22));
  /** The device's share of LIVE_BUDGET (calibrate.ts: weaker GPUs keep fewer puffs alive). */
  budget = 1;
  /** Default dust colour (linear rgb): pale concrete. */
  readonly dustTint = new THREE.Color(0.36, 0.33, 0.29);

  private readonly renderer: THREE.WebGPURenderer;
  private readonly capacity: number;
  private readonly material: THREE.SpriteNodeMaterial;
  private readonly puff: THREE.DataTexture;

  /** Per particle: position, radius | velocity, age | life, temperature, seed, kind | tint, opacity | growth, cooling, sunlight, skylight. */
  private readonly P: THREE.StorageBufferNode<'vec4'>;
  private readonly V: THREE.StorageBufferNode<'vec4'>;
  private readonly A: THREE.StorageBufferNode<'vec4'>;
  private readonly C: THREE.StorageBufferNode<'vec4'>;
  private readonly D: THREE.StorageBufferNode<'vec4'>;
  private readonly spawn: THREE.StorageBufferNode<'vec4'>;
  private readonly bursts: THREE.StorageBufferNode<'vec4'>;

  private readonly dt = uniform(0);
  private readonly time = uniform(0);
  private readonly wind = uniform(new THREE.Vector3());
  /**
   * Earth's pull on the particles (1) or none (0, in orbit): sparks falling, dust settling, hot
   * smoke rising. And the air (1) or none (0): its drag and swirl; in vacuum a puff flies on
   * as it was thrown.
   */
  readonly gravity = uniform(1);
  readonly air = uniform(1);
  private readonly sun = uniform(new THREE.Vector3(0, 0, 1));
  private readonly ground = uniform(0);
  private readonly spawnBase = uniform(0, 'uint');
  private readonly spawnCount = uniform(0, 'uint');
  private readonly spawnSeed = uniform(0, 'uint');
  private readonly gridOrigin = uniform(new THREE.Vector3());
  private readonly lightAt = uniformArray(Array.from({ length: LIGHTS }, () => new THREE.Vector4()), 'vec4');
  private readonly lightColour = uniformArray(Array.from({ length: LIGHTS }, () => new THREE.Vector4()), 'vec4');

  private readonly emitNode: THREE.ComputeNode;
  private readonly stepNodes: THREE.ComputeNode[];
  private readonly clearNode: THREE.ComputeNode;

  private quality = QUALITY.medium;
  private head = 0;
  private seed = 1;
  private clock = 0;
  private readonly pending: Pending[] = [];
  private readonly batches: Batch[] = [];
  private readonly fires: FireLight[] = [];
  private readonly smoulders: { at: THREE.Vector3; until: number; smoke: number; flame: number }[] = [];
  private readonly spawnData: Float32Array;
  private readonly burstData: Float32Array;

  /** The opaque scene's depth (a pass's depth texture), or null: this pass's own. */
  private readonly sceneDepth: THREE.TextureNode | null;

  constructor(renderer: THREE.WebGPURenderer, capacity = 65536, sceneDepth: THREE.TextureNode | null = null) {
    this.renderer = renderer;
    this.sceneDepth = sceneDepth;
    this.capacity = capacity;
    this.P = instancedArray(capacity, 'vec4');
    this.V = instancedArray(capacity, 'vec4');
    this.A = instancedArray(capacity, 'vec4');
    this.C = instancedArray(capacity, 'vec4');
    this.D = instancedArray(capacity, 'vec4');
    this.spawn = instancedArray(MAX_SPAWN, 'vec4');
    this.bursts = instancedArray(MAX_BURSTS * 3, 'vec4');
    this.spawnData = this.spawn.value.array as Float32Array;
    this.burstData = this.bursts.value.array as Float32Array;
    const puff = puffTexture();
    this.puff = puff.texture;

    this.emitNode = this.emitKernel();
    this.stepNodes = this.stepKernels();
    // Every particle's age past any life: all dead
    this.clearNode = Fn(() => {
      const v = this.V.element(instanceIndex);
      v.assign(vec4(v.xyz, 1e9));
    })().compute(capacity, [64]);
    this.material = this.drawMaterial(puff.slope);
    const sprite = new THREE.Sprite(this.material);
    sprite.count = capacity;
    sprite.frustumCulled = false;
    sprite.castShadow = false;
    sprite.receiveShadow = false;
    this.object = sprite;
  }

  /** A blast at `at`: fireball + smoke + dust ring on the ground, scaled by radius (m). */
  explosion(at: ArrayLike<number>, radius: number): void {
    const r = Math.max(radius, 0.25);
    const f = (r / 2) ** 1.25 * this.quality.count;
    const s = this.quality.size;
    const life = Math.min(Math.max(Math.sqrt(r / 3), 0.6), 1.8);
    const speed = Math.sqrt(r);
    const [x, y, z] = [at[0], at[1], at[2]];
    const height = z - this.groundHeight;
    // Drawn in this order (unsorted): the ground dust, then smoke behind the fire, sparks last
    if (height < r * 1.5) {
      const d = this.dustTint;
      this.queue([Burst.DustRing, r, 5 * speed, life, d.r, d.g, d.b, 0.8, x, y, this.groundHeight, s], [x, y, this.groundHeight], Math.round(13 * f * Math.max(0.3, 1 - height / (r * 1.5))));
    }
    // Smoke the colour of the dust it's full of (a building's, not an oil fire's soot). In
    // vacuum there's no air to hold it: a flash, and a thin cloud racing out and gone
    const vacuum = this.air.value === 0;
    const [held, thick] = vacuum ? [0.3, 0.3] : [1, 1];
    this.queue([Burst.Smoke, r, 3.5 * speed * (vacuum ? 2 : 1), life * held, 0.4, 0.38, 0.35, 0.85 * thick, x, y, z, s], [x, y, z], Math.round(14 * f * (vacuum ? 0.5 : 1)));
    this.queue([Burst.Fireball, r, 7 * speed, life * (vacuum ? 0.25 : 1), 0.3, 0.29, 0.27, 0.9 * thick, x, y, z, s], [x, y, z], Math.round(16 * f));
    this.queue([Burst.Sparks, r, 13 * speed, 1, 1, 1, 1, 0.9, x, y, z, 1], [x, y, z], Math.round(30 * f));
    this.fires.push({ at: new THREE.Vector3(x, y, z), t0: this.clock, radius: r, until: this.clock + 1.2 * Math.sqrt(r) });
  }

  /**
   * Dust thrown up where voxels broke loose or crumbled: xyz triples, `amount` particles per
   * point (fractional amounts emit by chance), tinted `tint` (linear rgb). A call makes at
   * most DUST_BUDGET particles (times the quality's count); past that they are fewer and
   * bigger, so a whole storey crumbling costs no more to draw than a wall.
   */
  /** How much of a new dust burst to spawn (keep) and how much bigger its puffs (grow), by the room left under LIVE_BUDGET. */
  private thin(): { keep: number; grow: number } {
    const room = 1 - this.live / (LIVE_BUDGET * this.quality.count * this.budget);
    const keep = Math.min(1, Math.max(0.06, room));
    return { keep, grow: Math.min(2, 1 / Math.sqrt(keep)) };
  }

  dust(points: ArrayLike<number>, amount = 1, tint?: THREE.Color): void {
    const n = Math.floor(points.length / 3);
    if (n === 0) return;
    const thin = this.thin();
    const budget = DUST_BUDGET * this.quality.count * thin.keep;
    const wanted = n * amount * this.quality.count * thin.keep;
    const per = Math.min(wanted, budget) / n;
    const grow = Math.min(Math.cbrt(Math.max(wanted / budget, 1)), 2.5);
    const origins: number[] = [];
    let cx = 0;
    let cy = 0;
    let cz = 0;
    for (let i = 0; i < n; i++) {
      const k = Math.floor(per + Math.random());
      for (let j = 0; j < k; j++) origins.push(points[i * 3], points[i * 3 + 1], points[i * 3 + 2]);
      cx += points[i * 3] / n;
      cy += points[i * 3 + 1] / n;
      cz += points[i * 3 + 2] / n;
    }
    const t = tint ?? this.dustTint;
    this.queue([Burst.Crumble, 1, 1.2, 1, t.r, t.g, t.b, 0.55, cx, cy, cz, this.quality.size * grow * thin.grow], origins, origins.length / 3);
  }

  /** Sparks at `at`: `count` of them (times the quality's), flung out at up to `speed` m/s. */
  sparks(at: ArrayLike<number>, count: number, speed: number): void {
    const [x, y, z] = [at[0], at[1], at[2]];
    const n = Math.max(1, Math.round(count * this.quality.count));
    this.queue([Burst.Sparks, 0.2, speed, 1, 1, 1, 1, 0.9, x, y, z, 1], [x, y, z], n);
  }

  /** A puff of dust where a piece of debris hit the ground hard (speed m/s). */
  impact(at: ArrayLike<number>, speed: number, tint?: THREE.Color): void {
    const thin = this.thin();
    const n = Math.floor(Math.min(Math.max(speed * 0.8, 2), 30) * this.quality.count * thin.keep + Math.random());
    if (n === 0) return;
    const t = tint ?? this.dustTint;
    const size = Math.min(Math.max(speed / 10, 0.3), 2.5);
    this.queue([Burst.Impact, size, speed, 1, t.r, t.g, t.b, 0.45, at[0], at[1], at[2], this.quality.size * thin.grow], [at[0], at[1], at[2]], Math.max(1, Math.round(n * 0.6)));
  }

  /** A wisp of exhaust smoke (`amount` 0..1): a couple of small, faint, pale grey puffs. */
  exhaust(at: ArrayLike<number>, amount: number): void {
    const n = Math.max(1, Math.round((1 + 2 * amount) * this.quality.count));
    this.queue([Burst.Exhaust, 0.1 + amount * 0.12, 1, 1, 0.86, 0.86, 0.88, 0.3 + amount * 0.15, at[0], at[1], at[2], this.quality.size], [at[0], at[1], at[2]], n);
  }

  /**
   * A frame of a suit's jets: from each vent (`vents`, xyz each) a small, dense wisp blown at `velocity` (m/s), so
   * frame on frame they string out into thin trails; `amount` (0..1) thickens them.
   */
  jet(vents: ArrayLike<number>, velocity: THREE.Vector3, amount: number): void {
    const speed = velocity.length();
    if (speed === 0) return;
    // (The burst's centre far back along the jet, so the wisps from every vent fly out alike, through their vent)
    const back = 1000 / speed;
    const [x, y, z] = [vents[0] - velocity.x * back, vents[1] - velocity.y * back, vents[2] - velocity.z * back];
    this.queue([Burst.Jet, 0.035, speed, 1, 0.95, 0.95, 0.97, 0.45 + 0.35 * amount, x, y, z, this.quality.size], vents, vents.length / 3);
  }

  /** Lingering smoke source (e.g. burning spot) for `seconds`. */
  smoulder(at: ArrayLike<number>, seconds: number): void {
    const p = new THREE.Vector3(at[0], at[1], at[2]);
    this.smoulders.push({ at: p, until: this.clock + seconds, smoke: 0, flame: 0 });
    this.fires.push({ at: p, t0: this.clock, radius: 0, until: this.clock + seconds });
  }

  /** Particles alive now: an estimate from the lifetimes spawned (the GPU isn't read back). */
  get live(): number {
    let n = 0;
    for (const b of this.batches) n += b.n * Math.min(Math.max((b.t0 + b.max - this.clock) / (b.max - b.min), 0), 1);
    return Math.min(Math.round(n), this.capacity);
  }

  /** Per frame, before rendering. `sun` points towards the sun. */
  update(dt: number, camera: THREE.Camera, sun: THREE.Vector3, wind?: THREE.Vector3): void {
    dt = Math.min(Math.max(dt, 0), 0.1);
    this.clock += dt;
    this.dt.value = dt;
    this.time.value = this.clock;
    this.ground.value = this.groundHeight;
    this.sun.value.copy(sun).normalize();
    if (wind) this.wind.value.copy(wind);
    else this.wind.value.set(0, 0, 0);
    this.smoulderStep(dt);
    this.lightStep();

    // The shadow grid sits ahead of the camera, on the ground, snapped to its cells
    const ahead = _ahead.set(0, 0, -1).transformDirection(camera.matrixWorld);
    const centre = _centre.setFromMatrixPosition(camera.matrixWorld).addScaledVector(ahead, GRID[0] * CELL * 0.3);
    this.gridOrigin.value.set(
      Math.round(centre.x / CELL - GRID[0] / 2) * CELL,
      Math.round(centre.y / CELL - GRID[1] / 2) * CELL,
      Math.max(this.groundHeight - CELL, Math.round(centre.z / CELL - GRID[2] / 2) * CELL),
    );

    this.flush();
    this.renderer.compute(this.stepNodes);
    for (let i = this.batches.length - 1; i >= 0; i--) if (this.batches[i].t0 + this.batches[i].max < this.clock) this.batches.splice(i, 1);
  }

  setQuality(quality: ParticleQuality): void {
    this.quality = QUALITY[quality];
  }

  /** Everything gone at once: live particles, bursts not yet spawned, fires and their light. */
  clear(): void {
    this.pending.length = 0;
    this.batches.length = 0;
    this.fires.length = 0;
    this.smoulders.length = 0;
    this.flash.intensity = 0;
    this.renderer.compute(this.clearNode);
  }

  dispose(): void {
    this.clearNode.dispose();
    this.material.dispose();
    this.puff.dispose();
    this.emitNode.dispose();
    for (const node of this.stepNodes) node.dispose();
    this.flash.dispose();
  }

  /** Queue a burst of `count` particles from `origins` (one origin for all, or one each). */
  private queue(params: number[], origins: ArrayLike<number>, count: number): void {
    if (count <= 0) return;
    this.pending.push({ params, origins: Float32Array.from(origins), count, done: 0 });
    // More waiting than the ring holds would only overwrite itself: drop the oldest
    let waiting = 0;
    for (const p of this.pending) waiting += p.count - p.done;
    while (waiting > this.capacity && this.pending.length > 1) {
      const p = this.pending.shift()!;
      waiting -= p.count - p.done;
    }
  }

  /** Upload this frame's share of the queue and run the spawn kernel over it. */
  private flush(): void {
    let n = 0;
    let bursts = 0;
    while (this.pending.length > 0 && n < MAX_SPAWN && bursts < MAX_BURSTS) {
      const p = this.pending[0];
      this.burstData.set(p.params, bursts * 12);
      const m = Math.min(p.count - p.done, MAX_SPAWN - n);
      const single = p.origins.length === 3;
      for (let i = 0; i < m; i++) {
        const o = single ? 0 : (p.done + i) * 3;
        const r = (n + i) * 4;
        this.spawnData[r] = p.origins[o];
        this.spawnData[r + 1] = p.origins[o + 1];
        this.spawnData[r + 2] = p.origins[o + 2];
        this.spawnData[r + 3] = bursts;
      }
      const [lo, hi] = LIFE[p.params[0]];
      this.batches.push({ n: m, t0: this.clock, min: lo * p.params[3], max: hi * p.params[3] });
      n += m;
      bursts++;
      p.done += m;
      if (p.done === p.count) this.pending.shift();
    }
    if (n === 0) return;
    const spawn = this.spawn.value;
    spawn.clearUpdateRanges();
    spawn.addUpdateRange(0, n * 4);
    spawn.needsUpdate = true;
    const table = this.bursts.value;
    table.clearUpdateRanges();
    table.addUpdateRange(0, bursts * 12);
    table.needsUpdate = true;
    this.spawnBase.value = this.head;
    this.spawnCount.value = n;
    this.spawnSeed.value = this.seed;
    this.head = (this.head + n) % this.capacity;
    this.seed = (this.seed + n) % 0x1000000;
    this.renderer.compute(this.emitNode, n);
  }

  /** Smouldering spots: a steady trickle of smoke and small flames at each. */
  private smoulderStep(dt: number): void {
    for (let i = this.smoulders.length - 1; i >= 0; i--) {
      const s = this.smoulders[i];
      if (s.until < this.clock) {
        this.smoulders.splice(i, 1);
        continue;
      }
      // Dying down over its last few seconds
      const strength = Math.min(1, (s.until - this.clock) / 3);
      s.smoke += dt * 7 * this.quality.count * strength;
      s.flame += dt * 9 * this.quality.count * strength;
      const { x, y, z } = s.at;
      const smoke = Math.floor(s.smoke);
      const flame = Math.floor(s.flame);
      s.smoke -= smoke;
      s.flame -= flame;
      this.queue([Burst.Smoulder, 1, 1, 1, 0.15, 0.15, 0.15, 0.5, x, y, z, this.quality.size], [x, y, z], smoke);
      this.queue([Burst.Flame, 1, 1, 1, 0.2, 0.2, 0.2, 0.8, x, y, z, this.quality.size], [x, y, z], flame);
    }
  }

  /** The fire lights for the smoke, and the optional flash, from the live blasts and smoulders. */
  private lightStep(): void {
    for (let i = this.fires.length - 1; i >= 0; i--) if (this.fires[i].until < this.clock) this.fires.splice(i, 1);
    const colour = _colour;
    let brightest = 0;
    for (let i = 0; i < LIGHTS; i++) {
      const at = this.lightAt.array[i] as THREE.Vector4;
      const col = this.lightColour.array[i] as THREE.Vector4;
      const fire = this.fires[this.fires.length - 1 - i];
      if (!fire) {
        col.set(0, 0, 0, 0);
        continue;
      }
      const t = this.clock - fire.t0;
      let strength: number;
      let reach: number;
      if (fire.radius > 0) {
        // A blast: a flash, then gone within a second (the fireball cools that fast: the
        // cloud after it is dust, not lit from inside)
        const r = fire.radius;
        const tau = 0.2 * Math.sqrt(r);
        strength = (t < 0.05 ? t / 0.05 : Math.exp(-(t - 0.05) / tau)) * 3.5;
        reach = 2.2 * r;
        at.set(fire.at.x, fire.at.y, fire.at.z + 1.5 * Math.sqrt(r) * Math.min(t, 1), reach);
        blackbodyCPU(Math.max(3000 - 2200 * t, 1300), colour);
      } else {
        // A smoulder: flickering low flames
        strength = 0.6 * (0.8 + 0.2 * Math.sin(t * 13) * Math.sin(t * 7.3)) * Math.min(1, (fire.until - this.clock) / 3);
        reach = 3;
        at.set(fire.at.x, fire.at.y, fire.at.z + 0.8, reach);
        blackbodyCPU(1700, colour);
      }
      col.set(colour.r * strength, colour.g * strength, colour.b * strength, 0);
      if (fire.radius > 0 && strength * fire.radius ** 2 > brightest) {
        brightest = strength * fire.radius ** 2;
        this.flash.position.set(at.x, at.y, at.z);
        this.flash.color.copy(colour);
      }
    }
    // (candela; set by eye against a sun of intensity about 3)
    this.flash.intensity = brightest * 12;
  }

  /** The spawn kernel: one thread per new particle, reading its record and its burst. */
  private emitKernel(): THREE.ComputeNode {
    const { P, V, A, C, D, spawn, bursts, capacity } = this;
    const lifeRange = uniformArray(LIFE.map(([lo, hi]) => new THREE.Vector2(lo, hi)), 'vec2');
    return Fn(() => {
      const i = instanceIndex;
      If(i.greaterThanEqual(this.spawnCount), () => {
        Return();
      });
      const record = spawn.element(i);
      const b = record.w.toUint().mul(3);
      const b0 = bursts.element(b).toVar(); // type, size, speed, life scale
      const b1 = bursts.element(b.add(1)).toVar(); // tint, opacity
      const b2 = bursts.element(b.add(2)).toVar(); // centre, size scale
      const seed = this.spawnSeed.add(i).mul(16);
      const rnd = (k: number) => hash(seed.add(uint(k)));
      const type = b0.x;
      const size = b0.y;
      const speed = b0.z;
      const scale = b2.w;

      // A random direction, a random point in the unit ball, and one on the ground's circle
      const cz = rnd(0).mul(2).sub(1);
      const phi = rnd(1).mul(2 * Math.PI);
      const sz = sqrt(float(1).sub(cz.mul(cz)));
      const dir = vec3(sz.mul(cos(phi)), sz.mul(sin(phi)), cz).toVar();
      const ball = dir.mul(pow(rnd(2), 1 / 3)).toVar();
      const flat = vec3(cos(phi), sin(phi), 0).toVar();

      const origin = record.xyz;
      const pos = vec3(origin).toVar();
      const vel = vec3(0).toVar();
      const radius = float(1).toVar();
      const temp = float(300).toVar();
      const kind = float(Kind.Smoke).toVar();
      const growth = float(0).toVar();
      const cooling = float(0.3).toVar();
      const range = lifeRange.element(type.toUint()) as unknown as THREE.Node<'vec2'>;
      const life = mix(range.x, range.y, rnd(6)).mul(b0.w).toVar();

      If(type.equal(Burst.Fireball), () => {
        pos.assign(origin.add(ball.mul(size.mul(0.35))));
        vel.assign(dir.mul(speed.mul(rnd(3).mul(rnd(3)).mul(0.9).add(0.3))).add(vec3(0, 0, speed.mul(0.2))));
        radius.assign(size.mul(rnd(4).mul(0.25).add(0.4)));
        // A brief white-hot flash that cools fast into dust-grey smoke (some sooner than others)
        temp.assign(rnd(5).mul(1100).add(2000));
        kind.assign(Kind.Fire);
        growth.assign(size.mul(0.5));
        cooling.assign(float(3.2).div(sqrt(size.mul(0.5))).mul(rnd(7).mul(1.2).add(0.6)));
      })
        .ElseIf(type.equal(Burst.Smoke), () => {
          pos.assign(origin.add(ball.mul(size.mul(0.6))));
          vel.assign(dir.mul(speed.mul(rnd(3).mul(0.6).add(0.4))).add(vec3(0, 0, speed.mul(0.3))));
          radius.assign(size.mul(rnd(4).mul(0.3).add(0.5)));
          temp.assign(rnd(5).mul(200).add(450));
          growth.assign(size.mul(0.3));
        })
        .ElseIf(type.equal(Burst.DustRing), () => {
          const reach = rnd(3).mul(0.7).add(0.3);
          pos.assign(origin.add(flat.mul(size.mul(reach))).add(vec3(0, 0, size.mul(rnd(5).mul(0.5).add(0.1)))));
          vel.assign(flat.mul(speed.mul(rnd(4).mul(0.6).add(0.6))).add(vec3(0, 0, speed.mul(rnd(5).mul(0.3).add(0.05)))));
          radius.assign(size.mul(rnd(7).mul(0.3).add(0.5)));
          kind.assign(Kind.Dust);
          growth.assign(size.mul(0.45));
        })
        .ElseIf(type.equal(Burst.Crumble), () => {
          // Out and away from the crumbling section's centre, slumping
          const out = origin.sub(b2.xyz).mul(vec3(1, 1, 0));
          const away = out.div(max(length(out), 0.5));
          pos.assign(origin.add(ball.mul(1.4)));
          vel.assign(away.mul(speed).add(dir.mul(rnd(3).mul(0.6))).add(vec3(0, 0, -0.5)));
          radius.assign(rnd(4).mul(1.1).add(1.5));
          kind.assign(Kind.Dust);
          growth.assign(1.1);
        })
        .ElseIf(type.equal(Burst.Impact), () => {
          pos.assign(origin.add(ball.mul(size.mul(0.3))));
          vel.assign(flat.mul(speed.mul(rnd(3).mul(0.2).add(0.1))).add(vec3(0, 0, speed.mul(rnd(4).mul(0.12).add(0.04)))));
          radius.assign(size.mul(rnd(5).mul(0.4).add(0.6)).add(0.5));
          kind.assign(Kind.Dust);
          growth.assign(size.mul(0.5).add(0.35));
        })
        .ElseIf(type.equal(Burst.Exhaust), () => {
          // Small and short-lived: out of the pipe, rising a little, spreading slowly
          pos.assign(origin.add(ball.mul(0.04)));
          vel.assign(ball.mul(0.4).add(vec3(0, 0, rnd(3).mul(0.3).add(0.2))));
          radius.assign(size.mul(rnd(5).mul(0.5).add(0.5)));
          kind.assign(Kind.Dust);
          growth.assign(size.mul(0.8));
        })
        .ElseIf(type.equal(Burst.Jet), () => {
          // Out of the vent along its line (from the burst's centre through it), in a tight cone: a thin trail,
          // widening a little as it goes
          pos.assign(origin.add(ball.mul(0.01)));
          vel.assign(normalize(origin.sub(b2.xyz)).add(ball.mul(0.06)).mul(speed.mul(rnd(3).mul(0.2).add(0.9))));
          radius.assign(size.mul(rnd(5).mul(0.4).add(0.8)));
          kind.assign(Kind.Dust);
          growth.assign(size.mul(1.5));
        })
        .ElseIf(type.equal(Burst.Smoulder), () => {
          pos.assign(origin.add(ball.mul(0.5)));
          vel.assign(dir.mul(0.4).add(vec3(0, 0, rnd(3).add(1.2))));
          radius.assign(rnd(4).mul(0.4).add(0.45));
          temp.assign(rnd(5).mul(200).add(600));
          growth.assign(0.5);
        })
        .ElseIf(type.equal(Burst.Flame), () => {
          pos.assign(origin.add(ball.mul(vec3(0.5, 0.5, 0.2))));
          vel.assign(dir.mul(0.3).add(vec3(0, 0, rnd(3).mul(0.8).add(0.8))));
          radius.assign(rnd(4).mul(0.2).add(0.3));
          temp.assign(rnd(5).mul(500).add(1700));
          kind.assign(Kind.Fire);
          growth.assign(0.25);
          cooling.assign(1.6);
        })
        .Else(() => {
          // Sparks: flung out, mostly upwards
          const up = vec3(dir.x, dir.y, dir.z.abs().mul(0.8).add(0.2));
          pos.assign(origin.add(ball.mul(size.mul(0.3))));
          vel.assign(normalize(up).mul(speed.mul(rnd(3).mul(0.9).add(0.3))));
          radius.assign(rnd(4).mul(0.05).add(0.05));
          temp.assign(rnd(5).mul(700).add(1900));
          kind.assign(Kind.Spark);
          cooling.assign(0.35);
        });

      const p = i.add(this.spawnBase).mod(capacity);
      // Tints vary a little from particle to particle
      const tint = b1.xyz.mul(rnd(8).mul(0.3).add(0.85));
      P.element(p).assign(vec4(pos, radius.mul(select(kind.equal(Kind.Spark), float(1), scale))));
      V.element(p).assign(vec4(vel, 0));
      A.element(p).assign(vec4(life, temp, rnd(9), kind));
      C.element(p).assign(vec4(tint, b1.w));
      D.element(p).assign(vec4(growth.mul(scale).mul(rnd(10).mul(0.8).add(0.6)), cooling, 1, 1));
    })().compute(MAX_SPAWN, [64]);
  }

  /** Per frame: clear the shadow grid, splat the particles into it, then advance them. */
  private stepKernels(): THREE.ComputeNode[] {
    const { P, V, A, C, D, capacity } = this;
    const cells = GRID[0] * GRID[1] * GRID[2];
    const grid = instancedArray(cells, 'uint').toAtomic();
    const table = (values: number[]) => {
      const array = uniformArray(values, 'float');
      return (k: THREE.Node<'uint'>) => array.element(k) as unknown as Float;
    };
    const drag = table(DRAG);
    const turbulence = table(TURBULENCE);
    const growthDecay = table(GROWTH_DECAY);

    /** A cell's index and whether it's in the grid. */
    const cell = (c: Vec3) => {
      const inside = c.x.greaterThanEqual(0).and(c.y.greaterThanEqual(0)).and(c.z.greaterThanEqual(0)).and(c.x.lessThan(GRID[0])).and(c.y.lessThan(GRID[1])).and(c.z.lessThan(GRID[2]));
      const index = c.x.add(c.y.mul(GRID[0])).add(c.z.mul(GRID[0] * GRID[1])).toUint();
      return { index: select(inside, index, uint(0)), inside };
    };
    /** Optical depth per cell crossed at `p` (nearest cell; outside the grid, clear air). */
    const depthAt = (p: Vec3): Float => {
      const { index, inside } = cell(floor(p.sub(this.gridOrigin).div(CELL)));
      return select(inside, atomicLoad(grid.element(index)).toFloat().div(FIXED), float(0));
    };

    const clear = Fn(() => {
      atomicStore(grid.element(instanceIndex), uint(0));
    })().compute(cells, [64]);

    // Each particle's shadow is its disc's cover of a cell's cross-section (at most all of
    // it) times its opacity, shared out trilinearly among the eight nearest cell centres
    const splat = Fn(() => {
      const i = instanceIndex;
      const v = V.element(i);
      const a = A.element(i);
      If(v.w.greaterThanEqual(a.x).or(a.w.equal(Kind.Spark)), () => {
        Return();
      });
      const p = P.element(i);
      const opacity = C.element(i).w.mul(lifeFade(a.w, v.w, a.x));
      const cover = opacity.mul(min(p.w.mul(p.w).mul(Math.PI / (CELL * CELL)), 1)).mul(FIXED);
      const q = p.xyz.sub(this.gridOrigin).div(CELL).sub(0.5);
      const base = floor(q);
      const f = q.sub(base);
      for (let k = 0; k < 8; k++) {
        const o = vec3(k & 1, (k >> 1) & 1, (k >> 2) & 1);
        const w = mix(vec3(1).sub(f), f, o);
        const { index, inside } = cell(base.add(o));
        atomicAdd(grid.element(index), select(inside, cover.mul(w.x.mul(w.y).mul(w.z)).toUint(), uint(0)));
      }
    })().compute(capacity, [64]);

    const step = Fn(() => {
      const i = instanceIndex;
      const pv = P.element(i);
      const vv = V.element(i);
      const av = A.element(i);
      const dv = D.element(i);
      If(vv.w.greaterThanEqual(av.x), () => {
        Return();
      });
      const dt = this.dt;
      const kind = av.w;
      const k = kind.toUint();
      const age = vv.w.toVar();
      const pos = pv.xyz.toVar();
      const vel = vv.xyz.toVar();
      const radius = pv.w.toVar();
      const ground = this.ground;

      // Cooling towards the air's 300 K
      const temp = av.y.sub(300).mul(exp(dv.y.negate().mul(dt))).add(300).toVar();

      // The flow: wind (stronger with height) plus two octaves of curl noise, stirred harder
      // while the blast is young; the noise drifts with the wind
      const height = clamp(pos.z.sub(ground).div(40), 0, 1);
      const drift = pos.sub(this.wind.mul(this.time));
      const stir = turbulence(k).mul(exp(age.mul(-2)).mul(1.2).add(1));
      const flow = this.wind
        .mul(height.mul(0.65).add(0.35))
        .add(curl(drift.div(24), this.time.mul(0.015)).mul(stir))
        .add(curl(drift.div(9).add(17.3), this.time.mul(0.03)).mul(stir.mul(0.25)));

      // Buoyancy from heat; fine dust barely settles; sparks fall
      const lift = select(
        kind.equal(Kind.Spark),
        float(-9.8),
        select(kind.equal(Kind.Dust), float(-0.18), min(temp.sub(300).mul(select(kind.equal(Kind.Fire), float(0.0025), float(0.0012))), 5).add(0.1)),
      ).mul(this.gravity);
      const relax = float(1).sub(exp(drag(k).negate().mul(dt))).mul(this.air);
      vel.addAssign(flow.sub(vel).mul(relax));
      vel.z.addAssign(lift.mul(dt));
      pos.addAssign(vel.mul(dt));

      // The ground: puffs rest on it (a third sunk in), sparks bounce
      const floorZ = ground.add(select(kind.equal(Kind.Spark), float(0.05), radius.mul(0.3)));
      If(pos.z.lessThan(floorZ), () => {
        pos.z.assign(floorZ);
        vel.z.assign(select(kind.equal(Kind.Spark), vel.z.abs().mul(0.3), max(vel.z, 0)));
        vel.xy.mulAssign(select(kind.equal(Kind.Spark), float(0.6), float(1)));
      });

      radius.addAssign(dv.x.div(age.mul(growthDecay(k)).add(1)).mul(dt));
      age.addAssign(dt);

      // Light through the grid: a march towards the sun and one straight up (clear air
      // outside it), smoothed over time so crossing cells doesn't flicker
      const sun = this.sun;
      let toSun: Float = depthAt(pos).mul(0.5);
      let toSky: Float = depthAt(pos).mul(0.5);
      for (const [s, w] of [
        [1, 1],
        [2, 1],
        [3.5, 1.5],
        [5.5, 2],
        [8, 2.5],
      ])
        toSun = toSun.add(depthAt(pos.add(sun.mul(s * CELL))).mul(w));
      for (const [s, w] of [
        [1, 1],
        [2.5, 1.5],
        [4.5, 2],
      ])
        toSky = toSky.add(depthAt(pos.add(vec3(0, 0, s * CELL))).mul(w));
      // (with a floor for light scattered in from around: more in pale dust than in soot)
      const scattered = select(kind.equal(Kind.Dust), float(0.6), float(0.4));
      const sunlight = exp(toSun.mul(-0.15)).mul(float(1).sub(scattered)).add(scattered);
      const skylight = exp(toSky.mul(-0.2)).mul(0.75).add(0.25);
      const settle = select(vv.w.equal(0), float(1), float(1).sub(exp(dt.mul(-5))));

      pv.assign(vec4(pos, radius));
      vv.assign(vec4(vel, age));
      av.y.assign(temp);
      dv.z.assign(mix(dv.z, sunlight, settle));
      dv.w.assign(mix(dv.w, skylight, settle));
    })().compute(capacity, [64]);

    return [clear, splat, step];
  }

  /**
   * The billboards' material: puff shape, lighting, fire, fading and soft edges. Like any
   * material it also writes the scene pass's other targets (the city's view normals: where a
   * puff is drawn they're the sprite's, facing the camera). A per-material mrtNode can't keep
   * the scene's normals instead: the targets' blending is set per pass, not per material.
   */
  private drawMaterial(slope: number): THREE.SpriteNodeMaterial {
    // Drawn in a pass of their own (sceneDepth given), the scene's depth isn't in the depth
    // buffer: the soft fade against it hides what's behind the scene instead
    const material = new THREE.SpriteNodeMaterial({ transparent: true, depthWrite: false, depthTest: this.sceneDepth === null });
    const p = this.P.toAttribute();
    const v = this.V.toAttribute();
    const a = this.A.toAttribute();
    const c = this.C.toAttribute();
    const d = this.D.toAttribute();
    const kind = a.w;
    const alive = v.w.lessThan(a.x);
    const isSpark = kind.equal(Kind.Spark);

    // Puffs fade out as the camera comes within two radii, and aren't drawn at all inside
    // one: a puff around the camera would fill the screen for little to see
    const centreView = cameraViewMatrix.mul(vec4(p.xyz, 1)).xyz;
    const near = clamp(centreView.z.negate().sub(p.w.mul(0.8)).div(max(p.w.mul(1.2), 0.5)), 0, 1);
    // Sparks stretch along their motion on screen (a 1/30 s streak)
    const onScreen = cameraViewMatrix.mul(vec4(v.xyz, 0)).xy;
    const streak = length(onScreen).mul(1 / 30);
    material.positionNode = p.xyz;
    material.rotationNode = select(isSpark, atan(onScreen.y, onScreen.x), float(0));
    material.scaleNode = select(alive.and(near.greaterThan(0)), vec2(p.w.mul(2).add(select(isSpark, streak, float(0))), p.w.mul(2)), vec2(0));

    // Per particle, in the vertex stage, packed four to a varying (there are only 16)
    const sunDir = normalize(cameraViewMatrix.mul(vec4(this.sun, 0)).xyz);
    const upDir = cameraViewMatrix.mul(vec4(0, 0, 1, 0)).xyz;
    // Forward scattering (Henyey–Greenstein, g = 0.4, relative to isotropic): bright rims
    // looking towards the sun
    const cosTheta = dot(normalize(centreView), sunDir);
    const phase = float(0.84).div(pow(float(1.16).sub(cosTheta.mul(0.8)), 1.5));
    const angle = a.z.mul(2 * Math.PI).add(v.w.mul(a.z.sub(0.5).mul(0.6)));
    // The fire lights, falling off with distance
    let glow: Vec3 = vec3(0);
    for (let i = 0; i < LIGHTS; i++) {
      const at = this.lightAt.element(i) as unknown as Vec4;
      const col = this.lightColour.element(i) as unknown as Vec4;
      const r2 = at.w.mul(at.w);
      const d2 = dot(p.xyz.sub(at.xyz), p.xyz.sub(at.xyz));
      glow = glow.add(col.xyz.mul(r2.div(r2.add(d2))));
    }
    const state = varying(vec4(kind, a.y, v.w.div(max(a.x, 1e-3)), c.w.mul(lifeFade(kind, v.w, a.x)).mul(near)));
    const vKind = state.x;
    const vTemp = state.y;
    const vAge = state.z;
    const vOpacity = state.w;
    const look = varying(vec4(c.xyz, p.w));
    const vTint = look.xyz;
    const vRadius = look.w;
    const light = varying(vec3(d.z, d.w, phase));
    const vSun = light.x;
    const vSky = light.y;
    const vPhase = light.z;
    const spin = varying(vec2(cos(angle), sin(angle)));
    const glowAndHeight = varying(vec4(glow, p.z.sub(this.ground)));
    const vGlow = glowAndHeight.xyz;
    // Where the billboard is: its view depth (a sprite's is its centre's), the height of its
    // centre above the ground, and how height changes across it
    const vHeight = glowAndHeight.w;
    const sunAndDepth = varying(vec4(sunDir, centreView.z));
    const sunView = sunAndDepth.xyz;
    const vDepth = sunAndDepth.w;
    const upView = varying(upDir);
    const offsets = varying(vec4(a.z.mul(7.31), a.z.mul(3.17), upDir.xy.mul(p.w)));
    const vOffset = offsets.xy;
    const vRise = offsets.zw;

    // Per fragment, cheapest tests first: outside the disc, eroded to nothing, or faded out
    material.colorNode = Fn(() => {
      // The puff: a sphere's thickness, lumped by billows turning with the particle
      const q = uv().mul(2).sub(1);
      const r2 = dot(q, q);
      Discard(r2.greaterThan(1));
      const h = sqrt(float(1).sub(r2));
      const ca = spin.x;
      const sa = spin.y;
      const turned = vec2(q.x.mul(ca).sub(q.y.mul(sa)), q.x.mul(sa).add(q.y.mul(ca)));
      // Low-frequency: a few broad billows across the puff, little fine detail (fine noise
      // read as busy, small-scale smoke)
      const tex = texture(this.puff, turned.mul(0.14).add(vOffset));
      const billow = tex.r;
      const detail = texture(this.puff, turned.mul(0.28).add(vOffset.yx).add(vec2(0, vAge.mul(-0.3)))).a;
      // Young puffs are full, lumpy billows; as they age they soften and thin, not go wispy
      const lumps = h.mul(billow.mul(mix(0.85, 0.6, vAge)).add(detail.mul(mix(0.15, 0.3, vAge))).add(0.22));
      const erode = mix(float(0.1), float(0.3), vAge);
      const density = select(vKind.equal(Kind.Spark), exp(r2.mul(-5)), smoothstep(erode, erode.add(mix(0.6, 0.9, vAge)), lumps));
      Discard(density.mul(vOpacity).lessThan(0.004));

      // Soft edges where the billboard meets the scene; towards the ground by height as
      // well, since at grazing views the depth difference grows too fast to hide the cut
      const fadeLength = max(vRadius.mul(0.25), 0.05);
      const depth = this.sceneDepth ? this.sceneDepth.sample(screenUV).x : viewportDepthTexture().x;
      const sceneZ = perspectiveDepthToViewZ(depth, cameraNear, cameraFar);
      const height = vHeight.add(dot(q, vRise));
      const soft = clamp(vDepth.sub(sceneZ).div(fadeLength), 0, 1).mul(smoothstep(0, 1, height.div(fadeLength.mul(2))));
      const alpha = density.mul(vOpacity).mul(soft);
      Discard(alpha.lessThan(0.004));

      // Lit by the sun (wrapped, through the grid's shadow), the sky and the ground (by the
      // normal's height), forward scattering at thin edges, and the fire lights from below.
      // The normal is the sphere's, bumped by the billows' slope (turned back to the
      // billboard's frame)
      const slopeT = tex.gb.sub(0.5).mul(2 * slope * 0.25);
      const bump = vec2(slopeT.x.mul(ca).add(slopeT.y.mul(sa)), slopeT.y.mul(ca).sub(slopeT.x.mul(sa)));
      const normal = normalize(vec3(q.sub(bump.mul(0.8)), h.add(0.15)));
      const wrap = clamp(dot(normal, sunView).mul(0.7).add(0.3), 0, 1);
      const diffuse = wrap.mul(sqrt(wrap)).mul(float(1).sub(density.mul(h).mul(0.15)));
      const upness = dot(normal, upView);
      const sunlit = this.sunColour.mul(vSun).mul(diffuse.add(vPhase.mul(0.25).mul(float(1).sub(density.mul(0.6)))));
      const ambient = mix(this.groundAmbient, this.skyAmbient, clamp(upness.mul(0.5).add(0.5), 0, 1)).mul(vSky);
      const below = vGlow.mul(clamp(float(0.55).sub(upness.mul(0.45)), 0.1, 1));
      const colour = vTint.mul(sunlit.add(ambient).add(below)).toVar();

      // Fire: hotter in the lumps and the core. Hot gas is optically thick, so fire is drawn
      // near opaque with its own radiance (overlapping flames don't add up to white);
      // cooling below about 1000 K it's left as lit soot. A branch: most fragments are smoke
      If(vKind.equal(Kind.Fire).or(vKind.equal(Kind.Spark)), () => {
        const heat = max(vTemp.mul(billow.mul(0.3).add(detail.mul(0.25)).add(0.6)).mul(h.mul(0.3).add(0.7)), 400);
        colour.addAssign(blackbody(heat).mul(fireRadiance(heat)));
      });
      return vec4(colour, alpha);
    })();
    return material;
  }
}

/** Opacity over a particle's life: a quick fade in, then thinning out to nothing. */
function lifeFade(kind: Float, age: Float, life: Float): Float {
  const fadeIn = select(kind.equal(Kind.Fire).or(kind.equal(Kind.Spark)), float(0.08), float(0.35));
  const x = clamp(age.div(max(life, 1e-3)), 0, 1);
  return smoothstep(0, fadeIn, age).mul(pow(float(1).sub(x), 1.2));
}

/**
 * Curl noise: the curl of a Perlin vector potential, so divergence free (puffs swirl without
 * collecting or emptying), by forward differences. `t` drifts the sample point so the flow
 * changes over time (a cheap stand-in for a fourth noise axis).
 */
function curl(q: Vec3, t: Float): Vec3 {
  const e = 0.3;
  const at = q.add(vec3(t, t.mul(0.7), t.mul(-0.4)));
  const psi = mx_noise_vec3(at);
  const dx = mx_noise_vec3(at.add(vec3(e, 0, 0))).sub(psi);
  const dy = mx_noise_vec3(at.add(vec3(0, e, 0))).sub(psi);
  const dz = mx_noise_vec3(at.add(vec3(0, 0, e))).sub(psi);
  return vec3(dy.z.sub(dz.y), dz.x.sub(dx.z), dx.y.sub(dy.x)).div(e) as unknown as Vec3;
}
