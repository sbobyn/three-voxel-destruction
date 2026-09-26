// The city's sky: a physical atmosphere (single scattering by air and haze, integrated along
// the view ray, so a low sun turns the horizon gold under a still-blue zenith), a sun disc with
// its glow, and a deck of raymarched cumulus between 900 m and 1800 m, lit by the sun through
// themselves and by the sky around them. Z is up, units are metres. It's all worked out once a
// frame on a full-screen quad at reduced resolution, and drawn on a dome that follows the
// camera, behind the city (depth untouched, no fog, no shadows).
//
// The clouds are shaped the way games have done it since Horizon Zero Dawn (Schneider 2015):
// a coverage map says where clouds may be, a height profile gives them flat bellies and round
// tops, a tiling Perlin-Worley volume gives the billows, and a finer Worley volume erodes their
// edges into wisps. The noise volumes are baked once on the CPU (a texture fetch is far cheaper
// than evaluating noise in the loop). Lighting per sample: a short march towards the sun for
// self-shadowing, Beer-Lambert with a powder term, a few octaves of forward-scattering phase
// standing in for multiple scattering (the silver lining), and the sky's colour as ambient.

import { Break, clamp, dot, exp, Fn, float, fract, If, length, Loop, max, min, mix, normalize, positionLocal, pow, rtt, screenCoordinate, screenSize, screenUV, select, smoothstep, sqrt, texture, texture3D, uniform, vec2, vec3, vec4 } from 'three/tsl';
import * as THREE from 'three/webgpu';

type Float = THREE.Node<'float'>;
type Vec3 = THREE.Node<'vec3'>;

export type SkyQuality = 'low' | 'medium' | 'high';

/** Raymarch budget per quality: steps through the cloud deck, steps towards the sun, and the
 * resolution the sky is worked out at (a fraction of the screen's, each way; the dome upsamples
 * it, and the soft clouds don't show it). */
const QUALITY: Record<SkyQuality, { steps: number; lightSteps: number; scale: number }> = {
  low: { steps: 16, lightSteps: 2, scale: 0.5 },
  medium: { steps: 32, lightSteps: 4, scale: 0.5 },
  high: { steps: 64, lightSteps: 6, scale: 0.75 },
};

/** The atmosphere, in kilometres (the scattering coefficients are per km). */
const ATMOSPHERE = {
  /** Planet radius and the atmosphere's depth. */
  radius: 6360,
  depth: 60,
  /** Scale heights of air (Rayleigh) and haze (Mie). */
  rayleighHeight: 8,
  mieHeight: 1.2,
  /** Rayleigh scattering at 680, 550 and 440 nm. */
  rayleigh: [5.802e-3, 13.558e-3, 33.1e-3],
  /** Ozone absorption (the Chappuis band keeps the zenith blue at sunset), carried on the air's
   * density profile with a column about as thick as the real one. */
  ozone: [1.3e-3, 3.76e-3, 0.17e-3],
  /** Haze scattering (a city's: half as much again as a clear day's), its extinction is 1.1
   * times that, and its forward lobe (Cornette-Shanks g), which makes the glow round the sun. */
  mie: 6e-3,
  mieExtinction: 1.1,
  mieG: 0.8,
  /** Samples along the view ray (spaced quadratically, densest near the eye). */
  steps: 8,
  /** The sun's irradiance as the sky sees it (sets how bright the sky reads under ACES). */
  exposure: 28,
} as const;

const LOOK = {
  /** The sun disc's angular radius (rad; about 2.4 times the real sun's, as games draw it), its
   * radiance, and the bright halo hugging it (a fraction of the disc's radiance). */
  sunRadius: 0.011,
  sunDisc: 60,
  sunHalo: 0.05,
  /** The DirectionalLight's intensity for a high sun, scaled by the air's transmittance. */
  sunLight: 3.2,
  /** Below the horizon the sky fades to `horizon` over this much of the view vector's z. */
  groundFade: 0.06,
  /** The cloud deck (m) and its noise: tile sizes of the shape, erosion and coverage maps. */
  cloudBase: 900,
  cloudTop: 1800,
  shapeTile: 3600,
  detailTile: 280,
  coverageTile: 24000,
  /** Mean coverage (0..1), and how far the coverage map moves it about (as a fraction of it,
   * so no cover means a clear sky). */
  coverage: 0.5,
  coverageSpread: 0.9,
  /** How far the erosion eats into the billows, and extinction (1/m) at full density. */
  erosion: 0.55,
  extinction: 0.15,
  /** Wind (m/s, towards +x a little north) and how far tops lean downwind (m). */
  wind: [11, 4],
  lean: 350,
  /** The small steps (m) are this divided by the square root of the step count. */
  fineSpan: 220,
  /** Distance the light march reaches towards the sun (m), in steps doubling in length from at
   * most `lightFirst` (so a short march still sees the shadow right under a cloud's skin). */
  lightReach: 720,
  lightFirst: 90,
  /** Clouds are marched no farther than this (m) along a ray, and not at all beyond `farthest`;
   * they fade into haze with distance over `haze` (m). */
  segment: 9000,
  farthest: 60000,
  haze: 14000,
  /** The erosion fades out towards this distance (m). */
  detailFar: 14000,
  /** Sunlight (irradiance) on the clouds and the sky's light around them, relative to the sky. */
  cloudSun: 9,
  cloudAmbient: 1,
} as const;

// ---------------------------------------------------------------------------------------------
// Noise volumes, baked once. Tiling Perlin and Worley noise on integer lattices with a period, so
// the textures wrap seamlessly.

/** A 32-bit integer hash of a lattice point to [0, 1). */
function hash(x: number, y: number, z: number, seed: number): number {
  let h = Math.imul(x, 0x27d4eb2d) ^ Math.imul(y, 0x165667b1) ^ Math.imul(z, 0x9e3779b1) ^ Math.imul(seed, 0x85ebca6b);
  h = Math.imul(h ^ (h >>> 15), 0x2c1b3c6d);
  h = Math.imul(h ^ (h >>> 12), 0x297a2d39);
  h ^= h >>> 15;
  return (h >>> 0) / 4294967296;
}

const wrap = (i: number, n: number) => ((i % n) + n) % n;

/** Tiling gradient noise with `period` lattice cells per tile; x, y, z in cells. About [-1, 1]. */
function perlin(period: number, seed: number): (x: number, y: number, z: number) => number {
  const grads = new Float32Array(period ** 3 * 3);
  for (let i = 0; i < period ** 3; i++) {
    // A random unit vector per lattice point
    const u = hash(i, 0, 0, seed) * 2 - 1;
    const a = hash(i, 1, 0, seed) * 2 * Math.PI;
    const s = Math.sqrt(1 - u * u);
    grads.set([s * Math.cos(a), s * Math.sin(a), u], i * 3);
  }
  const fade = (t: number) => t * t * t * (t * (t * 6 - 15) + 10);
  return (x, y, z) => {
    const [x0, y0, z0] = [Math.floor(x), Math.floor(y), Math.floor(z)];
    const [fx, fy, fz] = [x - x0, y - y0, z - z0];
    const [u, v, w] = [fade(fx), fade(fy), fade(fz)];
    let n = 0;
    for (let c = 0; c < 8; c++) {
      const [dx, dy, dz] = [c & 1, (c >> 1) & 1, c >> 2];
      const g = (wrap(x0 + dx, period) + period * (wrap(y0 + dy, period) + period * wrap(z0 + dz, period))) * 3;
      const d = grads[g] * (fx - dx) + grads[g + 1] * (fy - dy) + grads[g + 2] * (fz - dz);
      n += d * (dx ? u : 1 - u) * (dy ? v : 1 - v) * (dz ? w : 1 - w);
    }
    return n * 1.6;
  };
}

/** Tiling Worley noise, inverted (1 at a feature point, 0 a cell away); x, y, z in cells. */
function worley(period: number, seed: number): (x: number, y: number, z: number) => number {
  const points = new Float32Array(period ** 3 * 3);
  for (let i = 0; i < period ** 3; i++) points.set([hash(i, 0, 1, seed), hash(i, 1, 1, seed), hash(i, 2, 1, seed)], i * 3);
  return (x, y, z) => {
    const [cx, cy, cz] = [Math.floor(x), Math.floor(y), Math.floor(z)];
    let best = 1;
    for (let dz = -1; dz <= 1; dz++)
      for (let dy = -1; dy <= 1; dy++)
        for (let dx = -1; dx <= 1; dx++) {
          const [ix, iy, iz] = [cx + dx, cy + dy, cz + dz];
          const p = (wrap(ix, period) + period * (wrap(iy, period) + period * wrap(iz, period))) * 3;
          const ox = ix + points[p] - x;
          const oy = iy + points[p + 1] - y;
          const oz = iz + points[p + 2] - z;
          best = Math.min(best, ox * ox + oy * oy + oz * oz);
        }
    return 1 - Math.sqrt(best);
  };
}

/** Stretch `values` to span [0, 1] and store them as bytes. */
function toBytes(values: Float32Array): Uint8Array<ArrayBuffer> {
  let [lo, hi] = [Infinity, -Infinity];
  for (const v of values) [lo, hi] = [Math.min(lo, v), Math.max(hi, v)];
  const bytes = new Uint8Array(values.length);
  for (let i = 0; i < values.length; i++) bytes[i] = Math.round(((values[i] - lo) / (hi - lo)) * 255);
  return bytes;
}

function volume(size: number, voxel: (u: number, v: number, w: number) => number): THREE.Data3DTexture {
  const values = new Float32Array(size ** 3);
  for (let k = 0, i = 0; k < size; k++)
    for (let j = 0; j < size; j++) for (let n = 0; n < size; n++, i++) values[i] = voxel((n + 0.5) / size, (j + 0.5) / size, (k + 0.5) / size);
  const tex = new THREE.Data3DTexture(toBytes(values), size, size, size);
  tex.format = THREE.RedFormat;
  tex.type = THREE.UnsignedByteType;
  tex.minFilter = tex.magFilter = THREE.LinearFilter;
  tex.wrapS = tex.wrapT = tex.wrapR = THREE.RepeatWrapping;
  tex.needsUpdate = true;
  return tex;
}

/**
 * The billows (64³): Perlin noise dilated by Worley cells (rounded heaps with Perlin's
 * connectedness), then carved by a finer Worley fbm so each heap is made of smaller ones.
 */
function shapeVolume(): THREE.Data3DTexture {
  const p = [perlin(4, 1), perlin(8, 2), perlin(16, 3)];
  const w = [worley(4, 4), worley(8, 5), worley(16, 6), worley(32, 7)];
  const fbm = (o: number, u: number, v: number, t: number) => w[o](u * 4 * 2 ** o, v * 4 * 2 ** o, t * 4 * 2 ** o);
  return volume(64, (u, v, t) => {
    const perl = (p[0](u * 4, v * 4, t * 4) + 0.5 * p[1](u * 8, v * 8, t * 8) + 0.25 * p[2](u * 16, v * 16, t * 16)) / 1.75;
    const w1 = fbm(0, u, v, t) * 0.625 + fbm(1, u, v, t) * 0.25 + fbm(2, u, v, t) * 0.125;
    const w2 = fbm(1, u, v, t) * 0.625 + fbm(2, u, v, t) * 0.25 + fbm(3, u, v, t) * 0.125;
    const perlinWorley = w1 + Math.min(Math.max(perl * 0.5 + 0.5, 0), 1) * (1 - w1);
    return Math.max((perlinWorley - (w2 - 1)) / (2 - w2), 0);
  });
}

/** The erosion (32³): a Worley fbm, cauliflower bumps at three scales. */
function detailVolume(): THREE.Data3DTexture {
  const w = [worley(2, 8), worley(4, 9), worley(8, 10)];
  return volume(32, (u, v, t) => w[0](u * 2, v * 2, t * 2) * 0.625 + w[1](u * 4, v * 4, t * 4) * 0.25 + w[2](u * 8, v * 8, t * 8) * 0.125);
}

/** The coverage map (128², tiling): where the sky is cloudy and where it's clear. */
function coverageMap(): THREE.DataTexture {
  const size = 128;
  const p = [perlin(3, 11), perlin(6, 12), perlin(12, 13), perlin(24, 14)];
  const values = new Float32Array(size * size);
  for (let j = 0, i = 0; j < size; j++)
    for (let n = 0; n < size; n++, i++) {
      const [u, v] = [(n + 0.5) / size, (j + 0.5) / size];
      values[i] = p.reduce((sum, f, o) => sum + f(u * 3 * 2 ** o, v * 3 * 2 ** o, 0.5) * 0.5 ** o, 0);
    }
  const tex = new THREE.DataTexture(toBytes(values), size, size, THREE.RedFormat, THREE.UnsignedByteType);
  tex.minFilter = tex.magFilter = THREE.LinearFilter;
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.needsUpdate = true;
  return tex;
}

// ---------------------------------------------------------------------------------------------
// The atmosphere, twice: in TSL for the dome and in plain numbers for the colours the caller
// needs (fog, light). Keep the two in step.

const A = ATMOSPHERE;
const TOP = A.radius + A.depth;
/** Extinction by air per unit of its density: scattering plus ozone. */
const AIR_EXTINCTION = A.rayleigh.map((b, i) => b + A.ozone[i]);
const MIE_EXTINCTION = A.mie * A.mieExtinction;
const RAYLEIGH_PHASE = 3 / (16 * Math.PI);
const MIE_PHASE = (3 / (8 * Math.PI)) * ((1 - A.mieG ** 2) / (2 + A.mieG ** 2));

/**
 * The Chapman function: how much more air a ray meets leaving height `r` (in scale heights from
 * the planet's centre) at cosine `mu` from the zenith than one going straight up. An erfcx-based
 * closed form, within about 3% down to the horizon (below it, it stays at the horizon's value).
 */
function chapman(x: number, mu: number): number {
  const y = Math.max(mu, 0) * Math.sqrt(x / 2);
  return Math.sqrt(2 * x) / (y + Math.sqrt(y * y + 4 / Math.PI));
}

/** The sky's radiance towards unit `dir` (z-up) with the sun towards `sun`; into `out`. */
function skyRadiance(dir: THREE.Vector3, sun: THREE.Vector3, out: THREE.Color): THREE.Color {
  const mu = Math.max(dir.z, 0);
  const far = Math.sqrt(A.radius * A.radius * mu * mu + (TOP * TOP - A.radius * A.radius)) - A.radius * mu;
  const cos = dir.dot(sun);
  const phaseR = RAYLEIGH_PHASE * (1 + cos * cos);
  const phaseM = (MIE_PHASE * (1 + cos * cos)) / Math.pow(1 + A.mieG ** 2 - 2 * A.mieG * cos, 1.5);
  const sum = [0, 0, 0];
  let [depthR, depthM] = [0, 0];
  for (let i = 0; i < A.steps; i++) {
    const t = far * ((i + 0.5) / A.steps) ** 2;
    const ds = (far * (2 * i + 1)) / A.steps ** 2;
    const [px, py, pz] = [dir.x * t, dir.y * t, A.radius + mu * t];
    const r = Math.hypot(px, py, pz);
    const h = r - A.radius;
    const [rhoR, rhoM] = [Math.exp(-h / A.rayleighHeight), Math.exp(-h / A.mieHeight)];
    const muSun = (px * sun.x + py * sun.y + pz * sun.z) / r;
    const lit = smooth(-Math.sqrt((2 * h) / A.radius) - 0.02, -Math.sqrt((2 * h) / A.radius), muSun);
    const tauR = depthR + rhoR * ds * 0.5 + A.rayleighHeight * rhoR * chapman(r / A.rayleighHeight, muSun);
    const tauM = depthM + rhoM * ds * 0.5 + A.mieHeight * rhoM * chapman(r / A.mieHeight, muSun);
    for (let c = 0; c < 3; c++) sum[c] += Math.exp(-AIR_EXTINCTION[c] * tauR - MIE_EXTINCTION * tauM) * (A.rayleigh[c] * rhoR * phaseR + A.mie * rhoM * phaseM) * ds * lit;
    depthR += rhoR * ds;
    depthM += rhoM * ds;
  }
  return out.setRGB(sum[0] * A.exposure, sum[1] * A.exposure, sum[2] * A.exposure);
}

/** The air's transmittance from the ground towards the sun (its colour at the ground). */
function sunTransmittance(sun: THREE.Vector3, out: THREE.Color): THREE.Color {
  const tauR = A.rayleighHeight * chapman(A.radius / A.rayleighHeight, sun.z);
  const tauM = A.mieHeight * chapman(A.radius / A.mieHeight, sun.z);
  const [r, g, b] = AIR_EXTINCTION.map((e) => Math.exp(-e * tauR - MIE_EXTINCTION * tauM));
  // The sun sets behind the planet's edge over a few tenths of a degree
  const up = smooth(-0.01, 0.01, sun.z);
  return out.setRGB(r * up, g * up, b * up);
}

function smooth(e0: number, e1: number, x: number): number {
  const t = Math.min(Math.max((x - e0) / (e1 - e0), 0), 1);
  return t * t * (3 - 2 * t);
}

function chapmanNode(x: Float, mu: Float): Float {
  const y = max(mu, 0).mul(sqrt(x.mul(0.5)));
  return sqrt(x.mul(2)).div(y.add(sqrt(y.mul(y).add(4 / Math.PI))));
}

/** skyRadiance in TSL (inside a Fn: it loops). */
function skyRadianceNode(dir: Vec3, sun: Vec3): Vec3 {
  const mu = max(dir.z, 0);
  const far = sqrt(mu.mul(mu).mul(A.radius * A.radius).add(TOP * TOP - A.radius * A.radius)).sub(mu.mul(A.radius));
  const cos = dot(dir, sun);
  const phaseR = cos.mul(cos).add(1).mul(RAYLEIGH_PHASE);
  const phaseM = cos.mul(cos).add(1).mul(MIE_PHASE).div(pow(cos.mul(-2 * A.mieG).add(1 + A.mieG ** 2), 1.5));
  const sum = vec3(0).toVar();
  const depthR = float(0).toVar();
  const depthM = float(0).toVar();
  Loop(A.steps, ({ i }: { i: THREE.Node<'int'> }) => {
    const n = float(i);
    const t = far.mul(n.add(0.5).div(A.steps).pow(2));
    const ds = far.mul(n.mul(2).add(1)).div(A.steps ** 2);
    const p = vec3(dir.xy.mul(t), mu.mul(t).add(A.radius));
    const r = length(p);
    const h = r.sub(A.radius);
    const rhoR = exp(h.div(-A.rayleighHeight));
    const rhoM = exp(h.div(-A.mieHeight));
    const muSun = dot(p, sun).div(r);
    const dip = sqrt(h.mul(2 / A.radius)).negate();
    const lit = smoothstep(dip.sub(0.02), dip, muSun);
    const tauR = depthR.add(rhoR.mul(ds).mul(0.5)).add(rhoR.mul(A.rayleighHeight).mul(chapmanNode(r.div(A.rayleighHeight), muSun)));
    const tauM = depthM.add(rhoM.mul(ds).mul(0.5)).add(rhoM.mul(A.mieHeight).mul(chapmanNode(r.div(A.mieHeight), muSun)));
    const transmittance = exp(vec3(...AIR_EXTINCTION).mul(tauR).add(tauM.mul(MIE_EXTINCTION)).negate());
    const scatter = vec3(...A.rayleigh).mul(rhoR.mul(phaseR)).add(rhoM.mul(phaseM).mul(A.mie));
    sum.addAssign(transmittance.mul(scatter).mul(ds.mul(lit)));
    depthR.addAssign(rhoR.mul(ds));
    depthM.addAssign(rhoM.mul(ds));
  });
  return sum.mul(A.exposure);
}

// ---------------------------------------------------------------------------------------------

/** Henyey-Greenstein phase for scattering angle cosine `cos` and anisotropy `g`. */
function henyeyGreenstein(cos: Float, g: number): Float {
  return float((1 - g * g) / (4 * Math.PI)).div(pow(cos.mul(-2 * g).add(1 + g * g), 1.5));
}

export class CitySky {
  readonly object: THREE.Mesh;
  readonly sun: THREE.Vector3;
  readonly horizon = new THREE.Color();
  readonly sunColor = new THREE.Color();

  private readonly material: THREE.MeshBasicNodeMaterial;
  private readonly textures: THREE.Texture[];
  /** The sky and clouds, rendered at reduced resolution once a frame (rgb), with how much of
   * the sun gets through the clouds (a). */
  private readonly skyPass: THREE.RTTNode;
  /** The sun direction the colours below were last worked out for. */
  private readonly lastSun = new THREE.Vector3(NaN, NaN, NaN);
  /** The air's transmittance towards the sun (unnormalised sunColor). */
  private readonly transmittance = new THREE.Color();
  private readonly uniforms = {
    /** The camera (the sky is worked out on a full-screen quad, with a camera of its own). */
    eye: uniform(new THREE.Vector3()),
    cameraWorld: uniform(new THREE.Matrix4()),
    projectionInverse: uniform(new THREE.Matrix4()),
    sun: uniform(new THREE.Vector3(0, 0, 1)),
    /** The sun disc's radiance, the sunlight on the clouds, the sky's light around them (top
     * and bottom of the deck), and the ground haze below the horizon. */
    disc: uniform(new THREE.Color()),
    cloudSun: uniform(new THREE.Color()),
    ambientTop: uniform(new THREE.Color()),
    ambientBottom: uniform(new THREE.Color()),
    horizon: uniform(new THREE.Color()),
    /** Where the wind has carried the clouds (xy, m), and how far the erosion noise has churned
     * upwards (z); the erosion drifts a little faster than the billows. */
    drift: uniform(new THREE.Vector3()),
    coverage: uniform(LOOK.coverage),
    steps: uniform(32, 'int'),
    lightSteps: uniform(4, 'int'),
    /** The first light step's length (m); each after it doubles. */
    lightStep: uniform(48),
    /** The resolution the sky is rendered at, as a fraction of the screen's. */
    scale: uniform(0.5),
  };

  constructor(sunDirection: THREE.Vector3) {
    this.sun = sunDirection.clone().normalize();
    const shape = shapeVolume();
    const detail = detailVolume();
    const coverage = coverageMap();
    this.textures = [shape, detail, coverage];

    this.skyPass = rtt(this.sky(shape, detail, coverage), null, null, { depthBuffer: false });
    // The dome: the sky as rendered, upsampled with a tent filter (four bilinear taps half a
    // texel out, which also smooths the raymarch's jitter), and the sun disc drawn over it at
    // full resolution, so its rim stays crisp
    const half = vec2(0.5).div(screenSize.mul(this.uniforms.scale));
    this.skyPass.uvNode = screenUV.add(half);
    const tap = (x: number, y: number) => texture(this.skyPass.value, screenUV.add(half.mul(vec2(x, y))));
    const rendered = this.skyPass.add(tap(-1, 1)).add(tap(1, -1)).add(tap(-1, -1)).mul(0.25);
    const dir = normalize(positionLocal);
    const cos = dot(dir, this.uniforms.sun);
    const onDisc = smoothstep(Math.cos(LOOK.sunRadius * 1.15), Math.cos(LOOK.sunRadius * 0.85), cos).add(pow(max(cos, 0), 3000).mul(LOOK.sunHalo));
    const sun = this.uniforms.disc.mul(onDisc.mul(rendered.a).mul(smoothstep(-0.005, 0.005, dir.z)));
    this.material = new THREE.MeshBasicNodeMaterial({ side: THREE.BackSide, depthWrite: false, fog: false });
    this.material.colorNode = rendered.rgb.add(sun) as unknown as THREE.Node<'color'>;
    this.object = new THREE.Mesh(new THREE.SphereGeometry(1, 48, 24), this.material);
    this.object.name = 'CitySky';
    this.object.renderOrder = -1;
    this.object.frustumCulled = false;
    this.object.castShadow = this.object.receiveShadow = false;
    this.setQuality('medium');
    this.refreshColours();
  }

  /** Mean cloud cover, 0 (clear) to 1 (overcast). */
  get coverage(): number {
    return this.uniforms.coverage.value;
  }

  set coverage(value: number) {
    this.uniforms.coverage.value = Math.min(Math.max(value, 0), 1);
  }

  update(camera: THREE.Camera, time: number): void {
    camera.updateMatrixWorld();
    this.uniforms.eye.value.setFromMatrixPosition(camera.matrixWorld);
    this.uniforms.cameraWorld.value.copy(camera.matrixWorld);
    this.uniforms.projectionInverse.value.copy(camera.projectionMatrixInverse);
    this.object.position.copy(camera.position);
    const far = (camera as THREE.PerspectiveCamera).far ?? 4000;
    this.object.scale.setScalar(far * 0.9);
    this.object.updateMatrixWorld();
    if (!this.lastSun.equals(this.sun)) this.refreshColours();
    const [wx, wy] = LOOK.wind;
    this.uniforms.drift.value.set(wx * time, wy * time, time * 0.8);
  }

  setQuality(quality: SkyQuality): void {
    const q = QUALITY[quality];
    this.uniforms.steps.value = q.steps;
    this.uniforms.lightSteps.value = q.lightSteps;
    this.uniforms.lightStep.value = Math.min(LOOK.lightReach / (2 ** q.lightSteps - 1), LOOK.lightFirst);
    this.skyPass.setResolutionScale(q.scale);
    this.uniforms.scale.value = q.scale;
  }

  sunIntensity(): number {
    return LOOK.sunLight * Math.max(this.transmittance.r, this.transmittance.g, this.transmittance.b);
  }

  dispose(): void {
    this.object.geometry.dispose();
    this.material.dispose();
    this.skyPass.dispose();
    for (const t of this.textures) t.dispose();
  }

  /** Work out the colours that follow the sun: fog, light, the clouds' lighting. */
  private refreshColours(): void {
    this.sun.normalize();
    this.lastSun.copy(this.sun);
    const u = this.uniforms;
    u.sun.value.copy(this.sun);
    sunTransmittance(this.sun, this.transmittance);
    const t = this.transmittance;
    this.sunColor.setRGB(t.r, t.g, t.b).multiplyScalar(1 / Math.max(t.r, t.g, t.b, 1e-4));
    u.disc.value.copy(t).multiplyScalar(LOOK.sunDisc);
    u.cloudSun.value.copy(t).multiplyScalar(LOOK.cloudSun);

    // The horizon, averaged round the compass but for the quarter facing the sun (its glare
    // would tint all the fog); and the sky overhead (what lights cloud tops), averaged over the
    // zenith and a ring 35° up
    const dir = new THREE.Vector3();
    const c = new THREE.Color();
    const toSun = Math.atan2(this.sun.y, this.sun.x);
    this.horizon.setRGB(0, 0, 0);
    const above = new THREE.Color();
    for (let i = 0; i < 12; i++) {
      const a = toSun + Math.PI / 4 + (i / 12) * 1.5 * Math.PI;
      this.horizon.add(skyRadiance(dir.set(Math.cos(a), Math.sin(a), 0.02).normalize(), this.sun, c));
      above.add(skyRadiance(dir.set(Math.cos(a), Math.sin(a), 0.7).normalize(), this.sun, c));
    }
    this.horizon.multiplyScalar(1 / 12);
    above.multiplyScalar(0.5 / 12).add(skyRadiance(dir.set(0, 0, 1), this.sun, c).multiplyScalar(0.5));
    u.horizon.value.copy(this.horizon);
    // Light inside a cloud has mostly bounced around in it, so it's greyer than the sky outside
    const grey = (above.r + above.g + above.b) / 3;
    above.lerp(c.setRGB(grey, grey, grey), 0.4);
    u.ambientTop.value.copy(above).multiplyScalar(LOOK.cloudAmbient);
    // Under the deck: less of the sky, and a little sunlight back off the ground
    u.ambientBottom.value.copy(above).multiplyScalar(0.5).add(c.copy(t).multiplyScalar(0.08 * Math.max(this.sun.z, 0)));
    u.ambientBottom.value.multiplyScalar(LOOK.cloudAmbient);
  }

  /** The sky, clouds and the haze below the horizon towards each pixel (rgb), and how much of
   * the sun would show through (a). */
  private sky(shape: THREE.Texture, detail: THREE.Texture, coverageTex: THREE.Texture): THREE.Node {
    const u = this.uniforms;
    const L = LOOK;
    const thickness = L.cloudTop - L.cloudBase;
    const earthRadius = A.radius * 1000;

    /** Cloud cover (0..1) over world point `p`. */
    const coverAt = (p: Vec3): Float => {
      const cover = texture(coverageTex, p.xy.sub(u.drift.xy).div(L.coverageTile), 0).r;
      return clamp(u.coverage.mul(cover.sub(0.5).mul(2 * L.coverageSpread).add(1)), 0, 1);
    };

    /** Cloud density (0..1) at world point `p` whose height through the deck is `h` (0..1),
     * under cover `cov`, eroded by the detail noise in proportion to `fine`. */
    const density = (p: Vec3, h: Float, cov: Float, fine: Float | null): Float => {
      const q = p.sub(u.drift.mul(vec3(1, 1, 0))).add(vec3(L.wind[0], L.wind[1], 0).normalize().mul(h.mul(L.lean)));
      // Flat bellies, round tops; heavier cover builds taller clouds
      const top = mix(float(0.45), float(1), cov);
      const profile = smoothstep(0, 0.08, h).mul(smoothstep(top, top.mul(0.45), h));
      const base = texture3D(shape, q.div(L.shapeTile), 0).r;
      const d = clamp(base.mul(profile).sub(float(1).sub(cov)).div(max(cov.mul(0.6), 0.03)), 0, 1).toVar();
      if (fine)
        If(d.greaterThan(0).and(fine.greaterThan(0)), () => {
          const e = texture3D(detail, q.sub(u.drift.mul(0.4)).div(L.detailTile), 0).r;
          // Wispy underneath, billowing on top
          const erode = mix(float(1).sub(e), e, smoothstep(0.1, 0.5, h)).mul(fine.mul(L.erosion));
          d.assign(clamp(d.sub(erode).div(float(1).sub(erode)), 0, 1));
        });
      return d;
    };

    return Fn(() => {
      // The view ray through this pixel (WebGPU's clip space: y up, screen y down)
      const clip = vec4(screenUV.x.mul(2).sub(1), screenUV.y.mul(-2).add(1), 0.5, 1);
      const view = u.projectionInverse.mul(clip);
      const dir = normalize(u.cameraWorld.mul(vec4(view.xyz.div(view.w), 0)).xyz);
      const sun = u.sun;
      const cos = dot(dir, sun);

      // Below the horizon look along it instead (the planet would be there); fade to haze lower
      const level = normalize(vec3(dir.xy, max(dir.z, 0.0)));
      const sky = skyRadianceNode(level, sun).toVar();

      const transmittance = float(1).toVar();
      const scattered = vec3(0).toVar();
      const fade = float(1).toVar();

      // The deck: altitudes on a curved planet, flattened into a tangent plane at the camera, so
      // a ray's height at distance t is z0 + t dz + (t |dxy|)² / 2R and meets the deck at a root
      // of a quadratic (written in its cancellation-free form)
      const z0 = u.eye.z;
      const curve = dot(dir.xy, dir.xy).div(2 * earthRadius);
      const rise = (height: number) => {
        const dh = float(height).sub(z0);
        return dh.mul(2).div(dir.z.add(sqrt(max(dir.z.mul(dir.z).add(curve.mul(4).mul(dh)), 0))));
      };
      const sinkDisc = dir.z.mul(dir.z).sub(curve.mul(4).mul(z0.sub(L.cloudBase)));
      const sink = z0.sub(L.cloudBase).mul(2).div(dir.z.negate().add(sqrt(max(sinkDisc, 0))));
      const inside = z0.greaterThanEqual(L.cloudBase);
      const start = select(inside, float(0), rise(L.cloudBase));
      const leavesBelow = inside.and(dir.z.lessThan(0)).and(sinkDisc.greaterThan(0));
      const end = min(min(select(leavesBelow, sink, rise(L.cloudTop)), start.add(L.segment)), float(L.farthest));

      If(z0.lessThan(L.cloudTop).and(dir.z.greaterThan(-0.02)).and(end.greaterThan(start)), () => {
        const steps = u.steps;
        // Big steps through clear air; on meeting cloud, back up one and go on in small ones (a
        // cloud's sunlit skin is only tens of metres deep), back to big ones after a clear run
        const coarse = end.sub(start).div(float(steps));
        const fine = min(coarse, float(L.fineSpan).div(sqrt(float(steps))));
        // Start each pixel a different fraction of a step in (interleaved gradient noise), so
        // banding becomes fine grain
        const jitter = fract(fract(dot(screenCoordinate.xy, vec2(0.06711056, 0.00583715))).mul(52.9829189));
        const t = start.add(coarse.mul(jitter)).toVar();
        const inCloud = float(0).toVar();
        const clear = float(0).toVar();
        const depth = float(0).toVar();
        const weight = float(0).toVar();
        const horizontal = sqrt(dot(dir.xy, dir.xy));
        // Phase towards the sun: a strong forward lobe (the silver lining) and a weak back one
        const phase = mix(henyeyGreenstein(cos, 0.8), henyeyGreenstein(cos, -0.25), 0.35).mul(4 * Math.PI);
        const phase2 = henyeyGreenstein(cos, 0.4).mul(4 * Math.PI);
        const phase3 = henyeyGreenstein(cos, 0.2).mul(4 * Math.PI);
        Loop(steps.mul(2), () => {
          If(t.greaterThan(end).or(transmittance.lessThan(0.01)), () => {
            Break();
          });
          const p = u.eye.add(dir.mul(t));
          const drop = t.mul(horizontal).pow(2).div(2 * earthRadius);
          const h = p.z.add(drop).sub(L.cloudBase).div(thickness);
          If(inCloud.lessThan(0.5), () => {
            If(density(p, h, coverAt(p), null).greaterThan(0), () => {
              inCloud.assign(1);
              clear.assign(0);
              t.assign(max(t.sub(coarse), start));
            }).Else(() => {
              t.addAssign(coarse);
            });
          }).Else(() => {
            // The light march stays under the same cover (the map's features are kilometres)
            const cov = coverAt(p).toVar();
            // Far off, the erosion's detail is finer than a pixel and would only sparkle
            const d = density(p, h, cov, smoothstep(L.detailFar, L.detailFar * 0.4, t));
            If(d.greaterThan(0.002), () => {
              clear.assign(0);
              // March towards the sun through the cloud: its optical depth
              const optical = float(0).toVar();
              const len = u.lightStep.toVar();
              const along = float(0).toVar();
              Loop(u.lightSteps, () => {
                const s = p.add(sun.mul(along.add(len.mul(0.5))));
                const hs = s.z.add(drop).sub(L.cloudBase).div(thickness);
                optical.addAssign(density(s, hs, cov, null).mul(len));
                along.addAssign(len);
                len.mulAssign(2);
              });
              optical.mulAssign(L.extinction);
              // Beer-Lambert plus two dimmer, wider-lobed octaves for light scattered many
              // times; the powder term darkens the sunward skin, where little light has
              // gathered yet
              const beer = exp(optical.negate()).mul(phase).add(exp(optical.mul(-0.25)).mul(phase2).mul(0.3)).add(exp(optical.mul(-0.03)).mul(phase3).mul(0.08));
              const powder = float(1).sub(exp(d.mul(L.extinction * -60)).mul(0.6));
              const ambient = mix(u.ambientBottom, u.ambientTop, clamp(h, 0, 1)).mul(d.mul(-0.35).add(1));
              const light = u.cloudSun.mul(beer.mul(powder)).add(ambient);
              const absorbed = exp(d.mul(fine).mul(-L.extinction));
              // Albedo 1: what a step scatters is what it takes out of the light passing through
              const gain = transmittance.mul(float(1).sub(absorbed));
              scattered.addAssign(light.mul(gain));
              depth.addAssign(t.mul(gain));
              weight.addAssign(gain);
              transmittance.mulAssign(absorbed);
            }).Else(() => {
              clear.addAssign(1);
              If(clear.greaterThan(5), () => {
                inCloud.assign(0);
              });
            });
            t.addAssign(fine);
          });
        });
        // Distant clouds sink into the haze, and none are drawn right down at the horizon
        const distance = depth.div(max(weight, 1e-4));
        fade.assign(exp(distance.div(-L.haze)).mul(smoothstep(-0.01, 0.1, dir.z)));
      });

      // The sun and sky behind the clouds; the clouds, and the haze in front of distant ones
      const opacity = float(1).sub(transmittance);
      const col = sky.mul(transmittance).add(scattered.mul(fade)).add(sky.mul(opacity.mul(float(1).sub(fade))));
      return vec4(mix(col, u.horizon, smoothstep(0, -L.groundFade, dir.z)), transmittance);
    })();
  }
}
