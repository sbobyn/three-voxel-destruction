// The city's renderer. Every voxel is one instance of a unit cube, placed in the vertex shader
// straight from the solver's body buffer (shared with it: nothing is copied per frame), and
// shaded procedurally in its own local frame, so a brick keeps its courses as it tumbles:
// concrete with weathering, brick and mortar, framed glass that mirrors the sky, windows lit
// from inside, steel and gravel roofs. The streets are drawn on the ground from world
// position: asphalt, lane markings, crossings, pavements, paved plazas. Then ambient
// occlusion, bloom, fog and ACES.

import { bloom } from 'three/addons/tsl/display/BloomNode.js';
import { depthAwareBlend } from 'three/addons/tsl/display/depthAwareBlend.js';
import { film } from 'three/addons/tsl/display/FilmNode.js';
import { godrays } from 'three/addons/tsl/display/GodraysNode.js';
import { smaa } from 'three/addons/tsl/display/SMAANode.js';
import { fxaa } from 'three/addons/tsl/display/FXAANode.js';
import { ao } from 'three/addons/tsl/display/GTAONode.js';
import {
  abs,
  Fn,
  Loop,
  perspectiveDepthToViewZ,
  rtt,
  add,
  cameraPosition,
  clamp,
  cross,
  equirectUV,
  exp,
  float,
  floor,
  fract,
  hash,
  instanceIndex,
  int,
  uint,
  max,
  min,
  mix,
  mod,
  mrt,
  mx_noise_float,
  normalGeometry,
  normalize,
  normalView,
  output,
  pass,
  positionGeometry,
  positionLocal,
  positionWorld,
  pow,
  renderOutput,
  select,
  sign,
  smoothstep,
  step,
  storage,
  texture,
  transformNormalToView,
  uniform,
  uv,
  varying,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import { HDRLoader } from 'three/examples/jsm/loaders/HDRLoader.js';
import * as THREE from 'three/webgpu';
import { B_POS, B_ROT, B_SIZE, BODY_FLOATS } from 'three-avbd/advanced';
import { edgeShade } from './look.ts';
import { PARKED_BELOW } from './physics.ts';
import { BASE, type City, Mat, VOXEL } from './world.ts';

type Vec3 = THREE.Node<'vec3'>;
type Vec4 = THREE.Node<'vec4'>;
type Float = THREE.Node<'float'>;
/** A read-only storage buffer of vec4s, as the voxel shaders read it. */
const vec4Storage = (a: THREE.StorageInstancedBufferAttribute, n: number) => storage(a, 'vec4', n).toReadOnly();
type StorageVec4 = ReturnType<typeof vec4Storage>;

/** v rotated by the unit quaternion q = (u, w). */
function rotate(q: Vec4, v: Vec3): Vec3 {
  const t = cross(q.xyz, v).mul(2);
  return v.add(t.mul(q.w)).add(cross(q.xyz, t));
}

export type RenderQuality = 'low' | 'medium' | 'high';

/**
 * Detail drawn on each voxel's faces, for its size: brick courses and bricks along a course
 * (a course is 8 cm, a brick 25 cm), and the voxel-art cells (an eighth of a metre).
 */
const COURSES = Math.max(1, Math.round(VOXEL / 0.083));
const BRICKS = Math.max(1, Math.round(VOXEL / 0.25));

/** Sun shaft density in clear air and in thick dust. */
/**
 * A scene pass that clears to transparent black whatever the renderer's clear colour: effects
 * that copy their input into a texture first (bloom, SMAA, FXAA) reset the renderer to clear
 * opaque black while they do, and a pass rendered inside them inherited it (the smoke's
 * coverage came out 1 everywhere, and the scene under it black).
 */
class TransparentPass extends THREE.PassNode {
  private readonly held = new THREE.Color();
  updateBefore(frame: THREE.NodeFrame): boolean | undefined {
    const renderer = frame.renderer as THREE.Renderer;
    const alpha = renderer.getClearAlpha();
    renderer.getClearColor(this.held);
    renderer.setClearColor(this.held, 0);
    const result = super.updateBefore(frame);
    renderer.setClearColor(this.held, alpha);
    return result;
  }
}

/** The smoke pass's resolution, as a share of the scene's, by quality. */
const SMOKE_SCALE: Record<RenderQuality, number> = { high: 0.5, medium: 0.5, low: 0.35 };

const RAYS_CLEAR = 0.04;
const RAYS_DUSTY = 0.14;

/**
 * Surfaces (Polyhaven, CC0: public/textures, credits there): a layer each in one array texture
 * per map (colour, normal, AO-roughness-metal), so a face samples its own with one lookup.
 * `metres`: the width one tile covers.
 */
const SETS = [
  { id: 'brick_wall_005', metres: 1.44 },
  { id: 'concrete_wall_008', metres: 2.71 },
  { id: 'plastered_wall_02', metres: 2.23 },
  { id: 'damaged_plaster', metres: 1.85 },
  { id: 'asphalt_03', metres: 2.05 },
];
const [L_BRICK, L_CONCRETE, L_PLASTER, L_BROKEN, L_ASPHALT] = [0, 1, 2, 3, 4];
const TEXTURE_SIZE = 1024;

interface Surfaces {
  diffuse: THREE.DataArrayTexture;
  normal: THREE.DataArrayTexture;
  arm: THREE.DataArrayTexture;
}

/**
 * The surface textures, decoded and stacked (row 0 at the bottom, so v runs up the image as the
 * normal map's green does). `onFile(done, total)` after each image.
 */
async function loadSurfaces(onFile: (done: number, total: number) => void = () => {}): Promise<Surfaces> {
  const maps = ['diff', 'nor_gl', 'arm'];
  const total = SETS.length * maps.length;
  let done = 0;
  const layers = await Promise.all(
    SETS.flatMap((set) =>
      maps.map(async (map) => {
        const blob = await (await fetch(`textures/${set.id}_${map}_1k.webp`)).blob();
        const bitmap = await createImageBitmap(blob, { imageOrientation: 'flipY' });
        const canvas = new OffscreenCanvas(TEXTURE_SIZE, TEXTURE_SIZE);
        const ctx = canvas.getContext('2d')!;
        ctx.drawImage(bitmap, 0, 0, TEXTURE_SIZE, TEXTURE_SIZE);
        bitmap.close();
        onFile(++done, total);
        return ctx.getImageData(0, 0, TEXTURE_SIZE, TEXTURE_SIZE).data;
      }),
    ),
  );
  const build = (k: number, srgb: boolean) => {
    const data = new Uint8Array(TEXTURE_SIZE * TEXTURE_SIZE * 4 * SETS.length);
    SETS.forEach((_, i) => data.set(layers[i * maps.length + k], i * TEXTURE_SIZE * TEXTURE_SIZE * 4));
    const t = new THREE.DataArrayTexture(data, TEXTURE_SIZE, TEXTURE_SIZE, SETS.length);
    t.wrapS = t.wrapT = THREE.RepeatWrapping;
    t.minFilter = THREE.LinearMipmapLinearFilter;
    t.magFilter = THREE.LinearFilter;
    t.generateMipmaps = true;
    t.anisotropy = 8;
    t.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
    t.needsUpdate = true;
    return t;
  };
  return { diffuse: build(0, true), normal: build(1, false), arm: build(2, false) };
}

/** A material's base colour (world.ts BASE), linear. */
function baseColour(m: Mat): Vec3 {
  const c = new THREE.Color(BASE[m]);
  return vec3(c.r, c.g, c.b);
}

/** Nearer than this (m) is the tool in hand, not the world: no ambient occlusion on it. */
const HAND_REACH = 0.9;

export class CityRenderer {
  readonly renderer: THREE.WebGPURenderer;
  readonly scene = new THREE.Scene();
  readonly camera = new THREE.PerspectiveCamera(75, 1, 0.05, 6000);
  readonly sun = new THREE.DirectionalLight(0xffffff, 3);
  readonly hemi = new THREE.HemisphereLight(0xbcd4ff, 0x6b5f52, 0.25);
  /** Unit vector towards the sun. */
  readonly sunDirection = new THREE.Vector3(0.45, -0.55, 0.42).normalize();
  private readonly post: THREE.RenderPipeline;
  private bodies: THREE.StorageInstancedBufferAttribute | null = null;
  private paints: THREE.StorageInstancedBufferAttribute | null = null;
  /** Per body: carried by a section (offset xyz, the section's body; w < 0: not carried). */
  private links: THREE.StorageInstancedBufferAttribute | null = null;
  private voxels: THREE.Mesh | null = null;
  /** The voxel buffers as the shaders read them (for the glass). */
  private nodes: { bodies: StorageVec4; paints: StorageVec4; links: StorageVec4 } | null = null;
  /** Each body's time burned by the laser (the world's clock, s), for the cut's glow. */
  private heats: THREE.StorageInstancedBufferAttribute | null = null;
  /** Each body's voxel's place in the city as built (xyz): its faces' texture stays put on it wherever it goes. */
  private homes: THREE.StorageInstancedBufferAttribute | null = null;
  /** The world's clock (s), which the cut's glow cools by. */
  readonly heatClock = uniform(0);
  private glass: THREE.Mesh | null = null;
  private readonly outputs: { high: THREE.Node; low: THREE.Node };
  quality: RenderQuality = 'high';
  /** The bloom (explosions push its strength up briefly). */
  private readonly bloomPass: ReturnType<typeof bloom>;
  /** Sun shafts, and their colour (the sun's). */
  private readonly rays: ReturnType<typeof godrays>;
  private readonly rayColour = uniform(new THREE.Color(1, 0.9, 0.75));
  /**
   * Crepuscular rays round what stands against the sky (GPU Gems 3 ch. 13, as the light-shafts
   * lab does them): the bright sky near the sun, where nothing covers it, blurred radially
   * toward the sun on screen at half resolution and added on. Where the sun is on screen (y
   * down), how much it faces the view (fading out as it turns away), and the strength.
   */
  private readonly sunScreen = uniform(new THREE.Vector2(0.5, 0.5));
  private readonly sunFade = uniform(0);
  private readonly aspect = uniform(1);
  readonly sunRays = uniform(1);
  /** Depth of field: the distance in focus (m), how far either side stays sharp (m), and the bokeh's size (0: off). */
  private readonly focus = uniform(30);
  private readonly focalRange = uniform(60);
  private readonly bokeh = uniform(0);
  /** One pixel, as a share of the screen (the blur's taps are placed in pixels). */
  private readonly pixel = uniform(new THREE.Vector2(1 / 1920, 1 / 1080));
  /**
   * Smoke, dust and fire: drawn by a pass of their own at a share of the resolution
   * (SMOKE_SCALE), against the scene's depth (`sceneDepth`), and laid over the scene after its
   * ambient occlusion and before the bloom (fire glows, smoke isn't darkened like a wall).
   */
  readonly smokeScene = new THREE.Scene();
  readonly sceneDepth: THREE.TextureNode;
  private readonly smokePass: TransparentPass;
  /** What GPU this is (vendor, architecture, device), for keeping a calibration per device. */
  adapterInfo = 'unknown';
  /** The surface textures (null while loading, or if they could not be: the surfaces are procedural then). */
  private surfaces: Surfaces | null = null;
  private readonly skies = new Map<string, Promise<THREE.Texture>>();

  constructor(canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGPURenderer({ canvas, antialias: false, requiredLimits: { maxStorageBufferBindingSize: 1 << 30, maxBufferSize: 1 << 30 } });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1;
    this.camera.up.set(0, 0, 1);

    const sun = this.sun;
    sun.castShadow = true;
    sun.shadow.mapSize.set(4096, 4096);
    const s = sun.shadow.camera as THREE.OrthographicCamera;
    [s.left, s.right, s.top, s.bottom, s.near, s.far] = [-110, 110, 110, -110, 1, 900];
    sun.shadow.bias = -0.0004;
    sun.shadow.normalBias = 0.02;
    this.scene.add(sun, sun.target, this.hemi);
    this.scene.fog = new THREE.FogExp2(0xc9d6e2, 0.0022);
    this.smokeScene.fog = this.scene.fog;
    this.renderer.setClearColor(0x000000, 0);
    this.scene.add(this.ground());

    this.post = new THREE.RenderPipeline(this.renderer);
    const scene = pass(this.scene, this.camera, { samples: 0 });
    scene.setMRT(mrt({ output, normal: normalView }));
    const colour = scene.getTextureNode('output');
    const depth = scene.getTextureNode('depth');
    // The smoke reads the scene's depth as a plain texture: through the pass's own texture
    // node, the smoke pass's render pulled the scene pass into it and the scene came out black
    this.sceneDepth = texture(scene.getTexture('depth'));
    // The smoke's pass: premultiplied colour and coverage over transparent black
    this.smokePass = new TransparentPass(THREE.PassNode.COLOR, this.smokeScene, this.camera, { samples: 0, depthBuffer: false });
    this.smokePass.setResolutionScale(SMOKE_SCALE.high);
    const smoke = this.smokePass.getTextureNode('output');
    const over = (under: Vec4) => vec4(add(under.rgb.mul(float(1).sub(smoke.a)), smoke.rgb), 1);
    const occlusion = ao(depth, scene.getTextureNode('normal'), this.camera);
    occlusion.resolutionScale = 0.5;
    occlusion.radius.value = 1.2;
    occlusion.thickness.value = 1.5;
    occlusion.scale.value = 1.4;
    // Sun shafts: the light marched through the shadow map, blended in away from depth edges
    const rays = godrays(depth, this.camera, this.sun);
    rays.raymarchSteps.value = 48;
    rays.density.value = RAYS_CLEAR;
    rays.maxDensity.value = 0.25;
    rays.distanceAttenuation.value = 1.5;
    this.rays = rays;
    const shafts = depthAwareBlend(colour, rays.getTextureNode(), depth, this.camera, { blendColor: this.rayColour, edgeRadius: 2, edgeStrength: 2 });
    // The game camera's clip planes (in these passes cameraNear/cameraFar are the full-screen
    // quad's own camera, 0 and 1: every pixel came out a metre away)
    const near = uniform(this.camera.near);
    const far = uniform(this.camera.far);
    // Sun rays: the sky's light near the sun where nothing stands in front of it (the sky
    // doesn't write depth: it keeps the far plane)
    const viewZ = perspectiveDepthToViewZ(depth, near, far);
    const skyShare = step(far.mul(0.999), viewZ.negate());
    const fromSun = uv().sub(this.sunScreen).mul(vec2(this.aspect, 1)).length();
    // Only the sky brighter than its surroundings (the sun's glow, lit cloud edges) makes rays,
    // falling off away from the sun: a plain bright sky makes haze, not shafts
    const bright = colour.rgb.sub(vec3(0.8)).max(0);
    const skyGlow = bright.mul(skyShare).mul(pow(float(1).sub(fromSun.div(0.6)).max(0), 3));
    const mask = rtt(vec4(skyGlow.min(vec3(40)), 1), null, null, { resolutionScale: 0.5 });
    const SAMPLES = 64;
    const radial = Fn(() => {
      const at = uv().toVar();
      const delta = uv().sub(this.sunScreen).mul(0.9 / SAMPLES);
      const sum = mask.sample(uv()).rgb.toVar();
      const falloff = float(1).toVar();
      Loop(SAMPLES, () => {
        at.subAssign(delta);
        sum.addAssign(mask.sample(at).rgb.mul(falloff).mul(0.4));
        falloff.mulAssign(0.965);
      });
      return vec4(sum.mul(0.05).mul(this.sunFade).mul(this.sunRays), 1);
    })();
    const sunRays = rtt(radial, null, null, { resolutionScale: 0.5 });
    // Laid on after the smoke, dimmed where it's thick
    const raysOver = vec4(sunRays.rgb.mul(float(1).sub(smoke.a.mul(0.7))), 0);
    // The tool in hand (under a metre away) gets no occlusion: at the AO's metre-wide reach it
    // darkens itself all over, its lit parts too
    const unoccluded = (z: THREE.Node<'float'>) => step(z.negate(), float(HAND_REACH));
    const litSharp = over(vec4(shafts.rgb.mul(occlusion.getTextureNode().r.max(unoccluded(viewZ))), 1));
    // Depth of field, gathered from what's already drawn (the scene, its occlusion, the smoke):
    // a disc of taps sized by the blur at this pixel, each weighted by its own blur so what's
    // in focus doesn't smear over the background behind it.
    const cocAt = (at: THREE.Node<'vec2'>) => {
      // Background only: sharp up to the focus distance, softening over focalRange beyond it
      // (nothing nearer is ever blurred: the blur draws the eye in, it doesn't hide the action)
      const z = perspectiveDepthToViewZ(depth.sample(at).r, near, far).negate();
      return smoothstep(float(0), this.focalRange, z.sub(this.focus));
    };
    const occTex = occlusion.getTextureNode();
    const litAt = (at: THREE.Node<'vec2'>) => {
      const s2 = smoke.sample(at);
      const occ = occTex.sample(at).r.max(unoccluded(perspectiveDepthToViewZ(depth.sample(at).r, near, far)));
      return colour.sample(at).rgb.mul(occ).mul(float(1).sub(s2.a)).add(s2.rgb);
    };
    const TAPS = 24;
    const disc = Array.from({ length: TAPS }, (_, i) => {
      const r = Math.sqrt((i + 0.5) / TAPS);
      const a = i * 2.39996323;
      return [r * Math.cos(a), r * Math.sin(a)] as const;
    });
    const here = uv();
    const cocHere = cocAt(here);
    const litBlur = Fn(() => {
      const radius = vec2(cocHere.mul(this.bokeh)).mul(this.pixel);
      const sum = vec3(0).toVar();
      const weight = float(0.0001).toVar();
      for (const [x, y] of disc) {
        const at = here.add(vec2(x, y).mul(radius));
        const w = cocAt(at).max(0.04);
        sum.addAssign(litAt(at).mul(w));
        weight.addAssign(w);
      }
      return sum.div(weight);
    })();
    const blend = smoothstep(float(0.05), float(0.4), cocHere).mul(step(float(0.01), this.bokeh));
    const lit = vec4(select(blend.greaterThan(0.001), mix(litSharp.rgb, litBlur, blend), litSharp.rgb), 1).add(raysOver);
    this.bloomPass = bloom(lit, 0.35, 0.4, 0.85);
    // A warm grade: highlights a touch gold, shadows a touch blue, a little more colour
    const hdr = add(lit, this.bloomPass).rgb;
    const lum = hdr.dot(vec3(0.2126, 0.7152, 0.0722));
    const graded = mix(vec3(lum), hdr, 1.12).mul(mix(vec3(0.94, 0.98, 1.06), vec3(1.05, 1.0, 0.93), smoothstep(0, 1.2, lum)));
    // Depth of field: the tool in hand (within a metre and a half) counts as in focus. Its
    // passes read the scene's depth as a plain texture (through the pass's own node it came
    // through as zero there: everything counted as in hand, and nothing blurred)
    const focused = graded;
    // After tone mapping: a soft vignette and fine film grain, then anti-aliasing
    const vignette = smoothstep(float(1.25), float(0.35), uv().sub(0.5).length().mul(1.6)).mul(0.28).add(0.72);
    const finished = film(vec4(renderOutput(vec4(focused, 1)).rgb.mul(vignette), 1), float(0.035));
    this.outputs = { high: smaa(finished), low: fxaa(renderOutput(over(colour))) };
    this.post.outputColorTransform = false;
    this.post.outputNode = this.outputs.high;
  }

  /**
   * Start the GPU. The device asks for big storage buffers (the solver's contacts), but no
   * bigger than this adapter offers: asking for more than it has fails the whole device (a
   * phone offers far less than a desktop, and the city never started there). The solver sizes
   * its buffers to what the device has. `onTextures(share)` as the surface textures come in.
   */
  async init(onTextures: (share: number) => void = () => {}): Promise<GPUDevice> {
    const gpu = (navigator as Navigator & { gpu?: GPU }).gpu;
    if (!gpu && !isSecureContext) throw new Error(`WebGPU needs a secure page (https:// or localhost), and this one is ${location.protocol}//${location.host}`);
    if (!gpu) throw new Error('WebGPU is not available in this browser');
    const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('No WebGPU adapter: the browser has WebGPU but could not find a usable GPU');
    const want = 1 << 30;
    const backend = this.renderer.backend as unknown as { parameters: { requiredLimits: Record<string, number> } };
    backend.parameters.requiredLimits = {
      maxStorageBufferBindingSize: Math.min(want, adapter.limits.maxStorageBufferBindingSize),
      maxBufferSize: Math.min(want, adapter.limits.maxBufferSize),
    };
    this.adapterInfo = adapter.info ? `${adapter.info.vendor}|${adapter.info.architecture}|${adapter.info.device}|${adapter.info.description}` : 'unknown';
    const [, surfaces] = await Promise.all([
      this.renderer.init(),
      loadSurfaces((done, total) => onTextures(done / total)).catch((e: unknown) => {
        console.warn('Surface textures not loaded; drawing procedural surfaces', e);
        return null;
      }),
    ]);
    this.surfaces = surfaces;
    return (this.renderer.backend as unknown as { device: GPUDevice }).device;
  }

  /**
   * How dusty the air is (0 clear, 1 thick after a big blast): the sun shafts thicken with it,
   * so light cuts through the dust of a collapse.
   */
  set dust(amount: number) {
    this.rays.density.value = RAYS_CLEAR + (RAYS_DUSTY - RAYS_CLEAR) * Math.min(1, amount);
  }

  /** Bloom strength (0.35 normally). */
  set glow(strength: number) {
    this.bloomPass.strength.value = strength;
  }

  setQuality(quality: RenderQuality): void {
    this.quality = quality;
    this.post.outputNode = quality === 'low' ? this.outputs.low : this.outputs.high;
    this.post.needsUpdate = true;
    this.sun.shadow.mapSize.setScalar(quality === 'high' ? 4096 : 2048);
    this.rays.raymarchSteps.value = quality === 'high' ? 48 : 24;
    this.smokePass.setResolutionScale(SMOKE_SCALE[quality]);
    this.sun.shadow.map?.dispose();
    this.sun.shadow.map = null;
    this.setResolutionScale(this.resolutionScale);
  }

  /**
   * Render resolution as a share of the quality's (1: full): the game lowers it while a thick
   * cloud of dust slows frames (fill rate is what a blast costs), and raises it again after.
   */
  private resolutionScale = 1;
  private width = 1;
  private height = 1;
  /** The device's own resolution share (calibrate.ts: below 1 where even low quality was slow). */
  baseResolution = 1;
  setResolutionScale(scale: number): void {
    this.resolutionScale = scale;
    const base = Math.min(devicePixelRatio, this.quality === 'high' ? 1.5 : 1) * this.baseResolution;
    const ratio = Math.max(0.5, base * scale);
    if (Math.abs(this.renderer.getPixelRatio() - ratio) < 1e-3) return;
    this.renderer.setPixelRatio(ratio);
    this.renderer.setSize(this.width, this.height, false);
  }

  /**
   * The voxels: `capacity` bodies drawn from a buffer the solver will own (returned), coloured
   * per voxel. Body index b is voxel voxelOf[b] (-1: the ground or a spare, not drawn as a voxel).
   */
  attachVoxels(capacity: number, city: City): GPUBuffer {
    if (this.voxels) {
      this.scene.remove(this.voxels);
      this.voxels.geometry.dispose();
      (this.voxels.material as THREE.Material).dispose();
    }
    const n = capacity;
    this.bodies = new THREE.StorageInstancedBufferAttribute(new Float32Array(n * BODY_FLOATS), 4);
    this.paints = new THREE.StorageInstancedBufferAttribute(new Float32Array(n * 4), 4);
    this.links = new THREE.StorageInstancedBufferAttribute(new Float32Array(n * 4).fill(-1), 4);
    this.heats = new THREE.StorageInstancedBufferAttribute(new Float32Array(n).fill(-1e4), 1);
    this.homes = new THREE.StorageInstancedBufferAttribute(new Float32Array(n * 4), 4);
    const bodies = storage(this.bodies, 'vec4', (n * BODY_FLOATS) / 4).toReadOnly();
    const paints = storage(this.paints, 'vec4', n).toReadOnly();
    const links = storage(this.links, 'vec4', n).toReadOnly();
    const body = instanceIndex;
    // A voxel carried by a rigid section (physics.ts proxy) is drawn at the section's pose plus
    // its offset in the section; its own body waits parked meanwhile
    const link = links.element(body);
    const linked = link.w.greaterThanEqual(0);
    const parent = select(linked, uint(link.w), body);
    const base = body.mul(BODY_FLOATS / 4);
    const parentBase = parent.mul(BODY_FLOATS / 4);
    const ownPos = bodies.element(base.add(B_POS / 4));
    const parentPos = bodies.element(parentBase.add(B_POS / 4));
    const rot = bodies.element(parentBase.add(B_ROT / 4)) as unknown as Vec4;
    const pos = select(linked, vec4(rotate(rot, link.xyz).add(parentPos.xyz), 1), ownPos) as unknown as Vec4;
    const size = bodies.element(base.add(B_SIZE / 4));
    const paint = paints.element(body);
    // The ground and bodies out of play aren't drawn: folded to a point
    const hidden = select(linked, parentPos.z.lessThan(PARKED_BELOW), ownPos.z.lessThan(PARKED_BELOW).or(size.x.greaterThan(50))).or(paint.w.lessThan(0));
    // Clear glass is drawn by a mesh of its own (setGlass): see-through, after the rest
    const clear = abs(mod(paint.w, 16).sub(Mat.Glass)).lessThan(0.5);
    const local = positionGeometry.mul(size.xyz);
    const material = new THREE.MeshStandardNodeMaterial();
    material.positionNode = select(hidden.or(clear), pos.xyz, rotate(rot, local).add(pos.xyz));
    this.nodes = { bodies, paints, links };
    const burnt = storage(this.heats, 'float', n).toReadOnly().element(body) as unknown as Float;
    const home = storage(this.homes, 'vec4', n).toReadOnly().element(body).xyz as unknown as Vec3;
    this.shadeVoxels(material, paint, body, size.xyz, home, rot, local, burnt);
    const geometry = new THREE.InstancedBufferGeometry();
    const box = new THREE.BoxGeometry(1, 1, 1);
    geometry.index = box.index;
    geometry.setAttribute('position', box.getAttribute('position'));
    geometry.setAttribute('normal', box.getAttribute('normal'));
    geometry.instanceCount = n;
    const mesh = new THREE.Mesh(geometry, material);
    mesh.frustumCulled = false;
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    this.scene.add(mesh);
    this.voxels = mesh;
    this.setPaints(city, null);
    const backend = this.renderer.backend as unknown as { createStorageAttribute(a: THREE.BufferAttribute): void; get(o: object): { buffer?: GPUBuffer } };
    backend.createStorageAttribute(this.bodies);
    const buffer = backend.get(this.bodies).buffer;
    if (!buffer) throw new Error('Three.js did not create a GPU buffer for the voxels');
    return buffer;
  }

  /**
   * Bodies carried by a section (`parent`, a body index) at `offsets` (xyz each, in the
   * section's frame), or let go (parent -1).
   */
  setLinks(bodies: ArrayLike<number>, parent: number, offsets?: ArrayLike<number>): void {
    if (!this.links) return;
    const a = this.links.array as Float32Array;
    for (let k = 0; k < bodies.length; k++) {
      const o = 4 * bodies[k];
      if (parent < 0) a.set([0, 0, 0, -1], o);
      else a.set([offsets![3 * k], offsets![3 * k + 1], offsets![3 * k + 2], parent], o);
    }
    this.links.needsUpdate = true;
  }

  /** Balls thrown (bodies `ids`, the spares): drawn as dark steel spheres from the same buffer. */
  attachBalls(ids: number[]): void {
    if (!this.bodies) return;
    this.balls?.removeFromParent();
    const n = this.bodies.count / (BODY_FLOATS / 4);
    const bodies = storage(this.bodies, 'vec4', (n * BODY_FLOATS) / 4).toReadOnly();
    const which = storage(new THREE.StorageInstancedBufferAttribute(Uint32Array.from(ids), 1), 'uint', ids.length).toReadOnly();
    const base = which.element(instanceIndex).mul(BODY_FLOATS / 4);
    const pos = bodies.element(base.add(B_POS / 4));
    const rot = bodies.element(base.add(B_ROT / 4)) as unknown as Vec4;
    const size = bodies.element(base.add(B_SIZE / 4));
    const material = new THREE.MeshStandardNodeMaterial({ color: 0x2e3034, metalness: 0.85, roughness: 0.4, flatShading: true });
    const hidden = pos.z.lessThan(PARKED_BELOW);
    material.positionNode = select(hidden, pos.xyz, rotate(rot, positionGeometry.mul(size.x)).add(pos.xyz));
    const source = new THREE.IcosahedronGeometry(0.5, 4);
    const geometry = new THREE.InstancedBufferGeometry();
    geometry.index = source.index;
    geometry.setAttribute('position', source.getAttribute('position'));
    geometry.setAttribute('normal', source.getAttribute('normal'));
    geometry.instanceCount = ids.length;
    const mesh = new THREE.Mesh(geometry, material);
    mesh.frustumCulled = false;
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    this.scene.add(mesh);
    this.balls = mesh;
  }
  private balls: THREE.Mesh | null = null;

  /** Bodies burned by the laser at `time` (the world's clock, s): they glow and cool (shadeVoxels). */
  heat(bodies: ArrayLike<number>, time: number): void {
    if (!this.heats || !bodies.length) return;
    const a = this.heats.array as Float32Array;
    for (let k = 0; k < bodies.length; k++) if (bodies[k] >= 0 && bodies[k] < a.length) a[bodies[k]] = time;
    this.heats.needsUpdate = true;
  }

  /** Each body's colour (linear) and material; `voxelOf` maps bodies to voxels (null: before the solver exists, all hidden). */
  setPaints(city: City, voxelOf: Int32Array | null): void {
    if (!this.paints) return;
    const a = this.paints.array as Float32Array;
    a.fill(-1);
    const homes = this.homes ? (this.homes.array as Float32Array) : null;
    if (voxelOf) {
      const c = new THREE.Color();
      for (let b = 0; b < voxelOf.length && b < a.length / 4; b++) {
        const v = voxelOf[b];
        if (v < 0) continue;
        c.setHex(city.color[v]);
        a.set([c.r, c.g, c.b, city.material[v] + 16 * city.exposed[v]], b * 4);
        if (homes) homes.set(city.position.subarray(3 * v, 3 * v + 3), b * 4);
      }
      this.setGlass(city, voxelOf);
    }
    this.paints.needsUpdate = true;
    if (this.homes) this.homes.needsUpdate = true;
  }

  /**
   * The clear glass: its bodies (those of `voxelOf` that are glass) drawn from the same buffers
   * as the voxels, each only by its faces open to the air (a pane's two sides, not the faces
   * between its voxels), after everything solid, so what's behind shows through: the rooms,
   * the floors, the far windows. Colour and coverage are premultiplied, as glass works: it reflects the sky (Fresnel: faint head on, a mirror at a
   * glancing angle) on top of what it lets through, darkened a little and tinted.
   */
  private setGlass(city: City, voxelOf: Int32Array): void {
    if (this.glass) {
      this.scene.remove(this.glass);
      this.glass.geometry.dispose();
      (this.glass.material as THREE.Material).dispose();
      this.glass = null;
    }
    const ids: number[] = [];
    for (let b = 0; b < voxelOf.length; b++) {
      const v = voxelOf[b];
      if (v >= 0 && city.material[v] === Mat.Glass && city.exposed[v]) ids.push(b);
    }
    if (!ids.length || !this.nodes) return;
    const { bodies, paints, links } = this.nodes;
    const which = storage(new THREE.StorageInstancedBufferAttribute(Uint32Array.from(ids), 1), 'uint', ids.length).toReadOnly();
    const body = which.element(instanceIndex);
    const link = links.element(body);
    const linked = link.w.greaterThanEqual(0);
    const parent = select(linked, uint(link.w), body);
    const ownPos = bodies.element(body.mul(BODY_FLOATS / 4).add(B_POS / 4));
    const parentPos = bodies.element(parent.mul(BODY_FLOATS / 4).add(B_POS / 4));
    const rot = bodies.element(parent.mul(BODY_FLOATS / 4).add(B_ROT / 4)) as unknown as Vec4;
    const pos = select(linked, vec4(rotate(rot, link.xyz).add(parentPos.xyz), 1), ownPos) as unknown as Vec4;
    const size = bodies.element(body.mul(BODY_FLOATS / 4).add(B_SIZE / 4));
    const paint = paints.element(body);
    const gone = select(linked, parentPos.z.lessThan(PARKED_BELOW), ownPos.z.lessThan(PARKED_BELOW)).or(paint.w.lessThan(0));
    // This vertex's face (bit of the open-face mask: -x, +x, -y, +y, -z, +z), and whether it's open
    const n = normalGeometry;
    const sideX = abs(n.x).greaterThan(0.5);
    const top = abs(n.z).greaterThan(0.5);
    const bit = select(sideX, sign(n.x).mul(0.5).add(0.5), select(top, sign(n.z).mul(0.5).add(4.5), sign(n.y).mul(0.5).add(2.5)));
    const open = mod(floor(floor(paint.w.div(16)).div(pow(float(2), bit))), 2).greaterThan(0.5);
    const material = new THREE.MeshPhysicalNodeMaterial({ transparent: true });
    material.blending = THREE.CustomBlending;
    material.blendSrc = THREE.OneFactor;
    material.blendDst = THREE.OneMinusSrcAlphaFactor;
    material.blendSrcAlpha = THREE.OneFactor;
    material.blendDstAlpha = THREE.OneMinusSrcAlphaFactor;
    material.positionNode = select(gone.or(open.not()), pos.xyz, rotate(rot, positionGeometry.mul(size.xyz)).add(pos.xyz));
    const normal = rotate(rot, n);
    material.normalNode = transformNormalToView(normal) as unknown as THREE.Node<'vec3'>;
    // A little grime: the reflection blurs a touch here and there
    const grime = mx_noise_float(varying(pos.xyz.add(rotate(rot, positionGeometry.mul(size.xyz)))).mul(1.7)).mul(0.5).add(0.5);
    // Office glazing is coated: it reflects some five times what plain glass does (F0 about
    // 0.2, not 0.04), tinted by the pane's colour
    material.colorNode = paint.rgb.mul(0.05) as unknown as THREE.Node<'color'>;
    material.metalnessNode = float(0);
    material.specularColorNode = mix(vec3(1), paint.rgb.mul(2.5), 0.5).mul(5) as unknown as THREE.Node<'color'>;
    material.roughnessNode = grime.mul(grime).mul(0.12).add(0.02);
    const facing = abs(normalize(cameraPosition.sub(positionWorld)).dot(normalize(varying(normal) as unknown as Vec3)));
    // (Clamped: the cosine rounds a hair past 1 head on, and a power of a negative is NaN)
    material.opacityNode = pow(float(1).sub(facing).max(0), 5).mul(0.55).add(0.4).add(grime.mul(0.06));
    // It writes its depth and normal, so the ambient occlusion sees a pane (with the room's
    // depth under its normal, the occlusion came out as noise; a zero normal, to leave the
    // room's, blacked the glass out: the normal target takes no blending)
    const source = new THREE.BoxGeometry(1, 1, 1);
    const geometry = new THREE.InstancedBufferGeometry();
    geometry.index = source.index;
    geometry.setAttribute('position', source.getAttribute('position'));
    geometry.setAttribute('normal', source.getAttribute('normal'));
    geometry.instanceCount = ids.length;
    const mesh = new THREE.Mesh(geometry, material);
    mesh.frustumCulled = false;
    mesh.renderOrder = 1;
    this.scene.add(mesh);
    this.glass = mesh;
  }

  /**
   * The voxels' surfaces, in each voxel's own frame: each face textured as what it is (paint.w:
   * material + 16 × the faces open to the air when built), a face bared since as a break, the
   * texture tied to the voxel's centre so it stays put as it tumbles; glass, steel and lit
   * windows procedural. Without textures, procedural brick courses and concrete streaks.
   */
  private shadeVoxels(material: THREE.MeshStandardNodeMaterial, paint: Vec4, body: THREE.Node<'uint'>, size: Vec3, home: Vec3, rot: Vec4, local: Vec3, burnt: Float): void {
    const id = mod(paint.w, 16);
    const mask = floor(paint.w.div(16));
    const is = (m: Mat) => abs(id.sub(m)).lessThan(0.5);
    const lp = varying(positionGeometry) as unknown as Vec3;
    const ln = varying(normalGeometry) as unknown as Vec3;
    const seed = varying(hash(float(body)).mul(97)) as unknown as Float;
    // The face's own coordinates: `across` along it, `up` the voxel's z (or y on top and bottom)
    const top = abs(ln.z).greaterThan(0.5);
    const sideX = abs(ln.x).greaterThan(0.5);
    const across = select(sideX, lp.y, lp.x) as unknown as Float;
    const up = select(top, lp.y, lp.z) as unknown as Float;
    const edge = min(float(0.5).sub(abs(across)), float(0.5).sub(abs(up)));
    const noise = (s: number, o = 0) => mx_noise_float(lp.mul(s).add(seed).add(o));
    const rgb = paint.rgb;
    const glassy = is(Mat.Glass).or(is(Mat.LitGlass));

    // Concrete and plaster: mottled, with streaks running down the faces
    const streaks = mx_noise_float(vec3(across.mul(6).add(seed), up.mul(0.8), seed)).mul(0.5).add(0.5);
    const concrete = rgb.mul(noise(3).mul(0.1).add(0.95)).mul(mix(float(1), streaks.mul(0.18).add(0.86), select(top, float(0), float(1))));
    // Brick: four courses a metre, half-brick bond, mortar joints, every brick its own shade
    const row = floor(up.add(0.5).mul(COURSES));
    const u = across.add(0.5).mul(BRICKS).add(mod(row, 2).mul(0.5));
    const brickId = row.mul(7).add(floor(u)).add(seed.mul(13));
    const mortar = step(fract(up.add(0.5).mul(COURSES)), float(0.08)).max(step(fract(u), float(0.05)));
    const brick = mix(rgb.mul(hash(brickId).mul(0.35).add(0.78)), vec3(0.58, 0.55, 0.5), mortar.mul(select(top, float(0), float(1))));
    // Glass in a slim frame; roof gravel; steel
    const frame = (VOXEL >= 0.5 ? step(edge, float(0.07)) : float(0)).max(select(top, float(1), float(0)));
    const gravel = rgb.mul(hash(floor(lp.mul(18)).dot(vec3(1, 37, 113)).add(seed)).mul(0.4).add(0.8));
    const glassColour = mix(rgb.mul(0.35), vec3(0.2, 0.21, 0.22), frame);
    // Leaves: each voxel its own shade, mottled, no texture
    const leaf = rgb.mul(noise(9).mul(0.25).add(0.9));
    // Wood: grain running along the piece (its long way is x or y), rings of darker late wood
    const grainAt = select(abs(ln.x).greaterThan(0.5), lp.yz, lp.xz);
    const grain = mx_noise_float(vec3(grainAt.x.mul(1.5).add(seed), grainAt.y.mul(14), seed)).mul(0.5).add(0.5);
    const wood = rgb.mul(grain.mul(0.35).add(0.78)).mul(select(top, float(1.05), float(1)));
    let procedural = concrete as unknown as Vec3;
    procedural = select(is(Mat.Leaf), leaf, procedural) as unknown as Vec3;
    procedural = select(is(Mat.Wood), wood, procedural) as unknown as Vec3;
    procedural = select(is(Mat.Brick), brick, procedural) as unknown as Vec3;
    procedural = select(is(Mat.Roof), gravel, procedural) as unknown as Vec3;
    procedural = select(glassy, glassColour, procedural) as unknown as Vec3;

    let albedo = procedural;
    let normal = ln;
    let roughness = select(is(Mat.Roof), float(0.95), float(0.82)) as unknown as Float;
    if (this.surfaces) {
      // Which face (bit of the mask: -x, +x, -y, +y, -z, +z), and whether it was open when built
      const [sx, sy, sz] = [sign(ln.x), sign(ln.y), sign(ln.z)];
      const bit = select(sideX, sx.mul(0.5).add(0.5), select(top, sz.mul(0.5).add(4.5), sy.mul(0.5).add(2.5)));
      const finished = mod(floor(mask.div(pow(float(2), bit))), 2).greaterThan(0.5);
      // Texture coordinates on the face from the voxel's centre and the point on it, the
      // frame (tangent along u, bitangent along v) right-handed about the face's normal
      // (From where the voxel was built, not where it is: on a falling piece the texture stays on its faces)
      const p = varying(home.add(local)) as unknown as Vec3;
      const st = select(sideX, vec2(p.y.mul(sx), p.z), select(top, vec2(p.x.mul(sz), p.y), vec2(p.x.mul(sy).negate(), p.z)));
      const tangent = select(sideX, vec3(0, sx, 0), select(top, vec3(sz, 0, 0), vec3(sy.negate(), 0, 0)));
      const bitangent = select(top, vec3(0, 1, 0), vec3(0, 0, 1));
      // The layer: the finished face's own surface, a break's rubble (bare concrete for concrete)
      // (Chosen with integer selects: WGSL wants the layer as an integer, and a float converted
      // late came through as a float.)
      const brick = is(Mat.Brick);
      const plaster = is(Mat.Plaster);
      const rubble = brick.or(plaster).or(is(Mat.Trim));
      const which = select(finished, select(brick, int(L_BRICK), select(plaster, int(L_PLASTER), int(L_CONCRETE))), select(rubble, int(L_BROKEN), int(L_CONCRETE)));
      const metres = select(
        finished,
        select(brick, float(SETS[L_BRICK].metres), select(plaster, float(SETS[L_PLASTER].metres), float(SETS[L_CONCRETE].metres))),
        select(rubble, float(SETS[L_BROKEN].metres), float(SETS[L_CONCRETE].metres)),
      );
      const scaled = st.div(metres);
      const diffuse = texture(this.surfaces.diffuse, scaled).depth(which);
      const bump = texture(this.surfaces.normal, scaled).depth(which);
      const arm = texture(this.surfaces.arm, scaled).depth(which);
      // The voxel's colour is its material's base shifted a little: keep the shift as a tint
      let base = baseColour(Mat.Concrete);
      for (const m of [Mat.Brick, Mat.Plaster, Mat.Trim, Mat.Roof, Mat.Steel] as Mat[]) base = select(is(m), baseColour(m), base) as unknown as Vec3;
      const tint = rgb.div(base.max(0.05));
      const shade = arm.r.mul(0.5).add(0.5).mul(select(finished, float(1), float(0.8))).mul(select(is(Mat.Roof), float(0.5), float(1)));
      const textured = glassy.or(is(Mat.Steel)).or(is(Mat.Leaf)).or(is(Mat.Wood)).not();
      albedo = select(textured, diffuse.rgb.mul(tint).mul(shade), procedural) as unknown as Vec3;
      const n = bump.rgb.mul(2).sub(1);
      const bumped = normalize(tangent.mul(n.x).add(bitangent.mul(n.y)).add(ln.mul(n.z)));
      normal = select(textured, bumped, ln) as unknown as Vec3;
      roughness = select(textured, clamp(arm.g, 0.4, 1), roughness) as unknown as Float;
    }
    // Edges darkened a touch (as on the demo's blocks), less on glass
    material.colorNode = albedo.mul(mix(edgeShade(lp, size, 0.05), float(1), select(glassy, float(0.7), float(0.35)))) as unknown as THREE.Node<'color'>;
    material.normalNode = transformNormalToView(rotate(rot, normal)) as unknown as THREE.Node<'vec3'>;
    material.metalnessNode = select(glassy, float(0.9), select(is(Mat.Steel), float(0.8), float(0)));
    material.roughnessNode = select(glassy, float(0.06), select(is(Mat.Steel), float(0.38), roughness));
    // Lit windows: warm light from inside, some rooms brighter than others, blinds drawn in some
    const room = hash(seed.add(3)).mul(0.8).add(0.4);
    const blind = step(float(0.35), fract(up.add(0.5).mul(1).add(hash(seed.add(5))))).mul(0.5).add(0.5);
    // Neon (signs, the canopy's fascia, traffic signals) glows its own colour
    const lit = select(is(Mat.LitGlass).and(frame.lessThan(0.5)), rgb.mul(room.mul(blind).mul(2.2)), vec3(0));
    // Where the laser cut: white hot at first, cooling through orange to a dull red ember
    const age = this.heatClock.sub(burnt).max(0);
    const heat = exp(age.mul(-1 / 0.8)).mul(6).add(exp(age.mul(-1 / 5)).mul(0.7));
    const glow = mix(vec3(1, 0.2, 0.03), vec3(1, 0.72, 0.4), exp(age.mul(-2.5))).mul(heat);
    material.emissiveNode = select(is(Mat.Neon), rgb.mul(1.4), lit).add(glow) as unknown as THREE.Node<'color'>;
  }

  /** The ground: streets, pavements and plazas drawn from world position (city grid of world.ts). */
  private ground(): THREE.Mesh {
    const material = new THREE.MeshStandardNodeMaterial({ roughness: 0.9, metalness: 0 });
    this.groundMaterial = material;
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(8000, 8000), material);
    mesh.receiveShadow = true;
    return mesh;
  }
  private groundMaterial: THREE.MeshStandardNodeMaterial | null = null;

  /** Draw the city's street grid on the ground. */
  setStreets(city: City): void {
    const material = this.groundMaterial!;
    const block = city.pitch - city.street;
    const half = (city.blocks * city.pitch - city.street) / 2;
    const p = positionWorld.xy;
    const q = p.add(half);
    const cell = floor(q.div(city.pitch));
    const f = q.sub(cell.mul(city.pitch));
    const inCity = cell.x.greaterThanEqual(0).and(cell.y.greaterThanEqual(0)).and(cell.x.lessThan(city.blocks)).and(cell.y.lessThan(city.blocks));
    const sx = f.x.sub(block);
    const sy = f.y.sub(block);
    const streetX = sx.greaterThanEqual(0);
    const streetY = sy.greaterThanEqual(0);
    const w = city.street;
    const kerb = 3;
    const roadX = streetX.and(sx.greaterThan(kerb)).and(sx.lessThan(w - kerb));
    const roadY = streetY.and(sy.greaterThan(kerb)).and(sy.lessThan(w - kerb));
    const road = roadX.or(roadY);
    const crossing = streetX.and(streetY);

    const n = mx_noise_float(vec3(p.mul(0.35), 0)).mul(0.5).add(0.5);
    // Fine grain up close, faded to its mean far off: finer than a pixel it only sparkles
    // (the distant fields and roads came out speckled)
    const fine = mix(float(0.5), mx_noise_float(vec3(p.mul(4), 1)).mul(0.5).add(0.5), smoothstep(float(90), float(20), positionWorld.sub(cameraPosition).length()));
    let asphalt = vec3(0.1, 0.095, 0.09).mul(n.mul(0.35).add(0.8)).mul(fine.mul(0.25).add(0.88)) as unknown as Vec3;
    let paving = vec3(0.5, 0.49, 0.46) as unknown as Vec3;
    let roadRough = float(0.92) as unknown as Float;
    if (this.surfaces) {
      // Real asphalt on the roads, cast concrete on the pavements and plazas, bumped
      const sa = p.div(SETS[L_ASPHALT].metres);
      const sc = p.div(SETS[L_CONCRETE].metres);
      const tar = texture(this.surfaces.diffuse, sa).depth(int(L_ASPHALT));
      const tarBump = texture(this.surfaces.normal, sa).depth(int(L_ASPHALT));
      const tarArm = texture(this.surfaces.arm, sa).depth(int(L_ASPHALT));
      const slabs = texture(this.surfaces.diffuse, sc).depth(int(L_CONCRETE));
      const slabBump = texture(this.surfaces.normal, sc).depth(int(L_CONCRETE));
      asphalt = tar.rgb.mul(0.8).mul(tarArm.r.mul(0.4).add(0.6)).mul(n.mul(0.2).add(0.9)) as unknown as Vec3;
      paving = slabs.rgb.mul(0.85) as unknown as Vec3;
      roadRough = clamp(tarArm.g, 0.5, 1) as unknown as Float;
      const onRoad = f.x.sub(block).greaterThan(kerb).and(f.x.sub(block).lessThan(w - kerb)).or(f.y.sub(block).greaterThan(kerb).and(f.y.sub(block).lessThan(w - kerb)));
      // The bump fades with distance (at a low sun it sparkled along the horizon)
      const bump = select(onRoad, tarBump.rgb, slabBump.rgb).mul(2).sub(1);
      const near = smoothstep(float(90), float(25), positionWorld.sub(cameraPosition).length());
      const bent = mix(vec3(0, 0, 1), vec3(bump.x, bump.y, bump.z.mul(2)), near.mul(select(inCity.or(streetX).or(streetY), float(1), float(0))));
      material.normalNode = transformNormalToView(normalize(bent)) as unknown as THREE.Node<'vec3'>;
    }
    // Lane markings: dashed centre lines and solid edges along each street, zebra crossings at junctions
    const line = (d: Float, width: number) => smoothstep(float(width), float(width * 0.6), abs(d));
    const dashX = step(fract(p.y.div(6)), float(0.5));
    const dashY = step(fract(p.x.div(6)), float(0.5));
    const along = select(streetX.and(streetY.not()), line(sx.sub(w / 2), 0.1).mul(dashX).max(line(sx.sub(kerb + 0.4), 0.07)).max(line(sx.sub(w - kerb - 0.4), 0.07)), float(0));
    const alongY = select(streetY.and(streetX.not()), line(sy.sub(w / 2), 0.1).mul(dashY).max(line(sy.sub(kerb + 0.4), 0.07)).max(line(sy.sub(w - kerb - 0.4), 0.07)), float(0));
    const zebra = (s: Float, t: Float) =>
      select(t.greaterThan(kerb + 0.3).and(t.lessThan(kerb + 2.6)).or(t.greaterThan(w - kerb - 2.6).and(t.lessThan(w - kerb - 0.3))), step(fract(s.div(1.2)), float(0.5)), float(0));
    const stripes = select(crossing, zebra(sx, sy).max(zebra(sy, sx)), float(0));
    const paint = clamp(along.max(alongY).max(stripes), 0, 1).mul(fine.mul(0.3).add(0.7));
    const roadColour = mix(asphalt, vec3(0.78, 0.78, 0.74), paint);
    // Pavements and plazas: slabs with joints
    const slab = (size: number) => {
      const g = fract(p.div(size));
      const joint = step(min(g.x, g.y), float(0.03)).max(step(float(0.97), max(g.x, g.y)));
      return mix(float(1), float(0.72), joint).mul(hash(floor(p.div(size)).dot(vec2(1, 57))).mul(0.12).add(0.92));
    };
    const pavement = paving.mul(slab(1.5)).mul(n.mul(0.2).add(0.9));
    const plaza = paving.mul(1.1).mul(slab(2)).mul(n.mul(0.2).add(0.9));
    const grass = mix(vec3(0.09, 0.16, 0.05), vec3(0.2, 0.26, 0.09), n).mul(fine.mul(0.3).add(0.8));
    const lot = select(inCity, plaza, grass);
    const colour = select(road, roadColour, select(streetX.or(streetY), pavement, lot));
    material.colorNode = colour as unknown as THREE.Node<'color'>;
    material.roughnessNode = select(road, roadRough.sub(paint.mul(0.2)), float(0.85));
  }

  /**
   * Draw the race track on the ground instead of streets (track.ts): `field` is its picture
   * (trackField: per texel of a `size` square over ±`extent` m, the signed distance to the
   * middle line, round-the-lap distance, corner and turn). Asphalt with white edge lines,
   * red-and-white kerbs in the bends, gravel run-off outside them, a chequered start line at
   * x = `startX` on the main straight, a paved pit lane, and mown grass everywhere else.
   */
  setTrack(field: Float32Array, size: number, extent: number, width: number, kerb: number, runoff: number, startX: number, pits: [number, number, number, number]): void {
    const material = this.groundMaterial!;
    // Half floats (filterable everywhere): distance, the kerb stripes' phase, corner × turn
    const half = new Uint16Array(size * size * 4);
    for (let t = 0; t < size * size; t++) {
      half[4 * t] = THREE.DataUtils.toHalfFloat(field[4 * t]);
      half[4 * t + 1] = THREE.DataUtils.toHalfFloat(Math.cos(Math.PI * field[4 * t + 1]));
      half[4 * t + 2] = THREE.DataUtils.toHalfFloat(field[4 * t + 2] * field[4 * t + 3]);
      half[4 * t + 3] = 0;
    }
    const tex = new THREE.DataTexture(half, size, size, THREE.RGBAFormat, THREE.HalfFloatType);
    tex.magFilter = THREE.LinearFilter;
    tex.minFilter = THREE.LinearFilter;
    tex.needsUpdate = true;
    const p = positionWorld.xy;
    const at = p.add(extent).div(2 * extent);
    const inside = at.x.greaterThan(0).and(at.y.greaterThan(0)).and(at.x.lessThan(1)).and(at.y.lessThan(1));
    const f = texture(tex, at);
    const sd = select(inside, f.r, float(30));
    const d = abs(sd);
    const bend = select(inside, f.b, float(0));
    const n = mx_noise_float(vec3(p.mul(0.35), 0)).mul(0.5).add(0.5);
    const far = positionWorld.sub(cameraPosition).length();
    const fine = mix(float(0.5), mx_noise_float(vec3(p.mul(4), 1)).mul(0.5).add(0.5), smoothstep(float(90), float(20), far));
    let asphalt = vec3(0.1, 0.095, 0.09).mul(n.mul(0.35).add(0.8)).mul(fine.mul(0.25).add(0.88)) as unknown as Vec3;
    let paving = vec3(0.5, 0.49, 0.46) as unknown as Vec3;
    const onTrack = d.lessThan(width / 2);
    if (this.surfaces) {
      const sa = p.div(SETS[L_ASPHALT].metres);
      const sc = p.div(SETS[L_CONCRETE].metres);
      const tar = texture(this.surfaces.diffuse, sa).depth(int(L_ASPHALT));
      const tarBump = texture(this.surfaces.normal, sa).depth(int(L_ASPHALT));
      const tarArm = texture(this.surfaces.arm, sa).depth(int(L_ASPHALT));
      asphalt = tar.rgb.mul(0.8).mul(tarArm.r.mul(0.4).add(0.6)).mul(n.mul(0.2).add(0.9)) as unknown as Vec3;
      paving = texture(this.surfaces.diffuse, sc).depth(int(L_CONCRETE)).rgb.mul(0.85) as unknown as Vec3;
      const bump = tarBump.rgb.mul(2).sub(1);
      const near = smoothstep(float(90), float(25), far);
      const bent = mix(vec3(0, 0, 1), vec3(bump.x, bump.y, bump.z.mul(2)), near.mul(select(onTrack, float(1), float(0))));
      material.normalNode = transformNormalToView(normalize(bent)) as unknown as THREE.Node<'vec3'>;
    }
    const line = (dd: Float, w: number) => smoothstep(float(w), float(w * 0.6), abs(dd));
    const edge = line(d.sub(width / 2 - 0.45), 0.12);
    // The start line: a chequered band across the main straight
    const startBand = abs(p.x.sub(startX)).lessThan(1.2).and(p.y.lessThan(-60)).and(onTrack);
    const checker = mod(floor(p.x.div(0.6)).add(floor(p.y.div(0.6))), float(2));
    let track = mix(asphalt, vec3(0.8, 0.8, 0.76), edge.mul(fine.mul(0.3).add(0.7))) as unknown as Vec3;
    track = select(startBand, mix(vec3(0.04, 0.04, 0.04), vec3(0.85, 0.85, 0.82), checker), track) as unknown as Vec3;
    // Kerbs: a metre of red, a metre of white, in the bends
    const isKerb = d.greaterThanEqual(width / 2).and(d.lessThan(width / 2 + kerb)).and(abs(bend).greaterThan(0.5));
    const stripe = step(float(0), f.g);
    const kerbColour = mix(vec3(0.62, 0.07, 0.05), vec3(0.85, 0.84, 0.8), stripe).mul(fine.mul(0.2).add(0.85));
    // Gravel run-off on the outside of the bends (the side the car would slide off)
    const outside = sd.mul(bend).lessThan(0);
    const isGravel = outside.and(abs(bend).greaterThan(0.3)).and(d.lessThan(width / 2 + kerb + runoff)).and(d.greaterThanEqual(width / 2 + kerb));
    const gravel = vec3(0.56, 0.5, 0.4).mul(mx_noise_float(vec3(p.mul(2.5), 3)).mul(0.25).add(0.85)).mul(fine.mul(0.3).add(0.8));
    // Grass, mown in stripes
    const mown = step(fract(p.x.add(p.y.mul(0.2)).div(12)), float(0.5));
    const grass = mix(vec3(0.09, 0.16, 0.05), vec3(0.2, 0.26, 0.09), n).mul(mix(float(0.88), float(1.08), mown)).mul(fine.mul(0.3).add(0.8));
    // The pit lane: paved, behind the pit wall
    const inPits = p.x.greaterThan(pits[0]).and(p.x.lessThan(pits[2])).and(p.y.greaterThan(pits[1])).and(p.y.lessThan(pits[3]));
    const colour = select(onTrack, track, select(isKerb, kerbColour, select(isGravel, gravel, select(inPits, paving, grass))));
    material.colorNode = colour as unknown as THREE.Node<'color'>;
    material.roughnessNode = select(onTrack.or(isKerb), float(0.8), float(0.9));
    material.needsUpdate = true;
  }

  /**
   * Reflections and sky light: a dome from the sky's own colours (its horizon, a deeper blue
   * overhead, the sun's glow and disc, dim ground below), prefiltered for rough and smooth
   * surfaces. The real sky can't be captured: its clouds are drawn from the player's camera.
   */
  setEnvironment(horizon: THREE.Color, sunColour: THREE.Color, sunIntensity: number): void {
    const env = new THREE.Scene();
    const dome = new THREE.MeshBasicNodeMaterial({ side: THREE.BackSide });
    const dir = normalize(positionLocal);
    const zenith = vec3(0.16, 0.32, 0.62);
    const sky = mix(vec3(horizon.r, horizon.g, horizon.b), zenith, smoothstep(0.0, 0.7, dir.z));
    const toSun = dir.dot(vec3(this.sunDirection.x, this.sunDirection.y, this.sunDirection.z)).max(0);
    const sun = vec3(sunColour.r, sunColour.g, sunColour.b).mul(toSun.pow(8).mul(0.6).add(toSun.pow(400).mul(60)).mul(sunIntensity / 3));
    const ground = vec3(0.09, 0.085, 0.08);
    dome.colorNode = mix(ground, sky.add(sun), smoothstep(-0.08, 0.02, dir.z)) as unknown as THREE.Node<'color'>;
    env.add(new THREE.Mesh(new THREE.SphereGeometry(10, 64, 32), dome));
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    const target = pmrem.fromScene(env, 0.01);
    this.scene.environment?.dispose();
    this.scene.environment = target.texture;
    this.scene.environmentIntensity = 0.55;
    pmrem.dispose();
  }

  /** A sky HDRI (public/hdri), loaded once. */
  loadSky(file: string): Promise<THREE.Texture> {
    let sky = this.skies.get(file);
    if (!sky) this.skies.set(file, (sky = new HDRLoader().loadAsync(`hdri/${file}`)));
    return sky;
  }

  /**
   * Sky light and reflections from a sky HDRI: equirectangular, its centre column facing +y
   * and +z up (as scripts/hdri-sun.ts reads it), turned `azimuth` radians about z so its sun
   * sits where the hour puts it, prefiltered for rough and smooth surfaces, at `intensity`.
   */
  setSkyLight(sky: THREE.Texture, azimuth: number, intensity: number): void {
    const env = new THREE.Scene();
    const dome = new THREE.MeshBasicNodeMaterial({ side: THREE.BackSide });
    const dir = normalize(positionLocal);
    const [c, s] = [Math.cos(-azimuth), Math.sin(-azimuth)];
    const turned = vec3(dir.x.mul(c).sub(dir.y.mul(s)), dir.x.mul(s).add(dir.y.mul(c)), dir.z);
    dome.colorNode = texture(sky, equirectUV(vec3(turned.y, turned.z, turned.x))).rgb as unknown as THREE.Node<'color'>;
    env.add(new THREE.Mesh(new THREE.SphereGeometry(10, 64, 32), dome));
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    const target = pmrem.fromScene(env, 0.01);
    this.scene.environment?.dispose();
    this.scene.environment = target.texture;
    this.scene.environmentIntensity = intensity;
    pmrem.dispose();
  }

  /** Light and shadow follow the player; the sun from `sunDirection`. */
  update(focus: THREE.Vector3, sunColour?: THREE.Color, sunIntensity?: number, fogColour?: THREE.Color): void {
    const d = this.sunDirection;
    this.sun.position.copy(focus).addScaledVector(d, 400);
    this.sun.target.position.copy(focus);
    // Snap the shadow camera to its texels so shadows don't crawl as the player walks
    const texel = 220 / this.sun.shadow.mapSize.x;
    this.sun.target.position.set(Math.round(focus.x / texel) * texel, Math.round(focus.y / texel) * texel, 0);
    this.sun.position.copy(this.sun.target.position).addScaledVector(d, 400);
    if (sunColour) {
      this.sun.color.copy(sunColour);
      this.rayColour.value.copy(sunColour);
    }
    if (sunIntensity !== undefined) this.sun.intensity = sunIntensity;
    if (fogColour) (this.scene.fog as THREE.FogExp2).color.copy(fogColour);
  }

  /**
   * One plain render first: it makes the sun's shadow map, which the sun shafts are built
   * against (they fail to build without it).
   */
  warm(): void {
    this.renderer.render(this.scene, this.camera);
  }

  resize(width: number, height: number): void {
    this.width = width;
    this.height = height;
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  /**
   * Depth of field for this frame: in focus at `distance` m, sharp for `range` m either side,
   * bokeh `amount` (0: none). Eased by the caller.
   */
  setFocus(distance: number, range: number, amount: number): void {
    this.focus.value = distance;
    this.focalRange.value = range;
    this.bokeh.value = amount;
  }

  render(): void {
    // Where the sun is on screen, for its rays, and how much it's in front
    const cam = this.camera;
    const forward = cam.getWorldDirection(this.scratchV);
    const facing = forward.dot(this.sunDirection);
    this.sunFade.value = THREE.MathUtils.smoothstep(facing, 0, 0.35);
    const p = this.scratchP.copy(cam.position).addScaledVector(this.sunDirection, 1000).project(cam);
    this.sunScreen.value.set((p.x + 1) / 2, (1 - p.y) / 2);
    this.aspect.value = cam.aspect;
    const size = this.renderer.getDrawingBufferSize(this.scratchSize);
    this.pixel.value.set(1 / Math.max(1, size.x), 1 / Math.max(1, size.y));
    this.post.render();
  }
  private readonly scratchV = new THREE.Vector3();
  private readonly scratchP = new THREE.Vector3();
  private readonly scratchSize = new THREE.Vector2();
}
