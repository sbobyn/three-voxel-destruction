// The Earth's clouds seen from the station: web-gpu-gems' volumetric cloud layer (./clouds, its Clouds lab and planet
// march: Perlin-Worley noise volumes made on the GPU, coverage and a height profile, detail erosion, a light march
// toward the sun, multiple-scattering octaves) over our scaled Earth, as its planet adapter draws it from orbit. The
// layer works in kilometres, in a frame whose +Y is up the camera's radial: each frame the camera's own radial is
// that frame's up, and the noise frame carries the march's camera-local offsets back to the world, so the clouds stay
// fixed to the Earth as the camera moves. It's drawn on a quad filling the camera's view, after the Earth and before
// the station, composited as the layer's premultiplied radiance plus the background times its transmittance.

import * as THREE from 'three/webgpu';
import { acos, asin, atan, cos, dot, float, log, max, mix, mx_noise_float, normalize, sin, smoothstep, texture, uniform, uv, vec2, vec3, vec4 } from 'three/tsl';
import { createCloudLayer } from './clouds/layer.ts';
import { PERIODS } from './clouds/model.ts';
import { BASE_SIZE, DETAIL_SIZE } from './clouds/noise.ts';
import { generateNoise, makeVolumeTexture } from './clouds/noise-gpu.ts';

/**
 * The clouds' Earth's radius (km). The real one's 6371, with the station 450 km up, and from there a cumulus field is
 * flat patches (its billows under a pixel), too small beside a station that at the Earth's scale here is 75 km long.
 * So, as the planet demo scales its world, a smaller Earth: the camera some 60 km above the clouds at the same
 * ratio of height to radius (so the cloud shell's horizon still lies on the drawn Earth's), the clouds seven times
 * bigger, their tops, sides and thickness showing.
 */
const EARTH_KM = 900;
/** The cloud noise, coarser than the layer's own: bigger clouds. */
const NOISE_SCALE = 0.3;
/**
 * Cyclones: where their eyes are on the Earth (directions from its middle, Earth-fixed, as it's turned at the start),
 * and how far their bands reach (km). The first is ahead of the station, coming towards it as the Earth turns.
 */
const CYCLONES = [new THREE.Vector3(0.05, Math.sin(0.3), Math.cos(0.3)), new THREE.Vector3(-0.2, Math.sin(2.4), Math.cos(2.4)), new THREE.Vector3(0.25, Math.sin(4.3), Math.cos(4.3))];
const CYCLONE = 280;
/** The coverage map: longitude by latitude about the spin axis (x), some 3 km a texel. */
const MAP = [2048, 1024];

/**
 * The cloud cover at `d` on the Earth (unit, Earth-fixed): what the layer keeps of its cloud noise is what rises
 * above 1 - cover, so cover makes the cloud type: near 0.8 every cell merges into one flat sheet, below 0.4 only
 * scattered puffs are left. So weather systems (fronts some 400 km across, their edges curled, drawn out along the
 * winds) of broken to closed decks (0.56 to 0.74), heaped and thinned within over some 40 km; and cyclones, spiral
 * bands wound into a dense eyewall round a clear eye. Nothing between.
 */
function cover(d: THREE.Node<'vec3'>): THREE.Node<'float'> {
  const w = d.mul(EARTH_KM).toVar();
  const warp = vec3(mx_noise_float(w.mul(1 / 500)), mx_noise_float(w.mul(1 / 500).add(17)), mx_noise_float(w.mul(1 / 500).add(31))).mul(160);
  const n = mx_noise_float(w.add(warp).mul(vec3(1 / 260, 1 / 480, 1 / 480))).mul(0.75).add(mx_noise_float(w.mul(1 / 110)).mul(0.25));
  let storms: THREE.Node<'float'> = float(0);
  let calm: THREE.Node<'float'> = float(1);
  for (const at of CYCLONES) {
    const c = at.clone().normalize();
    const t1 = new THREE.Vector3(0, 0, 1).cross(c).normalize();
    const t2 = c.clone().cross(t1);
    const r = acos(dot(d, vec3(c.x, c.y, c.z)).clamp(-1, 1)).mul(EARTH_KM);
    const around = atan(dot(d, vec3(t2.x, t2.y, t2.z)), dot(d, vec3(t1.x, t1.y, t1.z)));
    // Two logarithmic spiral arms, thinning and breaking up outwards, round a dense disc with a small clear eye
    const arm = sin(around.mul(2).add(log(r.add(4)).mul(7)));
    const bands = smoothstep(0.1, 0.8, arm).mul(mix(0.72, 0.45, smoothstep(60, CYCLONE, r))).mul(smoothstep(CYCLONE, CYCLONE * 0.4, r));
    const disc = smoothstep(80, 45, r).mul(0.78);
    storms = max(storms, max(bands, disc).mul(smoothstep(7, 14, r)));
    // (and the air between its bands mostly clear)
    calm = calm.mul(float(1).sub(smoothstep(CYCLONE * 1.05, CYCLONE * 0.75, r).mul(0.85)));
  }
  const systems = smoothstep(-0.12, -0.02, n).mul(mix(0.56, 0.74, smoothstep(-0.02, 0.35, n))).mul(calm);
  const weather = mx_noise_float(w.mul(1 / 40)).mul(0.13).add(mx_noise_float(w.mul(1 / 13)).mul(0.05));
  return max(systems, storms).add(weather).clamp(0, 1);
}

export class OrbitClouds {
  /** The quad the clouds are drawn on (the camera's child: add it to the camera). */
  readonly object: THREE.Mesh;
  ready = false;
  private readonly layer: ReturnType<typeof createCloudLayer>;
  private readonly base: THREE.Data3DTexture;
  private readonly detail: THREE.Data3DTexture;
  /** The cloud cover, baked once from cover() (it's far too dear to work out at every step of every ray). */
  private readonly map = new THREE.RenderTarget(MAP[0], MAP[1], { format: THREE.RedFormat, type: THREE.UnsignedByteType, wrapS: THREE.RepeatWrapping, depthBuffer: false });
  private readonly km: number;
  private readonly centre: THREE.Vector3;
  /** Where the camera is from the Earth's middle (km), and its local frame, in the Earth's own (unturned) axes. */
  private readonly earthOrigin = uniform(new THREE.Vector3());
  private readonly earthFrame = uniform(new THREE.Matrix3());

  /** Over the Earth of `radius` (scene units) centred at `centre`. */
  constructor(centre: THREE.Vector3, radius: number) {
    this.centre = centre.clone();
    this.km = EARTH_KM / radius;
    this.base = makeVolumeTexture(new Uint8Array(4 * BASE_SIZE ** 3), BASE_SIZE);
    this.detail = makeVolumeTexture(new Uint8Array(4 * DETAIL_SIZE ** 3), DETAIL_SIZE);
    // The march's camera-local offsets (km) as a place on the Earth, looked up in the map
    const coverage = (offset: THREE.Node<'vec3'>) => {
      const d = normalize(this.earthOrigin.add(this.earthFrame.mul(offset)));
      const at = vec2(atan(d.y, d.z).div(2 * Math.PI).add(0.5), asin(d.x.clamp(-1, 1)).div(Math.PI).add(0.5));
      return texture(this.map.texture, at).level(float(0)).r;
    };
    // (The weather is in the map: the layer's own repeats every 120 km, which would tile the view)
    this.layer = createCloudLayer({ base: this.base, detail: this.detail, worldFrame: true, coverage, weather: () => float(0.5) });
    const u = this.layer.uniforms;
    u.radius.value = EARTH_KM;
    // A deep layer: heaps with tops and sides (deeper still, a deck's edge stood up as a sunlit wall with a dark side)
    u.bottom.value = 1.5;
    u.top.value = 5.5;
    u.density.value = 24;
    // The fine erosion: billowed edges, not frayed scatters
    u.erosion.value = 0.3;
    // The planet adapter's display settings, at its orbit quality
    u.early.value = 1;
    u.skip.value = 1;
    u.adaptive.value = 1;
    u.stepPerKm.value = 0.002;
    u.viewSteps.value = 32;
    u.lightSteps.value = 2;
    u.minStep.value = 0.25;
    u.lod.value = 1;
    u.lightLod.value = 1;
    const material = new THREE.MeshBasicNodeMaterial({ depthTest: false, depthWrite: false, fog: false });
    // Premultiplied: the cloud's light, plus what's behind it times what gets through
    material.blending = THREE.CustomBlending;
    material.blendEquation = THREE.AddEquation;
    material.blendSrc = THREE.OneFactor;
    material.blendDst = THREE.SrcAlphaFactor;
    const march = this.layer.fastMarch;
    material.colorNode = march.rgb as unknown as THREE.Node<'color'>;
    material.opacityNode = march.a as unknown as THREE.Node<'float'>;
    this.object = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), material);
    this.object.frustumCulled = false;
    // After the Earth (-1), before everything else
    this.object.renderOrder = -0.5;
    this.object.visible = false;
  }

  /** Bake the cover map and fill the noise volumes on the GPU; the clouds show once it resolves. */
  async load(renderer: THREE.WebGPURenderer): Promise<void> {
    const bake = new THREE.MeshBasicNodeMaterial();
    const lon = uv().x.sub(0.5).mul(2 * Math.PI), lat = uv().y.sub(0.5).mul(Math.PI);
    bake.outputNode = vec4(cover(vec3(sin(lat), cos(lat).mul(sin(lon)), cos(lat).mul(cos(lon)))), 0, 0, 1);
    const quad = new THREE.QuadMesh(bake);
    renderer.setRenderTarget(this.map);
    quad.render(renderer);
    renderer.setRenderTarget(null);
    bake.dispose();
    const bytes = await generateNoise(renderer);
    this.base.image.data = bytes.base;
    this.base.needsUpdate = true;
    this.detail.image.data = bytes.detail;
    this.detail.needsUpdate = true;
    this.ready = true;
    this.object.visible = true;
  }

  /** Each frame: the march's camera, sun and noise frame from the scene's camera and sun (world, unit), lit by `sunColour`. */
  update(camera: THREE.PerspectiveCamera, sun: THREE.Vector3, sunColour: THREE.Color, brightness: number, spin: THREE.Quaternion): void {
    const u = this.layer.uniforms;
    camera.updateMatrixWorld();
    const eye = new THREE.Vector3().setFromMatrixPosition(camera.matrixWorld);
    const rel = eye.clone().sub(this.centre);
    // The local frame: up the camera's radial, any tangent across
    const y = rel.clone().normalize();
    const x = new THREE.Vector3(Math.abs(y.x) < 0.9 ? 1 : 0, Math.abs(y.x) < 0.9 ? 0 : 1, 0).cross(y).normalize();
    const z = new THREE.Vector3().crossVectors(x, y);
    const frame = new THREE.Matrix3().setFromMatrix4(new THREE.Matrix4().makeBasis(x, y, z));
    const toLocal = frame.clone().transpose();
    u.altitude.value = (rel.length() - EARTH_KM / this.km) * this.km;
    const forward = camera.getWorldDirection(new THREE.Vector3());
    const right = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 0).normalize();
    const up = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 1).normalize();
    u.forward.value.copy(forward).applyMatrix3(toLocal);
    u.right.value.copy(right).applyMatrix3(toLocal);
    // (The quad's uv runs up; the march's screen quad has it running down)
    u.up.value.copy(up).applyMatrix3(toLocal).negate();
    const tanHalf = Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2);
    u.tanHalf.value = tanHalf;
    u.aspect.value = camera.aspect;
    u.sun.value.copy(sun).applyMatrix3(toLocal).normalize();
    u.sunColour.value.set(sunColour.r, sunColour.g, sunColour.b).multiplyScalar(brightness);
    // The sky's blue from above and the Earth's light from below (seen from orbit, a cloud's shaded side isn't dark)
    u.ambient.value.set(0.2, 0.22, 0.26).multiplyScalar(brightness * 1.1);
    // The noise is fixed to the Earth, which turns by `spin`: the local frame and the camera's place in Earth-fixed axes
    const unspin = new THREE.Matrix3().setFromMatrix4(new THREE.Matrix4().makeRotationFromQuaternion(spin.clone().invert()));
    const fixedFrame = unspin.clone().multiply(frame);
    const fixedRel = rel.clone().applyMatrix3(unspin);
    u.noiseFrame.value.copy(fixedFrame).multiplyScalar(NOISE_SCALE);
    // The noise origins: where the camera is from the Earth's middle (km, scaled as the noise), wrapped to each volume's period
    const origin = fixedRel.clone().multiplyScalar(this.km * NOISE_SCALE);
    const wrap = (target: THREE.Vector3, period: number) => target.set(((origin.x % period) + period) % period, ((origin.y % period) + period) % period, ((origin.z % period) + period) % period);
    wrap(u.originBase.value, PERIODS.base);
    wrap(u.originDetail.value, PERIODS.detail);
    this.earthOrigin.value.copy(fixedRel).multiplyScalar(this.km);
    this.earthFrame.value.copy(fixedFrame);
    // The quad fills the view just past the near plane
    const d = camera.near * 2;
    this.object.position.set(0, 0, -d);
    this.object.scale.set(2 * d * tanHalf * camera.aspect, 2 * d * tanHalf, 1);
  }

  dispose(): void {
    this.layer.dispose();
    this.map.dispose();
    this.base.dispose();
    this.detail.dispose();
  }
}
