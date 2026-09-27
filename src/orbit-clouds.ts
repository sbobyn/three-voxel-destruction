// The Earth's clouds seen from the station: web-gpu-gems' volumetric cloud layer (./clouds, its Clouds lab and planet
// march: Perlin-Worley noise volumes made on the GPU, coverage and a height profile, detail erosion, a light march
// toward the sun, multiple-scattering octaves) over our scaled Earth, as its planet adapter draws it from orbit. The
// layer works in kilometres, in a frame whose +Y is up the camera's radial: each frame the camera's own radial is
// that frame's up, and the noise frame carries the march's camera-local offsets back to the world, so the clouds stay
// fixed to the Earth as the camera moves. It's drawn on a quad filling the camera's view, after the Earth and before
// the station, composited as the layer's premultiplied radiance plus the background times its transmittance.

import * as THREE from 'three/webgpu';
import { acos, atan, dot, float, log, max, mix, mx_noise_float, normalize, sin, smoothstep, uniform, vec3 } from 'three/tsl';
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

export class OrbitClouds {
  /** The quad the clouds are drawn on (the camera's child: add it to the camera). */
  readonly object: THREE.Mesh;
  ready = false;
  private readonly layer: ReturnType<typeof createCloudLayer>;
  private readonly base: THREE.Data3DTexture;
  private readonly detail: THREE.Data3DTexture;
  private readonly km: number;
  private readonly centre: THREE.Vector3;
  private readonly weatherOrigin = uniform(new THREE.Vector3());
  private readonly weatherFrame = uniform(new THREE.Matrix3());

  /** Over the Earth of `radius` (scene units) centred at `centre`. */
  constructor(centre: THREE.Vector3, radius: number) {
    this.centre = centre.clone();
    this.km = EARTH_KM / radius;
    this.base = makeVolumeTexture(new Uint8Array(4 * BASE_SIZE ** 3), BASE_SIZE);
    this.detail = makeVolumeTexture(new Uint8Array(4 * DETAIL_SIZE ** 3), DETAIL_SIZE);
    // Where the clouds are, on the Earth itself (the offsets turned into Earth-fixed kilometres: the clouds turn with
    // it). The layer keeps whatever of its cloud noise rises above 1 - cover, so cover is what makes a cloud type:
    // near 0.8 every cell merges into one flat sheet, near 0.35 only scattered puffs are left. So weather systems
    // (fronts, warped and drawn out east-west, some 400 km across) with a quick fringe of broken cells, then broken
    // to closed decks within (0.56 to 0.74: cells that join up, with gaps and texture); cyclones, spirals of cloud
    // bands wound into a dense eyewall round a clear eye, the air between their bands mostly clear; nothing between.
    const world = (offset: THREE.Node<'vec3'>) => this.weatherOrigin.add(this.weatherFrame.mul(offset));
    const cyclone = (w: THREE.Node<'vec3'>, at: THREE.Vector3) => {
      const c = at.clone().normalize();
      const t1 = new THREE.Vector3(0, 0, 1).cross(c).normalize();
      const t2 = c.clone().cross(t1);
      const p = normalize(w);
      const r = acos(dot(p, vec3(c.x, c.y, c.z)).clamp(-1, 1)).mul(EARTH_KM);
      const around = atan(dot(p, vec3(t2.x, t2.y, t2.z)), dot(p, vec3(t1.x, t1.y, t1.z)));
      // A hurricane: a dense central disc round a small clear eye, and bands spiralling out of it (two arms,
      // logarithmic spirals), thinning and breaking up towards the edge
      const arm = sin(around.mul(2).add(log(r.add(4)).mul(7)));
      const bands = smoothstep(0.1, 0.8, arm).mul(mix(0.72, 0.45, smoothstep(60, CYCLONE, r))).mul(smoothstep(CYCLONE, CYCLONE * 0.4, r));
      const disc = smoothstep(80, 45, r).mul(0.78);
      const eye = smoothstep(7, 14, r);
      return { cover: max(bands, disc).mul(eye), zone: smoothstep(CYCLONE * 1.05, CYCLONE * 0.75, r) };
    };
    const coverage = (offset: THREE.Node<'vec3'>) => {
      const w = world(offset).toVar();
      // Fronts: the broad field's domain warped (so its edges curl) and squashed along the spin axis (x: north-south
      // here), so systems are drawn out east-west as the winds draw them
      const warp = vec3(mx_noise_float(w.mul(1 / 500)), mx_noise_float(w.mul(1 / 500).add(17)), mx_noise_float(w.mul(1 / 500).add(31))).mul(160);
      const q = w.add(warp).mul(vec3(1 / 260, 1 / 480, 1 / 480));
      const n = mx_noise_float(q).mul(0.75).add(mx_noise_float(w.mul(1 / 110)).mul(0.25)).toVar();
      let storms = cyclone(w, CYCLONES[0]).cover;
      let zone = cyclone(w, CYCLONES[0]).zone;
      for (const c of CYCLONES.slice(1)) {
        const one = cyclone(w, c);
        storms = max(storms, one.cover);
        zone = max(zone, one.zone);
      }
      const inside = smoothstep(-0.12, -0.02, n);
      const systems = inside.mul(mix(0.56, 0.74, smoothstep(-0.02, 0.35, n))).mul(float(1).sub(zone.mul(0.85)));
      return max(systems, storms);
    };
    // The weather within: the cover heaped and thinned over some 40 km (the layer adds (weather - 0.5) * 0.8), so a
    // deck has thicker and thinner parts and a broken field clumps. (Not the layer's own, which repeats every 120 km.)
    const weather = (offset: THREE.Node<'vec3'>) => {
      const w = world(offset);
      return mx_noise_float(w.mul(1 / 40)).mul(0.16).add(mx_noise_float(w.mul(1 / 13)).mul(0.06)).add(0.5);
    };
    this.layer = createCloudLayer({ base: this.base, detail: this.detail, worldFrame: true, coverage, weather });
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

  /** Fill the noise volumes on the GPU; the clouds show once it resolves. */
  async load(renderer: THREE.WebGPURenderer): Promise<void> {
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
    // The noise origins: where the camera is from the Earth's middle (km, scaled as the noise), wrapped to each field's period
    const origin = fixedRel.clone().multiplyScalar(this.km * NOISE_SCALE);
    const wrap = (target: THREE.Vector3, period: number) => target.set(((origin.x % period) + period) % period, ((origin.y % period) + period) % period, ((origin.z % period) + period) % period);
    wrap(u.originWeather.value, PERIODS.weather);
    wrap(u.originBase.value, PERIODS.base);
    wrap(u.originDetail.value, PERIODS.detail);
    this.weatherOrigin.value.copy(fixedRel).multiplyScalar(this.km);
    this.weatherFrame.value.copy(fixedFrame);
    // The quad fills the view just past the near plane
    const d = camera.near * 2;
    this.object.position.set(0, 0, -d);
    this.object.scale.set(2 * d * tanHalf * camera.aspect, 2 * d * tanHalf, 1);
  }

  dispose(): void {
    this.layer.dispose();
    this.base.dispose();
    this.detail.dispose();
  }
}
