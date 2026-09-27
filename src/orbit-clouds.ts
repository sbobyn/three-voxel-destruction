// The Earth's clouds seen from the station: web-gpu-gems' volumetric cloud layer (./clouds, its Clouds lab and planet
// march: Perlin-Worley noise volumes made on the GPU, coverage and a height profile, detail erosion, a light march
// toward the sun, multiple-scattering octaves) over our scaled Earth, as its planet adapter draws it from orbit. The
// layer works in kilometres, in a frame whose +Y is up the camera's radial: each frame the camera's own radial is
// that frame's up, and the noise frame carries the march's camera-local offsets back to the world, so the clouds stay
// fixed to the Earth as the camera moves. It's drawn on a quad filling the camera's view, after the Earth and before
// the station, composited as the layer's premultiplied radiance plus the background times its transmittance.

import * as THREE from 'three/webgpu';
import { mx_noise_float, smoothstep, uniform } from 'three/tsl';
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
/** The cloud noise at the layer's own scale. */
const NOISE_SCALE = 1;

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
    // Regional cover from broad noise over the globe: systems a hundred or two kilometres across with clear sea
    // between them. And the weather within them from noise too (the layer's own weather field repeats every 120 km,
    // which would tile the view): cells of 15 to 40 km, heaping and thinning the cover
    const world = (offset: THREE.Node<'vec3'>) => this.weatherOrigin.add(this.weatherFrame.mul(offset));
    const coverage = (offset: THREE.Node<'vec3'>) => {
      const w = world(offset);
      const n = mx_noise_float(w.mul(1 / 260)).mul(0.65).add(mx_noise_float(w.mul(1 / 90)).mul(0.35));
      return smoothstep(-0.2, 0.45, n).mul(0.66).add(0.04);
    };
    const weather = (offset: THREE.Node<'vec3'>) => {
      const w = world(offset);
      return mx_noise_float(w.mul(1 / 40)).mul(0.35).add(mx_noise_float(w.mul(1 / 15)).mul(0.2)).add(0.5);
    };
    this.layer = createCloudLayer({ base: this.base, detail: this.detail, worldFrame: true, coverage, weather });
    const u = this.layer.uniforms;
    u.radius.value = EARTH_KM;
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
  update(camera: THREE.PerspectiveCamera, sun: THREE.Vector3, sunColour: THREE.Color, brightness: number): void {
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
    u.ambient.value.set(0.16, 0.2, 0.27).multiplyScalar(brightness * 0.6);
    u.noiseFrame.value.copy(frame).multiplyScalar(NOISE_SCALE);
    // The noise origins: where the camera is from the Earth's middle (km, scaled as the noise), wrapped to each field's period
    const origin = rel.clone().multiplyScalar(this.km * NOISE_SCALE);
    const wrap = (target: THREE.Vector3, period: number) => target.set(((origin.x % period) + period) % period, ((origin.y % period) + period) % period, ((origin.z % period) + period) % period);
    wrap(u.originWeather.value, PERIODS.weather);
    wrap(u.originBase.value, PERIODS.base);
    wrap(u.originDetail.value, PERIODS.detail);
    this.weatherOrigin.value.copy(rel).multiplyScalar(this.km);
    this.weatherFrame.value.copy(frame);
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
