// Fine debris: when voxels break, chips of them fly (a fraction of a voxel, in its colour),
// bounce off the ground and whatever fixed voxels they meet, tumble to rest and shrink away.
// Purely for the eye: they make a break look finer than the voxels. A few thousand on the
// CPU, drawn as one instanced mesh. Glass (Chips.glass) the same way: shards, bright and
// sharp in the sky's reflection, that stay a while glittering on the pavement.

import * as THREE from 'three/webgpu';
import { type City, VOXEL, voxelAt } from './world.ts';

const MAX = 8000;
const GRAVITY = 14;

export interface ChipLook {
  material: THREE.Material;
  geometry: THREE.BufferGeometry;
  /** Seconds before a chip is gone (it shrinks away over the last 1.2). */
  life: number;
  shadows: boolean;
  /** Size, as a share of a voxel: least, and the most added. */
  size: [number, number];
  /** How the air slows them (per second, plus per metre of size). */
  drag: number;
}

export class Chips {
  readonly object: THREE.InstancedMesh;
  private readonly p = new Float32Array(MAX * 3);
  private readonly v = new Float32Array(MAX * 3);
  private readonly spin = new Float32Array(MAX * 3);
  private readonly angle = new Float32Array(MAX * 3);
  private readonly size = new Float32Array(MAX);
  private readonly age: Float32Array;
  private readonly life: number;
  private readonly look: ChipLook;
  private readonly colour = new Float32Array(MAX * 3);
  private next = 0;
  private readonly matrix = new THREE.Matrix4();
  private readonly q = new THREE.Quaternion();
  private readonly e = new THREE.Euler();
  private readonly s = new THREE.Vector3();
  private readonly t = new THREE.Vector3();
  private readonly c = new THREE.Color();
  city: City;

  constructor(city: City, look?: Partial<ChipLook>) {
    this.city = city;
    // Thin plates: broken render and brick fly as flakes and panels, not dice
    this.look = {
      material: new THREE.MeshStandardMaterial({ roughness: 0.85, metalness: 0, flatShading: true }),
      geometry: new THREE.BoxGeometry(1, 1, 0.16),
      life: 6,
      shadows: true,
      size: [0.25, 0.75],
      drag: 0.6,
      ...look,
    };
    this.life = this.look.life;
    this.age = new Float32Array(MAX).fill(this.life);
    this.object = new THREE.InstancedMesh(this.look.geometry, this.look.material, MAX);
    this.object.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.object.frustumCulled = false;
    this.object.castShadow = this.look.shadows;
    this.object.receiveShadow = this.look.shadows;
    this.object.count = 0;
    this.object.setColorAt(0, this.c.setRGB(0.5, 0.5, 0.5));
  }

  /**
   * Glass: slivers, near-mirrors of the sky (a sliver facing the sun flashes), tinted as the
   * pane was, lasting long enough to glitter on the ground.
   */
  static glass(city: City): Chips {
    // A sliver: a flat triangle, a little thickness so it catches light edge on
    const shape = new THREE.Shape([new THREE.Vector2(-0.5, -0.4), new THREE.Vector2(0.55, -0.2), new THREE.Vector2(-0.1, 0.6)]);
    const geometry = new THREE.ExtrudeGeometry(shape, { depth: 0.04, bevelEnabled: false });
    geometry.center();
    const material = new THREE.MeshStandardMaterial({ roughness: 0.12, metalness: 0.75, flatShading: true, envMapIntensity: 2.2, emissive: 0x1a2226 });
    return new Chips(city, { material, geometry, life: 14, shadows: false, size: [0.2, 0.5], drag: 1.2 });
  }

  /**
   * Chips from voxels (their indices) broken by a blast at `from`: `each` per voxel (fractions
   * round at random), thrown outward at up to `speed` m/s, in their own colour or `colour`.
   */
  burst(voxels: number[], from: ArrayLike<number>, each: number, speed: number, colour?: number): void {
    const { city } = this;
    for (const vx of voxels) {
      const n = Math.floor(each) + (Math.random() < each % 1 ? 1 : 0);
      this.c.setHex(colour ?? city.color[vx]);
      for (let k = 0; k < n; k++) {
        const i = this.next;
        this.next = (this.next + 1) % MAX;
        const o = 3 * i;
        for (let a = 0; a < 3; a++) this.p[o + a] = city.position[3 * vx + a] + (Math.random() - 0.5) * 0.9 * VOXEL;
        const d = [this.p[o] - from[0], this.p[o + 1] - from[1], this.p[o + 2] - from[2] + 0.6];
        const len = Math.hypot(d[0], d[1], d[2]) || 1;
        const f = speed * (0.3 + Math.random() * 0.8);
        for (let a = 0; a < 3; a++) {
          this.v[o + a] = (d[a] / len) * f + (Math.random() - 0.5) * speed * 0.4;
          this.spin[o + a] = (Math.random() - 0.5) * 11;
          this.angle[o + a] = Math.random() * 6.3;
        }
        this.size[i] = VOXEL * (this.look.size[0] + Math.random() ** 2 * this.look.size[1]);
        this.age[i] = Math.random() * 0.5;
        const shade = 0.8 + Math.random() * 0.35;
        this.colour.set([this.c.r * shade, this.c.g * shade, this.c.b * shade], o);
      }
    }
  }

  update(dt: number): void {
    const { p, v, city } = this;
    let live = 0;
    for (let i = 0; i < MAX; i++) {
      if (this.age[i] >= this.life) continue;
      this.age[i] += dt;
      const o = 3 * i;
      // Plates catch the air: they slow and flutter down rather than drop
      v[o + 2] -= GRAVITY * dt;
      const air = Math.exp(-dt * (this.look.drag + this.size[i] * 3));
      v[o] *= air;
      v[o + 1] *= air;
      v[o + 2] = v[o + 2] < -4 ? v[o + 2] * air : v[o + 2];
      const half = this.size[i] / 2;
      for (let a = 0; a < 3; a++) {
        const next = p[o + a] + v[o + a] * dt;
        // Bounce off the ground and off fixed voxels, losing most of the speed
        const q = [p[o], p[o + 1], p[o + 2]];
        q[a] = next;
        const hitGround = a === 2 && next - half < 0;
        if (hitGround || voxelAt(city, q[0], q[1], q[2] - (a === 2 && v[o + 2] < 0 ? half : 0)) >= 0) {
          v[o + a] *= -0.3;
          for (let b = 0; b < 3; b++) if (b !== a) v[o + b] *= 0.7;
          for (let b = 0; b < 3; b++) this.spin[o + b] *= 0.6;
          if (hitGround) p[o + 2] = half;
        } else p[o + a] = next;
      }
      for (let a = 0; a < 3; a++) this.angle[o + a] += this.spin[o + a] * dt;
      // Resting chips stop spinning
      if (p[o + 2] <= half + 1e-3 && Math.abs(v[o + 2]) < 0.5) for (let a = 0; a < 3; a++) this.spin[o + a] *= 0.9;
      const fade = Math.min(1, (this.life - this.age[i]) / 1.2);
      this.s.setScalar(this.size[i] * fade);
      this.q.setFromEuler(this.e.set(this.angle[o], this.angle[o + 1], this.angle[o + 2]));
      this.matrix.compose(this.t.set(p[o], p[o + 1], p[o + 2]), this.q, this.s);
      // The live chips pack to the front of the instances, each with its own colour
      this.object.setMatrixAt(live, this.matrix);
      this.object.setColorAt(live, this.c.setRGB(this.colour[o], this.colour[o + 1], this.colour[o + 2]));
      live++;
    }
    this.object.count = live;
    this.object.instanceMatrix.needsUpdate = true;
    if (live && this.object.instanceColor) this.object.instanceColor.needsUpdate = true;
  }
}
