// Bits of the race car knocked off it: little cubes of its paint, lights and frame thrown out where it was hit,
// tumbling, bouncing along the ground and shrinking away after a few seconds. One instanced mesh, moved on the
// CPU (a few hundred at most).

import * as THREE from 'three/webgpu';

/** Bits at most, their size (m, before each's own scale), life (s) and how they bounce. */
const POOL = 600;
const SIZE = 0.0625;
const LIFE = [3, 6];
const BOUNCE = 0.3;
const GRAVITY = 9.81;

interface Bit {
  at: THREE.Vector3;
  velocity: THREE.Vector3;
  spin: THREE.Vector3;
  turn: THREE.Euler;
  size: number;
  age: number;
  life: number;
}

export class Bits {
  readonly object: THREE.InstancedMesh;
  private readonly bits: Bit[] = [];

  constructor() {
    const material = new THREE.MeshStandardMaterial({ roughness: 0.45, metalness: 0.5 });
    this.object = new THREE.InstancedMesh(new THREE.BoxGeometry(SIZE, SIZE, SIZE), material, POOL);
    this.object.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(POOL * 3), 3);
    this.object.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.object.count = 0;
    this.object.frustumCulled = false;
    this.object.castShadow = true;
  }

  /**
   * Bits thrown from `points` (world), each its `colours`' (linear rgb), out along `away` (unit, world) and
   * spreading, on top of what they were moving at (`carried`, m/s), harder for a harder `blow` (0..1).
   */
  burst(points: THREE.Vector3[], colours: THREE.Color[], away: THREE.Vector3, carried: THREE.Vector3, blow: number): void {
    points.forEach((at, i) => {
      if (this.bits.length >= POOL) return;
      const spread = new THREE.Vector3(Math.random() - 0.5, Math.random() - 0.5, Math.random() * 0.6).multiplyScalar(2);
      const velocity = away.clone().add(spread).normalize().multiplyScalar(2 + Math.random() * (3 + 8 * blow)).add(carried);
      const spin = new THREE.Vector3(Math.random() - 0.5, Math.random() - 0.5, Math.random() - 0.5).multiplyScalar(20);
      const bit = { at: at.clone(), velocity, spin, turn: new THREE.Euler(), size: 1 + Math.random() * 1.4, age: 0, life: LIFE[0] + Math.random() * (LIFE[1] - LIFE[0]) };
      this.bits.push(bit);
      this.object.setColorAt(this.bits.length - 1, colours[i]);
    });
    this.object.instanceColor!.needsUpdate = true;
  }

  /** Each frame: fly them on, off the ground, and lay out their cubes. */
  update(dt: number): void {
    if (!this.bits.length && this.object.count === 0) return;
    const m = new THREE.Matrix4();
    const q = new THREE.Quaternion();
    const s = new THREE.Vector3();
    const colour = new THREE.Color();
    let n = 0;
    for (let i = 0; i < this.bits.length; i++) {
      const b = this.bits[i];
      b.age += dt;
      if (b.age >= b.life) continue;
      b.velocity.z -= GRAVITY * dt;
      b.at.addScaledVector(b.velocity, dt);
      const half = (SIZE * b.size) / 2;
      if (b.at.z < half) {
        // Off the ground: a bounce, skidding to a stop, spinning down
        b.at.z = half;
        b.velocity.z = Math.abs(b.velocity.z) * BOUNCE;
        b.velocity.x *= 0.7;
        b.velocity.y *= 0.7;
        b.spin.multiplyScalar(0.6);
      }
      b.turn.set(b.turn.x + b.spin.x * dt, b.turn.y + b.spin.y * dt, b.turn.z + b.spin.z * dt);
      // Shrinking away over its last second
      const scale = b.size * Math.min(1, b.life - b.age);
      m.compose(b.at, q.setFromEuler(b.turn), s.setScalar(scale));
      if (n !== i) {
        this.bits[n] = b;
        this.object.getColorAt(i, colour);
        this.object.setColorAt(n, colour);
      }
      this.object.setMatrixAt(n, m);
      n++;
    }
    this.bits.length = n;
    this.object.count = n;
    this.object.instanceMatrix.needsUpdate = true;
    this.object.instanceColor!.needsUpdate = true;
  }

  /** None left. */
  clear(): void {
    this.bits.length = 0;
    this.object.count = 0;
  }
}
