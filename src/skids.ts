// Skid marks: dark rubber laid on the track under the race car's rear tyres while they slide, a strip of short
// segments per tyre. One instanced mesh on the GPU, used as a ring (the oldest segments are reused), each
// segment carrying when it was laid and how hard, so the shader fades it; a frame writes a segment or two and
// uploads just those.

import * as THREE from 'three/webgpu';
import { abs, attribute, float, mx_noise_float, positionWorld, smoothstep, uniform, uv, vec3 } from 'three/tsl';

/** Segments at most, how long each is (m), a tyre's width (m), and how long the marks last (s: then fade out). */
const CAPACITY = 4096;
const STEP = 0.35;
const WIDTH = 0.27;
const LAST = [25, 50];

export class SkidMarks {
  readonly object: THREE.InstancedMesh;
  private readonly marks: THREE.InstancedBufferAttribute;
  private readonly now = uniform(0);
  private readonly last: (THREE.Vector3 | null)[] = [];
  private next = 0;

  constructor(tyres: number) {
    const geometry = new THREE.PlaneGeometry(1, 1);
    // Per segment: when it was laid (s) and how hard the tyre was sliding (0..1)
    this.marks = new THREE.InstancedBufferAttribute(new Float32Array(CAPACITY * 2), 2);
    this.marks.setUsage(THREE.DynamicDrawUsage);
    geometry.setAttribute('mark', this.marks);
    const mark = attribute('mark', 'vec2');
    const age = this.now.sub(mark.x);
    // Soft at the tyre's edges, mottled along it, darker the harder it slid, fading as it ages
    const across = smoothstep(0.5, 0.3, abs(uv().y.sub(0.5)));
    const mottle = mx_noise_float(vec3(positionWorld.xy.mul(3), 0)).mul(0.25).add(0.85);
    const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthWrite: false, fog: true, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2 });
    material.colorNode = vec3(0.012, 0.012, 0.012) as unknown as THREE.Node<'color'>;
    material.opacityNode = mark.y.mul(0.85).mul(across).mul(mottle).mul(float(1).sub(smoothstep(LAST[0], LAST[1], age))) as unknown as THREE.Node<'float'>;
    this.object = new THREE.InstancedMesh(geometry, material, CAPACITY);
    this.object.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.object.count = 0;
    this.object.frustumCulled = false;
    this.object.renderOrder = 1;
    for (let i = 0; i < tyres; i++) this.last.push(null);
  }

  /** Each frame: where each tyre touches the ground (world), how hard it's skidding (0..1), and the clock (s). */
  update(tyres: THREE.Vector3[], skid: number, time: number): void {
    this.now.value = time;
    const m = new THREE.Matrix4();
    const dir = new THREE.Vector3();
    const side = new THREE.Vector3();
    tyres.forEach((at, i) => {
      const from = this.last[i];
      if (skid < 0.05) {
        this.last[i] = null;
        return;
      }
      if (!from) {
        this.last[i] = at.clone();
        return;
      }
      const length = Math.hypot(at.x - from.x, at.y - from.y);
      if (length < STEP) return;
      // (Metres at once: the car was put somewhere new, not driven there)
      if (length > 3) {
        from.copy(at);
        return;
      }
      // A segment from where the last ended to here, flat on the ground (a little longer: no gaps in bends)
      dir.set((at.x - from.x) / length, (at.y - from.y) / length, 0);
      side.set(-dir.y, dir.x, 0).multiplyScalar(WIDTH);
      m.makeBasis(dir.clone().multiplyScalar(length + WIDTH * 0.3), side, new THREE.Vector3(0, 0, 1));
      m.setPosition((at.x + from.x) / 2, (at.y + from.y) / 2, 0.012);
      const k = this.next;
      this.object.setMatrixAt(k, m);
      this.marks.setXY(k, time, skid);
      this.object.instanceMatrix.addUpdateRange(k * 16, 16);
      this.marks.addUpdateRange(k * 2, 2);
      this.object.instanceMatrix.needsUpdate = true;
      this.marks.needsUpdate = true;
      this.next = (k + 1) % CAPACITY;
      this.object.count = Math.max(this.object.count, k + 1);
      from.copy(at);
    });
  }

  /** All the marks gone (a fresh track). */
  clear(): void {
    this.object.count = 0;
    this.next = 0;
    this.last.fill(null);
  }
}
