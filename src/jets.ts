// The space suit's jets: while they fire, a plume of gas from each vent, a soft cone flaring out of the nozzle and
// gone within a metre and a half (in vacuum the gas spreads at once), brightest at the nozzle, its turbulence streaming out
// along it. One small cone a vent, drawn additively (the bloom gives it its glow): no particles.

import * as THREE from 'three/webgpu';
import { abs, dot, float, mx_noise_float, normalView, positionLocal, positionViewDirection, pow, smoothstep, time, uniform, vec3 } from 'three/tsl';

/** A plume's length and its radius at the nozzle and at the end (m). */
const LENGTH = 1.6;
const NOZZLE = 0.02;
const FLARE = 0.5;
/** How quickly a plume lights and goes out (s). */
const ATTACK = 0.03;
const RELEASE = 0.07;
const UP = new THREE.Vector3(0, 1, 0);

interface Plume {
  mesh: THREE.Mesh;
  level: THREE.UniformNode<'float', number>;
}

export class SuitJets {
  readonly object = new THREE.Group();
  private readonly plumes: Plume[] = [];

  constructor(vents: number) {
    // The cone along +y, its nozzle at the origin
    const geometry = new THREE.CylinderGeometry(FLARE, NOZZLE, LENGTH, 20, 1, true).translate(0, LENGTH / 2, 0);
    for (let i = 0; i < vents; i++) {
      const level = uniform(0);
      const along = positionLocal.y.div(LENGTH);
      // Soft at its sides (seen edge on, the gas is thin), fading from the nozzle out, and streaked by turbulence
      // blown out along it (each vent its own)
      const facing = pow(abs(dot(normalView, positionViewDirection)), 2.5);
      const fade = pow(float(1).sub(along), 1.6).mul(smoothstep(0, 0.03, along));
      const flow = mx_noise_float(vec3(positionLocal.x.mul(14), positionLocal.y.mul(9).sub(time.mul(18)), positionLocal.z.mul(14).add(i * 7)));
      const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthWrite: false, side: THREE.DoubleSide, blending: THREE.AdditiveBlending, fog: false });
      material.colorNode = vec3(0.9, 0.93, 1).mul(level.mul(facing).mul(fade).mul(flow.mul(0.8).add(0.7).clamp(0, 1.5)).mul(1.6)) as unknown as THREE.Node<'color'>;
      const mesh = new THREE.Mesh(geometry, material);
      mesh.visible = false;
      mesh.frustumCulled = false;
      this.object.add(mesh);
      this.plumes.push({ mesh, level });
    }
  }

  /** Each frame: the vents (world), how hard each fires (0..1, 0 when it's off) and the way the gas goes (unit, world). */
  update(dt: number, vents: THREE.Vector3[], amounts: number[], away: THREE.Vector3): void {
    this.plumes.forEach(({ mesh, level }, i) => {
      const amount = amounts[i];
      level.value += (amount - level.value) * (1 - Math.exp(-dt / (amount > level.value ? ATTACK : RELEASE)));
      mesh.visible = level.value > 0.01;
      if (!mesh.visible) return;
      mesh.position.copy(vents[i]);
      if (amount > 0) mesh.quaternion.setFromUnitVectors(UP, away);
      // Longer and wider the harder they push, with a little flutter
      const size = (0.55 + 0.45 * level.value) * (0.94 + Math.random() * 0.12);
      mesh.scale.set(size, size * (0.9 + 0.2 * level.value), size);
    });
  }
}
