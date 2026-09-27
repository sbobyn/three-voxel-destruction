// The race car's afterfire: at each upshift a flame bursts out of every exhaust and is gone in a tenth of a second.
// A cone a pipe, flaring out of it: white-hot at the nozzle with a touch of blue, orange along it, a red tip
// fading out, its turbulence flickering outward. Drawn additively (the bloom gives it its glow): no particles.

import * as THREE from 'three/webgpu';
import { abs, dot, float, mix, mx_noise_float, normalView, positionLocal, positionViewDirection, pow, smoothstep, time, uniform, vec3 } from 'three/tsl';

/** A flame's length and its radius at the nozzle and at its end (m), and how quickly it dies away (s). */
const LENGTH = 0.7;
const NOZZLE = 0.05;
const FLARE = 0.13;
const FADE = 0.09;
const UP = new THREE.Vector3(0, 1, 0);

interface Flame {
  mesh: THREE.Mesh;
  heat: THREE.UniformNode<'float', number>;
  /** This burst's length (a share of LENGTH). */
  reach: number;
}

export class Flames {
  readonly object = new THREE.Group();
  private readonly flames: Flame[] = [];

  constructor(pipes: number) {
    // The cone along +y, its nozzle at the origin
    const geometry = new THREE.CylinderGeometry(FLARE, NOZZLE, LENGTH, 16, 1, true).translate(0, LENGTH / 2, 0);
    for (let i = 0; i < pipes; i++) {
      const heat = uniform(0);
      const along = positionLocal.y.div(LENGTH);
      // Soft at its sides, strongest at the nozzle, its turbulence streaming out along it
      const facing = pow(abs(dot(normalView, positionViewDirection)), 1.5);
      const flicker = mx_noise_float(positionLocal.mul(vec3(18, 10, 18)).add(vec3(0, time.mul(-30), i * 5))).mul(0.5).add(0.75);
      const fade = pow(float(1).sub(along), 1.4).mul(smoothstep(0, 0.05, along));
      // White-hot (a little blue) at the nozzle, orange along it, red at the tip
      const hot = mix(vec3(0.75, 0.85, 1), vec3(1, 0.55, 0.12), smoothstep(0, 0.3, along));
      const colour = mix(hot, vec3(0.9, 0.18, 0.04), smoothstep(0.35, 0.9, along));
      const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthWrite: false, side: THREE.DoubleSide, blending: THREE.AdditiveBlending, fog: false });
      material.colorNode = colour.mul(heat.mul(facing).mul(fade).mul(flicker).mul(3)) as unknown as THREE.Node<'color'>;
      const mesh = new THREE.Mesh(geometry, material);
      mesh.visible = false;
      mesh.frustumCulled = false;
      this.object.add(mesh);
      this.flames.push({ mesh, heat, reach: 1 });
    }
  }

  /** A burst out of every pipe (`strength` 0..1), each its own length. */
  fire(strength = 1): void {
    for (const flame of this.flames) {
      flame.heat.value = strength * (0.8 + Math.random() * 0.4);
      flame.reach = 0.6 + Math.random() * 0.5;
    }
  }

  /** Each frame: the pipes (world) and the way out of them (unit, world). */
  update(dt: number, pipes: THREE.Vector3[], back: THREE.Vector3): void {
    this.flames.forEach((flame, i) => {
      flame.heat.value *= Math.exp(-dt / FADE);
      const { mesh } = flame;
      mesh.visible = flame.heat.value > 0.02;
      if (!mesh.visible) return;
      mesh.position.copy(pipes[i]);
      mesh.quaternion.setFromUnitVectors(UP, back);
      // Shrinking back into the pipe as it dies
      const s = flame.reach * (0.5 + 0.5 * Math.min(1, flame.heat.value));
      mesh.scale.set(0.8 + 0.4 * s, s, 0.8 + 0.4 * s);
    });
  }
}
