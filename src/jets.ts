// The space suit's jets: while they fire, a plume of gas from each vent, a soft cone flaring out of the nozzle and
// gone within a metre and a half (in vacuum the gas spreads at once), brightest at the nozzle, its turbulence
// streaming out along it; a flare as a vent lights, and when it shuts its last gas drifts off. Through the gas,
// glints: specks of frozen propellant thrown out ahead of it, catching the light as short streaks. One small cone
// a vent and a pool of streaks, all drawn additively (the bloom gives them their glow): no particle system.

import * as THREE from 'three/webgpu';
import { abs, dot, float, mx_noise_float, normalView, positionLocal, positionView, positionViewDirection, pow, smoothstep, time, uniform, uv, vec3 } from 'three/tsl';

/** A plume's length and its radius at the nozzle and at the end (m). */
const LENGTH = 1.6;
const NOZZLE = 0.02;
const FLARE = 0.5;
/** How quickly a plume lights and goes out (s), and how long a vent's flare as it lights lasts. */
const ATTACK = 0.03;
const RELEASE = 0.1;
const FLASH = 0.12;
/** The gas's speed out of a vent (m/s): its last gas drifts off at it, the glints fly at about twice it. */
const GAS = 3;
/** Glints: how many at most, how many a second from a vent at full push, their life (s) and width (m). */
const GLINTS = 96;
const GLINT_RATE = 40;
const GLINT_LIFE = [0.25, 0.5];
const GLINT_WIDTH = 0.006;
const UP = new THREE.Vector3(0, 1, 0);

interface Plume {
  mesh: THREE.Mesh;
  /** Its brightness in the shader: how hard it's firing, with the flare. */
  glow: THREE.UniformNode<'float', number>;
  level: number;
  flash: number;
  /** Where its gas goes (m/s, world): its last drifts off at this. */
  drift: THREE.Vector3;
  /** Glints due from it (they come a fraction at a time). */
  due: number;
}

interface Glint {
  at: THREE.Vector3;
  velocity: THREE.Vector3;
  age: number;
  life: number;
  bright: number;
}

export class SuitJets {
  readonly object = new THREE.Group();
  private readonly plumes: Plume[] = [];
  private readonly glints: Glint[] = [];
  private readonly streaks: THREE.InstancedMesh;

  constructor(vents: number) {
    // The cone along +y, its nozzle at the origin
    const geometry = new THREE.CylinderGeometry(FLARE, NOZZLE, LENGTH, 20, 1, true).translate(0, LENGTH / 2, 0);
    for (let i = 0; i < vents; i++) {
      const glow = uniform(0);
      const along = positionLocal.y.div(LENGTH);
      // Soft at its sides (seen edge on, the gas is thin), fading from the nozzle out and near the eye (the gas
      // right by you would only wash over the view), streaked by turbulence blown out along it at two scales
      const facing = pow(abs(dot(normalView, positionViewDirection)), 2.5);
      const fade = pow(float(1).sub(along), 1.6).mul(smoothstep(0, 0.03, along)).mul(smoothstep(0.15, 0.6, positionView.length()));
      const p = positionLocal.add(vec3(0, time.mul(-16), i * 7));
      const flow = mx_noise_float(p.mul(vec3(14, 9, 14))).mul(0.6).add(mx_noise_float(p.mul(vec3(40, 22, 40))).mul(0.3));
      const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthWrite: false, side: THREE.DoubleSide, blending: THREE.AdditiveBlending, fog: false });
      material.colorNode = vec3(0.9, 0.93, 1).mul(glow.mul(facing).mul(fade).mul(flow.add(0.7).clamp(0, 1.5)).mul(1.6)) as unknown as THREE.Node<'color'>;
      const mesh = new THREE.Mesh(geometry, material);
      mesh.visible = false;
      mesh.frustumCulled = false;
      this.object.add(mesh);
      this.plumes.push({ mesh, glow, level: 0, flash: 0, drift: new THREE.Vector3(), due: 0 });
    }
    // The glints: a thin quad each, along its flight, bright at its head and fading down its tail and to its edges
    const streak = new THREE.MeshBasicNodeMaterial({ transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, fog: false });
    const across = float(1).sub(abs(uv().x.mul(2).sub(1)));
    streak.colorNode = vec3(0.95, 0.97, 1).mul(across.mul(across).mul(pow(uv().y, 1.5)).mul(3)) as unknown as THREE.Node<'color'>;
    this.streaks = new THREE.InstancedMesh(new THREE.PlaneGeometry(1, 1).translate(0, 0.5, 0), streak, GLINTS);
    this.streaks.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(GLINTS * 3), 3);
    this.streaks.count = 0;
    this.streaks.frustumCulled = false;
    this.object.add(this.streaks);
  }

  /**
   * Each frame: the eye and the suit's velocity (world), the vents (world), how hard each fires (0..1, 0 when it's
   * off) and the way the gas goes (unit, world).
   */
  update(dt: number, eye: THREE.Vector3, velocity: THREE.Vector3, vents: THREE.Vector3[], amounts: number[], away: THREE.Vector3): void {
    this.plumes.forEach((plume, i) => {
      const { mesh } = plume;
      const amount = amounts[i];
      if (amount > 0 && plume.level < 0.05) plume.flash = 1;
      plume.level += (amount - plume.level) * (1 - Math.exp(-dt / (amount > plume.level ? ATTACK : RELEASE)));
      plume.flash *= Math.exp(-dt / FLASH);
      plume.glow.value = plume.level * (1 + 0.6 * plume.flash);
      mesh.visible = plume.level > 0.01;
      if (!mesh.visible) return;
      if (amount > 0) {
        // Firing: at its vent, along the gas; shut, what's left drifts off
        mesh.position.copy(vents[i]);
        mesh.quaternion.setFromUnitVectors(UP, away);
        plume.drift.copy(away).multiplyScalar(GAS).add(velocity);
        plume.due += amount * GLINT_RATE * dt;
        for (; plume.due >= 1; plume.due--) this.glint(vents[i], away, velocity);
      } else mesh.position.addScaledVector(plume.drift, dt);
      // Longer and wider the harder they push and as they light, with a little flutter
      const size = (0.55 + 0.45 * plume.level) * (1 + 0.12 * plume.flash) * (0.94 + Math.random() * 0.12);
      mesh.scale.set(size, size * (0.9 + 0.2 * plume.level), size);
    });
    this.flyGlints(dt, eye);
  }

  /** A glint thrown from `at` along `away` (within a narrow cone), on top of the suit's own `velocity`. */
  private glint(at: THREE.Vector3, away: THREE.Vector3, velocity: THREE.Vector3): void {
    if (this.glints.length >= GLINTS) return;
    const spread = new THREE.Vector3(Math.random() - 0.5, Math.random() - 0.5, Math.random() - 0.5).multiplyScalar(0.35);
    const v = away.clone().add(spread).normalize().multiplyScalar(GAS * (1.6 + Math.random() * 1.2)).add(velocity);
    const life = GLINT_LIFE[0] + Math.random() * (GLINT_LIFE[1] - GLINT_LIFE[0]);
    this.glints.push({ at: at.clone().addScaledVector(away, 0.05), velocity: v, age: 0, life, bright: 0.3 + Math.random() * 0.7 });
  }

  /** Move the glints on and lay out their streaks: each along its flight, turned to face the eye. */
  private flyGlints(dt: number, eye: THREE.Vector3): void {
    const m = new THREE.Matrix4();
    const [dir, side, face, tail] = [new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3()];
    const colour = new THREE.Color();
    let n = 0;
    for (const g of this.glints) {
      g.age += dt;
      g.at.addScaledVector(g.velocity, dt);
      if (g.age >= g.life) continue;
      this.glints[n] = g;
      const speed = g.velocity.length();
      // A streak as long as it moves in about 1/30 s (its blur), from its tail up to where it is
      const length = Math.min(0.3, speed / 30);
      dir.copy(g.velocity).divideScalar(speed);
      face.subVectors(eye, g.at).normalize();
      side.crossVectors(dir, face).normalize().multiplyScalar(GLINT_WIDTH);
      face.crossVectors(side, dir).normalize();
      tail.copy(g.at).addScaledVector(dir, -length);
      m.makeBasis(side, dir.multiplyScalar(length), face).setPosition(tail);
      this.streaks.setMatrixAt(n, m);
      // Twinkling as it tumbles, gone as it spreads
      const fade = (1 - g.age / g.life) ** 2 * g.bright * (0.6 + 0.4 * Math.sin(g.age * 60 + g.bright * 20));
      this.streaks.setColorAt(n, colour.setScalar(fade));
      n++;
    }
    this.glints.length = n;
    this.streaks.count = n;
    this.streaks.instanceMatrix.needsUpdate = true;
    this.streaks.instanceColor!.needsUpdate = true;
  }
}
