// Satellites under the station (space.ts): a few small ones on orbits a little below ours, each drifting past
// every minute or so, following the Earth's curve (up from far off, under us, and away again). A gold-foil body,
// blue solar wings, a slow tumble and a blinking light. In the station's frame: space.ts sways them with the Earth.
// Hit one (main.ts looks for them along rockets and the laser, and in blasts) and it bursts: foil chunks, wing
// plates and struts tumbling off, shrinking away to nothing in a few seconds; its next pass it's back.

import * as THREE from 'three/webgpu';
import { float, fract, smoothstep, time, vec3 } from 'three/tsl';

interface Pass {
  /** How far below the station it passes (m), its heading (rad), how far to the side (m), speed (m/s), where it starts. */
  depth: number;
  heading: number;
  offset: number;
  speed: number;
  phase: number;
  /** Its size (m, the body's). */
  size: number;
}

const PASSES: Pass[] = [
  { depth: 45, heading: 0.3, offset: 30, speed: 22, phase: 0.45, size: 2.5 },
  { depth: 90, heading: 2.1, offset: -60, speed: 35, phase: 0.1, size: 3.5 },
  { depth: 120, heading: -1.2, offset: 80, speed: 30, phase: 0.75, size: 4 },
  { depth: 65, heading: 2.8, offset: -20, speed: 18, phase: 0.3, size: 3 },
];
/** How far each goes before it comes round again (m): so far off at both ends it's gone. */
const TRACK = 2400;
/** A craft's reach round its middle, for hits (its sizes: out along its wings, most of the way). */
const REACH = 1.8;
/** Its wreckage: how fast the pieces fly apart (m/s) and how long they last (s; the last of it shrinking away). */
const BURST = [3, 9];
const WRECK_LIFE = [2.5, 4];
const SHRINK = 1;

interface Craft {
  mesh: THREE.Group;
  pass: Pass;
  spin: number;
  /** Where along its pass it was (m), and its velocity (m/s, the station's frame). */
  along: number;
  velocity: THREE.Vector3;
  broken: boolean;
}

interface Piece {
  mesh: THREE.Mesh;
  velocity: THREE.Vector3;
  spin: THREE.Vector3;
  size: THREE.Vector3;
  age: number;
  life: number;
}

export class Satellites {
  readonly object = new THREE.Group();
  private readonly craft: Craft[] = [];
  private readonly pieces: Piece[] = [];
  private readonly block = new THREE.BoxGeometry(1, 1, 1);
  private readonly materials: THREE.Material[];
  private last = -1;
  private readonly centre: THREE.Vector3;
  /** The Earth's curve under them: how far below the station the Earth's middle is (m). */
  private readonly curve: number;

  /** Passing under `centre` (the station's middle), `curve` above the Earth's middle. */
  constructor(centre: THREE.Vector3, curve: number) {
    this.centre = centre.clone();
    this.curve = curve;
    const foil = new THREE.MeshStandardMaterial({ color: new THREE.Color(0.85, 0.6, 0.22), metalness: 0.6, roughness: 0.35 });
    const cells = new THREE.MeshStandardMaterial({ color: new THREE.Color(0.08, 0.13, 0.32), metalness: 0.4, roughness: 0.3 });
    const frame = new THREE.MeshStandardMaterial({ color: new THREE.Color(0.7, 0.7, 0.72), metalness: 0.5, roughness: 0.5 });
    this.materials = [foil, cells, frame];
    PASSES.forEach((pass, i) => {
      const s = pass.size;
      const mesh = new THREE.Group();
      mesh.add(new THREE.Mesh(new THREE.BoxGeometry(s, s, s * 1.5), foil));
      // Its wings on a boom either side, and a dish on its end
      for (const side of [-1, 1]) {
        const wing = new THREE.Mesh(new THREE.BoxGeometry(s * 2.6, s * 0.9, s * 0.04), cells);
        wing.position.x = side * s * 2;
        mesh.add(wing);
      }
      const boom = new THREE.Mesh(new THREE.BoxGeometry(s * 1.4, s * 0.06, s * 0.06), frame);
      mesh.add(boom);
      const dish = new THREE.Mesh(new THREE.CylinderGeometry(s * 0.45, s * 0.1, s * 0.2, 16), frame);
      dish.rotation.x = Math.PI / 2;
      dish.position.z = s * 0.85;
      mesh.add(dish);
      // The light: a red blink a second or so apart, each its own time
      const blink = new THREE.MeshBasicNodeMaterial({ fog: false });
      const on = smoothstep(0.08, 0, fract(time.mul(0.8).add(i * 0.37)));
      blink.colorNode = vec3(1, 0.12, 0.08).mul(on.mul(8).add(float(0.3))) as unknown as THREE.Node<'color'>;
      const light = new THREE.Mesh(new THREE.SphereGeometry(s * 0.08, 8, 6), blink);
      light.position.set(0, s * 0.55, -s * 0.75);
      mesh.add(light);
      this.object.add(mesh);
      this.craft.push({ mesh, pass, spin: (i % 2 ? 1 : -1) * (0.05 + 0.04 * i), along: 0, velocity: new THREE.Vector3(), broken: false });
    });
  }

  /** Each frame: where each is along its pass at `time` (s), and the wreckage flying. */
  update(time: number): void {
    const dt = this.last < 0 ? 0 : Math.max(0, time - this.last);
    this.last = time;
    for (const craft of this.craft) {
      const { mesh, pass, spin } = craft;
      const along = ((((time * pass.speed) / TRACK + pass.phase) % 1) - 0.5) * TRACK;
      // Round again (far off, out of sight): a new one, whole
      if (along < craft.along) {
        craft.broken = false;
        mesh.visible = true;
      }
      craft.along = along;
      const [c, s] = [Math.cos(pass.heading), Math.sin(pass.heading)];
      // Along its heading, off to the side, and down the Earth's curve the further off it is
      mesh.position.set(this.centre.x + c * along - s * pass.offset, this.centre.y + s * along + c * pass.offset, this.centre.z - pass.depth - (along * along) / (2 * this.curve));
      mesh.rotation.set(0.3 + spin * time, 0, pass.heading + 0.4 * Math.sin(spin * time));
      craft.velocity.set(c * pass.speed, s * pass.speed, (-along * pass.speed) / this.curve);
    }
    this.fly(dt);
  }

  /** The nearest craft along the ray (world) within `reach` (m): how far, and which; or null. */
  hit(from: THREE.Vector3, dir: THREE.Vector3, reach: number): { distance: number; craft: number } | null {
    let best: { distance: number; craft: number } | null = null;
    const at = new THREE.Vector3();
    this.craft.forEach((craft, i) => {
      if (craft.broken) return;
      craft.mesh.getWorldPosition(at).sub(from);
      const along = at.dot(dir);
      const r = craft.pass.size * REACH;
      const miss = at.lengthSq() - along * along;
      if (miss > r * r) return;
      const distance = along - Math.sqrt(r * r - miss);
      if (distance >= 0 && distance <= reach && (!best || distance < best.distance)) best = { distance, craft: i };
    });
    return best;
  }

  /** Break every craft within `radius` (m) of `at` (world): where each was (world). */
  blast(at: THREE.Vector3, radius: number): THREE.Vector3[] {
    const hit: THREE.Vector3[] = [];
    this.craft.forEach((craft, i) => {
      const where = craft.mesh.getWorldPosition(new THREE.Vector3());
      if (!craft.broken && where.distanceTo(at) < radius + craft.pass.size * REACH) {
        this.smash(i, at);
        hit.push(where);
      }
    });
    return hit;
  }

  /**
   * Break craft `i`, the blow from `from` (world): gone, and in its place its wreck flying apart, away from the blow,
   * carried on at the craft's own speed. Foil chunks from its body, plates of its wings, bits of strut.
   */
  smash(i: number, from: THREE.Vector3): void {
    const craft = this.craft[i];
    if (craft.broken) return;
    craft.broken = true;
    craft.mesh.visible = false;
    const s = craft.pass.size;
    const [foil, cells, frame] = this.materials;
    const parts: [THREE.Material, number, number, number, number][] = [
      ...Array.from({ length: 8 }, (): [THREE.Material, number, number, number, number] => [foil, s * 0.4, s * 0.4, s * 0.5, s * 0.5]),
      ...Array.from({ length: 8 }, (): [THREE.Material, number, number, number, number] => [cells, s * 0.8, s * 0.6, s * 0.04, s * 2.2]),
      ...Array.from({ length: 4 }, (): [THREE.Material, number, number, number, number] => [frame, s * 0.5, s * 0.06, s * 0.06, s]),
    ];
    // (The blow, into the craft's own frame: the wreck flies in the station's frame, as the craft did)
    const centre = craft.mesh.position;
    const away = this.object.worldToLocal(from.clone()).sub(centre).negate().normalize();
    for (const [material, x, y, z, spread] of parts) {
      const mesh = new THREE.Mesh(this.block, material);
      const size = new THREE.Vector3(x, y, z).multiplyScalar(0.6 + Math.random() * 0.6);
      mesh.scale.copy(size);
      mesh.position.copy(centre).add(new THREE.Vector3(Math.random() - 0.5, Math.random() - 0.5, Math.random() - 0.5).multiplyScalar(spread));
      mesh.quaternion.copy(craft.mesh.quaternion);
      const out = mesh.position.clone().sub(centre).normalize().add(away).normalize();
      const velocity = out.multiplyScalar(BURST[0] + Math.random() * (BURST[1] - BURST[0])).add(craft.velocity);
      const spin = new THREE.Vector3(Math.random() - 0.5, Math.random() - 0.5, Math.random() - 0.5).multiplyScalar(8);
      this.object.add(mesh);
      this.pieces.push({ mesh, velocity, spin, size, age: 0, life: WRECK_LIFE[0] + Math.random() * (WRECK_LIFE[1] - WRECK_LIFE[0]) });
    }
  }

  /** The wreckage flies on, tumbling; the last of each piece's life it shrinks away, then it's gone. */
  private fly(dt: number): void {
    let n = 0;
    for (const piece of this.pieces) {
      piece.age += dt;
      if (piece.age >= piece.life) {
        this.object.remove(piece.mesh);
        continue;
      }
      this.pieces[n++] = piece;
      piece.mesh.position.addScaledVector(piece.velocity, dt);
      piece.mesh.rotation.x += piece.spin.x * dt;
      piece.mesh.rotation.y += piece.spin.y * dt;
      piece.mesh.rotation.z += piece.spin.z * dt;
      piece.mesh.scale.copy(piece.size).multiplyScalar(Math.min(1, (piece.life - piece.age) / SHRINK));
    }
    this.pieces.length = n;
  }
}
