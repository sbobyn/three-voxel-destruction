// Satellites under the station (space.ts): a few small ones on orbits a little below ours, each drifting past
// every minute or so, following the Earth's curve (up from far off, under us, and away again). A gold-foil body,
// blue solar wings, a slow tumble and a blinking light. In the station's frame: space.ts sways them with the Earth.

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

export class Satellites {
  readonly object = new THREE.Group();
  private readonly craft: { mesh: THREE.Group; pass: Pass; spin: number }[] = [];
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
      this.craft.push({ mesh, pass, spin: (i % 2 ? 1 : -1) * (0.05 + 0.04 * i) });
    });
  }

  /** Each frame: where each is along its pass at `time` (s). */
  update(time: number): void {
    for (const { mesh, pass, spin } of this.craft) {
      const along = ((((time * pass.speed) / TRACK + pass.phase) % 1) - 0.5) * TRACK;
      const [c, s] = [Math.cos(pass.heading), Math.sin(pass.heading)];
      // Along its heading, off to the side, and down the Earth's curve the further off it is
      mesh.position.set(this.centre.x + c * along - s * pass.offset, this.centre.y + s * along + c * pass.offset, this.centre.z - pass.depth - (along * along) / (2 * this.curve));
      mesh.rotation.set(0.3 + spin * time, 0, pass.heading + 0.4 * Math.sin(spin * time));
    }
  }
}
