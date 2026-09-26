// The tool in hand, drawn in front of the camera: a sledgehammer, a laser, a rocket tube, a
// detonator, a wrecking ball on a chain. Simple shapes, lit like the world; it sways against
// the mouse, bobs with each step, dips while sprinting and kicks (or swings) when used.

import * as THREE from 'three/webgpu';

const metal = (c: number, rough = 0.35) => new THREE.MeshStandardMaterial({ color: c, metalness: 0.85, roughness: rough });
const matte = (c: number, rough = 0.7) => new THREE.MeshStandardMaterial({ color: c, metalness: 0, roughness: rough });
const glow = (c: number) => new THREE.MeshStandardMaterial({ color: 0x111111, emissive: c, emissiveIntensity: 3 });

function part(geometry: THREE.BufferGeometry, material: THREE.Material, x: number, y: number, z: number, rx = 0, ry = 0, rz = 0): THREE.Mesh {
  const m = new THREE.Mesh(geometry, material);
  m.position.set(x, y, z);
  m.rotation.set(rx, ry, rz);
  m.castShadow = false;
  return m;
}

/** The detonator's button: lit (armed) and dark, its light off (pressed, until armed again). */
const BUTTON_LIT = new THREE.MeshStandardMaterial({ color: 0x3a0000, emissive: 0xff0c08, emissiveIntensity: 2 });
const BUTTON_DARK = matte(0x4a0806, 0.4);
/** The models, in camera space (x right, y up, -z ahead), their grip at the origin. */
function models(): THREE.Group[] {
  // A sledgehammer: a long fibreglass handle with a rubber grip, a heavy octagonal steel head
  // across the top in the plane of the swing (a face leads into the blow), its faces worn bright
  const hammer = new THREE.Group();
  hammer.add(part(new THREE.CylinderGeometry(0.017, 0.02, 0.82, 12), matte(0xd9a21b, 0.45), 0, 0.36, 0));
  hammer.add(part(new THREE.CylinderGeometry(0.023, 0.021, 0.2, 12), matte(0x151515, 0.95), 0, 0.02, 0));
  hammer.add(part(new THREE.CylinderGeometry(0.021, 0.024, 0.08, 12), matte(0x151515, 0.95), 0, 0.72, 0));
  hammer.add(part(new THREE.CylinderGeometry(0.058, 0.058, 0.26, 8), metal(0x3a3d42, 0.5), 0, 0.8, 0, Math.PI / 2, 0, Math.PI / 8));
  for (const z of [-0.133, 0.133]) hammer.add(part(new THREE.CylinderGeometry(0.052, 0.058, 0.012, 8), metal(0xb8bcc2, 0.25), 0, 0.8, z, Math.PI / 2, 0, Math.PI / 8));
  hammer.add(part(new THREE.BoxGeometry(0.05, 0.03, 0.12), metal(0x2a2c30, 0.6), 0, 0.745, 0));

  // A laser: a squat emitter body, a finned barrel ringed with red coils, a glowing cell
  const blaster = new THREE.Group();
  blaster.add(part(new THREE.BoxGeometry(0.08, 0.1, 0.32), metal(0x23262b, 0.35), 0, 0.05, -0.12));
  blaster.add(part(new THREE.BoxGeometry(0.05, 0.13, 0.06), matte(0x151515), 0, -0.05, -0.01, 0.25));
  blaster.add(part(new THREE.CylinderGeometry(0.03, 0.034, 0.3, 16), metal(0x5a5f66, 0.3), 0, 0.075, -0.42, Math.PI / 2));
  for (let k = 0; k < 4; k++) blaster.add(part(new THREE.TorusGeometry(0.036, 0.007, 8, 20), glow(0xff2a14), 0, 0.075, -0.32 - k * 0.06));
  blaster.add(part(new THREE.CylinderGeometry(0.018, 0.018, 0.02, 16), glow(0xff5030), 0, 0.075, -0.575, Math.PI / 2));
  blaster.add(part(new THREE.BoxGeometry(0.07, 0.025, 0.16), glow(0xff3a20), 0, 0.11, -0.1));
  for (let k = 0; k < 3; k++) blaster.add(part(new THREE.BoxGeometry(0.1, 0.012, 0.02), metal(0x3a3e45, 0.4), 0, 0.05, -0.02 - k * 0.05));
  const muzzle = new THREE.Object3D();
  muzzle.name = 'muzzle';
  muzzle.position.set(0, 0.075, -0.59);
  blaster.add(muzzle);

  const rocket = new THREE.Group();
  rocket.add(part(new THREE.CylinderGeometry(0.075, 0.075, 0.95, 20), matte(0x4e5a3a, 0.65), 0, 0.09, -0.2, Math.PI / 2));
  rocket.add(part(new THREE.CylinderGeometry(0.085, 0.085, 0.07, 20), metal(0x3a3a3a), 0, 0.09, -0.66, Math.PI / 2));
  rocket.add(part(new THREE.BoxGeometry(0.04, 0.12, 0.05), matte(0x1a1a1a), 0, -0.02, 0.02, 0.2));
  rocket.add(part(new THREE.ConeGeometry(0.06, 0.12, 16), matte(0x8a2a1a), 0, 0.09, -0.74, -Math.PI / 2));

  // A detonator: a chunky remote held flat, a big red button on top in a guard ring (lit when
  // armed, dark once pressed), a status light, and the antenna at the back
  const charge = new THREE.Group();
  charge.add(part(new THREE.BoxGeometry(0.095, 0.045, 0.15), matte(0x2a2c2f, 0.55), 0, 0, -0.08));
  charge.add(part(new THREE.BoxGeometry(0.1, 0.012, 0.155), matte(0xd9a21b, 0.5), 0, -0.018, -0.08));
  charge.add(part(new THREE.TorusGeometry(0.03, 0.006, 8, 24), metal(0x9a9ea4, 0.35), 0, 0.024, -0.1, Math.PI / 2));
  // The button: lit red while armed, dark (its light off) from the press until armed again
  const button = part(new THREE.CylinderGeometry(0.024, 0.026, 0.018, 24), BUTTON_LIT, 0, 0.028, -0.1);
  button.name = 'button';
  charge.add(button);
  charge.add(part(new THREE.BoxGeometry(0.012, 0.006, 0.012), glow(0xffb020), 0.032, 0.024, -0.035));
  charge.add(part(new THREE.CylinderGeometry(0.005, 0.005, 0.12, 8), metal(0x888888), -0.03, 0.08, -0.145));
  charge.add(part(new THREE.SphereGeometry(0.008, 8, 6), matte(0xd02a1a, 0.4), -0.03, 0.14, -0.145));

  const ball = new THREE.Group();
  ball.add(part(new THREE.SphereGeometry(0.16, 24, 16), metal(0x2e3034, 0.5), 0, 0.1, -0.3));
  ball.add(part(new THREE.TorusGeometry(0.03, 0.008, 8, 16), metal(0x6a6a6a), 0, 0.28, -0.3, 0, Math.PI / 2));
  return [hammer, blaster, rocket, charge, ball];
}

/** The hammer's swing (s): wind-up, blow, recovery. */
const SWING = 0.31;
const ease = (x: number) => x * x * (3 - 2 * x);

export class ViewModel {
  readonly object = new THREE.Group();
  private readonly tools = models();
  private current = 0;
  /** Recoil / swing (0..1, decays), sway from the mouse, and the switch-in lift. */
  private kick = 0;
  private sway = new THREE.Vector2();
  private raise = 1;

  constructor() {
    for (const t of this.tools) {
      t.visible = false;
      this.object.add(t);
    }
    this.tools[0].visible = true;
  }

  select(i: number): void {
    if (i === this.current) return;
    this.tools[this.current].visible = false;
    this.current = i;
    this.tools[i].visible = true;
    this.raise = 1;
  }

  /**
   * Used the tool: a swing for the hammer, the detonator's button pressed (armed again after
   * `ready` s), a kick for the rest.
   */
  use(ready = 0): void {
    if (this.current === 3) {
      this.press = 0;
      this.rearm = ready;
    } else this.kick = 1;
  }
  /** The detonator: seconds since its button was pressed, and until it's armed again (lit). */
  private press = Infinity;
  private rearm = 0;

  /** The blow landed: the swing holds still for a moment (a hit-stop sells the weight). */
  hitstop(): void {
    this.stopped = 0.06;
  }
  private stopped = 0;

  /** Whether the laser is firing (it trembles in the hand). */
  firing = false;

  /** Seconds from starting a swing to the hammer landing (the blow lands then). */
  static readonly STRIKE = 0.12;

  /** Where the laser's beam leaves the barrel (world m). */
  muzzle(out: THREE.Vector3): THREE.Vector3 {
    const m = this.tools[1].getObjectByName('muzzle')!;
    m.updateWorldMatrix(true, false);
    return m.getWorldPosition(out);
  }

  /** Mouse movement this frame (pixels): the tool lags behind the look. */
  look(dx: number, dy: number): void {
    this.sway.x = Math.max(-1, Math.min(1, this.sway.x - dx * 0.004));
    this.sway.y = Math.max(-1, Math.min(1, this.sway.y + dy * 0.004));
  }

  /** Place it for this frame: walking phase (radians), speed share (0..1), sprinting. */
  update(dt: number, walk: number, speed: number, sprint: boolean): void {
    // The hammer's swing takes SWING s (held still a beat on the blow); the rest kick back briefly
    this.stopped = Math.max(0, this.stopped - dt);
    if (!this.stopped) this.kick = Math.max(0, this.kick - dt * (this.current === 0 ? 1 / SWING : 6));
    this.raise = Math.max(0, this.raise - dt * 4);
    this.press += dt;
    this.rearm = Math.max(0, this.rearm - dt);
    this.sway.multiplyScalar(Math.exp(-dt * 9));
    const t = this.tools[this.current];
    const bob = Math.sin(walk) * 0.012 * speed;
    const side = Math.cos(walk * 0.5) * 0.01 * speed;
    const run = sprint ? 1 : 0;
    t.position.set(0.22 + side + this.sway.x * 0.03 - run * 0.04, -0.2 + bob - this.raise * 0.25 + this.sway.y * 0.02 - run * 0.05, -0.38);
    if (this.current === 0) {
      // The hammer rests on the shoulder; a swing draws it back, brings it down hard (landing
      // at STRIKE s), holds a beat on the blow, and lifts it back to the shoulder
      const u = this.kick > 0 ? 1 - this.kick : 1;
      const rest = -0.35;
      const angle =
        u < 0.2 ? rest + (0.45 - rest) * ease(u / 0.2)
        : u < 0.39 ? 0.45 + (-1.75 - 0.45) * (((u - 0.2) / 0.19) ** 2)
        : u < 0.52 ? -1.75 + 0.08 * Math.sin(((u - 0.39) / 0.13) * Math.PI)
        : -1.75 + (rest + 1.75) * ease((u - 0.52) / 0.48);
      const lean = u < 0.52 ? Math.sin(Math.min(1, u / 0.39) * Math.PI * 0.5) : 1 - ease((u - 0.52) / 0.48);
      t.rotation.set(angle + run * 0.4, 0.15 + this.sway.x * 0.2 - lean * 0.15, -0.35 + lean * 0.25);
      t.position.y += lean * 0.12;
      t.position.x -= lean * 0.17;
    } else {
      t.rotation.set(this.kick * 0.25 + this.sway.y * 0.1 - run * 0.5, this.sway.x * 0.15 + run * 0.6, 0);
      t.position.z += this.kick * 0.06;
      if (this.current === 3) {
        // A thumb pressing the button: the remote tips forward and rolls a touch as it goes in,
        // then eases back; the button sinks and goes dark, and lights again once armed
        const p = this.press;
        const tip = p < 0.07 ? ease(p / 0.07) : p < 0.16 ? 1 : p < 0.45 ? 1 - ease((p - 0.16) / 0.29) : 0;
        t.rotation.x += tip * 0.16;
        t.rotation.z -= tip * 0.07;
        t.position.y -= tip * 0.012;
        const button = t.getObjectByName('button') as THREE.Mesh;
        button.position.y = 0.028 - tip * 0.009;
        button.material = this.rearm > 0 ? BUTTON_DARK : BUTTON_LIT;
      }
      if (this.current === 1 && this.firing) {
        t.position.x += (Math.random() - 0.5) * 0.004;
        t.position.y += (Math.random() - 0.5) * 0.004;
        t.position.z += 0.012;
      }
    }
  }
}
