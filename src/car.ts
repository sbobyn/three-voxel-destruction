// The race car: a low papaya-orange supercar built of voxels (half the world's size, so it
// reads as a car and not a crate), driven on the CPU with arcade handling (grip that lets go
// under the handbrake, speed-sensitive steering, a boost), and a pair of cannons that rise on
// yellow arms from under the rear deck. The world's physics sees it as a box moved each step
// (physics.ts drive): it shoves debris aside; what it drives into that still stands, it breaks
// or is stopped by (main.ts), by the momentum it carries into it.

import * as THREE from 'three/webgpu';
import { attribute, float, vec3 } from 'three/tsl';
import { edgeShade } from './look.ts';
import { type City, State, VOXEL } from './world.ts';

/** The model's voxel (m): half the world's. */
const CELL = VOXEL / 2;
/** The car's box for collisions and the solver: length, width, height (m), and its underside's height. */
export const CAR_SIZE = [4.6, 2.0, 1.15];
export const CLEARANCE = 0.12;
/** kg: what a crash weighs its momentum against. */
export const CAR_MASS = 1300;

export interface Drive {
  /** -1 (brake, then reverse) to 1 (throttle). */
  throttle: number;
  /** -1 (right) to 1 (left). */
  steer: number;
  handbrake: boolean;
  boost: boolean;
}

// The body, in cells: x from the tail (0) to the nose (38), y across (0 to 16), z up
const LEN = 38;
const WID = 16;
const PAPAYA = 0xff7a12;
const CARBON = 0x17181a;
const GLASS = 0x121c24;

type Finish = { colour: number; rough: number; metal: number; glow: number };
const PAINT: Finish = { colour: PAPAYA, rough: 0.28, metal: 0.35, glow: 0 };
const BLACK: Finish = { colour: CARBON, rough: 0.5, metal: 0.2, glow: 0 };
const TINT: Finish = { colour: GLASS, rough: 0.08, metal: 0.6, glow: 0 };
const HEAD: Finish = { colour: 0xe8f2ff, rough: 0.3, metal: 0, glow: 1.2 };
const TAIL: Finish = { colour: 0xe0140c, rough: 0.3, metal: 0, glow: 0.9 };
const WHITE: Finish = { colour: 0xf2efe8, rough: 0.35, metal: 0.1, glow: 0 };
const YELLOW: Finish = { colour: 0xf2c21a, rough: 0.45, metal: 0.3, glow: 0 };
const GUNMETAL: Finish = { colour: 0x2c3036, rough: 0.35, metal: 0.8, glow: 0 };
const TYRE: Finish = { colour: 0x141414, rough: 0.85, metal: 0, glow: 0 };
const RIM: Finish = { colour: 0xb8bcc2, rough: 0.25, metal: 0.9, glow: 0 };

/** Cells → a mesh of their open faces, edge-shaded like the world's voxels (centred on `origin`, in cells). */
function voxelMesh(cells: Map<string, Finish>, origin: [number, number, number]): THREE.Mesh {
  const pos: number[] = [];
  const nor: number[] = [];
  const col: number[] = [];
  const local: number[] = [];
  const rough: number[] = [];
  const metal: number[] = [];
  const glow: number[] = [];
  const colour = new THREE.Color();
  const faces: [number[], number[][]][] = [
    [[1, 0, 0], [[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]]],
    [[-1, 0, 0], [[0, 1, 0], [0, 0, 0], [0, 0, 1], [0, 1, 1]]],
    [[0, 1, 0], [[1, 1, 0], [0, 1, 0], [0, 1, 1], [1, 1, 1]]],
    [[0, -1, 0], [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]]],
    [[0, 0, 1], [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]],
    [[0, 0, -1], [[0, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, 0]]],
  ];
  for (const [key, f] of cells) {
    const [x, y, z] = key.split(',').map(Number);
    colour.setHex(f.colour, THREE.SRGBColorSpace);
    for (const [n, quad] of faces) {
      if (cells.has(`${x + n[0]},${y + n[1]},${z + n[2]}`)) continue;
      for (const k of [0, 1, 2, 0, 2, 3]) {
        const c = quad[k];
        pos.push((x + c[0] - origin[0]) * CELL, (y + c[1] - origin[1]) * CELL, (z + c[2] - origin[2]) * CELL);
        nor.push(n[0], n[1], n[2]);
        col.push(colour.r, colour.g, colour.b);
        local.push(c[0] - 0.5, c[1] - 0.5, c[2] - 0.5);
        rough.push(f.rough);
        metal.push(f.metal);
        glow.push(f.glow);
      }
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3));
  g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  g.setAttribute('local', new THREE.Float32BufferAttribute(local, 3));
  g.setAttribute('rough', new THREE.Float32BufferAttribute(rough, 1));
  g.setAttribute('metal', new THREE.Float32BufferAttribute(metal, 1));
  g.setAttribute('glow', new THREE.Float32BufferAttribute(glow, 1));
  const m = new THREE.MeshStandardNodeMaterial();
  // A lighter edge than the world's voxels: at half their size, the full darkening reads as tiles
  const shade = edgeShade(attribute('local', 'vec3') as unknown as THREE.Node<'vec3'>, vec3(1, 1, 1), 0.08).mul(0.5).add(0.5);
  const base = attribute('color', 'vec3').mul(shade);
  m.colorNode = base as unknown as THREE.Node<'color'>;
  m.roughnessNode = attribute('rough', 'float') as unknown as THREE.Node<'float'>;
  m.metalnessNode = attribute('metal', 'float') as unknown as THREE.Node<'float'>;
  m.emissiveNode = attribute('color', 'vec3').mul(attribute('glow', 'float')).add(float(0)) as unknown as THREE.Node<'color'>;
  const mesh = new THREE.Mesh(g, m);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

/** The body's cells (wheels and cannons apart). */
function bodyCells(): Map<string, Finish> {
  const cells = new Map<string, Finish>();
  const put = (x: number, y: number, z: number, f: Finish) => cells.set(`${x},${y},${z}`, f);
  const mirror = (x: number, y: number, z: number, f: Finish) => {
    put(x, y, z, f);
    put(x, WID - 1 - y, z, f);
  };
  // Wheel arches: front wheels at x 29-35, rear at 5-12 (cells), open to the outer 3 cells
  const arch = (x: number, y: number, z: number) => {
    const inner = y >= 3 && y <= WID - 4;
    if (inner) return false;
    const front = Math.hypot(x + 0.5 - 32, z + 0.5 - 2.7) < 3.6;
    const rear = Math.hypot(x + 0.5 - 8.5, z + 0.5 - 2.9) < 3.8;
    return front || rear;
  };
  // The body's top at each station along it: a low nose, the windscreen rising to the roof,
  // the roof falling away over the engine to a high tail
  const top = (x: number): number => {
    if (x >= 34) return 3;
    if (x >= 27) return 4;
    if (x >= 24) return 5 + (27 - x) * 0.7;
    if (x >= 14) return 8;
    if (x >= 5) return 8 - (14 - x) * 0.34;
    return 5;
  };
  // How far in from the side the body's top narrows (the cabin is narrower than the hips)
  const inset = (x: number, z: number): number => (z >= 6 ? (x >= 14 && x < 24 ? 3 : 4) : z >= 5 ? 1 : 0);
  for (let x = 0; x < LEN; x++) {
    const t = Math.floor(top(x));
    for (let z = 1; z <= t; z++) {
      const i = inset(x, z) + (x >= 36 ? 1 : 0) + (x === 37 ? 1 : 0);
      for (let y = i; y < WID - i; y++) {
        if (arch(x, y, z)) continue;
        // The glass: the windscreen and the cabin's sides and back window
        const cabin = x >= 14 && x < 27 && z >= 5;
        const pillar = cabin && (x === 14 || x === 26) && (y === i || y === WID - 1 - i);
        const glass = cabin && !pillar && z < t + 1 && (z >= 6 || x >= 24) && !(z === t && x < 24 && y > i && y < WID - 1 - i);
        let f: Finish = glass ? TINT : PAINT;
        // Carbon: the sills, the splitter, the diffuser, the roof scoop
        if (z === 1) f = BLACK;
        if (x >= 36 && z <= 2) f = BLACK;
        if (x < 3 && z <= 2) f = BLACK;
        if (z === t && x >= 15 && x < 21 && y >= 6 && y <= 9 && z >= 8) f = BLACK;
        // Side intakes behind the doors
        if ((y === 0 || y === WID - 1) && x >= 12 && x < 16 && z >= 2 && z <= 4) f = BLACK;
        put(x, y, z, f);
      }
    }
  }
  // A white stripe up the middle over the nose and the roof
  for (let x = 0; x < LEN; x++) {
    const t = Math.floor(top(x));
    for (const y of [7, 8]) {
      const key = `${x},${y},${t}`;
      if (cells.get(key) === PAINT) cells.set(key, WHITE);
    }
  }
  // Headlights, tail lights
  for (let y = 2; y < 5; y++) mirror(37, y, 2, HEAD);
  for (let y = 1; y < WID - 1; y++) put(0, y, 4, y % 5 === 2 ? BLACK : TAIL);
  // The rear wing on two struts
  for (let x = 0; x < 4; x++) for (let y = 0; y < WID; y++) put(x, y, 10, y === 0 || y === WID - 1 ? BLACK : PAINT);
  for (const y of [4, 11]) for (let z = 6; z < 10; z++) put(2, y, z, BLACK);
  // Mirrors
  mirror(24, -1, 6, BLACK);
  mirror(25, -1, 6, BLACK);
  return cells;
}

/** A wheel's cells: tyre round a silver rim, `width` cells wide along y. */
function wheelCells(radius: number, width: number): Map<string, Finish> {
  const cells = new Map<string, Finish>();
  const r = Math.ceil(radius);
  for (let x = -r; x < r; x++) {
    for (let z = -r; z < r; z++) {
      const d = Math.hypot(x + 0.5, z + 0.5);
      if (d > radius) continue;
      for (let y = 0; y < width; y++) {
        const face = y === 0 || y === width - 1;
        const spoke = face && d < radius - 1 && (Math.abs(x + 0.5) < 0.8 || Math.abs(z + 0.5) < 0.8);
        cells.set(`${x},${y},${z}`, d > radius - 1.2 ? TYRE : face ? (spoke || d < 1 ? RIM : BLACK) : RIM);
      }
    }
  }
  return cells;
}

/** One cannon: a yellow arm, the gun on it, its barrel pointing forward (+x). */
function cannonCells(): Map<string, Finish> {
  const cells = new Map<string, Finish>();
  const put = (x: number, y: number, z: number, f: Finish) => cells.set(`${x},${y},${z}`, f);
  for (let z = 0; z < 4; z++) for (let x = 0; x < 2; x++) put(x, 0, z, YELLOW);
  for (let x = -1; x < 4; x++) for (let y = -1; y < 2; y++) for (let z = 4; z < 6; z++) put(x, y, z, GUNMETAL);
  for (let x = 4; x < 11; x++) put(x, 0, 5, GUNMETAL);
  put(11, 0, 5, BLACK);
  return cells;
}

/** The race car: its model, its handling, and where it is. */
export class Car {
  readonly object = new THREE.Group();
  /** Where it is (m; z up), its heading (rad from +x), its velocity (m/s, world), its turning (rad/s). */
  readonly position = new THREE.Vector3();
  heading = 0;
  readonly velocity = new THREE.Vector2();
  yawRate = 0;
  /** The front wheels' angle (rad), and how far the wheels have rolled (rad). */
  steer = 0;
  roll = 0;
  /** Sliding sideways (m/s): screeching tyres. */
  slip = 0;
  /** 0 (stowed) to 1 (up and firing): the cannons' rise. */
  deployed = 0;
  wantGuns = false;
  /** The pose at the start of the world's step, to draw it between steps; and at the start of the sub-step, to undo it. */
  private readonly prev = new THREE.Vector3();
  private prevHeading = 0;
  private readonly sub = new THREE.Vector3();
  private subHeading = 0;
  private readonly wheels: THREE.Object3D[] = [];
  private readonly guns: THREE.Object3D[] = [];
  private readonly body: THREE.Object3D;
  /** The body's lean in corners and pitch under braking (rad), eased. */
  private lean = 0;
  private pitch = 0;

  constructor() {
    const cells = bodyCells();
    this.body = voxelMesh(cells, [LEN / 2, WID / 2, 0]);
    this.body.position.z = CLEARANCE - CELL;
    this.object.add(this.body);
    const wheel = wheelCells(2.7, 3);
    for (const [x, y] of [
      [32, -0.5],
      [32, WID - 2.5],
      [8.5, -0.5],
      [8.5, WID - 2.5],
    ]) {
      const hub = new THREE.Group();
      hub.position.set((x - LEN / 2) * CELL, (y + 1.5 - WID / 2) * CELL, 2.7 * CELL);
      const spin = voxelMesh(wheel, [0, 1.5, 0]);
      hub.add(spin);
      this.object.add(hub);
      this.wheels.push(hub);
    }
    for (const y of [3.5, WID - 4.5]) {
      const gun = voxelMesh(cannonCells(), [0, 0, 0]);
      gun.position.set((7 - LEN / 2) * CELL, (y - WID / 2 + 0.5) * CELL, 0);
      this.body.add(gun);
      this.guns.push(gun);
    }
  }

  /** Put it at (x, y) facing `heading`, at rest. */
  place(x: number, y: number, heading: number): void {
    this.position.set(x, y, 0);
    this.prev.copy(this.position);
    this.heading = this.prevHeading = heading;
    this.velocity.set(0, 0);
    this.yawRate = 0;
  }

  get forward(): THREE.Vector2 {
    return new THREE.Vector2(Math.cos(this.heading), Math.sin(this.heading));
  }
  /** m/s along its heading (negative backing up). */
  get speed(): number {
    return this.velocity.dot(this.forward);
  }

  /** A world step starts: it's drawn from here to where the step leaves it. */
  begin(): void {
    this.prev.copy(this.position);
    this.prevHeading = this.heading;
  }

  /**
   * `dt` of handling (a world step in a few parts, so a fast car can't pass through a thin
   * wall between checks): engine, brakes, grip and steering. Only the motion: what it runs
   * into is main.ts's (`sweep` below).
   */
  step(dt: number, d: Drive): void {
    this.sub.copy(this.position);
    this.subHeading = this.heading;
    const f = this.forward;
    const side = new THREE.Vector2(-f.y, f.x);
    let along = this.velocity.dot(f);
    const across = this.velocity.dot(side);
    // Engine and brakes: strong off the line, fading towards top speed; braking, then reverse
    const top = d.boost ? 95 : 78;
    if (d.throttle > 0) {
      const push = along < 0 ? 30 : 17 * Math.max(0, 1 - (along / top) ** 2) * (d.boost ? 1.5 : 1);
      along += push * d.throttle * dt;
    } else if (d.throttle < 0) {
      if (along > 0.5) along = Math.max(0, along + 32 * d.throttle * dt);
      else along = Math.max(-16, along + 9 * d.throttle * dt);
    }
    // Rolling and air: coasting slows it
    along -= Math.sign(along) * Math.min(Math.abs(along), (0.9 + 0.00045 * along * along) * dt);
    if (d.handbrake) along -= Math.sign(along) * Math.min(Math.abs(along), 7 * dt);
    // Steering: quick at low speed, gentle at speed
    const most = 0.62 / (1 + Math.abs(along) / 20);
    const want = d.steer * most;
    this.steer += Math.max(-3.5 * dt, Math.min(3.5 * dt, want - this.steer));
    // Yaw from the steering (a bicycle, 2.7 m between the axles); the handbrake lets the back step out
    const wheelbase = 2.7;
    const target = (along / wheelbase) * Math.tan(this.steer) * (d.handbrake ? 1.45 : 1);
    this.yawRate += (target - this.yawRate) * Math.min(1, dt * (d.handbrake ? 3 : 9));
    this.heading += this.yawRate * dt;
    // The velocity keeps its way as the car turns under it: what's now sideways to the new
    // heading is sliding, and the grip takes that away (fast; slowly under the handbrake: a drift)
    this.velocity.copy(f.multiplyScalar(along)).add(side.multiplyScalar(across));
    const nf = this.forward;
    const ns = new THREE.Vector2(-nf.y, nf.x);
    const on = this.velocity.dot(nf);
    const slide = this.velocity.dot(ns) * Math.exp(-dt * (d.handbrake ? 1.6 : 11));
    this.slip = Math.abs(slide);
    this.velocity.copy(nf.multiplyScalar(on)).add(ns.multiplyScalar(slide));
    this.position.x += this.velocity.x * dt;
    this.position.y += this.velocity.y * dt;
    this.roll += (along / (2.7 * CELL)) * dt;
    // Body lean and pitch, for the look of it
    this.lean += (Math.max(-0.07, Math.min(0.07, -this.yawRate * along * 0.004)) - this.lean) * Math.min(1, dt * 6);
    this.pitch += (Math.max(-0.05, Math.min(0.05, d.throttle < 0 && along > 1 ? 0.04 : d.throttle > 0 ? -0.02 : 0)) - this.pitch) * Math.min(1, dt * 5);
    // The cannons rise while wanted
    this.deployed = Math.max(0, Math.min(1, this.deployed + (this.wantGuns ? dt * 3 : -dt * 1.5)));
  }

  /** Back to where it was before this part of the step (it ran into something that held), bouncing off at `bounce` of its speed. */
  undo(bounce: number): void {
    this.position.copy(this.sub);
    this.heading = this.subHeading;
    this.velocity.multiplyScalar(-bounce);
    this.yawRate *= 0.3;
  }

  /** Draw it `alpha` of the way from the last step's pose to this one's. */
  draw(alpha: number): void {
    this.object.position.lerpVectors(this.prev, this.position, alpha);
    let dh = this.heading - this.prevHeading;
    dh = Math.atan2(Math.sin(dh), Math.cos(dh));
    this.object.rotation.set(0, 0, this.prevHeading + dh * alpha);
    this.body.rotation.set(this.lean, this.pitch, 0);
    for (const [k, w] of this.wheels.entries()) {
      w.rotation.set(0, this.roll, k < 2 ? this.steer : 0, 'ZYX');
    }
    // Cannons: up out of the deck on their arms, then tilted level
    const up = THREE.MathUtils.smoothstep(this.deployed, 0, 1);
    for (const g of this.guns) {
      // Stowed under the deck (5-6 cells up there), raised to stand on it
      g.position.z = (-1 + 7 * up) * CELL;
      g.visible = this.deployed > 0.02;
    }
  }

  /** Where the cannons' muzzles are now (world, m). */
  muzzles(): THREE.Vector3[] {
    return this.guns.map((g) => g.localToWorld(new THREE.Vector3(11.5 * CELL, 0.5 * CELL, 5.5 * CELL)));
  }

  /** The body's corners' rotation as a quaternion (xyzw), for the solver. */
  quaternion(): number[] {
    const h = this.heading / 2;
    return [0, 0, Math.sin(h), Math.cos(h)];
  }
}

/**
 * The fixed voxels the car's box overlaps at (x, y) facing `heading`: what it's running into.
 * (Each object's grid, over the part of it under the car's bounds; a voxel counts if its
 * centre is inside the box grown by half a voxel.)
 */
export function sweep(city: City, x: number, y: number, heading: number): number[] {
  const [L, W, H] = CAR_SIZE;
  const [c, s] = [Math.cos(heading), Math.sin(heading)];
  const hx = L / 2 + VOXEL / 2;
  const hy = W / 2 + VOXEL / 2;
  const ex = Math.abs(c) * hx + Math.abs(s) * hy;
  const ey = Math.abs(s) * hx + Math.abs(c) * hy;
  const z0 = CLEARANCE + 0.05;
  const z1 = CLEARANCE + H;
  const out: number[] = [];
  for (const b of city.buildings) {
    const bx0 = b.x0 * VOXEL;
    const by0 = b.y0 * VOXEL;
    if (x + ex < bx0 || x - ex > bx0 + b.w * VOXEL || y + ey < by0 || y - ey > by0 + b.d * VOXEL) continue;
    const i0 = Math.max(0, Math.floor((x - ex) / VOXEL) - b.x0);
    const i1 = Math.min(b.w - 1, Math.floor((x + ex) / VOXEL) - b.x0);
    const j0 = Math.max(0, Math.floor((y - ey) / VOXEL) - b.y0);
    const j1 = Math.min(b.d - 1, Math.floor((y + ey) / VOXEL) - b.y0);
    const k0 = Math.max(0, Math.floor(z0 / VOXEL));
    const k1 = Math.min(b.h - 1, Math.floor(z1 / VOXEL));
    for (let k = k0; k <= k1; k++) {
      for (let j = j0; j <= j1; j++) {
        for (let i = i0; i <= i1; i++) {
          const v = b.cells[i + b.w * (j + b.d * k)];
          if (v < 0 || city.state[v] !== State.Fixed) continue;
          const dx = (b.x0 + i + 0.5) * VOXEL - x;
          const dy = (b.y0 + j + 0.5) * VOXEL - y;
          if (Math.abs(dx * c + dy * s) <= hx && Math.abs(-dx * s + dy * c) <= hy) out.push(v);
        }
      }
    }
  }
  return out;
}
