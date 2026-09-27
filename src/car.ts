// The race car: a low silver-blue mid-engined supercar built of voxels (0.1 m, finer than the
// world's quarter metre, so it reads as a car and not a crate): a long nose between raised
// front wings, a teardrop canopy, wide hips over the rear wheels, twin exhausts high in the
// middle of the tail, bars of tail lights that brighten under braking. Driven on the CPU with
// arcade handling (grip that lets go under the handbrake, speed-sensitive steering, seven
// gears with a beat of lost drive at each change, a boost). Armed: two small machine guns rise
// from the bonnet, and two rocket launchers swing up and out from the hips on yellow arms. The
// world's physics sees it as a box moved each step (physics.ts drive): it shoves debris aside;
// what it drives into that still stands, it breaks or is stopped by (main.ts).

import * as THREE from 'three/webgpu';
import { attribute, float, select, uniform, vec3 } from 'three/tsl';
import { edgeShade } from './look.ts';
import { type City, State, VOXEL } from './world.ts';

/** The model's voxel (m): a quarter of the world's, fine enough for the curves of a car. */
const CELL = 0.0625;
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

// The body in the car's frame (m): x forward from the middle, y to the left, z up from the ground
const HALF_L = 2.3;
const HALF_W = 1.0;
const WHEEL_R = 0.34;
/** Wheel centres along it, and their middle's distance out from the centre line. */
const FRONT_AXLE = 1.42;
const REAR_AXLE = -1.3;
const TRACK = 0.84;
/** The exhausts' middles (m: across, up), two either side under the tail lights. */
const PIPES = [-0.59375, -0.40625, 0.40625, 0.59375].map((y) => [y, 0.40625]);
const WHEEL_W = 0.3;

/** A cell's look; a lamp's glow switched by the brakes or reversing (else always on). */
type Finish = { colour: number; rough: number; metal: number; glow: number; lamp?: 'brake' | 'reverse' };
const PAINT: Finish = { colour: 0x8fa6bb, rough: 0.22, metal: 0.75, glow: 0 };
const SHADOW: Finish = { colour: 0x6f8397, rough: 0.25, metal: 0.75, glow: 0 };
const BLACK: Finish = { colour: 0x141517, rough: 0.5, metal: 0.25, glow: 0 };
const TINT: Finish = { colour: 0x0f161d, rough: 0.06, metal: 0.7, glow: 0 };
const HEAD: Finish = { colour: 0xe6f0ff, rough: 0.3, metal: 0, glow: 0.9 };
const TAIL: Finish = { colour: 0xc40806, rough: 0.3, metal: 0, glow: 1, lamp: 'brake' };
// (A grey lens: white paint would read as a light in the sun even when it's off)
const REVERSE: Finish = { colour: 0x8c929c, rough: 0.2, metal: 0, glow: 1, lamp: 'reverse' };
const PIPE: Finish = { colour: 0x6b6f75, rough: 0.3, metal: 0.9, glow: 0 };
const YELLOW: Finish = { colour: 0xe8b810, rough: 0.4, metal: 0.3, glow: 0 };
const GUNMETAL: Finish = { colour: 0x2a2e33, rough: 0.35, metal: 0.8, glow: 0 };
const TIP: Finish = { colour: 0xb3261e, rough: 0.4, metal: 0.3, glow: 0 };
const TYRE: Finish = { colour: 0x121212, rough: 0.9, metal: 0, glow: 0 };
const RIM: Finish = { colour: 0x3a3e44, rough: 0.25, metal: 0.9, glow: 0 };

/** Straight lines between (x, value) points (x ascending), clamped at the ends. */
function table(points: [number, number][], x: number): number {
  if (x <= points[0][0]) return points[0][1];
  for (let k = 1; k < points.length; k++) {
    const [x1, v1] = points[k];
    if (x <= x1) {
      const [x0, v0] = points[k - 1];
      const f = (x - x0) / (x1 - x0);
      // Smoothed between the points (no creases along the body)
      return v0 + (v1 - v0) * f * f * (3 - 2 * f);
    }
  }
  return points[points.length - 1][1];
}

// Seen from above: half its width along it
const PLAN: [number, number][] = [
  [-2.3, 0.88],
  [-2.1, 0.97],
  [-1.6, 1.0],
  [-0.85, 1.0],
  [-0.35, 0.93],
  [0.5, 0.92],
  [1.1, 0.96],
  [1.75, 0.96],
  [2.05, 0.86],
  [2.3, 0.58],
];
// Height of the shoulders at its sides: high over the wheels, low along the doors
const SHOULDER: [number, number][] = [
  [-2.3, 0.74],
  [-1.9, 0.84],
  [-1.35, 0.9],
  [-0.75, 0.8],
  [-0.2, 0.66],
  [0.6, 0.64],
  [1.15, 0.72],
  [1.5, 0.76],
  [1.95, 0.64],
  [2.3, 0.4],
];
// Height down its middle (away from the canopy): the engine cover's spine, the bonnet's valley
const SPINE: [number, number][] = [
  [-2.3, 0.8],
  [-1.7, 0.88],
  [-1.0, 0.92],
  [-0.85, 0.94],
  [1.0, 0.6],
  [1.6, 0.54],
  [2.1, 0.46],
  [2.3, 0.34],
];
// The canopy: roof height and half width along it, from the back of the cabin to the windscreen's foot
const ROOF: [number, number][] = [
  [-0.95, 0.92],
  [-0.6, 1.08],
  [0.0, 1.14],
  [0.5, 1.07],
  [1.05, 0.62],
];
const CABIN: [number, number][] = [
  [-0.95, 0.34],
  [-0.5, 0.6],
  [0.1, 0.66],
  [0.7, 0.62],
  [1.05, 0.56],
];

/** What's at (x, y, z) m in the car's frame: its finish, or null (air). Symmetric: only |y| counts. */
function shape(x: number, y: number, z: number): Finish | null {
  const ay = Math.abs(y);
  if (Math.abs(x) > HALF_L || z < CLEARANCE) return null;
  const hw = table(PLAN, x);
  if (ay > hw) return null;
  // Wheel arches: the body is cut round each wheel, on the outside
  for (const axle of [FRONT_AXLE, REAR_AXLE]) if (ay > TRACK - WHEEL_W / 2 - 0.1 && Math.hypot(x - axle, z - WHEEL_R) < WHEEL_R + 0.07) return null;
  // The top here: the spine in the middle rolling out to the shoulders, rounded off at the edge
  const r = ay / hw;
  let top = table(SPINE, x) + (table(SHOULDER, x) - table(SPINE, x)) * Math.min(1, Math.max(0, (r - 0.25) / 0.6)) ** 2 * (3 - 2 * Math.min(1, Math.max(0, (r - 0.25) / 0.6)));
  if (r > 0.9) top -= ((r - 0.9) / 0.1) ** 2 * 0.12;
  // The lower sides tuck in (the sills are black, the flanks shadowed)
  if (z < 0.22) return r > 0.97 ? null : BLACK;
  // The canopy, a teardrop over the middle
  const inCabinX = x > -0.95 && x < 1.05;
  if (inCabinX) {
    const cw = table(CABIN, x);
    const base = table(SPINE, x);
    const roof = table(ROOF, x);
    if (ay < cw && z >= base - 0.02) {
      const rise = base + (roof - base) * Math.sqrt(Math.max(0, 1 - (ay / cw) ** 2));
      if (z <= rise) {
        // Glass: the windscreen, the side windows; the roof and the pillars painted
        const onTop = z > rise - 0.1 && ay < cw * 0.45 && x < 0.55;
        const pillar = Math.abs(x - 0.55) < 0.06 || x < -0.55;
        return onTop || pillar ? PAINT : TINT;
      }
    }
  }
  if (z > top) return null;
  // Details, front to back
  if (x > 2.08 && z < 0.26) return BLACK; // splitter
  if (x > 2.12 && ay < 0.45 && z < 0.34) return BLACK; // the grille
  if (x > 2.02 && x < 2.2 && ay > 0.56 && ay < 0.78 && z > top - 0.08 && z < top - 0.01) return HEAD; // headlights: a thin strip at the corners
  if (ay > hw - 0.07 && x > -0.8 && x < -0.3 && z > 0.3 && z < 0.6) return BLACK; // side intakes
  if (ay > hw - 0.05 && z > 0.36 && z < 0.44 && x > -0.3 && x < 1.1) return SHADOW; // a crease along the door
  // The tail: the diffuser under it, a dark band with a row of short light bars (two cells tall)
  // across each side and a white reversing light standing at its outer end; between them a black
  // valance the four exhausts stand out of (bodyCells adds those)
  if (x < -2.2) {
    if (z < 0.34) return ay < 0.75 && Math.floor(ay / 0.19) % 2 === 1 && z < 0.3 ? SHADOW : BLACK;
    if (z < 0.5 && ay > 0.3 && ay < 0.72) return BLACK;
    const band = z > 0.5 && z < 0.7 && ay > 0.28 && ay < 0.9;
    if (band) {
      if (ay > 0.82) return REVERSE;
      const row = z > 0.56 && z < 0.68;
      const bar = ((ay - 0.32) / 0.11) % 1 < 0.62 && ay > 0.32 && ay < 0.8;
      return row && bar ? TAIL : BLACK;
    }
  }
  if (x < -2.24 && z > top - 0.04) return BLACK; // the ducktail's lip
  return PAINT;
}

/** Cells of the body, from the shape sampled at each cell's middle. */
function bodyCells(): Map<string, Finish> {
  const cells = new Map<string, Finish>();
  const nx = Math.round((2 * HALF_L) / CELL);
  const ny = Math.round((2 * HALF_W) / CELL);
  const nz = Math.round(1.2 / CELL);
  for (let i = 0; i < nx; i++)
    for (let j = 0; j < ny; j++)
      for (let k = 0; k < nz; k++) {
        const f = shape(-HALF_L + (i + 0.5) * CELL, -HALF_W + (j + 0.5) * CELL, (k + 0.5) * CELL);
        if (f) cells.set(`${i},${j},${k}`, f);
      }
  // Four exhausts, two a side, standing out of the tail: a steel ring round a black bore, three cells across
  for (const [j, k] of PIPES.map(([y, z]) => [Math.round((y + HALF_W) / CELL - 0.5), Math.round(z / CELL - 0.5)]))
    for (let dj = -1; dj <= 1; dj++)
      for (let dk = -1; dk <= 1; dk++)
        for (const i of [-2, -1]) cells.set(`${i},${j + dj},${k + dk}`, i === -2 && dj === 0 && dk === 0 ? BLACK : PIPE);
  // Mirrors on stalks by the windscreen, just outside the body
  const i0 = Math.round((0.72 + HALF_L) / CELL);
  const k0 = Math.round(0.74 / CELL);
  for (const j of [-1, -2, ny, ny + 1])
    for (let i = i0; i < i0 + 3; i++) {
      for (let k = k0; k < k0 + 2; k++) cells.set(`${i},${j},${k}`, PAINT);
      cells.set(`${i},${j},${k0 - 1}`, BLACK);
    }
  return cells;
}

/** Cells → a mesh of their open faces, edge-shaded like the world's voxels (`origin` in cells). */
function voxelMesh(cells: Map<string, Finish>, origin: [number, number, number], lights: Lights): THREE.Mesh {
  const pos: number[] = [];
  const nor: number[] = [];
  const col: number[] = [];
  const local: number[] = [];
  const rough: number[] = [];
  const metal: number[] = [];
  const glow: number[] = [];
  const lamp: number[] = [];
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
        lamp.push(f.lamp === 'brake' ? 1 : f.lamp === 'reverse' ? 2 : 0);
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
  g.setAttribute('lamp', new THREE.Float32BufferAttribute(lamp, 1));
  const m = new THREE.MeshStandardNodeMaterial();
  // A much lighter edge than the world's voxels: at a quarter of their size, the full darkening draws contour lines
  const shade = edgeShade(attribute('local', 'vec3') as unknown as THREE.Node<'vec3'>, vec3(1, 1, 1), 0.1).mul(0.18).add(0.82);
  m.colorNode = attribute('color', 'vec3').mul(shade) as unknown as THREE.Node<'color'>;
  m.roughnessNode = attribute('rough', 'float') as unknown as THREE.Node<'float'>;
  m.metalnessNode = attribute('metal', 'float') as unknown as THREE.Node<'float'>;
  // Lights glow: the tail lights by the brakes (dim running lights to bright), the reversing lights in reverse
  const which = attribute('lamp', 'float');
  const glowing = attribute('glow', 'float').mul(select(which.lessThan(0.5), float(1), select(which.lessThan(1.5), lights.brake, lights.reverse)));
  m.emissiveNode = attribute('color', 'vec3').mul(glowing) as unknown as THREE.Node<'color'>;
  const mesh = new THREE.Mesh(g, m);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

/** A wheel's cells: tyre round a dark five-spoke rim, along y. */
function wheelCells(): Map<string, Finish> {
  const cells = new Map<string, Finish>();
  const radius = WHEEL_R / CELL;
  const width = Math.round(WHEEL_W / CELL);
  const r = Math.ceil(radius);
  for (let x = -r; x < r; x++) {
    for (let z = -r; z < r; z++) {
      const d = Math.hypot(x + 0.5, z + 0.5);
      if (d > radius) continue;
      const a = Math.atan2(z + 0.5, x + 0.5);
      const spoke = Math.cos(a * 5) > 0.55 || d < 1;
      for (let y = 0; y < width; y++) {
        const face = y === 0 || y === width - 1;
        cells.set(`${x},${y},${z}`, d > radius - 1 ? TYRE : face ? (spoke ? RIM : BLACK) : RIM);
      }
    }
  }
  return cells;
}

/** A machine gun: a small breech on a post, a long thin barrel forward (+x). */
function gunCells(): Map<string, Finish> {
  const cells = new Map<string, Finish>();
  const put = (x: number, y: number, z: number, f: Finish) => cells.set(`${x},${y},${z}`, f);
  for (let z = 0; z < 2; z++) put(0, 0, z, BLACK);
  for (let x = -2; x < 3; x++) for (let y = -1; y < 1; y++) for (let z = 2; z < 4; z++) put(x, y, z, GUNMETAL);
  for (let x = 3; x < 9; x++) put(x, 0, 3, GUNMETAL);
  put(9, 0, 3, BLACK);
  return cells;
}

/** A rocket launcher: a box with a red-tipped rocket in its mouth, pointing forward (+x). */
function launcherCells(): Map<string, Finish> {
  const cells = new Map<string, Finish>();
  const put = (x: number, y: number, z: number, f: Finish) => cells.set(`${x},${y},${z}`, f);
  for (let x = -3; x < 5; x++) for (let y = -1; y < 2; y++) for (let z = -1; z < 2; z++) put(x, y, z, x === -3 ? BLACK : GUNMETAL);
  for (let y = -1; y < 2; y++) for (let z = -1; z < 2; z++) put(5, y, z, y === 0 && z === 0 ? TIP : BLACK);
  return cells;
}

/** A yellow arm, `n` cells long up its z, two thick. */
function armCells(n: number): Map<string, Finish> {
  const cells = new Map<string, Finish>();
  for (let z = 0; z < n; z++) for (let x = 0; x < 2; x++) for (let y = 0; y < 2; y++) cells.set(`${x},${y},${z}`, YELLOW);
  return cells;
}
/** The launchers' arms (cells). */
const ARM = 8;

/** Top gear speeds (km/h) of each of the seven gears (the first from rest). */
/** The tyres' grip across (m/s²): how hard it can corner before it runs wide. */
const GRIP = 11;
/**
 * The tail and reversing lights' glow: off (the tail lights' dim running glow) and on. (Brighter tail lights than
 * this only turn orange through the tone mapping: they look brighter against a dimmer running glow instead.)
 */
const BRAKE_LIGHT = [0.12, 2];
const REVERSE_LIGHT = [0, 6];
interface Lights {
  brake: THREE.UniformNode<'float', number>;
  reverse: THREE.UniformNode<'float', number>;
}

const GEARS = [0, 55, 90, 125, 160, 195, 230, 300];
/** Seconds of lost drive at an upshift. */
const SHIFT = 0.16;

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
  /** How hard the rear tyres are skidding (0..1): sliding sideways, or locked by the handbrake. */
  skid = 0;
  /** The gear (1-7), the revs in it (0 idle to 1 the limiter), and upshifts so far (a new one: an afterfire). */
  gear = 1;
  revs = 0;
  shifts = 0;
  private shifting = 0;
  /** 0 (stowed) to 1 (up and firing): the machine guns and the rocket launchers, and whether each is wanted. */
  guns = 0;
  rockets = 0;
  wantGuns = false;
  wantRockets = false;
  /** How hard the tail lights glow (dim running lights; bright when braking), and the reversing lights (in reverse). */
  readonly lights: Lights = { brake: uniform(BRAKE_LIGHT[0]), reverse: uniform(REVERSE_LIGHT[0]) };
  /** The pose at the start of the world's step, to draw it between steps; and at the start of the sub-step, to undo it. */
  private readonly prev = new THREE.Vector3();
  private prevHeading = 0;
  private readonly sub = new THREE.Vector3();
  private subHeading = 0;
  private readonly wheels: THREE.Object3D[] = [];
  private readonly mgs: THREE.Object3D[] = [];
  private readonly arms: { pivot: THREE.Object3D; launcher: THREE.Object3D; side: number }[] = [];
  private readonly body: THREE.Object3D;
  /** The body's lean in corners and pitch under braking and at each shift (rad), eased. */
  private lean = 0;
  private pitch = 0;
  private braking = false;
  private reversing = false;

  constructor() {
    const brake = this.lights;
    this.body = voxelMesh(bodyCells(), [HALF_L / CELL, HALF_W / CELL, 0], brake);
    this.object.add(this.body);
    const wheel = wheelCells();
    const half = WHEEL_W / CELL / 2;
    for (const [x, s] of [
      [FRONT_AXLE, 1],
      [FRONT_AXLE, -1],
      [REAR_AXLE, 1],
      [REAR_AXLE, -1],
    ]) {
      const hub = new THREE.Group();
      hub.position.set(x, s * TRACK, WHEEL_R);
      hub.add(voxelMesh(wheel, [0, half, 0], brake));
      this.object.add(hub);
      this.wheels.push(hub);
    }
    // Machine guns in the bonnet either side of its valley
    for (const s of [1, -1]) {
      const gun = voxelMesh(gunCells(), [0.5, 0.5, 0], brake);
      gun.position.set(1.25, s * 0.5, 0.5);
      this.body.add(gun);
      this.mgs.push(gun);
    }
    // Rocket launchers on yellow arms from the hips: stowed flat inside, swung up and out
    for (const s of [1, -1]) {
      const pivot = new THREE.Group();
      pivot.position.set(-1.35, s * 0.62, 0.72);
      const arm = voxelMesh(armCells(ARM), [1, 1, 0], brake);
      pivot.add(arm);
      const launcher = voxelMesh(launcherCells(), [0.5, 0.5, 0.5], brake);
      launcher.position.set(0, 0, (ARM + 1) * CELL);
      pivot.add(launcher);
      this.body.add(pivot);
      this.arms.push({ pivot, launcher, side: s });
    }
  }

  /** Put it at (x, y) facing `heading`, at rest. */
  place(x: number, y: number, heading: number): void {
    this.position.set(x, y, 0);
    this.prev.copy(this.position);
    this.heading = this.prevHeading = heading;
    this.velocity.set(0, 0);
    this.yawRate = 0;
    this.gear = 1;
    this.guns = this.rockets = 0;
    this.wantGuns = this.wantRockets = false;
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
   * wall between checks): engine and gears, brakes, grip and steering. Only the motion: what
   * it runs into is main.ts's (`sweep` below).
   */
  step(dt: number, d: Drive): void {
    this.sub.copy(this.position);
    this.subHeading = this.heading;
    const f = this.forward;
    const side = new THREE.Vector2(-f.y, f.x);
    let along = this.velocity.dot(f);
    const across = this.velocity.dot(side);
    // Gears: up at the top of each (a beat of lost drive, an afterfire), down as it slows
    const kmh = Math.abs(along) * 3.6;
    if (this.gear < GEARS.length - 1 && kmh > GEARS[this.gear]) {
      this.gear++;
      if (d.throttle > 0.3) {
        this.shifting = SHIFT;
        this.shifts++;
        this.pitch += 0.025;
      }
    } else if (this.gear > 1 && kmh < GEARS[this.gear - 1] * 0.92) this.gear--;
    this.shifting = Math.max(0, this.shifting - dt);
    const low = GEARS[this.gear - 1];
    const within = Math.min(1, (kmh - low) / (GEARS[this.gear] - low));
    this.revs = 0.25 + 0.75 * (this.gear === 1 ? within : 0.35 + within * 0.65);
    // Engine and brakes: strong off the line, fading towards top speed (about 210 km/h, 250 boosting: a
    // circuit's pace); braking, then reverse
    const top = d.boost ? 70 : 58;
    this.braking = (d.throttle < 0 && along > 0.5) || (d.handbrake && Math.abs(along) > 0.5);
    this.reversing = d.throttle < 0 && along <= 0.5;
    if (d.throttle > 0) {
      const cut = this.shifting > 0 ? 0.25 : 1;
      const push = along < 0 ? 30 : 12 * Math.max(0, 1 - (along / top) ** 2) * (d.boost ? 1.4 : 1) * cut;
      along += push * d.throttle * dt;
    } else if (d.throttle < 0) {
      if (along > 0.5) along = Math.max(0, along + 26 * d.throttle * dt);
      else along = Math.max(-16, along + 9 * d.throttle * dt);
    }
    // Rolling and air: coasting slows it
    along -= Math.sign(along) * Math.min(Math.abs(along), (0.9 + 0.00045 * along * along) * dt);
    if (d.handbrake) along -= Math.sign(along) * Math.min(Math.abs(along), 7 * dt);
    // Steering: the lock falls away with speed; the wheel winds on steadily and comes back to the middle
    // quicker (a weighted rack, not a switch)
    const most = 0.55 / (1 + Math.abs(along) / 14);
    const want = d.steer * most;
    const rate = (Math.abs(want) < Math.abs(this.steer) || want * this.steer < 0 ? 3.2 : 1.9) * dt;
    this.steer += Math.max(-rate, Math.min(rate, want - this.steer));
    // Yaw from the steering (a bicycle, 2.7 m between the axles), up to what the tyres can hold (about 1.1 g
    // across: turn harder at speed and it runs wide, not round); the car's weight takes a moment to turn, and
    // the handbrake lets the back step out
    const wheelbase = FRONT_AXLE - REAR_AXLE;
    const grip = GRIP / Math.max(Math.abs(along), 1);
    const target = Math.max(-grip, Math.min(grip, (along / wheelbase) * Math.tan(this.steer))) * (d.handbrake ? 1.45 : 1);
    this.yawRate += (target - this.yawRate) * Math.min(1, dt * (d.handbrake ? 3 : 5.5));
    this.heading += this.yawRate * dt;
    // The velocity keeps its way as the car turns under it: what's now sideways to the new
    // heading is sliding, and the grip takes that away (fast; slowly under the handbrake: a drift)
    this.velocity.copy(f.multiplyScalar(along)).add(side.multiplyScalar(across));
    const nf = this.forward;
    const ns = new THREE.Vector2(-nf.y, nf.x);
    const on = this.velocity.dot(nf);
    const slide = this.velocity.dot(ns) * Math.exp(-dt * (d.handbrake ? 1.6 : 11));
    this.slip = Math.abs(slide);
    this.skid = Math.max(Math.min(1, Math.max(0, (this.slip - 1.5) / 5)), d.handbrake ? Math.min(1, Math.abs(on) / 12) : 0);
    this.velocity.copy(nf.multiplyScalar(on)).add(ns.multiplyScalar(slide));
    this.position.x += this.velocity.x * dt;
    this.position.y += this.velocity.y * dt;
    this.roll += (along / WHEEL_R) * dt;
    // Body lean and pitch, for the look of it: squat under power, dive under brakes, a nod at each shift
    this.lean += (Math.max(-0.06, Math.min(0.06, -this.yawRate * along * 0.004)) - this.lean) * Math.min(1, dt * 6);
    const squat = this.braking ? 0.035 : d.throttle > 0 && this.shifting === 0 ? -0.015 : 0;
    this.pitch += (squat - this.pitch) * Math.min(1, dt * 5);
    // The weapons come up while wanted
    this.guns = Math.max(0, Math.min(1, this.guns + (this.wantGuns ? dt * 5 : -dt * 2)));
    this.rockets = Math.max(0, Math.min(1, this.rockets + (this.wantRockets ? dt * 2.5 : -dt * 1.5)));
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
    for (const [k, w] of this.wheels.entries()) w.rotation.set(0, this.roll, k < 2 ? this.steer : 0, 'ZYX');
    this.lights.brake.value = BRAKE_LIGHT[this.braking ? 1 : 0];
    this.lights.reverse.value = REVERSE_LIGHT[this.reversing ? 1 : 0];
    // Machine guns: up out of the bonnet
    const g = THREE.MathUtils.smoothstep(this.guns, 0, 1);
    for (const m of this.mgs) {
      m.position.z = 0.5 + g * 0.16;
      m.visible = this.guns > 0.02;
    }
    // Launchers: the arm rises from inside the hip, then swings out to the side at 40 degrees;
    // the launcher on its end stays level, pointing forward
    const rise = THREE.MathUtils.smoothstep(this.rockets, 0, 0.5);
    const swing = THREE.MathUtils.smoothstep(this.rockets, 0.4, 1);
    for (const { pivot, launcher, side } of this.arms) {
      pivot.position.z = 0.35 + rise * 0.37;
      pivot.rotation.set(-side * swing * 0.7, -swing * 0.25, 0);
      launcher.rotation.set(side * swing * 0.7, swing * 0.25, 0, 'ZYX');
      pivot.visible = this.rockets > 0.02;
    }
  }

  /** Where the machine guns' and the launchers' muzzles are now (world, m). */
  muzzles(): { guns: THREE.Vector3[]; rockets: THREE.Vector3[] } {
    this.object.updateMatrixWorld(true);
    return {
      guns: this.mgs.map((m) => m.localToWorld(new THREE.Vector3(9.5 * CELL, 0, 3.5 * CELL))),
      rockets: this.arms.map(({ launcher }) => launcher.localToWorld(new THREE.Vector3(6 * CELL, 0, 0))),
    };
  }

  /** Where the rear tyres touch the ground (world, m). */
  rearTyres(): THREE.Vector3[] {
    this.object.updateMatrixWorld(true);
    return [1, -1].map((s) => this.object.localToWorld(new THREE.Vector3(REAR_AXLE, s * TRACK, 0)));
  }

  /** Where the exhausts are (world, m), and the way out of them (backwards). */
  exhausts(): { at: THREE.Vector3[]; back: THREE.Vector3 } {
    this.object.updateMatrixWorld(true);
    const at = PIPES.map(([y, z]) => this.body.localToWorld(new THREE.Vector3(-HALF_L - 2 * CELL, y, z)));
    const f = this.forward;
    return { at, back: new THREE.Vector3(-f.x, -f.y, 0.15).normalize() };
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
