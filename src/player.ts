// The player: a box 0.6 m wide, 1.8 m tall (1.1 m crouched) that walks the scene on the CPU,
// sliding along fixed voxels and loose debris (as last read back from the GPU) axis by axis.
// Steps of up to half a metre are climbed on their own, smoothed so the view glides up, which
// makes scrambling over rubble easy. Walk, sprint, crouch, jump (with a moment's grace after
// running off an edge and a remembered early press), and a noclip flight for looking around.

import * as THREE from 'three/webgpu';
import { type City, State, VOXEL } from './world.ts';

export interface Input {
  forward: number;
  right: number;
  jump: boolean;
  sprint: boolean;
  crouch: boolean;
  /** Fly: up (+1) or down (-1). */
  rise: number;
  /** Fly: creep (fine positioning). */
  slow: boolean;
}

const WIDTH = 0.3; // half
const TALL = 1.8;
const LOW = 1.1;
const EYE = 0.12; // below the head
const GRAVITY = 20;
const JUMP = 6.6;
const WALK = 6;
const RUN = 10.5;
const CREEP = 3;
/** Ground acceleration (per second, as a share of the top speed): near instant, as in Quake-likes. */
const ACCEL = 12;
const AIR = 3;
const FRICTION = 9;
/** Flying: seconds to reach the wanted velocity (and to stop), boost and creep factors. */
const FLY_RESPONSE = 0.09;
const BOOST = 3.5;
const CREEP_FLY = 0.25;
const STEP = 0.55;
const COYOTE = 0.12;
const BUFFER = 0.15;

export class Player {
  readonly position = new THREE.Vector3(0, 0, 0); // feet
  readonly velocity = new THREE.Vector3();
  yaw = 0;
  pitch = 0;
  onGround = false;
  flying = true;
  /** Flying speed (m/s), set with the mouse wheel. */
  flySpeed = 9;
  /** How much faster boosting flies (Shift, or the touch stick pushed past its ring). */
  boost = BOOST;
  height = TALL;
  /** The eye's lag behind a step climbed (m, eased back to 0). */
  private stepLag = 0;
  private sinceGround = 0;
  private sinceJumpPress = 1;
  private bob = 0;
  /** Walking phase (radians): the tool in hand bobs with it. */
  get walk(): number {
    return this.bob;
  }
  /** Debris boxes to stand on and bump into: centres (xyz), refreshed from readbacks. */
  debris: Float32Array = new Float32Array(0);
  debrisCount = 0;
  /** Speed (m/s) of the last landing, this frame only (0 otherwise). */
  landed = 0;

  private readonly city: City;

  constructor(city: City) {
    this.city = city;
  }

  /** The eye, for the camera; with a gentle bob when walking. */
  eye(out: THREE.Vector3): THREE.Vector3 {
    const bob = this.onGround ? Math.sin(this.bob) * 0.04 * Math.min(1, Math.hypot(this.velocity.x, this.velocity.y) / RUN) : 0;
    return out.set(this.position.x, this.position.y, this.position.z + this.height - EYE - this.stepLag + bob);
  }

  /** Unit look direction. */
  look(out: THREE.Vector3): THREE.Vector3 {
    return out.set(Math.cos(this.pitch) * Math.cos(this.yaw), Math.cos(this.pitch) * Math.sin(this.yaw), Math.sin(this.pitch));
  }

  update(dt: number, input: Input): void {
    const fwd = new THREE.Vector2(Math.cos(this.yaw), Math.sin(this.yaw));
    const right = new THREE.Vector2(Math.sin(this.yaw), -Math.cos(this.yaw));
    const wish = fwd.multiplyScalar(input.forward).add(right.multiplyScalar(input.right));
    if (wish.lengthSq() > 1) wish.normalize();

    if (this.flying) {
      // Where you look is forward; strafing stays level; up and down are the world's
      const speed = this.flySpeed * (input.sprint ? this.boost : 1) * (input.slow ? CREEP_FLY : 1);
      const target = this.look(new THREE.Vector3())
        .multiplyScalar(input.forward)
        .add(new THREE.Vector3(Math.sin(this.yaw), -Math.cos(this.yaw), 0).multiplyScalar(input.right))
        .add(new THREE.Vector3(0, 0, input.rise));
      if (target.lengthSq() > 1) target.normalize();
      target.multiplyScalar(speed);
      // Critically damped: responsive, no drift once the keys are up
      this.velocity.lerp(target, 1 - Math.exp(-dt / FLY_RESPONSE));
      this.position.addScaledVector(this.velocity, dt);
      if (this.position.z < 0.1) {
        this.position.z = 0.1;
        this.velocity.z = Math.max(0, this.velocity.z);
      }
      this.height = TALL;
      this.onGround = false;
      this.landed = 0;
      return;
    }

    // Crouch (stand up only if there's room)
    const want = input.crouch ? LOW : TALL;
    if (want > this.height && this.blocked(this.position.x, this.position.y, this.position.z, want)) {
      // stay low
    } else this.height = want;

    // Quake-style: friction on the ground, then accelerate along the wished direction up to
    // the top speed (full control on the ground, a little in the air)
    const top = input.crouch ? CREEP : input.sprint && input.forward > 0 ? RUN : WALK;
    const v2 = new THREE.Vector2(this.velocity.x, this.velocity.y);
    if (this.onGround) {
      const speed = v2.length();
      const drop = Math.max(speed, 2) * FRICTION * dt;
      v2.multiplyScalar(speed > 0 ? Math.max(0, speed - drop) / speed : 0);
    }
    if (wish.lengthSq() > 0) {
      const dirn = wish.clone().normalize();
      const along = v2.dot(dirn);
      const add = Math.min(Math.max(0, top * wish.length() - along), (this.onGround ? ACCEL : AIR) * top * dt);
      v2.addScaledVector(dirn, add);
    }
    [this.velocity.x, this.velocity.y] = [v2.x, v2.y];

    this.sinceGround = this.onGround ? 0 : this.sinceGround + dt;
    this.sinceJumpPress = input.jump ? 0 : this.sinceJumpPress + dt;
    if (this.sinceJumpPress < BUFFER && this.sinceGround < COYOTE && this.velocity.z <= 0.5) {
      this.velocity.z = JUMP;
      this.sinceGround = COYOTE;
      this.sinceJumpPress = BUFFER;
    }
    this.velocity.z -= GRAVITY * dt;

    // Move a little at a time so fast falls don't skip through a voxel
    const steps = Math.max(1, Math.ceil((this.velocity.length() * dt) / 0.3));
    const wasGround = this.onGround;
    const falling = -this.velocity.z;
    this.onGround = false;
    for (let k = 0; k < steps; k++) this.move(dt / steps);
    this.landed = this.onGround && !wasGround ? falling : 0;
    if (this.onGround) this.bob += dt * Math.hypot(this.velocity.x, this.velocity.y) * 1.7;
    this.stepLag = Math.max(0, this.stepLag - dt * 6 * Math.max(0.3, this.stepLag));
  }

  private move(dt: number): void {
    const p = this.position;
    for (const axis of [0, 1] as const) {
      const d = (axis === 0 ? this.velocity.x : this.velocity.y) * dt;
      if (!d) continue;
      const [x, y] = axis === 0 ? [p.x + d, p.y] : [p.x, p.y + d];
      if (!this.blocked(x, y, p.z, this.height)) {
        [p.x, p.y] = [x, y];
        continue;
      }
      // Climb it if there's a step no higher than STEP with room above
      const rise = this.stepUp(x, y, p.z);
      if (rise > 0 && (this.onGround || this.sinceGround < COYOTE)) {
        [p.x, p.y] = [x, y];
        p.z += rise;
        this.stepLag += rise;
        continue;
      }
      if (axis === 0) this.velocity.x = 0;
      else this.velocity.y = 0;
    }
    const dz = this.velocity.z * dt;
    if (!this.blocked(p.x, p.y, p.z + dz, this.height)) p.z += dz;
    else {
      if (dz < 0) {
        this.onGround = true;
        // Settle onto the top of what's below
        p.z = this.floorBelow(p.x, p.y, p.z + dz, p.z);
      }
      this.velocity.z = 0;
    }
    if (p.z <= 0) {
      p.z = 0;
      if (this.velocity.z < 0) this.velocity.z = 0;
      this.onGround = true;
    }
  }

  /** The least rise (m, ≤ STEP) that clears the box at (x, y), or 0. */
  private stepUp(x: number, y: number, z: number): number {
    for (let rise = VOXEL; rise <= STEP + 1e-6; rise += VOXEL) {
      if (!this.blocked(x, y, z + rise, this.height)) {
        // Rest on what's there: the lowest clear height
        return this.floorBelow(x, y, z, z + rise) - z;
      }
    }
    return 0;
  }

  /** Highest clear footing between `from` and `to` (to ≥ from) at (x, y), found by halving. */
  private floorBelow(x: number, y: number, from: number, to: number): number {
    let [lo, hi] = [from, to];
    for (let k = 0; k < 10; k++) {
      const mid = (lo + hi) / 2;
      if (this.blocked(x, y, mid, this.height)) lo = mid;
      else hi = mid;
    }
    return hi;
  }

  /** The player's box at feet (x, y, z) overlaps a fixed voxel or debris. */
  blocked(x: number, y: number, z: number, height: number): boolean {
    if (z < -1e-4) return true;
    const cell = (m: number) => Math.floor(m / VOXEL);
    const x0 = cell(x - WIDTH);
    const x1 = cell(x + WIDTH - 1e-6);
    const y0 = cell(y - WIDTH);
    const y1 = cell(y + WIDTH - 1e-6);
    const z0 = cell(z);
    const z1 = cell(z + height - 1e-6);
    const { city } = this;
    for (const b of city.buildings) {
      if (x1 < b.x0 || x0 >= b.x0 + b.w || y1 < b.y0 || y0 >= b.y0 + b.d || z0 >= b.h) continue;
      for (let zi = Math.max(z0, 0); zi <= Math.min(z1, b.h - 1); zi++) {
        for (let yi = Math.max(y0, b.y0); yi <= Math.min(y1, b.y0 + b.d - 1); yi++) {
          for (let xi = Math.max(x0, b.x0); xi <= Math.min(x1, b.x0 + b.w - 1); xi++) {
            const v = b.cells[xi - b.x0 + b.w * (yi - b.y0 + b.d * zi)];
            if (v >= 0 && city.state[v] === State.Fixed) return true;
          }
        }
      }
    }
    // Debris as unturned voxels (a little smaller, as they sit tilted and settle)
    const d = this.debris;
    const r = VOXEL * 0.45;
    for (let k = 0; k < this.debrisCount; k++) {
      const [cx, cy, cz] = [d[3 * k], d[3 * k + 1], d[3 * k + 2]];
      if (Math.abs(cx - x) < WIDTH + r && Math.abs(cy - y) < WIDTH + r && cz + r > z && cz - r < z + height) return true;
    }
    return false;
  }
}
