// The race track: a closed circuit on open grass (a spline through a few corners, 14 m of
// asphalt with kerbs where it bends), and things along it to drive through or shoot: a brick
// wall across the main straight, the start gantry, a footbridge, the grandstand and the pit
// garages, tyre walls on the outside of the corners, crates, oil drums, cones, billboards and
// a truck. Every piece is voxels, as breakable as the city (world.ts builds them).

import { type City, M, Mat, tree, type WorldBuilder, worldBuilder } from './world.ts';

/** The circuit's middle line, sampled every metre round the lap (for the ground and the car). */
export interface TrackLine {
  /** x, y (m) per sample; the last joins back to the first. */
  points: Float32Array;
  /** Unit direction of travel per sample (x, y). */
  tangents: Float32Array;
  /** Per sample: how much it's a corner (0 straight, 1 in a bend: kerbs and run-off there), and which way it turns (+1 left). */
  corner: Float32Array;
  turn: Int8Array;
  /** Asphalt width (m). */
  width: number;
  /** Lap length (m) and where the start line is along it. */
  length: number;
  start: number;
}

/** Asphalt width, the kerbs' width outside it, and the run-off beyond them in the bends (m). */
export const TRACK_WIDTH = 14;
export const KERB = 1.2;
export const RUNOFF = 8;

/** Corners of the lap (m), in order of travel (anticlockwise seen from above). */
const CONTROL: [number, number][] = [
  [-150, -90],
  [-60, -90],
  [40, -90],
  [120, -88],
  [165, -65],
  [178, -20],
  [160, 20],
  [120, 35],
  [90, 60],
  [40, 70],
  [-10, 55],
  [-40, 90],
  [-95, 100],
  [-150, 80],
  [-180, 40],
  [-185, -20],
  [-175, -65],
];
/** The start line (m along the main straight). */
const START_X = -60;

/** Centripetal Catmull-Rom through the control points, resampled every metre. */
export function trackLine(): TrackLine {
  const n = CONTROL.length;
  const dense: number[] = [];
  const at = (i: number) => CONTROL[((i % n) + n) % n];
  for (let i = 0; i < n; i++) {
    const [p0, p1, p2, p3] = [at(i - 1), at(i), at(i + 1), at(i + 2)];
    const knot = (a: [number, number], b: [number, number]) => Math.sqrt(Math.hypot(b[0] - a[0], b[1] - a[1]));
    const t1 = knot(p0, p1);
    const t2 = t1 + knot(p1, p2);
    const t3 = t2 + knot(p2, p3);
    for (let k = 0; k < 200; k++) {
      const t = t1 + ((t2 - t1) * k) / 200;
      const lerp = (a: [number, number], b: [number, number], ta: number, tb: number) => {
        const f = (t - ta) / (tb - ta);
        return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f];
      };
      const a1 = lerp(p0, p1, 0, t1);
      const a2 = lerp(p1, p2, t1, t2);
      const a3 = lerp(p2, p3, t2, t3);
      const b1 = [a1[0] + (a2[0] - a1[0]) * ((t - 0) / t2), a1[1] + (a2[1] - a1[1]) * ((t - 0) / t2)];
      const b2 = [a2[0] + (a3[0] - a2[0]) * ((t - t1) / (t3 - t1)), a2[1] + (a3[1] - a2[1]) * ((t - t1) / (t3 - t1))];
      const f = (t - t1) / (t2 - t1);
      dense.push(b1[0] + (b2[0] - b1[0]) * f, b1[1] + (b2[1] - b1[1]) * f);
    }
  }
  // Every metre along it
  const cum = [0];
  const m = dense.length / 2;
  for (let k = 1; k <= m; k++) {
    const [a, b] = [(k - 1) % m, k % m];
    cum.push(cum[k - 1] + Math.hypot(dense[2 * b] - dense[2 * a], dense[2 * b + 1] - dense[2 * a + 1]));
  }
  const length = cum[m];
  const count = Math.round(length);
  const points = new Float32Array(2 * count);
  let j = 0;
  for (let k = 0; k < count; k++) {
    const s = (k * length) / count;
    while (cum[j + 1] < s) j++;
    const f = (s - cum[j]) / Math.max(1e-9, cum[j + 1] - cum[j]);
    const [a, b] = [j % m, (j + 1) % m];
    points[2 * k] = dense[2 * a] + (dense[2 * b] - dense[2 * a]) * f;
    points[2 * k + 1] = dense[2 * a + 1] + (dense[2 * b + 1] - dense[2 * a + 1]) * f;
  }
  const tangents = new Float32Array(2 * count);
  const curvature = new Float32Array(count);
  for (let k = 0; k < count; k++) {
    const [a, b] = [(k - 1 + count) % count, (k + 1) % count];
    const dx = points[2 * b] - points[2 * a];
    const dy = points[2 * b + 1] - points[2 * a + 1];
    const l = Math.hypot(dx, dy) || 1;
    tangents[2 * k] = dx / l;
    tangents[2 * k + 1] = dy / l;
  }
  for (let k = 0; k < count; k++) {
    // Turn of the direction over ±4 m
    const [a, b] = [(k - 4 + count) % count, (k + 4) % count];
    const cross = tangents[2 * a] * tangents[2 * b + 1] - tangents[2 * a + 1] * tangents[2 * b];
    curvature[k] = Math.asin(Math.max(-1, Math.min(1, cross))) / 8;
  }
  // A corner: tighter than an 80 m radius, grown 8 m each way (kerbs run in and out of a bend)
  const corner = new Float32Array(count);
  const turn = new Int8Array(count);
  for (let k = 0; k < count; k++) {
    let most = 0;
    let way = 0;
    for (let o = -8; o <= 8; o++) {
      const c = curvature[(k + o + count) % count];
      if (Math.abs(c) > Math.abs(way)) way = c;
      most = Math.max(most, Math.abs(c));
    }
    corner[k] = Math.max(0, Math.min(1, (most * 80 - 1) * 2));
    turn[k] = way >= 0 ? 1 : -1;
  }
  // The start: where the main straight crosses x = START_X
  let start = 0;
  for (let k = 0; k < count; k++) if (Math.abs(points[2 * k] - START_X) < Math.abs(points[2 * start] - START_X) && points[2 * k + 1] < -60) start = k;
  return { points, tangents, corner, turn, width: TRACK_WIDTH, length: count, start };
}

/** Distance (m) from (x, y) to the middle line, brute force (for building, not per frame). */
export function distanceToTrack(line: TrackLine, x: number, y: number): number {
  let best = Infinity;
  const n = line.length;
  for (let k = 0; k < n; k++) {
    const [ax, ay] = [line.points[2 * k], line.points[2 * k + 1]];
    const [bx, by] = [line.points[2 * ((k + 1) % n)], line.points[2 * ((k + 1) % n) + 1]];
    const [dx, dy] = [bx - ax, by - ay];
    const f = Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy || 1)));
    best = Math.min(best, Math.hypot(x - ax - dx * f, y - ay - dy * f));
  }
  return best;
}

/** The world at the race track, and its line. */
export function buildTrack(seed = 11): { city: City; line: TrackLine } {
  const line = trackLine();
  const w = worldBuilder(seed);
  const n = line.length;
  const P = (k: number) => [line.points[2 * (((k % n) + n) % n)], line.points[2 * (((k % n) + n) % n) + 1]];
  const T = (k: number) => [line.tangents[2 * (((k % n) + n) % n)], line.tangents[2 * (((k % n) + n) % n) + 1]];
  /** The point `side` m to the left of the line at sample k. */
  const beside = (k: number, side: number) => {
    const [x, y] = P(k);
    const [tx, ty] = T(k);
    return [x - ty * side, y + tx * side];
  };
  const half = TRACK_WIDTH / 2;
  const s0 = line.start;

  // The brick wall right across the main straight, 160 m after the start: the first thing to drive through
  brickWall(w, P(s0 + 160)[0], P(s0 + 160)[1]);
  w.last().kind = 'obstacle';
  // The start gantry over the line
  gantry(w, P(s0)[0], P(s0)[1]);
  // The grandstand along the outside of the main straight, the pit garages inside it behind the pit wall
  grandstand(w, P(s0 - 10)[0] - 22, P(s0)[1] - half - 16);
  pitWall(w, P(s0 - 80)[0], P(s0 + 60)[0], P(s0)[1] + half + 2.5);
  garages(w, P(s0 - 70)[0], P(s0)[1] + half + 11);
  // Tyre walls round the outside of the bends, clear of any other part of the track
  tyreWalls(w, line, half + KERB + RUNOFF - 1);
  // A crate pyramid on the exit of the fast right-hander, oil drums at the chicane, cones in a slalom
  crates(w, ...(beside(Math.round(n * 0.47), 1.5) as [number, number]));
  w.last().kind = 'obstacle';
  for (const [f, side] of [
    [0.4, -4],
    [0.42, 3],
  ] as [number, number][])
  {
    drums(w, ...(beside(Math.round(n * f), side) as [number, number]));
    w.last().kind = 'obstacle';
  }
  for (let k = 0; k < 10; k++) {
    cone(w, ...(beside(s0 + 60 + k * 8, k % 2 ? 3 : -3) as [number, number]));
    w.last().kind = 'obstacle';
  }
  // A truck broadside across the track on the back section
  truck(w, ...(beside(Math.round(n * 0.62), 0) as [number, number]), T(Math.round(n * 0.62)));
  w.last().kind = 'obstacle';
  // A footbridge over the straight at the top left
  const bridgeAt = Math.round(n * 0.75);
  footbridge(w, P(bridgeAt)[0], P(bridgeAt)[1], T(bridgeAt));
  // Billboards on the outside of three bends
  for (const f of [0.2, 0.33, 0.9]) {
    const k = Math.round(n * f);
    const out = -line.turn[k] * (half + KERB + RUNOFF + 4);
    billboard(w, ...(beside(k, out) as [number, number]), Math.round(f * 7));
  }
  // Trees scattered round the infield and outside, off the asphalt and run-off
  const rnd = w.rnd;
  let placed = 0;
  for (let k = 0; k < 400 && placed < 26; k++) {
    const [x, y] = [-200 + rnd() * 400, -130 + rnd() * 250];
    const d = distanceToTrack(line, x, y);
    if (d < half + KERB + RUNOFF + 6 || (y < -60 && x > -170 && x < 60)) continue;
    tree(w, x, y, 100 + k);
    placed++;
  }
  return { city: w.finish({ pitch: 46, street: 16, blocks: 0 }), line };
}

/** Voxel corner of a thing `w` × `d` voxels centred at (cx, cy) m. */
const corner = (cx: number, cy: number, wv: number, dv: number): [number, number] => [Math.round(cx * M - wv / 2), Math.round(cy * M - dv / 2)];

/** A brick wall across the track (along y), 3 m high and half a metre thick, a target painted on its face. */
function brickWall(w: WorldBuilder, cx: number, cy: number): void {
  const [L, D, H] = [(TRACK_WIDTH + 4) * M, 2, 12];
  const [x0, y0] = corner(cx, cy, D, L);
  w.place(x0, y0, D, L, H, (put) => {
    for (let z = 0; z < H; z++) {
      for (let y = 0; y < L; y++) {
        for (let x = 0; x < D; x++) {
          // A target: rings round the middle, on the side the cars come from
          const r = Math.hypot(y - L / 2 + 0.5, z - H / 2 + 0.5);
          const ring = x === 0 && r < 5.5 && Math.floor(r / 1.4) % 2 === 0;
          put(x, y, z, Mat.Brick, ring ? (r < 1.4 ? 0xf0e8dc : 0xc8322a) : undefined);
        }
      }
    }
  });
}

/** The start gantry: two steel posts outside the track, a beam across with the start lights and a banner. */
function gantry(w: WorldBuilder, cx: number, cy: number): void {
  const span = (TRACK_WIDTH + 4) * M;
  const [x0, y0] = corner(cx, cy, 4, span);
  w.place(x0, y0, 4, span, 32, (put) => {
    for (const y of [0, span - 3]) w.box(put, 1, y, 0, 3, y + 3, 30, Mat.Steel, 0x3a3f45);
    w.box(put, 0, 0, 26, 4, span, 30, Mat.Steel, 0x2a2e33);
    // The lights: five pairs of red, and green
    for (let k = 0; k < 5; k++) {
      const y = span / 2 - 12 + k * 5;
      w.box(put, 0, y, 24, 1, y + 3, 26, Mat.Neon, k === 4 ? 0x2bd45a : 0xd01c14);
    }
    // A banner under the beam on the far side: chequered
    for (let y = 3; y < span - 3; y++) for (let z = 21; z < 26; z++) put(3, y, z, Mat.Neon, (Math.floor(y / 2) + Math.floor(z / 2)) % 2 ? 0x101010 : 0xf2f2f2);
  });
}

/**
 * The grandstand: stepped concrete tiers of coloured seats rising away from the track, a roof
 * on steel columns over them, stairs at the ends.
 */
function grandstand(w: WorldBuilder, cx: number, cy: number): void {
  const [L, D, H] = [36 * M, 12 * M, 11 * M];
  const [x0, y0] = corner(cx, cy, L, D);
  const seats = [0x1f4fa8, 0xd8d4cc, 0xc8322a];
  w.place(x0, y0, L, D, H, (put) => {
    const tiers = 14;
    for (let t = 0; t < tiers; t++) {
      // Tier t: 3 voxels deep, 2 higher than the last, from the track side (y high) back
      const y1 = D - 2 - t * 3;
      const z = 2 + t * 2;
      for (let x = 2; x < L - 2; x++) {
        for (let y = y1 - 3; y < y1; y++) {
          put(x, y, z - 1, Mat.Concrete);
          put(x, y, z - 2, Mat.Concrete);
        }
        // A seat on each tier every metre (a gap between for the feet)
        if (x % 4 !== 0) put(x, y1 - 2, z, Mat.Plaster, seats[Math.floor(x / 36) % 3]);
      }
      // The tiers stand on walls under their fronts
      for (let x = 2; x < L - 2; x += 24) for (let zz = 0; zz < z - 2; zz++) for (let y = y1 - 3; y < y1; y++) put(x, y, zz, Mat.Concrete);
    }
    // The front wall and the back wall
    w.box(put, 2, D - 2, 0, L - 2, D, 3, Mat.Concrete);
    w.box(put, 2, 0, 0, L - 2, 2, 2 + tiers * 2, Mat.Concrete);
    // Roof columns at the back, the roof leaning out over the tiers
    for (let x = 2; x < L - 2; x += 32) w.box(put, x, 0, 0, x + 2, 2, H - 1, Mat.Steel, 0x3a3f45);
    for (let y = 0; y < D; y++) {
      const z = H - 1 - Math.floor(y / 12);
      for (let x = 0; x < L; x++) put(x, y, z, Mat.Steel, y === D - 1 ? 0xc8322a : 0xd8dad6);
    }
    // Hangers from the roof's front edge down to the top tier: it rests on the tiers
    for (let x = 2; x < L - 2; x += 32) for (let z = 2 + tiers * 2; z < H - 3; z++) put(x, 3, z, Mat.Steel, 0x3a3f45);
  });
}

/** The pit wall: a low concrete wall along the inside of the main straight, a wire fence on it. */
function pitWall(w: WorldBuilder, xa: number, xb: number, cy: number): void {
  // In pieces 16 m long (a whole wall as one object would come down as one)
  for (let x = Math.min(xa, xb); x < Math.max(xa, xb); x += 16) {
    const [x0, y0] = corner(x + 8, cy, 16 * M, 2);
    w.place(x0, y0, 16 * M, 2, 9, (put) => {
      w.box(put, 0, 0, 0, 16 * M, 2, 4, Mat.Concrete, 0xd8d4cc);
      for (let xx = 0; xx < 16 * M; xx++) {
        if (xx % 8 === 0) for (let z = 4; z < 9; z++) put(xx, 0, z, Mat.Steel, 0x5a5f66);
        put(xx, 0, 8, Mat.Steel, 0x5a5f66);
        // Red and white on the wall's face
        put(xx, 1, 3, Mat.Neon, Math.floor(xx / 8) % 2 ? 0xd01c14 : 0xf0f0f0);
      }
    });
  }
}

/** The pit garages: a long low building, four bays with roller doors facing the track, a terrace on the roof. */
function garages(w: WorldBuilder, cx: number, cy: number): void {
  const [L, D, H] = [40 * M, 10 * M, 20];
  const [x0, y0] = corner(cx, cy, L, D);
  w.place(x0, y0, L, D, H, (put) => {
    const bay = L / 4;
    for (let z = 0; z < H - 4; z++) {
      for (let x = 0; x < L; x++) {
        for (const y of [0, 1, D - 1]) {
          const front = y <= 1;
          const inBay = x % bay > 3 && x % bay < bay - 3;
          if (front && inBay && z < 14) {
            if (y === 1) put(x, y, z, Mat.Steel, z % 2 ? 0x9aa0a6 : 0x7f858c);
          } else put(x, y, z, Mat.Plaster, 0xe6e2da);
        }
        if (x === 0 || x === L - 1 || x % bay === 0) for (let y = 2; y < D - 2; y++) put(x, y, z, Mat.Plaster, 0xe6e2da);
      }
    }
    // Roof slab, a sign band along the front, a glass balustrade
    w.box(put, 0, 0, H - 4, L, D, H - 3, Mat.Concrete);
    for (let x = 0; x < L; x++) {
      put(x, 0, H - 5, Mat.Neon, Math.floor(x / 20) % 2 ? 0xff7a00 : 0x1a1a1a);
      put(x, 0, H - 3, Mat.Glass);
      put(x, 0, H - 2, Mat.Steel, 0x3a3f45);
    }
  });
}

/** Tyre walls round the outside of the bends (`at` m from the middle line), in 8 m pieces. */
function tyreWalls(w: WorldBuilder, line: TrackLine, at: number): void {
  const n = line.length;
  let run: [number, number][] = [];
  const flush = () => {
    if (run.length < 2) return (run = []);
    const xs = run.map((p) => Math.round(p[0] * M));
    const ys = run.map((p) => Math.round(p[1] * M));
    const [lx, ly] = [Math.min(...xs) - 1, Math.min(...ys) - 1];
    const [wx, wy] = [Math.max(...xs) - lx + 2, Math.max(...ys) - ly + 2];
    const stacks = xs.map((x, k) => [x - lx, ys[k] - ly]);
    w.place(lx, ly, wx, wy, 3, (put) => {
      stacks.forEach(([x, y], k) => {
        for (let z = 0; z < 3; z++) {
          const top = z === 2 ? (k % 2 ? 0xd01c14 : 0xefefef) : 0x1b1b1c;
          for (const [dx, dy] of [
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1],
          ])
            put(x + dx, y + dy, z, Mat.Steel, top);
        }
      });
    });
    run = [];
  };
  for (let k = 0; k < n; k++) {
    if (line.corner[k] < 0.5) {
      flush();
      continue;
    }
    // Every half metre along the outside of the bend
    for (const f of [0, 0.5]) {
      const side = -line.turn[k] * at;
      const [x, y] = [line.points[2 * k] + line.tangents[2 * k] * f, line.points[2 * k + 1] + line.tangents[2 * k + 1] * f];
      const p: [number, number] = [x - line.tangents[2 * k + 1] * side, y + line.tangents[2 * k] * side];
      // Not across another part of the track
      if (distanceToTrack(line, p[0], p[1]) < at - 1) {
        flush();
        continue;
      }
      run.push(p);
      if (run.length >= 16) flush();
    }
  }
  flush();
}

/** A pyramid of wooden crates, 1 m each. */
function crates(w: WorldBuilder, cx: number, cy: number): void {
  const base = 4;
  const [x0, y0] = corner(cx, cy, base * M, 2 * M);
  w.place(x0, y0, base * M, 2 * M, base * M, (put) => {
    for (let level = 0; level < base; level++) {
      for (let c = 0; c < base - level; c++) {
        for (let r = 0; r < 2; r++) {
          const [bx, by, bz] = [level * 2 + c * M, r * M, level * M];
          for (let z = 0; z < M; z++)
            for (let y = 0; y < M; y++)
              for (let x = 0; x < M; x++) {
                const edge = (x === 0 || x === M - 1 ? 1 : 0) + (y === 0 || y === M - 1 ? 1 : 0) + (z === 0 || z === M - 1 ? 1 : 0) >= 2;
                put(bx + x, by + y, bz + z, Mat.Wood, edge ? 0x6b4a2c : 0xa87c4c);
              }
        }
      }
    }
  });
}

/** Oil drums: red steel, a pyramid of three rows. */
function drums(w: WorldBuilder, cx: number, cy: number): void {
  const [x0, y0] = corner(cx, cy, 9, 3);
  w.place(x0, y0, 9, 3, 12, (put) => {
    for (let level = 0; level < 3; level++) {
      for (let k = 0; k < 3 - level; k++) {
        const x = level * 1.5 + k * 3;
        for (let z = level * 4; z < level * 4 + 4; z++)
          for (let y = 0; y < 3; y++) for (let dx = 0; dx < 3; dx++) put(Math.round(x) + dx, y, z, Mat.Steel, z % 4 === 1 ? 0xe8e2d4 : 0xb3261e);
      }
    }
  });
}

/** A traffic cone. */
function cone(w: WorldBuilder, cx: number, cy: number): void {
  const [x0, y0] = corner(cx, cy, 2, 2);
  w.place(x0, y0, 2, 2, 4, (put) => {
    w.box(put, 0, 0, 0, 2, 2, 1, Mat.Steel, 0x1b1b1c);
    // Painted, not lit (lit orange blooms to a white blob)
    put(0, 0, 1, Mat.Steel, 0xff6a00);
    put(0, 0, 2, Mat.Steel, 0xf2f2f2);
    put(0, 0, 3, Mat.Steel, 0xff6a00);
  });
}

/** A lorry parked across the track (`dir`: the track's way there): a cab and a box trailer, on black wheels. */
function truck(w: WorldBuilder, cx: number, cy: number, dir: number[]): void {
  const alongY = Math.abs(dir[0]) > Math.abs(dir[1]);
  const [L, D, H] = [10 * M, 10, 16];
  const [wv, dv] = alongY ? [D, L] : [L, D];
  const [x0, y0] = corner(cx, cy, wv, dv);
  w.place(x0, y0, wv, dv, H, (put) => {
    const at = (b: number, a: number, z: number, m: Mat, c?: number) => (alongY ? put(b, a, z, m, c) : put(a, b, z, m, c));
    const box = (b0: number, a0: number, z0: number, b1: number, a1: number, z1: number, m: Mat, c?: number) => {
      for (let z = z0; z < z1; z++) for (let a = a0; a < a1; a++) for (let b = b0; b < b1; b++) at(b, a, z, m, c);
    };
    // Along its length: the trailer, then the cab
    box(1, 0, 3, D - 1, 30, 15, Mat.Steel, 0xd8dad6);
    for (let a = 0; a < 30; a++) at(0, a, 9, Mat.Neon, 0x1f4fa8);
    box(1, 31, 3, D - 1, L, 11, Mat.Steel, 0x1f4fa8);
    box(2, L - 1, 7, D - 2, L, 10, Mat.Glass);
    box(1, 0, 2, D - 1, L, 3, Mat.Steel, 0x2a2d31);
    for (const a of [3, 8, 24, 34]) for (const b of [0, D - 2]) box(b, a, 0, b + 2, a + 3, 3, Mat.Steel, 0x151515);
  });
}

/**
 * A footbridge over the track, square to it: a concrete deck on two piers each side of the
 * asphalt, glass balustrades, stair towers at the ends.
 */
function footbridge(w: WorldBuilder, cx: number, cy: number, dir: number[]): void {
  // Square to the track: the deck runs across it, along whichever axis is nearer the normal
  const acrossX = Math.abs(dir[1]) > Math.abs(dir[0]);
  const span = (TRACK_WIDTH + 12) * M;
  const [L, D, H] = [span, 3 * M, 26];
  const [wv, dv] = acrossX ? [L, D] : [D, L];
  const [x0, y0] = corner(cx, cy, wv, dv);
  w.place(x0, y0, wv, dv, H, (put) => {
    const at = (a: number, b: number, z: number, m: Mat, c?: number) => (acrossX ? put(a, b, z, m, c) : put(b, a, z, m, c));
    for (let a = 0; a < L; a++) {
      for (let b = 0; b < D; b++) {
        at(a, b, 20, Mat.Concrete);
        at(a, b, 21, Mat.Concrete);
        if (b === 0 || b === D - 1) {
          for (let z = 22; z < 25; z++) at(a, b, z, Mat.Glass);
          at(a, b, 25, Mat.Steel, 0x3a3f45);
        }
      }
    }
    // Piers each side of the asphalt, and stair towers at the ends
    for (const a of [5 * M, L - 5 * M - 2]) for (let b = 2; b < D - 2; b++) for (let z = 0; z < 20; z++) for (let da = 0; da < 2; da++) at(a + da, b, z, Mat.Concrete);
    for (const a0 of [0, L - 3 * M]) for (let a = a0; a < a0 + 3 * M; a++) for (let b = 0; b < D; b++) for (let z = 0; z < 20; z++) if (a === a0 || a === a0 + 3 * M - 1 || b === 0 || b === D - 1) at(a, b, z, Mat.Plaster, 0xd4cfc4);
  });
}

/** A billboard: two steel legs and a painted board (colour scheme `k`). */
function billboard(w: WorldBuilder, cx: number, cy: number, k: number): void {
  const [L, H] = [8 * M, 28];
  const schemes = [
    [0xff7a00, 0x111111],
    [0xd01c14, 0xf2f2f2],
    [0x1f4fa8, 0xf2d21c],
    [0x2bd45a, 0x111111],
  ];
  const [a, b] = schemes[k % schemes.length];
  const [x0, y0] = corner(cx, cy, L, 2);
  w.place(x0, y0, L, 2, H, (put) => {
    for (const x of [4, L - 5]) w.box(put, x, 1, 0, x + 1, 2, 16, Mat.Steel, 0x3a3f45);
    for (let x = 0; x < L; x++) for (let z = 16; z < H; z++) put(x, 0, z, Mat.Neon, (x + z) % 12 < 6 ? a : b);
    w.box(put, 0, 1, 16, L, 2, H, Mat.Steel, 0x3a3f45);
  });
}

/** Where the car starts: on the grid behind the start line, facing along the track. */
export function gridSlot(line: TrackLine): { position: [number, number]; heading: number } {
  const k = (line.start - 14 + line.length) % line.length;
  return { position: [line.points[2 * k], line.points[2 * k + 1]], heading: Math.atan2(line.tangents[2 * k + 1], line.tangents[2 * k]) };
}

/**
 * The circuit as a picture for the ground to be drawn from: per texel of a `size` × `size`
 * grid over ±`extent` m, the signed distance to the middle line (m, + to its left), how far
 * round the lap (m), how much of a corner it is there, and which way it turns; FAR m and more
 * away, just "far". Each metre of the line marks the texels near it (a texel keeps the nearest).
 */
export function trackField(line: TrackLine, size: number, extent: number): Float32Array {
  const FAR = 18;
  const out = new Float32Array(size * size * 4);
  const best = new Float32Array(size * size).fill(FAR * FAR);
  for (let t = 0; t < size * size; t++) out[4 * t] = FAR;
  const n = line.length;
  const texel = (2 * extent) / size;
  for (let k = 0; k < n; k++) {
    const [ax, ay] = [line.points[2 * k], line.points[2 * k + 1]];
    const [bx, by] = [line.points[2 * ((k + 1) % n)], line.points[2 * ((k + 1) % n) + 1]];
    const [dx, dy] = [bx - ax, by - ay];
    const l2 = dx * dx + dy * dy || 1;
    const i0 = Math.max(0, Math.floor((Math.min(ax, bx) - FAR + extent) / texel));
    const i1 = Math.min(size - 1, Math.ceil((Math.max(ax, bx) + FAR + extent) / texel));
    const j0 = Math.max(0, Math.floor((Math.min(ay, by) - FAR + extent) / texel));
    const j1 = Math.min(size - 1, Math.ceil((Math.max(ay, by) + FAR + extent) / texel));
    for (let j = j0; j <= j1; j++) {
      const y = -extent + (j + 0.5) * texel;
      for (let i = i0; i <= i1; i++) {
        const x = -extent + (i + 0.5) * texel;
        const f = Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / l2));
        const [ex, ey] = [x - ax - dx * f, y - ay - dy * f];
        const d2 = ex * ex + ey * ey;
        const t = i + size * j;
        if (d2 >= best[t]) continue;
        best[t] = d2;
        const d = Math.sqrt(d2);
        const o = 4 * t;
        out[o] = dx * (y - ay) - dy * (x - ax) >= 0 ? d : -d;
        out[o + 1] = k + f;
        out[o + 2] = line.corner[k];
        out[o + 3] = line.turn[k];
      }
    }
  }
  return out;
}
