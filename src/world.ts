// The scene: a tall, slim tower on a street corner, after the Hong Kong corner blocks of the
// 1950s and 60s: a rounded corner, ribbon windows between projecting floor bands, a glass
// shopfront under a cantilevered canopy with a glowing sign, a blade sign up the front, AC
// units hung under windows, a water tank on legs and a mast on the roof. Eleven storeys on a
// 13 x 11 m footprint, so taking out the bottom brings it down. Inside: a slab each floor,
// columns and a stair core. Outside: parked cars, a tree, street lamps.
//
// Built of VOXEL-sized voxels (a quarter metre: walls two voxels thick, blasts chew ragged
// holes). Every voxel is a body in the solver, fixed until something knocks it loose
// (destruction.ts); each object keeps a grid of its voxels for raycasts, the player's footing
// and the check of what still stands. Glass is grouped into panes (a window each), which
// shatter whole, and whose openings breathe out dust when a storey fails.

/** Voxel edge (m). */
export const VOXEL = 0.25;
/** The tower's floors (voxels): the shop floor's height, then each storey's. */
export const FLOORS = { ground: 18, storey: 12 } as const;

/** What a voxel is made of: its look (render.ts) and the dust it throws (particles). */
export const Mat = { Concrete: 0, Glass: 1, Brick: 2, Steel: 3, Roof: 4, LitGlass: 5, Plaster: 6, Trim: 7, Leaf: 8, Neon: 9, Wood: 10 } as const;
export type Mat = (typeof Mat)[keyof typeof Mat];

/**
 * A voxel's life: fixed in place, loose (simulated), gone (blown away), rubble (once loose,
 * now at rest and fixed where it lies until disturbed: physics.ts freeze), or carried by a
 * rigid section falling whole (physics.ts proxy).
 */
export const State = { Fixed: 0, Loose: 1, Gone: 2, Rubble: 3, Carried: 4 } as const;

export interface Building {
  /** Min corner and size, in voxels (the corner's world position is x0 · VOXEL). */
  x0: number;
  y0: number;
  w: number;
  d: number;
  h: number;
  /** Voxel index at each cell (x + w (y + d z)), or -1. */
  cells: Int32Array;
  /** What it is, where that matters (a car's alarm goes off when it's hit). */
  kind?: 'car';
}

export interface City {
  count: number;
  /** Per voxel: centre (xyz, m), material, object, and its cell in that object. */
  position: Float32Array;
  material: Uint8Array;
  building: Uint16Array;
  cell: Int32Array;
  /** Per voxel colour (0xrrggbb): the material's, varied per object and per voxel. */
  color: Uint32Array;
  state: Uint8Array;
  /**
   * Which faces of each voxel were open to the air when built (bits: -x, +x, -y, +y, -z, +z):
   * those are finished surfaces (brick, plaster, paint); a face bared later is a break.
   */
  exposed: Uint8Array;
  /** Each glass voxel's pane (index into panes), or -1. */
  pane: Int32Array;
  /** Windows: their glass, centre (m) and the way they face (unit, horizontal). */
  panes: Pane[];
  buildings: Building[];
  /** The street grid (render.ts draws it): block pitch and street width (m), blocks a side. */
  pitch: number;
  street: number;
  blocks: number;
}

export interface Pane {
  voxels: number[];
  centre: [number, number, number];
  normal: [number, number, number];
}

/** Glass (a lit window is glass too). */
export const isGlass = (m: number): boolean => m === Mat.Glass || m === Mat.LitGlass;

/** Deterministic pseudo-random numbers in [0, 1). */
export function random(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 2 ** 32;
  };
}

/** The block the building stands on and the streets round it (m). */
const BLOCK = 30;
const STREET = 16;
/** Voxels a metre. */
export const M = Math.round(1 / VOXEL);

/** Base colours by material; each object shifts its own a little. */
export const BASE: Record<Mat, number> = {
  [Mat.Concrete]: 0xb9b4aa,
  [Mat.Glass]: 0x3d5566,
  [Mat.Brick]: 0x9a4a35,
  [Mat.Steel]: 0x4a4e54,
  [Mat.Roof]: 0x5b5953,
  [Mat.LitGlass]: 0xffd9a0,
  [Mat.Plaster]: 0xd8cdb8,
  [Mat.Trim]: 0xe2dccf,
  [Mat.Leaf]: 0x4a7a30,
  [Mat.Neon]: 0xd8302a,
  [Mat.Wood]: 0x6b4a30,
};

type Put = (x: number, y: number, z: number, m: Mat, colour?: number) => void;

/** Builds a world object by object (buildCity, and the other scenes): `finish` makes the City. */
export interface WorldBuilder {
  rnd: () => number;
  /** An object of w × d × h voxels with its min corner at voxel (x0, y0, 0); `fill` puts its voxels. */
  place(x0: number, y0: number, w: number, d: number, h: number, fill: (put: Put) => void): void;
  /** A block of voxels, x0..x1 × y0..y1 × z0..z1 (ends excluded). */
  box(put: Put, x0: number, y0: number, z0: number, x1: number, y1: number, z1: number, m: Mat, colour?: number): void;
  /** The object placed last. */
  last(): Building;
  /** The world: what's been placed, its faces and windows, and the street grid the ground draws. */
  finish(grid: { pitch: number; street: number; blocks: number }): City;
}

export function worldBuilder(seed: number): WorldBuilder {
  const rnd = random(seed);
  const pos: number[] = [];
  const mat: number[] = [];
  const bld: number[] = [];
  const cellOf: number[] = [];
  const col: number[] = [];
  const buildings: Building[] = [];

  /** An object of w × d × h voxels with its min corner at voxel (x0, y0, 0). */
  const place = (x0: number, y0: number, w: number, d: number, h: number, fill: (put: Put) => void) => {
    const b: Building = { x0, y0, w, d, h, cells: new Int32Array(w * d * h).fill(-1) };
    const id = buildings.length;
    buildings.push(b);
    const shift = [rnd() - 0.5, rnd() - 0.5, rnd() - 0.5].map((v) => v * 0.1);
    const tint = (m: Mat) => {
      const base = BASE[m];
      // Bricks vary a lot, one in twelve much darker (fired harder); the rest a little
      const n = m === Mat.Brick ? (rnd() - 0.5) * 0.22 - (rnd() < 0.08 ? 0.18 : 0) : (rnd() - 0.5) * (m === Mat.LitGlass ? 0.2 : 0.05);
      const ch = (s: number, k: number) => Math.max(0, Math.min(255, Math.round(((base >> s) & 255) * (1 + shift[k] + n))));
      return (ch(16, 0) << 16) | (ch(8, 1) << 8) | ch(0, 2);
    };
    const first = pos.length / 3;
    fill((x, y, z, m, colour) => {
      if (x < 0 || y < 0 || z < 0 || x >= w || y >= d || z >= h) return;
      const c = x + w * (y + d * z);
      if (b.cells[c] >= 0) return;
      b.cells[c] = pos.length / 3;
      pos.push((x0 + x + 0.5) * VOXEL, (y0 + y + 0.5) * VOXEL, (z + 0.5) * VOXEL);
      mat.push(m);
      bld.push(id);
      cellOf.push(c);
      col.push(colour ?? tint(m));
    });
    dropFloating(b, first);
  };

  /**
   * Leave out voxels of the object just placed (from voxel `first` on) not joined to the
   * ground through its others: nothing starts out hanging in the air.
   */
  const dropFloating = (b: Building, first: number) => {
    const layer = b.w * b.d;
    const reached = new Uint8Array(b.cells.length);
    const queue: number[] = [];
    for (let c = 0; c < layer; c++) if (b.cells[c] >= 0) (reached[c] = 1), queue.push(c);
    while (queue.length) {
      const c = queue.pop()!;
      const x = c % b.w;
      const y = Math.floor(c / b.w) % b.d;
      for (const [ok, m] of [
        [x > 0, c - 1],
        [x + 1 < b.w, c + 1],
        [y > 0, c - b.w],
        [y + 1 < b.d, c + b.w],
        [c >= layer, c - layer],
        [c + layer < b.cells.length, c + layer],
      ] as [boolean, number][]) {
        if (ok && !reached[m] && b.cells[m] >= 0) (reached[m] = 1), queue.push(m);
      }
    }
    // Keep the reached ones, in order, renumbered
    const keep: number[] = [];
    for (let v = first; v < pos.length / 3; v++) if (reached[cellOf[v]]) keep.push(v);
    if (keep.length === pos.length / 3 - first) return;
    for (let v = first; v < pos.length / 3; v++) b.cells[cellOf[v]] = -1;
    const lists = [pos, mat, bld, cellOf, col];
    const copies = lists.map((a) => a.slice());
    for (const a of lists) a.length = first * (a === pos ? 3 : 1);
    for (const v of keep) {
      b.cells[copies[3][v]] = pos.length / 3;
      pos.push(copies[0][3 * v], copies[0][3 * v + 1], copies[0][3 * v + 2]);
      mat.push(copies[1][v]);
      bld.push(copies[2][v]);
      cellOf.push(copies[3][v]);
      col.push(copies[4][v]);
    }
  };
  const box = (put: Put, x0: number, y0: number, z0: number, x1: number, y1: number, z1: number, m: Mat, colour?: number) => {
    for (let z = z0; z < z1; z++) for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) put(x, y, z, m, colour);
  };
  const finish = (grid: { pitch: number; street: number; blocks: number }): City => {
    const count = pos.length / 3;
    const exposed = new Uint8Array(count);
    for (let v = 0; v < count; v++) {
      const b = buildings[bld[v]];
      const c = cellOf[v];
      const [x, y, z] = [c % b.w, Math.floor(c / b.w) % b.d, Math.floor(c / (b.w * b.d))];
      const open = (ok: boolean, m: number) => !ok || b.cells[m] < 0;
      const layer = b.w * b.d;
      exposed[v] =
        (open(x > 0, c - 1) ? 1 : 0) |
        (open(x + 1 < b.w, c + 1) ? 2 : 0) |
        (open(y > 0, c - b.w) ? 4 : 0) |
        (open(y + 1 < b.d, c + b.w) ? 8 : 0) |
        (z > 0 && open(true, c - layer) ? 16 : 0) |
        (open(z + 1 < b.h, c + layer) ? 32 : 0);
    }
    // Panes: each window's glass, connected face to face, facing away from its object's middle
    const pane = new Int32Array(count).fill(-1);
    const panes: Pane[] = [];
    for (let v = 0; v < count; v++) {
      if (!isGlass(mat[v]) || pane[v] >= 0) continue;
      const b = buildings[bld[v]];
      const id = panes.length;
      const voxels = [v];
      pane[v] = id;
      for (let n = 0; n < voxels.length; n++) {
        const c = cellOf[voxels[n]];
        const [x, y] = [c % b.w, Math.floor(c / b.w) % b.d];
        const layer = b.w * b.d;
        for (const [ok, m] of [
          [x > 0, c - 1],
          [x + 1 < b.w, c + 1],
          [y > 0, c - b.w],
          [y + 1 < b.d, c + b.w],
          [c >= layer, c - layer],
          [c + layer < b.cells.length, c + layer],
        ] as [boolean, number][]) {
          const u = ok ? b.cells[m] : -1;
          if (u >= 0 && pane[u] < 0 && isGlass(mat[u])) {
            pane[u] = id;
            voxels.push(u);
          }
        }
      }
      const centre: [number, number, number] = [0, 0, 0];
      for (const u of voxels) for (let a = 0; a < 3; a++) centre[a] += pos[3 * u + a] / voxels.length;
      const mid = [(b.x0 + b.w / 2) * VOXEL, (b.y0 + b.d / 2) * VOXEL];
      const out = [centre[0] - mid[0], centre[1] - mid[1]];
      const len = Math.hypot(out[0], out[1]) || 1;
      panes.push({ voxels, centre, normal: [out[0] / len, out[1] / len, 0] });
    }
    return {
      count,
      position: Float32Array.from(pos),
      material: Uint8Array.from(mat),
      building: Uint16Array.from(bld),
      cell: Int32Array.from(cellOf),
      color: Uint32Array.from(col),
      state: new Uint8Array(count),
      exposed,
      pane,
      panes,
      buildings,
      ...grid,
    };
  };
  return { rnd, place, box, last: () => buildings[buildings.length - 1], finish };
}

export function buildCity(seed = 7): City {
  const w = worldBuilder(seed);
  const { rnd, place, box } = w;
  // The tower: 13 x 11 m, a 4.5 m shop floor and ten 3 m storeys, a rounded front-left corner.
  // Its walls stand in from the edges of its grid by margins that leave room for the canopy
  // and signs out front and left, and the AC units on the other sides.
  const W = 13 * M;
  const D = 11 * M;
  const [ML, MF, MR, MB] = [7, 7, 3, 3];
  const GROUND = FLOORS.ground;
  const STOREY = FLOORS.storey;
  const UPPER = 10;
  const TOP = GROUND + UPPER * STOREY;
  const R = 14;
  const T = 2;
  const tw = ML + W + MR;
  const td = MF + D + MB;
  const RENDER = 0xdcd6c4;
  place(-Math.round(tw / 2), -Math.round(td / 2), tw, td, TOP + 30, (put) => {
    // How far inside the footprint a cell is (negative outside), the rounded corner included
    const depth = (x: number, y: number): number => {
      const [fx, fy] = [x - ML + 0.5, y - MF + 0.5];
      if (fx < R && fy < R) return R - Math.hypot(fx - R, fy - R);
      return Math.min(fx, fy, W - fx, D - fy);
    };
    // Where along the facade (voxels), for the window rhythm: straight sides by x or y, the
    // corner by its arc
    const along = (x: number, y: number): number => {
      const [fx, fy] = [x - ML + 0.5, y - MF + 0.5];
      if (fx < R && fy < R) return R * Math.atan2(R - fy, R - fx);
      return fy < 1.5 || fy > D - 1.5 ? fx : fy;
    };
    const storeyOf = (z: number) => (z < GROUND ? -1 : Math.floor((z - GROUND) / STOREY));
    const lit = new Map<number, boolean>();
    for (let z = 0; z < TOP; z++) {
      const s = storeyOf(z);
      const inStorey = s < 0 ? z : (z - GROUND) % STOREY;
      const slab = z === 0 || (s >= 0 && inStorey === 0);
      for (let y = 0; y < td; y++) {
        for (let x = 0; x < tw; x++) {
          const d = depth(x, y);
          const front = y - MF < 2;
          const left = x - ML < 2;
          if (d < 0) {
            // Outside the walls: floor bands, the canopy, nothing else
            if (d >= -1 && s >= 0 && inStorey <= 1) put(x, y, z, Mat.Trim);
            else if ((front || left) && (z === GROUND - 2 || z === GROUND - 1) && d >= -6) put(x, y, z, z === GROUND - 1 && d < -5 ? Mat.Neon : Mat.Concrete, z === GROUND - 1 && d < -5 ? 0xf0c030 : undefined);
            continue;
          }
          const a = along(x, y);
          const outer = d < 1;
          if (d >= T) {
            // Inside: slabs, columns on a 4 m grid, the stair core at the back right
            if (slab) put(x, y, z, Mat.Concrete);
            const cx = (x - ML) % 16;
            const cy = (y - MF) % 16;
            if (d >= 4 && cx < 2 && cy < 2 && x - ML > 4 && y - MF > 4) put(x, y, z, Mat.Concrete);
            const [kx, ky] = [x - ML - (W - 18), y - MF - (D - 18)];
            const core = kx >= 0 && kx < 14 && ky >= 0 && ky < 16;
            const coreWall = core && (kx === 0 || kx === 13 || ky === 0);
            const door = coreWall && ky === 0 && kx > 4 && kx < 9 && inStorey > 0 && inStorey < 9;
            if (coreWall && !door) put(x, y, z, Mat.Concrete);
            // Offices on the upper storeys: a desk with a chair and a screen between desks in
            // each 4 m bay, a strip light on the ceiling (seen through the windows)
            if (s >= 0 && !core && d >= 4) {
              const [bx, by] = [(x - ML) % 16, (y - MF) % 16];
              const deskTop = bx >= 6 && bx < 12 && by >= 7 && by < 10 && inStorey === 3;
              const deskLeg = (bx === 6 || bx === 11) && (by === 7 || by === 9) && inStorey >= 1 && inStorey < 3;
              const seat = bx >= 8 && bx < 10 && by >= 11 && by < 13 && inStorey === 2;
              const post = bx === 8 && by === 11 && inStorey === 1;
              const back = bx >= 8 && bx < 10 && by === 13 && inStorey >= 3 && inStorey < 5;
              const screen = bx >= 5 && bx < 13 && by === 5 && inStorey >= 1 && inStorey < 6;
              const monitor = bx >= 8 && bx < 10 && by === 7 && inStorey >= 4 && inStorey < 6;
              const light = bx >= 4 && bx < 12 && by === 3 && inStorey === 11;
              if (deskTop) put(x, y, z, Mat.Wood, 0xc8ab82);
              else if (deskLeg || post) put(x, y, z, Mat.Steel, 0x2b2e33);
              else if (seat || back) put(x, y, z, Mat.Steel, 0x1d2024);
              else if (screen) put(x, y, z, Mat.Wood, 0x7d8a92);
              else if (monitor) put(x, y, z, Mat.Steel, 0x111316);
              else if (light) put(x, y, z, Mat.Neon, 0xfff1d6);
            }
            continue;
          }
          // The walls
          if (s < 0) {
            // Shop floor: piers every 3 m, tall glass between, a sign band above; the back
            // and right are solid, with a roller shutter
            const street = front || left;
            const pier = street ? a % 12 < 2 || a > (front ? W : D) - 2 : false;
            if (!street) {
              const shutter = y - MF > D - 3 && x - ML > 10 && x - ML < 24 && z < 12;
              put(x, y, z, shutter ? Mat.Steel : Mat.Concrete, shutter ? (z % 2 ? 0x8a8e92 : 0x6f7377) : undefined);
            } else if (pier || z === 0) put(x, y, z, Mat.Concrete);
            else if (z >= 14) put(x, y, z, outer ? Mat.Neon : Mat.Concrete, outer ? (Math.floor(a / 3) % 2 ? 0xc42a22 : 0xe8e0d0) : undefined);
            else if (!outer && !(Math.abs(a - 26) < 4 && z < 11)) put(x, y, z, Mat.Glass);
            continue;
          }
          // Upper storeys: render spandrels, ribbon windows between slim mullions
          const inWindow = inStorey >= 4 && inStorey < 10 && a % 8 >= 1 && a > 1 && a < (front ? W : D) + R - 1;
          if (inWindow) {
            if (!outer) {
              const key = s * 1000 + Math.floor(a / 8) + (front ? 0 : left ? 300 : 600);
              if (!lit.has(key)) lit.set(key, rnd() < 0.2);
              put(x, y, z, lit.get(key) ? Mat.LitGlass : Mat.Glass);
            }
            continue;
          }
          const mullion = inStorey >= 4 && inStorey < 10;
          put(x, y, z, mullion ? Mat.Trim : outer ? Mat.Plaster : Mat.Plaster, outer && !mullion ? RENDER : undefined);
        }
      }
    }
    // The roof: slab, parapet, the stair core's head, a water tank on legs, a mast
    for (let y = 0; y < td; y++) {
      for (let x = 0; x < tw; x++) {
        const d = depth(x, y);
        if (d < 0) continue;
        put(x, y, TOP, Mat.Roof);
        if (d < T) for (let z = TOP + 1; z <= TOP + 3; z++) put(x, y, z, z === TOP + 3 ? Mat.Trim : Mat.Plaster, z === TOP + 3 ? undefined : RENDER);
      }
    }
    box(put, ML + W - 18, MF + D - 18, TOP + 1, ML + W - 4, MF + D - 2, TOP + 12, Mat.Concrete);
    const [tx, ty] = [ML + 12, MF + D - 10];
    for (const [lx, ly] of [
      [-4, -4],
      [4, -4],
      [-4, 4],
      [4, 4],
    ])
      box(put, tx + lx, ty + ly, TOP + 1, tx + lx + 1, ty + ly + 1, TOP + 8, Mat.Steel, 0x3c4146);
    for (let z = TOP + 8; z < TOP + 17; z++) {
      for (let y = ty - 6; y <= ty + 6; y++) {
        for (let x = tx - 6; x <= tx + 6; x++) {
          const r = Math.hypot(x - tx + 0.5, y - ty + 0.5);
          if (r < 6 && (r > 4.8 || z === TOP + 8 || z === TOP + 16)) put(x, y, z, Mat.Steel, z % 3 ? 0x8a969a : 0x77838a);
        }
      }
    }
    box(put, ML + 3, MF + 3, TOP + 1, ML + 4, MF + 4, TOP + 28, Mat.Steel, 0x2d3034);
    // The blade sign up the front, standing out from the wall
    for (let z = GROUND + 6; z < GROUND + 66; z++) {
      for (let y = MF - 6; y < MF; y++) {
        const x = ML + W - 12;
        const edge = y === MF - 6 || z === GROUND + 6 || z === GROUND + 65;
        const glyph = !edge && (Math.imul(Math.floor(z / 3) * 7 + (y - MF) * 13, 2654435761) >>> 28) < 7 && z % 12 > 1;
        put(x, y, z, Mat.Neon, edge ? 0xe8e0d0 : glyph ? 0xf4ead8 : 0xb4201c);
        if (y === MF - 1) put(x, y, z, Mat.Neon, 0xb4201c);
      }
    }
    // AC units under some windows, on brackets (all four sides)
    for (let s = 0; s < UPPER; s++) {
      const z0 = GROUND + s * STOREY + 1;
      for (let k = 0; k < 14; k++) {
        if (rnd() > 0.35) continue;
        const side = Math.floor(rnd() * 4);
        const along = 4 + Math.floor(rnd() * ((side < 2 ? W : D) - 12));
        const colour = rnd() < 0.5 ? 0xd4d6d2 : 0xbfc3c0;
        for (let u = 0; u < 4; u++) {
          for (let o = 1; o <= 2; o++) {
            for (let h = 0; h < 3; h++) {
              const [x, y] =
                side === 0 ? [ML + along + u, MF - o] : side === 1 ? [ML + along + u, MF + D - 1 + o] : side === 2 ? [ML - o, MF + along + u] : [ML + W - 1 + o, MF + along + u];
              if (depth(x, y) < 0) put(x, y, z0 + h, Mat.Steel, h === 1 && u > 0 && u < 3 ? 0x5a5e62 : colour);
            }
          }
        }
      }
    }
  });

  // Outside: cars at the kerb of the street in front, a tree on the pavement
  const kerb = -BLOCK / 2 - 3 - 0.9;
  car(w, -8, kerb, 0xa51d1d);
  car(w, 1.5, kerb, 0x1d3f8a);
  car(w, 10, kerb, 0xd8d4cc);
  // Trees round the block's pavement (clear of the view in from the corner)
  for (const [x, y, k] of [
    [-13.3, -13.3, 1],
    [12.8, -13.3, 2],
    [-13.3, 4.5, 3],
    [-13.3, 12, 4],
    [13.3, 1.5, 5],
    [13.3, 10.5, 6],
    [-2, 13.3, 7],
    [7, 13.3, 8],
  ])
    tree(w, x, y, k);
  for (const lx of [-12, 0, 12]) lamp(w, lx, kerb + 2.4);

  bench(w, -9.5, -13.6, true);
  bin(w, -11.8, -13.4);
  bench(w, -13.6, 8.5, false);
  bin(w, -13.6, 6.3);
  shelter(w, 5.5, -16.6);
  bin(w, 9.2, -15.6);
  hydrant(w, 14.2, -16.4);
  for (const bx of [-3, -1.5, 1.5, 3, 12, 13.5]) bollard(w, bx, -12);
  signal(w, -15.5, -15.5, 0, -1);
  signal(w, 15.5, -15.5, 1, 0);
  signal(w, 15.5, 15.5, 0, 1);
  signal(w, -15.5, 15.5, -1, 0);
  sign(w, -16.8, -14.2);

  return w.finish({ pitch: BLOCK + STREET, street: STREET, blocks: 1 });
}

/** A parked car along x: painted body, glass cabin on pillars, black tyres. */
export function car(w: WorldBuilder, cx: number, cy: number, paint: number): void {
  const { place, box } = w;
  const [L, Wc] = [18, 7];
  place(Math.round(cx * M - L / 2), Math.round(cy * M - Wc / 2), L, Wc, 6, (put) => {
    box(put, 1, 0, 1, L - 1, Wc, 3, Mat.Steel, paint);
    for (const wx of [3, L - 5]) {
      box(put, wx, 0, 0, wx + 2, 1, 2, Mat.Steel, 0x151515);
      box(put, wx, Wc - 1, 0, wx + 2, Wc, 2, Mat.Steel, 0x151515);
    }
    // Pillars at the cabin's corners hold the roof (glass carries nothing), glass round them
    for (const px of [6, 11]) for (const py of [1, Wc - 2]) box(put, px, py, 3, px + 1, py + 1, 5, Mat.Steel, paint);
    box(put, 5, 0, 3, 13, Wc, 5, Mat.Glass);
    box(put, 6, 1, 5, 12, Wc - 1, 6, Mat.Steel, paint);
    box(put, 0, 1, 1, 1, Wc - 1, 2, Mat.Trim, 0xd0d0c8);
    box(put, L - 1, 1, 1, L, Wc - 1, 2, Mat.Steel, 0x8a1c14);
  });
  w.last().kind = 'car';
}
/**
 * A street tree: a slim trunk forking into a few branches, each ending in a clump of leaves,
 * the clumps lumpy and overlapping into a loose crown, darker underneath, catching the light
 * on top. Every leaf is joined to a branch (no leaf left hanging in the air).
 */
export function tree(w: WorldBuilder, cx: number, cy: number, seed: number): void {
  const { place, box } = w;
  const R = 11;
  const H = 34;
  const r = random(seed);
  place(Math.round(cx * M - R), Math.round(cy * M - R), 2 * R, 2 * R, H, (put) => {
    const bark = [0x3e2c1f, 0x4a3524, 0x55402c];
    const fork = 13 + Math.floor(r() * 3);
    box(put, R - 1, R - 1, 0, R + 1, R + 1, 7, Mat.Wood, bark[1]);
    box(put, R - 1, R - 1, 7, R, R, fork + 1, Mat.Wood, bark[0]);
    // Clumps: one crowning the trunk, four or five round it
    const clumps: [number, number, number, number][] = [[R - 0.5 + (r() - 0.5) * 2, R - 0.5 + (r() - 0.5) * 2, fork + 11, 4.6]];
    const n = 4 + Math.floor(r() * 2);
    for (let k = 0; k < n; k++) {
      const a = (k / n) * Math.PI * 2 + r() * 0.8;
      const out = 4.5 + r() * 2;
      clumps.push([R - 0.5 + Math.cos(a) * out, R - 0.5 + Math.sin(a) * out, fork + 5 + r() * 5, 3.4 + r() * 1.3]);
    }
    // Branches: from the fork out to each clump's middle, face to face (a diagonal step
    // joins only at an edge, and what isn't joined by faces to the ground is left out)
    for (const [x, y, z] of clumps) {
      const [x0, y0, z0] = [R - 0.5, R - 0.5, fork];
      const steps = Math.ceil(Math.hypot(x - x0, y - y0, z - z0) * 1.5);
      let at = [Math.floor(x0), Math.floor(y0), Math.floor(z0)];
      for (let k = 0; k <= steps; k++) {
        const f = k / steps;
        const to = [Math.floor(x0 + (x - x0) * f), Math.floor(y0 + (y - y0) * f), Math.floor(z0 + (z - z0) * f)];
        for (const a of [2, 0, 1]) {
          while (at[a] !== to[a]) {
            at = [...at];
            at[a] += Math.sign(to[a] - at[a]);
            put(at[0], at[1], at[2], Mat.Wood, bark[2]);
          }
        }
      }
    }
    // Leaves: inside a clump (a little squashed), its edge lumpy; shade by height in it
    const greens = [0x2c5220, 0x365f24, 0x3f6b2a, 0x4d7a30, 0x5f8d36, 0x729c3e];
    for (let z = fork; z < H; z++) {
      for (let y = 0; y < 2 * R; y++) {
        for (let x = 0; x < 2 * R; x++) {
          let best = -Infinity;
          let height = 0;
          for (const [px, py, pz, pr] of clumps) {
            const d = pr - Math.hypot(x + 0.5 - px - 0.5, y + 0.5 - py - 0.5, (z + 0.5 - pz) * 1.25);
            if (d > best) [best, height] = [d, (z - pz) / pr];
          }
          const lump = ((Math.imul(x * 73 + y * 151 + z * 283 + seed * 17, 2654435761) >>> 0) / 2 ** 32) * 1.4;
          if (best <= (best < 1.2 ? lump : 0)) continue;
          const shade = Math.max(0, Math.min(greens.length - 1, Math.floor((height + 1) * 2.4 + r() * 1.6)));
          put(x, y, z, Mat.Leaf, greens[shade]);
        }
      }
    }
  });
}


/** A street lamp: a steel pole, an arm over the road, a warm lamp. */
export function lamp(w: WorldBuilder, cx: number, cy: number): void {
  const { place, box } = w;
  place(Math.round(cx * M) - 1, Math.round(cy * M) - 8, 3, 9, 26, (put) => {
    box(put, 1, 7, 0, 2, 8, 25, Mat.Steel, 0x2c3034);
    box(put, 1, 1, 24, 2, 7, 25, Mat.Steel, 0x2c3034);
    box(put, 0, 0, 23, 3, 3, 24, Mat.LitGlass, 0xffd9a0);
  });
}

/** A bench facing -y (or -x): wooden slats on steel frames, a backrest. */
export function bench(w: WorldBuilder, cx: number, cy: number, alongX: boolean): void {
  const { place } = w;
  const [L, D] = alongX ? [8, 3] : [3, 8];
  place(Math.round(cx * M - L / 2), Math.round(cy * M - D / 2), L, D, 4, (put) => {
    const at = (a: number, b: number, z: number, m: Mat, c: number) => (alongX ? put(a, b, z, m, c) : put(b, a, z, m, c));
    for (let a = 0; a < 8; a++) {
      for (const leg of [0, 7]) if (a === leg) for (let b = 0; b < 3; b++) at(a, b, 0, Mat.Steel, 0x24272b);
      at(a, 0, 1, Mat.Wood, 0x8a5a32);
      at(a, 1, 1, Mat.Wood, 0x7a4f2c);
      at(a, 2, 2, Mat.Wood, 0x8a5a32);
      at(a, 2, 3, Mat.Wood, 0x7a4f2c);
      if (a === 0 || a === 7) at(a, 2, 1, Mat.Steel, 0x24272b);
    }
  });
}
/** A litter bin: a green steel drum with a dark lid. */
export function bin(w: WorldBuilder, cx: number, cy: number): void {
  const { place, box } = w;
  place(Math.round(cx * M) - 1, Math.round(cy * M) - 1, 2, 2, 5, (put) => {
    box(put, 0, 0, 0, 2, 2, 4, Mat.Steel, 0x2f5a3a);
    box(put, 0, 0, 4, 2, 2, 5, Mat.Steel, 0x1c1f22);
  });
}
/** A fire hydrant. */
export function hydrant(w: WorldBuilder, cx: number, cy: number): void {
  const { place, box } = w;
  place(Math.round(cx * M), Math.round(cy * M), 1, 1, 4, (put) => {
    box(put, 0, 0, 0, 1, 1, 3, Mat.Steel, 0xb3261e);
    put(0, 0, 3, Mat.Steel, 0xd9d2c4);
  });
}
/** A short steel post keeping cars off the pavement. */
export function bollard(w: WorldBuilder, cx: number, cy: number): void {
  const { place, box } = w;
  place(Math.round(cx * M), Math.round(cy * M), 1, 1, 4, (put) => {
    box(put, 0, 0, 0, 1, 1, 3, Mat.Steel, 0x2a2d31);
    put(0, 0, 3, Mat.Neon, 0xe8e2d4);
  });
}
/**
 * Traffic lights on the corner: a pole, an arm out over the road (dx, dy), a signal head at
 * its end (red, amber, green; green lit), a crossing signal on the pole.
 */
export function signal(w: WorldBuilder, cx: number, cy: number, dx: number, dy: number): void {
  const { place, box } = w;
  const arm = 10;
  const [x0, y0] = [Math.round(cx * M) - (dx < 0 ? arm : 0), Math.round(cy * M) - (dy < 0 ? arm : 0)];
  place(x0, y0, dx ? arm + 1 : 1, dy ? arm + 1 : 1, 20, (put) => {
    const [px, py] = [dx < 0 ? arm : 0, dy < 0 ? arm : 0];
    const [ex, ey] = [px + dx * arm, py + dy * arm];
    box(put, px, py, 0, px + 1, py + 1, 19, Mat.Steel, 0x2a2e33);
    for (let k = 1; k <= arm; k++) put(px + dx * k, py + dy * k, 18, Mat.Steel, 0x2a2e33);
    box(put, ex, ey, 13, ex + 1, ey + 1, 18, Mat.Steel, 0x121416);
    put(ex, ey, 16, Mat.Neon, 0x3a0c08);
    put(ex, ey, 15, Mat.Neon, 0x3a2a06);
    put(ex, ey, 14, Mat.Neon, 0x33ff66);
    put(px, py, 9, Mat.Neon, 0xff7a1a);
  });
}
/** A street name sign: a pole with two green plates. */
export function sign(w: WorldBuilder, cx: number, cy: number): void {
  const { place, box } = w;
  place(Math.round(cx * M) - 2, Math.round(cy * M) - 2, 5, 5, 13, (put) => {
    box(put, 2, 2, 0, 3, 3, 12, Mat.Steel, 0x3a3f45);
    box(put, 0, 2, 11, 5, 3, 12, Mat.Steel, 0x1f6b3c);
    box(put, 2, 0, 12, 3, 5, 13, Mat.Steel, 0x1f6b3c);
  });
}
/**
 * A bus shelter facing the road (-y): steel posts, a roof, a glass back and side that
 * shatter like any window, a bench, a lit advertising panel; its stop sign by the kerb.
 */
export function shelter(w: WorldBuilder, cx: number, cy: number): void {
  const { place, box } = w;
  const [L, D, H] = [16, 6, 11];
  place(Math.round(cx * M - L / 2), Math.round(cy * M - D / 2), L + 3, D, H + 5, (put) => {
    const steel = 0x3b4046;
    for (const [x, y] of [
      [0, 0],
      [L - 1, 0],
      [0, D - 1],
      [L - 1, D - 1],
    ])
      box(put, x, y, 0, x + 1, y + 1, H, Mat.Steel, steel);
    box(put, 0, 0, H, L, D, H + 1, Mat.Steel, 0x2c3035);
    box(put, 1, D - 1, 1, L - 5, D, H, Mat.Glass);
    box(put, L - 5, D - 1, 1, L - 1, D, H, Mat.LitGlass, 0xf3e6c8);
    box(put, 0, 1, 1, 1, D - 1, H, Mat.Glass);
    box(put, 2, D - 3, 2, L - 2, D - 1, 3, Mat.Wood, 0x7a4f2c);
    for (const x of [2, L - 3]) box(put, x, D - 2, 0, x + 1, D - 1, 2, Mat.Steel, steel);
    // The stop sign: a pole by the kerb with a red disc
    box(put, L + 1, 0, 0, L + 2, 1, 12, Mat.Steel, 0x3a3f45);
    box(put, L, 0, 12, L + 3, 1, 15, Mat.Neon, 0xc8201a);
  });
}

/** No voxels at all: the world before the city is built (the player and chips start with it). */
export function emptyCity(): City {
  return {
    count: 0,
    position: new Float32Array(0),
    material: new Uint8Array(0),
    building: new Uint16Array(0),
    cell: new Int32Array(0),
    color: new Uint32Array(0),
    state: new Uint8Array(0),
    exposed: new Uint8Array(0),
    pane: new Int32Array(0),
    panes: [],
    buildings: [],
    pitch: 46,
    street: 16,
    blocks: 1,
  };
}

/** The fixed voxel at world point (x, y, z) (m), or -1. */
export function voxelAt(city: City, x: number, y: number, z: number): number {
  if (z < 0) return -1;
  const [xv, yv, zi] = [Math.floor(x / VOXEL), Math.floor(y / VOXEL), Math.floor(z / VOXEL)];
  for (const b of city.buildings) {
    const xi = xv - b.x0;
    const yi = yv - b.y0;
    if (xi < 0 || yi < 0 || xi >= b.w || yi >= b.d || zi >= b.h) continue;
    const v = b.cells[xi + b.w * (yi + b.d * zi)];
    return v >= 0 && city.state[v] === State.Fixed ? v : -1;
  }
  return -1;
}

/** Objects whose box (grown by `pad`, m) holds any of the box lo..hi (m). */
export function buildingsNear(city: City, lo: ArrayLike<number>, hi: ArrayLike<number>, pad = 0): number[] {
  const out: number[] = [];
  city.buildings.forEach((b, i) => {
    const [x0, y0, x1, y1, z1] = [b.x0 * VOXEL, b.y0 * VOXEL, (b.x0 + b.w) * VOXEL, (b.y0 + b.d) * VOXEL, b.h * VOXEL];
    if (hi[0] < x0 - pad || lo[0] > x1 + pad || hi[1] < y0 - pad || lo[1] > y1 + pad || lo[2] > z1 + pad) return;
    out.push(i);
  });
  return out;
}

/**
 * First fixed voxel along a ray from `o` (m) along unit `d`, within `far` m (Amanatides–Woo
 * through each object's grid, in voxel units), or the ground (voxel -1); null for sky.
 */
export function raycast(city: City, o: ArrayLike<number>, d: ArrayLike<number>, far: number): { t: number; voxel: number; normal: [number, number, number] } | null {
  let best: { t: number; voxel: number; normal: [number, number, number] } | null = null;
  if (d[2] < 0) {
    const t = -o[2] / d[2];
    if (t >= 0 && t <= far) best = { t, voxel: -1, normal: [0, 0, 1] };
  }
  const ov = [o[0] / VOXEL, o[1] / VOXEL, o[2] / VOXEL];
  for (const b of city.buildings) {
    const lo = [b.x0, b.y0, 0];
    const dims = [b.w, b.d, b.h];
    const hi = [b.x0 + b.w, b.y0 + b.d, b.h];
    let [t0, t1] = [0, (best ? best.t : far) / VOXEL];
    let axis = -1;
    for (let k = 0; k < 3; k++) {
      if (Math.abs(d[k]) < 1e-9) {
        if (ov[k] < lo[k] || ov[k] > hi[k]) t0 = Infinity;
        continue;
      }
      let [a, c] = [(lo[k] - ov[k]) / d[k], (hi[k] - ov[k]) / d[k]];
      if (a > c) [a, c] = [c, a];
      if (a > t0) [t0, axis] = [a, k];
      t1 = Math.min(t1, c);
    }
    if (t0 > t1) continue;
    const p = [0, 1, 2].map((k) => ov[k] + d[k] * (t0 + 1e-6));
    const cell = [Math.floor(p[0]) - b.x0, Math.floor(p[1]) - b.y0, Math.floor(p[2])].map((v, k) => Math.min(Math.max(v, 0), dims[k] - 1));
    const step = [Math.sign(d[0]), Math.sign(d[1]), Math.sign(d[2])];
    const next = [0, 1, 2].map((k) => (step[k] === 0 ? Infinity : (cell[k] + (step[k] > 0 ? 1 : 0) + lo[k] - ov[k]) / d[k]));
    const delta = [0, 1, 2].map((k) => (step[k] === 0 ? Infinity : Math.abs(1 / d[k])));
    let t = t0;
    let face = axis;
    for (;;) {
      const v = b.cells[cell[0] + b.w * (cell[1] + b.d * cell[2])];
      if (v >= 0 && city.state[v] === State.Fixed) {
        const normal: [number, number, number] = [0, 0, 0];
        if (face >= 0) normal[face] = -step[face];
        if (!best || t * VOXEL < best.t) best = { t: t * VOXEL, voxel: v, normal };
        break;
      }
      const k = next[0] < next[1] ? (next[0] < next[2] ? 0 : 2) : next[1] < next[2] ? 1 : 2;
      t = next[k];
      if (t > t1) break;
      cell[k] += step[k];
      if (cell[k] < 0 || cell[k] >= dims[k]) break;
      next[k] += delta[k];
      face = k;
    }
  }
  return best;
}
