// The space station: after the ISS, in voxels, in orbit. A lattice truss 64 m long across the
// middle, four pairs of long golden-brown solar array wings on it, white radiators, and a line
// of pressurised modules under it with a crew capsule docked at one end. Every part is its own
// object, held where it joins the part inward of it (world.ts anchors): cut a wing's mast, or a
// module's neck, and what's beyond floats off. Nothing weighs anything (weightless).
//
// The station sits 20 m up in the world's grid (whose objects all start at z = 0); there is no
// ground under it, only the Earth, far below (space.ts draws it).

import { type City, Mat, VOXEL, type WorldBuilder, worldBuilder } from './world.ts';

/** The station's middle (m): where the truss crosses the line of modules. */
export const HUB: [number, number, number] = [0, 0, 20];
/** The modules' axis height and the truss's middle height (m). */
const AXIS_Z = 16.8;
const TRUSS_Z = 20.5;
/** The truss's half length and half width (m). */
const TRUSS_HALF = 32;
const TRUSS_W = 1;

type Paint = [Mat, number] | null;

/**
 * An object filling the box lo..hi (m), each cell's contents by `what` at its middle, held
 * in place where `holds` says (m, the cell's middle).
 */
function solid(w: WorldBuilder, lo: number[], hi: number[], what: (x: number, y: number, z: number) => Paint, holds: (x: number, y: number, z: number) => boolean): void {
  const [x0, y0] = [Math.floor(lo[0] / VOXEL), Math.floor(lo[1] / VOXEL)];
  const [x1, y1, z1] = [Math.ceil(hi[0] / VOXEL), Math.ceil(hi[1] / VOXEL), Math.ceil(hi[2] / VOXEL)];
  const z0 = Math.max(0, Math.floor(lo[2] / VOXEL));
  const centre = (i: number, j: number, k: number) => [(x0 + i + 0.5) * VOXEL, (y0 + j + 0.5) * VOXEL, (k + 0.5) * VOXEL];
  w.place(
    x0,
    y0,
    x1 - x0,
    y1 - y0,
    z1,
    (put) => {
      for (let k = z0; k < z1; k++)
        for (let j = 0; j < y1 - y0; j++)
          for (let i = 0; i < x1 - x0; i++) {
            const [x, y, z] = centre(i, j, k);
            const p = what(x, y, z);
            if (p) put(i, j, k, p[0], p[1]);
          }
    },
    (i, j, k) => {
      const [x, y, z] = centre(i, j, k);
      return holds(x, y, z);
    },
  );
}

/** Deterministic noise per cell (0..1). */
const grain = (x: number, y: number, z: number) => (Math.imul(Math.floor(x * 4) * 73 + Math.floor(y * 4) * 151 + Math.floor(z * 4) * 283, 2654435761) >>> 0) / 2 ** 32;

/** A white module's skin: blankets with seams every metre and a darker ring at each end. */
function skin(x: number, x0: number, x1: number, tint = 0xe6e4de): number {
  if (x - x0 < 0.5 || x1 - x < 0.5) return 0xb9b7b0;
  return Math.abs(((x - x0) % 1.5) - 0.75) > 0.62 ? 0xcbc9c2 : tint;
}

/**
 * A pressurised module along x from x0 to x1: a cylinder shell of `radius` (m) round the
 * module axis, closed at the ends except for a hatch ring, windows in a band on its side,
 * handrails. Held at its end nearer the hub.
 */
function module(w: WorldBuilder, x0: number, x1: number, radius: number, tint: number, options: { windows?: boolean; foil?: boolean } = {}): void {
  const inner = x0 > 0 ? x0 : x1;
  solid(
    w,
    [x0, -radius - 0.3, AXIS_Z - radius - 0.3],
    [x1, radius + 0.3, AXIS_Z + radius + 0.3],
    (x, y, z) => {
      const r = Math.hypot(y, z - AXIS_Z);
      const end = x - x0 < VOXEL || x1 - x < VOXEL;
      // The ends: a disc with the hatch's collar standing proud of it, a docking ring
      if (end) return r < radius ? [Mat.Steel, r < 0.9 ? 0x9a9a96 : 0xbdbbb4] : null;
      if (r > radius || r < radius - VOXEL * 1.1) return null;
      const around = Math.atan2(z - AXIS_Z, y);
      // Windows: a band of dark ports on the Earth side
      if (options.windows && Math.abs(around + Math.PI / 2) < 0.35 && Math.abs(((x - x0) % 2.5) - 1.25) < 0.4) return [Mat.Glass, 0x1a2a38];
      // Gold foil round the ends (the Russian segment's)
      if (options.foil && (x - x0 < 1.2 || x1 - x < 1.2)) return [Mat.Steel, grain(x, y, z) < 0.5 ? 0xc79a3a : 0xb08430];
      // Handrails: short yellow bars along the top
      if (Math.abs(around - Math.PI / 2) < 0.08 && Math.abs(((x - x0) % 1.5) - 0.2) < 0.13) return [Mat.Steel, 0xd8b030];
      return [Mat.Plaster, skin(x, x0, x1, tint)];
    },
    (x) => Math.abs(x - inner) < 0.3,
  );
}

/** The core: the central node, the truss's middle 16 m over it, and the pylon joining them. Held throughout. */
function core(w: WorldBuilder): void {
  const R = 2.3;
  solid(
    w,
    [-4.2, -8, AXIS_Z - R - 0.3],
    [4.2, 8, TRUSS_Z + TRUSS_W + 0.3],
    (x, y, z) => {
      const truss = trussCell(x, y, z);
      if (truss) return truss;
      const r = Math.hypot(y, z - AXIS_Z);
      if (Math.abs(x) <= 4 && r <= R && (r > R - VOXEL * 1.1 || Math.abs(x) > 4 - VOXEL)) {
        if (Math.abs(x) > 4 - VOXEL) return [Mat.Steel, r < 1 ? 0x9a9a96 : 0xbdbbb4];
        // Ports on the node's sides (where modules and the airlock would join)
        if (Math.abs(y) > R - 0.4 && Math.abs(x) < 1 && Math.abs(z - AXIS_Z) < 1) return [Mat.Steel, 0x8f8f8a];
        return [Mat.Plaster, skin(x, -4, 4)];
      }
      // The pylon from the node's top up into the truss
      if (Math.abs(x) < 0.75 && Math.abs(y) < 1.25 && z > AXIS_Z + R - 0.3 && z < TRUSS_Z) return [Mat.Steel, 0xc9c9c4];
      return null;
    },
    () => true,
  );
}

/**
 * The truss at (x, y, z), if a cell of it's there: four longerons along its corners, a zigzag
 * of braces on each face every 2 m, bulkheads every 8 m (the lattice's joints), and at
 * |y| = 13 the rotary joint the outer wings turn on (a thick ring).
 */
function trussCell(x: number, y: number, z: number): Paint {
  const [ax, az] = [Math.abs(x), Math.abs(z - TRUSS_Z)];
  if (Math.abs(y) > TRUSS_HALF || ax > TRUSS_W + 0.6 || az > TRUSS_W + 0.6) return null;
  // The rotary joint: a drum wider than the truss
  if (Math.abs(Math.abs(y) - 13) < 0.5) return Math.hypot(x, z - TRUSS_Z) < TRUSS_W + 0.55 ? [Mat.Steel, 0xa9aaa6] : null;
  if (ax > TRUSS_W || az > TRUSS_W) return null;
  const edgeX = ax > TRUSS_W - VOXEL;
  const edgeZ = az > TRUSS_W - VOXEL;
  const grey = 0xd4d4cf;
  if (edgeX && edgeZ) return [Mat.Steel, grey];
  if (Math.abs(((Math.abs(y) + 4) % 8) - 4) < VOXEL / 2 && (edgeX || edgeZ)) return [Mat.Steel, 0xbcbcb6];
  // Braces: the zigzag across each face
  const t = ((Math.abs(y) % 2) / 2) * 2 * TRUSS_W - TRUSS_W;
  const zig = Math.floor(Math.abs(y) / 2) % 2 === 0 ? t : -t;
  if (edgeX && Math.abs(z - TRUSS_Z - zig) < VOXEL * 0.7) return [Mat.Steel, grey];
  if (edgeZ && Math.abs(x - zig) < VOXEL * 0.7) return [Mat.Steel, grey];
  return null;
}

/** An outer truss segment (y from y0 to y1, away from the hub): held at its inner end. */
function truss(w: WorldBuilder, y0: number, y1: number): void {
  const inner = Math.abs(y0) < Math.abs(y1) ? y0 : y1;
  solid(w, [-TRUSS_W - 0.6, Math.min(y0, y1), TRUSS_Z - TRUSS_W - 0.6], [TRUSS_W + 0.6, Math.max(y0, y1), TRUSS_Z + TRUSS_W + 0.6], trussCell, (_, y) => Math.abs(y - inner) < 0.5);
}

/**
 * A solar array wing at y on the truss, reaching `up` (+1) or down (-1): a white mast out from
 * a box on the truss, two blankets of cells either side of it, framed; held at its box.
 */
function wing(w: WorldBuilder, y: number, up: number): void {
  // The blankets start past a stretch of bare mast (they hang off the mast alone)
  const [length, half, start] = [15, 5, 2];
  const from = up > 0 ? TRUSS_Z + TRUSS_W : TRUSS_Z - TRUSS_W;
  const to = from + up * (length + start);
  const [zlo, zhi] = [Math.min(from, to), Math.max(from, to)];
  solid(
    w,
    [-half - 0.3, y - VOXEL, zlo],
    [half + 0.3, y + VOXEL, zhi],
    (x, _, z) => {
      const along = (z - from) * up;
      if (along < 0) return null;
      // The box on the truss and the mast out of it
      if (along < 1.5) return Math.abs(x) < 0.75 ? [Mat.Steel, 0xe6e4de] : null;
      if (Math.abs(x) < 0.2) return [Mat.Steel, 0xd8d8d2];
      if (Math.abs(x) > half || along < start) return null;
      // The blankets: framed; cells in panels, a dark line between each
      if (Math.abs(x) < 0.45 || Math.abs(x) > half - VOXEL || along > length + start - VOXEL || along < start + VOXEL) return [Mat.Steel, 0xcfcfc9];
      const panel = Math.floor((along - start) / 1.25);
      const seam = (along - start) % 1.25 < VOXEL;
      // (Wood: the matte, light material; steel would mirror the black of space)
      if (seam) return [Mat.Wood, 0x2a1810];
      const shade = (panel % 3) * 0.04 + grain(x, y, z) * 0.06;
      const c = (n: number) => Math.round(n * (0.9 + shade));
      return [Mat.Wood, (c(0xa0) << 16) | (c(0x4c) << 8) | c(0x24)];
    },
    (_, __, z) => (z - from) * up < 1.6,
  );
}

/** A white radiator panel off the truss at y, reaching back (-x): ribbed, held where it meets the truss. */
function radiator(w: WorldBuilder, y: number): void {
  const [length, half] = [9, 1.6];
  const z = TRUSS_Z - TRUSS_W - 0.5;
  solid(
    w,
    [-TRUSS_W - length, y - half, z - VOXEL],
    [-TRUSS_W + 0.3, y + half, z + VOXEL],
    (x, yy) => (Math.abs(((yy - y + half) % 0.8) - 0.4) < VOXEL / 2 ? [Mat.Wood, 0xb4b4ae] : x > -TRUSS_W - 0.6 ? [Mat.Steel, 0x9a9a95] : [Mat.Wood, 0xdcdbd6]),
    (x) => x > -TRUSS_W - 0.4,
  );
}

/** Small gold solar wings on the far module's sides (at x, spanning `span` m each way). */
function sideWings(w: WorldBuilder, x: number, span: number): void {
  for (const s of [1, -1]) {
    const from = s * 2.1;
    const to = s * (2.1 + span);
    solid(
      w,
      [x - 1.6, Math.min(from, to), AXIS_Z - VOXEL],
      [x + 1.6, Math.max(from, to), AXIS_Z + VOXEL],
      (xx, yy) => {
        const out = Math.abs(yy) - 2.1;
        if (out < 0.6) return Math.abs(xx - x) < 0.25 ? [Mat.Steel, 0xbdbbb4] : null;
        return Math.abs(xx - x) > 1.45 || out > span - VOXEL ? [Mat.Steel, 0xcfcfc9] : [Mat.Wood, grain(xx, yy, 0) < 0.5 ? 0x2b3a6a : 0x24325c];
      },
      (_, yy) => Math.abs(yy) - 2.1 < 0.3,
    );
  }
}

/** The crew capsule docked at the +x end: a cone on a short trunk, a black heat shield behind. */
function capsule(w: WorldBuilder, x0: number): void {
  const len = 4.2;
  solid(
    w,
    [x0, -2.2, AXIS_Z - 2.2],
    [x0 + len, 2.2, AXIS_Z + 2.2],
    (x, y, z) => {
      const along = x - x0;
      const r = Math.hypot(y, z - AXIS_Z);
      // The docking nose (its hatch towards the station), the cone widening, the trunk
      const outer = along < 0.5 ? 0.8 : along < 3 ? 0.8 + ((along - 0.5) / 2.5) * 1.2 : 2.0;
      if (r > outer || (r < outer - VOXEL * 1.2 && along > VOXEL && along < len - VOXEL)) return null;
      if (along > len - 0.5) return [Mat.Steel, 0x1c1c1c];
      if (along > 3) return [Mat.Steel, Math.abs(Math.atan2(z - AXIS_Z, y)) < 0.4 ? 0x1b2b50 : 0xe8e8e4];
      return [Mat.Plaster, along < 0.5 ? 0xbdbbb4 : 0xf2f2ee];
    },
    (x) => x - x0 < 0.3,
  );
}

/** The station, and where to look at it from. */
export function buildStation(seed = 21): City {
  const w = worldBuilder(seed);
  core(w);
  for (const s of [1, -1]) {
    truss(w, s * 8, s * TRUSS_HALF);
    for (const y of [18, 29]) for (const up of [1, -1]) wing(w, s * y, up);
    radiator(w, s * 9.5);
  }
  // The modules: a lab and a node towards +x with the capsule docked beyond, two older modules
  // towards -x (gold-foiled ends), the last with its own small wings
  module(w, 4.25, 12.5, 2.1, 0xe8e6e0, { windows: true });
  module(w, 12.75, 16.5, 2.0, 0xe2e0da);
  capsule(w, 16.75);
  module(w, -12.5, -4.25, 2.0, 0xe4e2d8, { foil: true, windows: true });
  module(w, -21.5, -12.75, 2.1, 0xe0ded4, { foil: true });
  sideWings(w, -17, 7);
  return w.finish({ pitch: 46, street: 16, blocks: 0, weightless: true });
}
