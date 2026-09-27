// Destruction, Teardown's way: a blast turns the voxels at its heart to dust, knocks the ones
// around them loose in clumps and throws them outward; then whatever in the objects it
// touched no longer stands on the ground falls. What still stands is known from the chunk
// graph (structure.ts), kept up to date as voxels leave, so a blast (or two dozen debris
// strikes in one readback) costs a walk over chunks, not a flood fill over every voxel.
//
// What falls is cut into fragments as three-destruction cuts a mesh: Voronoi cells, their
// seeds packed near the break, so fragments are small where it broke and big further off. The
// cells are made of the structure's chunks (a fragment is a few dozen chunks, so the cut costs
// nothing), and the bonds follow (physics.ts Bond): strong inside a chunk, a little weaker
// across the chunk seams inside a fragment (a piece cracks there first), weak between
// fragments (a section comes down whole and splits along them where it lands). The shell
// round a blast is cut finer still, into clumps of voxels seeded near the blast.

import { Bond, type CityPhysics, type Motion, turning } from './physics.ts';
import type { Structure } from './structure.ts';
import { buildingsNear, type City, FLOORS, isGlass, State, VOXEL } from './world.ts';

export interface Hit {
  at: number[];
  /** Metres: dust within `core` of the radius, loose within the radius, thrown at up to `push` m/s. */
  radius: number;
  push: number;
  core?: number;
  /**
   * A cut (the laser, whose way this is: horizontal, unit): what it brings down tips over
   * rather than dropping, turning about what's left uncut toward the side that was cut.
   */
  topple?: number[];
}

/** How fast a cut piece starts to tip (rad/s). */
const TOPPLE = 0.32;

export interface BlastResult {
  /** Voxels turned to dust, knocked loose, and falling in detached pieces, over all the hits. */
  gone: number[];
  loose: number[];
  falling: number[];
  /** The same per hit (for effects at each). */
  hits: { gone: number[]; loose: number[] }[];
  /** How many of the falling voxels fall carried by rigid sections. */
  carried: number;
  /** Windows shattered (whole panes: their glass is gone, for shards and sound), and what broke them: a blast (its index in the hits) or the frame giving way (-1). */
  panes: { pane: number; voxels: number[]; by: number }[];
}

/** A blast's pressure breaks windows out to this many times its radius. */
const PRESSURE = 3;

/** Fixed voxels within `radius` of `at`. */
function within(city: City, at: ArrayLike<number>, radius: number): number[] {
  const out: number[] = [];
  const lo = [at[0] - radius, at[1] - radius, at[2] - radius];
  const hi = [at[0] + radius, at[1] + radius, at[2] + radius];
  for (const i of buildingsNear(city, lo, hi)) {
    const b = city.buildings[i];
    const cell = (m: number) => Math.floor(m / VOXEL);
    const [x0, x1] = [Math.max(0, cell(lo[0]) - b.x0), Math.min(b.w - 1, cell(hi[0]) - b.x0)];
    const [y0, y1] = [Math.max(0, cell(lo[1]) - b.y0), Math.min(b.d - 1, cell(hi[1]) - b.y0)];
    const [z0, z1] = [Math.max(0, cell(lo[2])), Math.min(b.h - 1, cell(hi[2]))];
    for (let z = z0; z <= z1; z++) {
      for (let y = y0; y <= y1; y++) {
        for (let x = x0; x <= x1; x++) {
          const v = b.cells[x + b.w * (y + b.d * z)];
          if (v < 0 || city.state[v] !== State.Fixed) continue;
          const p = city.position.subarray(3 * v, 3 * v + 3);
          if ((p[0] - at[0]) ** 2 + (p[1] - at[1]) ** 2 + (p[2] - at[2]) ** 2 <= radius * radius) out.push(v);
        }
      }
    }
  }
  return out;
}

/** The share of a crushed storey that goes straight to dust. */
const CRUSHED_TO_DUST = 0.4;
/** Blocks of a falling piece at least this big (voxels) fall whole as one body (physics.ts proxy), and how wide the blocks are (m). */
const SECTION_MIN = 60;
const SECTION_SPAN = 6;
/** Fragment sizes (voxels): clumps a blast knocks out of the shell, and the most a falling section keeps together. */
const CLUMP = Math.round(1.5 / VOXEL ** 2);
const SLAB = Math.round(160 / VOXEL ** 2);

/**
 * Voronoi fragments of `voxels`, about `mean` voxels each: three quarters of the seeds drawn
 * near `at` (weighted by (1 + d/reach)^-2·falloff), the rest anywhere; every voxel joins its
 * nearest seed. Returns each voxel's fragment.
 */
export function fragments(city: City, voxels: number[], at: ArrayLike<number>, mean: number, falloff = 2, reach = 4): Map<number, number> {
  const n = voxels.length;
  const count = Math.max(1, Math.round(n / mean));
  const p = city.position;
  const dist = voxels.map((v) => Math.hypot(p[3 * v] - at[0], p[3 * v + 1] - at[1], p[3 * v + 2] - at[2]));
  const weight = dist.map((d) => (1 + d / reach) ** (-2 * falloff));
  const total = weight.reduce((a, b) => a + b, 0);
  const seeds: number[] = [];
  for (let k = 0; k < count; k++) {
    let pick = Math.floor(Math.random() * n);
    if (k < count * 0.75) {
      let r = Math.random() * total;
      for (pick = 0; pick < n - 1 && (r -= weight[pick]) > 0; pick++);
    }
    seeds.push(voxels[pick]);
  }
  const of = new Map<number, number>();
  for (const v of voxels) {
    let [best, bestD] = [0, Infinity];
    for (let k = 0; k < seeds.length; k++) {
      const s = seeds[k];
      const d = (p[3 * v] - p[3 * s]) ** 2 + (p[3 * v + 1] - p[3 * s + 1]) ** 2 + (p[3 * v + 2] - p[3 * s + 2]) ** 2;
      if (d < bestD) [best, bestD] = [k, d];
    }
    of.set(v, best);
  }
  return of;
}

/**
 * Voronoi fragments of `chunks` (of a falling piece), about `mean` voxels each, the seeds
 * drawn near `at` as in `fragments` but among the chunks' centres. Returns each chunk's fragment.
 */
function fragmentChunks(city: City, structure: Structure, chunks: number[], at: ArrayLike<number>, mean: number, falloff: number, reach: number): Map<number, number> {
  const centre = new Map<number, number[]>();
  let voxels = 0;
  for (const c of chunks) {
    const list = structure.chunks[c].voxels;
    const sum = [0, 0, 0];
    for (const v of list) for (let a = 0; a < 3; a++) sum[a] += city.position[3 * v + a];
    centre.set(c, sum.map((x) => x / Math.max(1, list.length)));
    voxels += list.length;
  }
  const count = Math.max(1, Math.round(voxels / mean));
  const dist = chunks.map((c) => Math.hypot(...centre.get(c)!.map((x, a) => x - at[a])));
  const weight = dist.map((d) => (1 + d / reach) ** (-2 * falloff));
  const total = weight.reduce((a, b) => a + b, 0);
  const seeds: number[] = [];
  for (let k = 0; k < count; k++) {
    let pick = Math.floor(Math.random() * chunks.length);
    if (k < count * 0.75) {
      let r = Math.random() * total;
      for (pick = 0; pick < chunks.length - 1 && (r -= weight[pick]) > 0; pick++);
    }
    seeds.push(chunks[pick]);
  }
  const of = new Map<number, number>();
  for (const c of chunks) {
    const p = centre.get(c)!;
    let [best, bestD] = [0, Infinity];
    seeds.forEach((s, k) => {
      const q = centre.get(s)!;
      const d = (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2;
      if (d < bestD) [best, bestD] = [k, d];
    });
    of.set(c, best);
  }
  return of;
}

/** Bonds of a falling piece cut into `frag` (chunk → fragment): by chunk and fragment of each pair. */
function bonds(structure: Structure, frag: Map<number, number>): (v: number, u: number) => number {
  return (v, u) => {
    const [cv, cu] = [structure.chunk[v], structure.chunk[u]];
    if (cv === cu) return Bond.Fragment;
    return frag.get(cv) === frag.get(cu) ? Bond.Crack : Bond.Seam;
  };
}

/**
 * Several blasts at once (debris strikes come in batches): each does its local damage, then
 * what no longer stands in the objects any of them touched falls, found once per object.
 */
export function blasts(city: City, structure: Structure, physics: CityPhysics, hits: Hit[]): BlastResult {
  const result: BlastResult = { gone: [], loose: [], falling: [], hits: [], carried: 0, panes: [] };
  const touched = new Set<number>();
  // Glass never falls as voxels: a window that's hit, or whose frame moves, shatters whole
  const broken = new Set<number>();
  // Taken out of the structure and the solver in one go (a collapse breaks a hundred windows)
  let glass: number[] = [];
  const shatter = (pane: number, by: number) => {
    if (broken.has(pane)) return;
    broken.add(pane);
    const voxels = city.panes[pane].voxels.filter((v) => city.state[v] === State.Fixed || city.state[v] === State.Loose);
    if (!voxels.length) return;
    // What the glass filled in may hang from it: that object is checked too
    touched.add(city.building[voxels[0]]);
    glass.push(...voxels);
    result.panes.push({ pane, voxels, by });
  };
  const shattered = () => {
    if (!glass.length) return;
    structure.leave(glass);
    physics.remove(glass);
    glass = [];
  };
  /** `voxels` less their glass, whose windows shatter. */
  const unglazed = (voxels: number[], by: number) => {
    const out: number[] = [];
    for (const v of voxels) {
      if (city.pane[v] >= 0) shatter(city.pane[v], by);
      else if (!isGlass(city.material[v]) && city.state[v] === State.Fixed) out.push(v);
    }
    shattered();
    return out;
  };
  hits.forEach(({ at, radius }, k) => {
    const reach = radius * PRESSURE;
    city.panes.forEach((p, i) => {
      if ((p.centre[0] - at[0]) ** 2 + (p.centre[1] - at[1]) ** 2 + (p.centre[2] - at[2]) ** 2 < reach * reach) shatter(i, k);
    });
  });
  shattered();
  const loosen = (voxels: number[], bond: (v: number, u: number) => number) => {
    structure.leave(voxels);
    physics.loosen(voxels, bond);
  };
  const clumped = (of: Map<number, number>, strength: number) => (v: number, u: number) => (of.get(v) === of.get(u) ? strength : 0);
  for (const [k, { at, radius, core = 0.5 }] of hits.entries()) {
    const gone: number[] = [];
    const loose: number[] = [];
    for (const v of unglazed(within(city, at, radius), k)) {
      touched.add(city.building[v]);
      const p = city.position.subarray(3 * v, 3 * v + 3);
      const d = Math.hypot(p[0] - at[0], p[1] - at[1], p[2] - at[2]);
      (d < radius * core ? gone : loose).push(v);
    }
    structure.leave(gone);
    physics.remove(gone);
    // The shell comes away in clumps, smallest nearest the blast
    const clumps = fragments(city, loose, at, CLUMP, 3, radius * 0.4);
    loosen(loose, clumped(clumps, Bond.Rubble));
    result.hits.push({ gone, loose });
    result.gone.push(...gone);
    result.loose.push(...loose);
  }
  const first = hits[0];
  const cut = hits.find((h) => h.topple)?.topple;
  for (const i of touched) {
    /** How what falls from this object starts to move (a cut tips it over). */
    let motion: Motion | undefined;
    // A storey that can't carry its load gives way (it may take a second one with it)
    for (let k = 0; k < 2; k++) {
      const crushed = unglazed(structure.overloaded(i), -1);
      if (!crushed.length) break;
      if (cut && !motion) motion = hinge(city, crushed, cut);
      // A crushed storey is mostly powder: some of it goes to dust at once, the rest crumbles
      const dust: number[] = [];
      const rest: number[] = [];
      for (const v of crushed) (Math.random() < CRUSHED_TO_DUST ? dust : rest).push(v);
      structure.leave(dust);
      physics.remove(dust);
      result.gone.push(...dust);
      const bits = fragments(city, rest, first.at, CLUMP, 1, first.radius);
      if (motion) {
        // Cut, it doesn't crumble: what's left of the storey is the hinge it tips over
        structure.leave(rest);
        physics.loosen(rest, clumped(bits, Bond.Rubble), motion);
      } else loosen(rest, clumped(bits, Bond.Rubble));
      result.falling.push(...rest);
    }
    for (const found of structure.unsupported(i)) {
      const piece = { voxels: unglazed(found.voxels, -1), chunks: found.chunks };
      if (!piece.voxels.length) continue;
      // Cut clean through (a tree's trunk): it tips over its lowest point, the cut's way. In
      // orbit nothing tips: what's cut off drifts away from the cut, turning slowly
      const moving = city.weightless ? drift(city, piece.voxels, first.at) : (motion ?? (cut ? hinge(city, lowest(city, piece.voxels), cut) : undefined));
      if (piece.voxels.length <= CLUMP) {
        structure.leave(piece.voxels);
        physics.loosen(piece.voxels, () => Bond.Rubble, moving);
      }
      else {
        const cut = fragmentChunks(city, structure, piece.chunks, first.at, Math.min(SLAB, Math.max(CLUMP * 3, piece.voxels.length / 10)), 1.5, 6);
        const bond = bonds(structure, cut);
        // It falls in blocks a storey tall and a few metres across (split on the floor slabs,
        // as a tower breaks), each one rigid body: a tower's storeys are far more voxels than
        // can be simulated one by one. What doesn't qualify falls as voxels, bonded.
        structure.leave(piece.voxels);
        const groups = new Map<number, number[]>();
        // (In orbit it goes whole: nothing makes it break into storeys)
        if (city.weightless) groups.set(0, piece.voxels);
        else for (const v of piece.voxels) {
          const z = Math.floor(city.position[3 * v + 2] / VOXEL);
          const band = z < FLOORS.ground ? -1 : Math.floor((z - FLOORS.ground) / FLOORS.storey);
          const key = (band + 2) * 10000 + Math.floor(city.position[3 * v] / SECTION_SPAN) * 100 + Math.floor(city.position[3 * v + 1] / SECTION_SPAN) + 5000;
          const g = groups.get(key);
          if (g) g.push(v);
          else groups.set(key, [v]);
        }
        const rest: number[] = [];
        for (const group of groups.values()) if (group.length < SECTION_MIN || !physics.proxy(group, bond, undefined, 0.08, moving)) rest.push(...group);
        physics.loosen(rest, bond, moving);
        result.carried += piece.voxels.length - rest.length;
      }
      result.falling.push(...piece.voxels);
    }
  }
  // Everything loose near each blast (freshly or already) is thrown
  for (const { at, radius, push } of hits) physics.blast(at, radius * 1.6, push);
  return result;
}

/**
 * A piece cut off in orbit: drifting away from `from` (where it was cut or blown) at half a
 * metre a second, turning slowly about its middle.
 */
function drift(city: City, voxels: number[], from: ArrayLike<number>): Motion {
  const c = [0, 0, 0];
  for (const v of voxels) for (let a = 0; a < 3; a++) c[a] += city.position[3 * v + a] / voxels.length;
  const d = [c[0] - from[0], c[1] - from[1], c[2] - from[2]];
  const l = Math.hypot(d[0], d[1], d[2]) || 1;
  const speed = 0.5;
  const v0 = d.map((x) => (x / l) * speed);
  const spin = [(Math.random() - 0.5) * 0.12, (Math.random() - 0.5) * 0.12, (Math.random() - 0.5) * 0.12];
  return {
    spin,
    at: (p) => {
      const r = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
      return [v0[0] + spin[1] * r[2] - spin[2] * r[1], v0[1] + spin[2] * r[0] - spin[0] * r[2], v0[2] + spin[0] * r[1] - spin[1] * r[0]];
    },
  };
}

/** The lowest voxels of `voxels` (within a voxel of the lowest). */
function lowest(city: City, voxels: number[]): number[] {
  let z = Infinity;
  for (const v of voxels) z = Math.min(z, city.position[3 * v + 2]);
  return voxels.filter((v) => city.position[3 * v + 2] < z + VOXEL * 1.5);
}

/**
 * How a piece cut through at the layer of `left` (what's left uncut there) starts to tip: about
 * the uncut part, toward the side that was cut (from the uncut part's middle toward the cut's,
 * or the cut's own way if the layer was cut clean through).
 */
function hinge(city: City, left: number[], way: number[]): Motion {
  const b = city.buildings[city.building[left[0]]];
  const layer = b.w * b.d;
  const z = Math.floor(city.cell[left[0]] / layer);
  const mean = (list: number[]) => {
    const m = [0, 0, 0];
    for (const v of list) for (let a = 0; a < 3; a++) m[a] += city.position[3 * v + a] / list.length;
    return m;
  };
  // What's gone from that layer: the cut
  const gone: number[] = [];
  for (let c = z * layer; c < (z + 1) * layer; c++) {
    const v = b.cells[c];
    if (v >= 0 && city.state[v] !== State.Fixed && !isGlass(city.material[v])) gone.push(v);
  }
  const pivot = mean(left);
  let dir = [way[0], way[1]];
  if (gone.length) {
    const g = mean(gone);
    const d = [g[0] - pivot[0], g[1] - pivot[1]];
    const n = Math.hypot(d[0], d[1]);
    if (n > 0.3) dir = [d[0] / n, d[1] / n];
  }
  // Turning about the horizontal axis across the way it tips: up × dir
  return turning([pivot[0], pivot[1], pivot[2] + VOXEL / 2], [-dir[1] * TOPPLE, dir[0] * TOPPLE, 0]);
}

/** One blast of `radius` metres at `at` (see Hit). */
export function blast(city: City, structure: Structure, physics: CityPhysics, at: ArrayLike<number>, radius: number, push: number, core = 0.5): BlastResult {
  return blasts(city, structure, physics, [{ at: Array.from(at), radius, push, core }]);
}
