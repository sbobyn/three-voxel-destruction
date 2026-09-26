// The city's chunk graph (src/city/structure.ts) must say exactly what a flood fill through the
// voxels says about what still stands, however the building is damaged, and cut the building
// into chunks that are whole (connected, one material) and repeatable.

import assert from 'node:assert/strict';
import { describe, test } from 'node:test';
import { MARGIN, Structure } from '../src/structure.ts';
import { buildCity, type City, isGlass, random, State, VOXEL } from '../src/world.ts';

/** Reference: unsupported pieces of object `i` by flood fill from the ground through fixed voxels. */
function unsupportedByVoxels(city: City, i: number): number[][] {
  const b = city.buildings[i];
  const n = b.w * b.d * b.h;
  const fixed = (c: number) => b.cells[c] >= 0 && city.state[b.cells[c]] === State.Fixed;
  const mark = new Uint8Array(n);
  const queue = new Int32Array(n);
  let [head, tail] = [0, 0];
  for (let c = 0; c < b.w * b.d; c++) if (fixed(c)) (mark[c] = 1), (queue[tail++] = c);
  const layer = b.w * b.d;
  const visit = (c: number, flag: number) => {
    const x = c % b.w;
    const y = Math.floor(c / b.w) % b.d;
    const tryCell = (m: number) => {
      if (!mark[m] && fixed(m)) (mark[m] = flag), (queue[tail++] = m);
    };
    if (x > 0) tryCell(c - 1);
    if (x + 1 < b.w) tryCell(c + 1);
    if (y > 0) tryCell(c - b.w);
    if (y + 1 < b.d) tryCell(c + b.w);
    if (c >= layer) tryCell(c - layer);
    if (c + layer < n) tryCell(c + layer);
  };
  while (head < tail) visit(queue[head++], 1);
  const pieces: number[][] = [];
  for (let c = 0; c < n; c++) {
    if (mark[c] || !fixed(c)) continue;
    [head, tail] = [0, 0];
    mark[c] = 2;
    queue[tail++] = c;
    while (head < tail) visit(queue[head++], 2);
    pieces.push(Array.from(queue.subarray(0, tail), (m) => b.cells[m]));
  }
  return pieces;
}

const key = (pieces: { voxels: number[] }[] | number[][]) =>
  pieces
    .map((p) => [...('voxels' in p ? p.voxels : p)].sort((a, b) => a - b).join(','))
    .sort()
    .join('|');

/** Fixed voxels of the building within `r` m of `at`. */
function ball(city: City, at: number[], r: number): number[] {
  const out: number[] = [];
  for (let v = 0; v < city.count; v++) {
    if (city.state[v] !== State.Fixed || city.building[v] !== 0) continue;
    const p = city.position.subarray(3 * v, 3 * v + 3);
    if ((p[0] - at[0]) ** 2 + (p[1] - at[1]) ** 2 + (p[2] - at[2]) ** 2 < r * r) out.push(v);
  }
  return out;
}

describe('city structure', () => {
  const city = buildCity();
  const structure = new Structure(city);

  test('cuts the objects into whole chunks of one material, the same every time', () => {
    for (let v = 0; v < city.count; v++) assert.ok(structure.chunk[v] >= 0, `voxel ${v} has a chunk`);
    const sizes = structure.chunks.map((c) => c.voxels.length);
    // Chunks, not voxels: a few thousand for the tower, its offices and the street furniture
    // (some 20 voxels a chunk), where the fill would otherwise walk 90,000 voxels
    assert.ok(structure.chunks.length > 500 && structure.chunks.length < 6000 && structure.chunks.length < city.count / 12, `${structure.chunks.length} chunks`);
    assert.ok(Math.max(...sizes) <= 800, `largest chunk ${Math.max(...sizes)} voxels`);
    structure.chunks.forEach((chunk, id) => {
      const b = city.buildings[chunk.building];
      const material = city.material[chunk.voxels[0]];
      const inChunk = new Set(chunk.voxels);
      const seen = new Set([chunk.voxels[0]]);
      const queue = [chunk.voxels[0]];
      const layer = b.w * b.d;
      for (let n = 0; n < queue.length; n++) {
        const c = city.cell[queue[n]];
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
          const u = ok ? b.cells[m] : -1;
          if (u >= 0 && inChunk.has(u) && !seen.has(u)) (seen.add(u), queue.push(u));
        }
      }
      assert.equal(seen.size, chunk.voxels.length, `chunk ${id} is connected`);
      for (const v of chunk.voxels) {
        assert.equal(structure.chunk[v], id);
        assert.equal(city.material[v], material, `chunk ${id} is one material`);
      }
    });
    const again = new Structure(buildCity());
    assert.deepEqual(Array.from(again.chunk), Array.from(structure.chunk), 'the cut is deterministic');
    // Edges are symmetric and count the faces between fixed voxels of different chunks
    let faces = 0;
    structure.chunks.forEach((chunk, id) => {
      for (const [o, n] of chunk.edges) {
        assert.equal(structure.chunks[o].edges.get(id), n, `edge ${id}-${o} is symmetric`);
        faces += n;
      }
    });
    let count = 0;
    for (let v = 0; v < city.count; v++) {
      const b = city.buildings[city.building[v]];
      const c = city.cell[v];
      const [x, y] = [c % b.w, Math.floor(c / b.w) % b.d];
      const layer = b.w * b.d;
      for (const u of [x + 1 < b.w ? b.cells[c + 1] : -1, y + 1 < b.d ? b.cells[c + b.w] : -1, c + layer < b.cells.length ? b.cells[c + layer] : -1]) {
        if (u >= 0 && structure.chunk[u] !== structure.chunk[v]) count++;
      }
    }
    assert.equal(faces, 2 * count, 'every chunk face counted once each way');
  });

  test('marks the faces open to the air when built (finished), and no others', () => {
    let open = 0;
    for (let v = 0; v < city.count; v++) {
      const b = city.buildings[city.building[v]];
      const c = city.cell[v];
      const [x, y, z] = [c % b.w, Math.floor(c / b.w) % b.d, Math.floor(c / (b.w * b.d))];
      const layer = b.w * b.d;
      const bits = [
        [x > 0, c - 1],
        [x + 1 < b.w, c + 1],
        [y > 0, c - b.w],
        [y + 1 < b.d, c + b.w],
        [z > 0, c - layer],
        [z + 1 < b.h, c + layer],
      ].map(([ok, m], k) => ((k === 4 && z === 0) || (ok && b.cells[m as number] >= 0) ? 0 : 1 << k));
      const mask = bits.reduce((a, b) => a | b, 0);
      assert.equal(city.exposed[v], mask, `voxel ${v} at (${x}, ${y}, ${z})`);
      if (mask) open++;
    }
    // A hollow building: most voxels are on a surface, none on the ground shows its underside
    assert.ok(open > city.count * 0.5 && open < city.count, `${open} of ${city.count} voxels have an open face`);
    for (let v = 0; v < city.count; v++) if (city.cell[v] < city.buildings[city.building[v]].w * city.buildings[city.building[v]].d) assert.equal(city.exposed[v] & 16, 0);
  });

  test('says what stands exactly as a flood fill through the voxels does, blast after blast', () => {
    const rnd = random(5);
    const b = city.buildings[0];
    assert.deepEqual(structure.unsupported(0), [], 'the building stands');
    let checked = 0;
    for (let round = 0; round < 10; round++) {
      // A blast somewhere on the building, big enough to cut things off
      const at = [(b.x0 + rnd() * b.w) * VOXEL, (b.y0 + rnd() * b.d) * VOXEL, (2 + rnd() * (b.h - 4)) * VOXEL];
      const gone = ball(city, at, 1 + rnd() * 2.2);
      structure.leave(gone);
      for (const v of gone) city.state[v] = State.Gone;
      const pieces = structure.unsupported(0);
      assert.equal(key(pieces), key(unsupportedByVoxels(city, 0)), `round ${round}: the same pieces`);
      for (const piece of pieces) {
        for (const c of piece.chunks) for (const v of structure.chunks[c].voxels) assert.equal(structure.chunk[v], c);
        checked++;
        structure.leave(piece.voxels);
        for (const v of piece.voxels) city.state[v] = State.Loose;
      }
      assert.deepEqual(structure.unsupported(0), [], `round ${round}: nothing left hanging`);
      // The layer counts agree with a recount
      const layer = b.w * b.d;
      const counts = new Int32Array(b.h);
      for (let c = 0; c < b.cells.length; c++) if (b.cells[c] >= 0 && city.state[b.cells[c]] === State.Fixed) counts[Math.floor(c / layer)]++;
      assert.deepEqual(Array.from(structure.layerCounts(0)), Array.from(counts), `round ${round}: layer counts`);
    }
    assert.ok(checked > 0, 'some pieces came off');
  });
});

describe('city loads', () => {
  test('glass carries nothing, and a storey gives way once a third of what carries it is gone', () => {
    const city = buildCity();
    const structure = new Structure(city);
    // Every window gone: everything still stands (a car's roof on its pillars too)
    const glass = Array.from({ length: city.count }, (_, v) => v).filter((v) => isGlass(city.material[v]));
    assert.ok(glass.length > 5000, `${glass.length} glass voxels`);
    structure.leave(glass);
    for (const v of glass) city.state[v] = State.Gone;
    city.buildings.forEach((_, i) => {
      assert.deepEqual(structure.overloaded(i), [], `object ${i}: no storey gives way`);
      assert.deepEqual(structure.unsupported(i), [], `object ${i}: nothing hangs`);
    });
    // A shop-floor layer: a quarter of what carries it gone holds, half doesn't (MARGIN: a
    // third, more with the windows gone and the load above lighter)
    assert.equal(MARGIN, 1.5);
    const b = city.buildings[0];
    const layer = b.w * b.d;
    const z = 6;
    const carrying: number[] = [];
    for (let c = z * layer; c < (z + 1) * layer; c++) if (b.cells[c] >= 0 && city.state[b.cells[c]] === State.Fixed) carrying.push(b.cells[c]);
    const take = (share: number, from: number) => {
      const out = carrying.filter((_, k) => k % 20 >= from && k % 20 < from + share * 20);
      structure.leave(out);
      for (const v of out) city.state[v] = State.Gone;
    };
    take(0.25, 0);
    assert.deepEqual(structure.overloaded(0), [], 'a quarter gone: it holds');
    take(0.25, 5);
    const crushed = structure.overloaded(0);
    assert.ok(crushed.length > 0, 'half gone: it gives way');
    assert.ok(crushed.every((v) => Math.floor(city.cell[v] / layer) === z), 'that layer');
  });
});
