// The space station (src/station.ts): every part held where it joins the rest (world.ts
// anchors), nothing loose as built, and what's cut off (a solar wing's mast) comes free whole,
// while nothing weighs on anything (no storey overloaded, however much is taken out).

import assert from 'node:assert/strict';
import { test } from 'node:test';
import { buildStation, HUB } from '../src/station.ts';
import { Structure } from '../src/structure.ts';
import { State } from '../src/world.ts';

test('every part of the station is held, and nothing is loose as built', () => {
  const city = buildStation();
  assert.ok(city.weightless && city.anchor, 'weightless, with anchors');
  assert.ok(city.count > 30000 && city.count < 90000, `${city.count} voxels`);
  const held = new Array(city.buildings.length).fill(0);
  for (let v = 0; v < city.count; v++) held[city.building[v]] += city.anchor![v];
  held.forEach((n, i) => assert.ok(n > 0, `part ${i} is held somewhere`));
  const structure = new Structure(city);
  for (let i = 0; i < city.buildings.length; i++) assert.deepStrictEqual(structure.unsupported(i), [], `part ${i} as built`);
});

test("cutting a wing's mast at its root frees the wing beyond it, whole; nothing is ever overloaded", () => {
  const city = buildStation();
  const structure = new Structure(city);
  // The first wing reaching up on the -y side: the part whose voxels are up there at y = -18
  const wing = city.buildings.findIndex((_, i) => {
    let [thin, top] = [true, -Infinity];
    for (let v = 0; v < city.count; v++) {
      if (city.building[v] !== i) continue;
      thin &&= Math.abs(city.position[3 * v + 1] + 18) < 0.5;
      top = Math.max(top, city.position[3 * v + 2]);
    }
    return thin && top > HUB[2] + 5;
  });
  assert.ok(wing >= 0, 'found the wing');
  const voxels: number[] = [];
  for (let v = 0; v < city.count; v++) if (city.building[v] === wing) voxels.push(v);
  // Cut through the mast just above its box on the truss (the cells within 0.2 m of x = 0, a
  // quarter metre band of height)
  const zs = voxels.map((v) => city.position[3 * v + 2]);
  const root = Math.min(...zs);
  const cut = voxels.filter((v) => Math.abs(city.position[3 * v]) < 0.2 && Math.abs(city.position[3 * v + 2] - (root + 1.625)) < 0.13);
  assert.ok(cut.length > 0, 'the mast is there to cut');
  structure.leave(cut);
  for (const v of cut) city.state[v] = State.Gone;
  const free = structure.unsupported(wing);
  const freed = free.reduce((n, p) => n + p.voxels.length, 0);
  // Everything above the cut: the mast and both blankets (they hang off the mast only)
  const above = voxels.filter((v) => city.position[3 * v + 2] > root + 1.75 && city.state[v] === State.Fixed).length;
  assert.strictEqual(freed, above, `${freed} voxels came free, ${above} above the cut`);
  assert.deepStrictEqual(structure.overloaded(wing), []);
});
