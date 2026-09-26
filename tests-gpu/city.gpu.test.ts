// The voxel city's physics (src/city) on a device with only WebGPU's default limits, as a phone
// offers: the renderer used to require 1 GB storage buffers, which a phone refuses, and the
// game never started there. The solver sizes its buffers to the device; this holds it to that.

import assert from 'node:assert/strict';
import { create } from 'webgpu';
import { BODY_FLOATS } from 'three-avbd/advanced';
import { blast } from '../src/destruction.ts';
import { bodyCapacity, CityPhysics } from '../src/physics.ts';
import { Structure } from '../src/structure.ts';
import { buildCity, isGlass, State, VOXEL } from '../src/world.ts';
import { gpuTest } from './device.ts';

gpuTest("the city's physics starts and steps on a device with only the default limits", async () => {
  // A second instance for the small device, kept reachable: collected, its callbacks stop
  // and the test's GPU promises never settle
  const gpu = create([]);
  const adapter = await gpu.requestAdapter();
  assert.ok(adapter, 'an adapter');
  const small = await adapter.requestDevice();
  (globalThis as { __dawnSmall?: unknown }).__dawnSmall = [gpu, adapter, small];
  const errors: string[] = [];
  small.onuncapturederror = (e) => errors.push(e.error.message);
  assert.ok(small.limits.maxStorageBufferBindingSize <= 128 * 2 ** 20, 'the default binding limit (128 MB)');
  const city = buildCity();
  const buffer = small.createBuffer({ size: (bodyCapacity(city)) * BODY_FLOATS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  const physics = new CityPhysics(small, city, buffer);
  // A section knocked loose falls: the solver works, not just builds
  const loose = Array.from({ length: 400 }, (_, k) => k * 7).filter((v) => v < city.count);
  physics.loosen(loose);
  for (let k = 0; k < 30; k++) physics.step();
  const { voxels, data } = await physics.readLoose();
  assert.equal(voxels.length, loose.length, 'the loosened voxels are simulated');
  assert.ok(data.every(Number.isFinite), 'no NaNs');
  physics.destroy();
  buffer.destroy();
  await small.queue.onSubmittedWorkDone();
  assert.deepEqual(errors, [], 'no GPU errors on the small device');
  small.destroy();
});

// Rubble frozen where it came to rest must fall again when what held it goes (a floor that
// collapsed under a heap left the heap hanging in the air): the support check wakes exactly
// the rubble with nothing under it, heaps held from their base.
gpuTest('rubble with nothing under it wakes, rubble on the ground or on held rubble stays', async (device) => {
  const city = buildCity();
  const buffer = device.createBuffer({ size: (bodyCapacity(city)) * BODY_FLOATS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  const physics = new CityPhysics(device, city, buffer);
  // Eight voxels off the roof, loosened then frozen where we say
  const top = Math.max(...Array.from({ length: city.count }, (_, v) => city.position[3 * v + 2]));
  const picks = Array.from({ length: city.count }, (_, v) => v)
    .filter((v) => city.position[3 * v + 2] > top - 0.5)
    .slice(0, 8);
  assert.equal(picks.length, 8);
  physics.loosen(picks);
  const h = VOXEL / 2;
  const at = (x: number, y: number, z: number) => [x, y, z, 0, 0, 0, 1];
  // Out on the street, clear of the building (x 20 m)
  const poses = [
    at(20, 0, h), at(20, 0, 3 * h), // on the ground, and on that
    at(20, 2, 30), at(20, 2, 30 + 2 * h), // in mid-air, and on that
    at(20, 4, h), at(20.2, 4, 3 * h), // on the ground, and leaning on that a little off centre
    at(20, 6, 12), at(20, 8, 20), // alone in mid-air
  ].flat();
  physics.freeze(picks, Float32Array.from(poses));
  assert.equal(physics.rubble.size, 8, 'all frozen');
  const woke = physics.settle();
  const wokenIdx = picks.map((v, k) => (city.state[v] === State.Loose ? k : -1)).filter((k) => k >= 0);
  assert.equal(woke, 4, 'four had nothing under them');
  assert.deepEqual(wokenIdx, [2, 3, 6, 7], 'the ones in mid-air, and the one stacked on them');
  assert.equal(physics.settle(), 0, 'what stays is held');
  physics.destroy();
  buffer.destroy();
});

// Glass never falls as voxels: a window a blast reaches, or whose frame gives way, shatters
// whole (its glass gone, for shards), so no pane is left half standing and no glass tumbles
// loose or rides a falling section. Demolition charges round the shop floor bring the tower
// down, which takes the rest of the windows with it.
gpuTest('windows shatter whole, and no glass falls as voxels', async (device) => {
  const city = buildCity();
  const structure = new Structure(city);
  const buffer = device.createBuffer({ size: bodyCapacity(city) * BODY_FLOATS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  const physics = new CityPhysics(device, city, buffer);
  const whole = () =>
    city.panes.forEach((p, i) => {
      const states = new Set(p.voxels.map((v) => city.state[v]));
      assert.ok(states.size === 1 && (states.has(State.Fixed) || states.has(State.Gone)), `pane ${i}: all standing or all gone (${[...states]})`);
    });
  // A rocket at an upper storey: the windows round it go, the rest stand
  const rocket = blast(city, structure, physics, [-3, -5.5, 9], 1.7, 12, 0.45);
  assert.ok(rocket.panes.length > 3 && rocket.panes.length < 40, `${rocket.panes.length} windows broken by the rocket`);
  assert.ok(rocket.panes.every((p) => p.by === 0), 'by its blast');
  whole();
  let carried = 0;
  for (const [x, y] of [[-4, -5], [2, -5], [-5, 1], [4, 2]]) carried += blast(city, structure, physics, [x, y, 2], 3.2, 14, 0.4).carried;
  assert.ok(carried > 40000, `the tower falls in sections (${carried} voxels carried)`);
  whole();
  for (let v = 0; v < city.count; v++) if (isGlass(city.material[v])) assert.ok(city.state[v] === State.Fixed || city.state[v] === State.Gone, `glass voxel ${v} is ${city.state[v]}`);
  physics.destroy();
  buffer.destroy();
});

