// The voxel city's destruction, headless (Dawn through the `webgpu` package): a charge on the
// building, the game's debris readback every six steps (freeze what rests, break what fast
// debris hits), then a second charge higher up. Times the main-thread cost of each part, the
// kind that makes a frame hitch: the blast itself, each step's CPU side, and each readback's
// continuation. GPU time is not measured (it stays out of the frame at these sizes).
//
//   node --experimental-transform-types --no-warnings scripts/city-bench.ts

import { create, globals } from 'webgpu';
import { blast, blasts, type Hit } from '../src/destruction.ts';
import { bodyCapacity, CityPhysics, PARKED_BELOW } from '../src/physics.ts';
import { Structure } from '../src/structure.ts';
import { buildCity, raycast } from '../src/world.ts';
import { BODY_FLOATS } from 'three-avbd/advanced';

Object.assign(globalThis, globals);
const gpu = create([]);
const adapter = (await gpu.requestAdapter())!;
const device = await adapter.requestDevice({
  requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize },
});
device.onuncapturederror = (e) => console.error('GPU error:', e.error.message);

const city = buildCity();
const structure = new Structure(city);
const capacity = bodyCapacity(city);
const buffer = device.createBuffer({ size: capacity * BODY_FLOATS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
const physics = new CityPhysics(device, city, buffer);

const ms = () => performance.now();
const stats = (name: string, xs: number[]) => {
  const s = [...xs].sort((a, b) => a - b);
  const q = (p: number) => s[Math.min(s.length - 1, Math.floor(s.length * p))];
  console.log(`${name.padEnd(22)} n ${String(xs.length).padStart(4)}  median ${q(0.5).toFixed(2).padStart(6)} ms  p95 ${q(0.95).toFixed(2).padStart(6)} ms  worst ${s[s.length - 1].toFixed(2).padStart(6)} ms  total ${xs.reduce((a, b) => a + b, 0).toFixed(0).padStart(5)} ms`);
};

// The game's readback (main.ts readDebris), without the effects
const REST_SPEED = 0.25;
const REST_SECONDS = 2.5;
const CRUSH_SPEED = 7;
const CRUSH_DAMAGE = 2500;
const LOOSE_CAP = 8000;
let capped = 0;
const stillFor = new Float32Array(city.count);
const damage = new Map<number, number>();
let lastRead = 0;
async function readDebris(now: number): Promise<{ read: number; work: number; strikes: number; fell: number }> {
  const t0 = ms();
  const counters = await physics.solver.readCounters();
  physics.solver.adapt(counters);
  const { voxels, data } = await physics.readLoose();
  const t1 = ms();
  const since = now - lastRead;
  lastRead = now;
  const resting: number[] = [];
  const poses: number[] = [];
  const crushes: { voxel: number; speed: number }[] = [];
  for (const [k, v] of voxels.entries()) {
    const o = 12 * k;
    const [px, py, pz] = [data[o], data[o + 1], data[o + 2]];
    if (pz < PARKED_BELOW) continue;
    const speed = Math.hypot(data[o + 4], data[o + 5], data[o + 6]);
    stillFor[v] = speed < REST_SPEED ? stillFor[v] + since : 0;
    if (stillFor[v] > REST_SECONDS && resting.length < 4000) {
      resting.push(v);
      poses.push(px, py, pz, data[o + 8], data[o + 9], data[o + 10], data[o + 11]);
    }
    if (speed > CRUSH_SPEED && crushes.length < 24) {
      const d = [data[o + 4] / speed, data[o + 5] / speed, data[o + 6] / speed];
      const hit = raycast(city, [px, py, pz], d, speed * 0.12 + 0.6);
      if (hit && hit.voxel >= 0) {
        const chunk = structure.chunk[hit.voxel];
        const total = (damage.get(chunk) ?? 0) + speed * speed;
        damage.set(chunk, total);
        if (total > CRUSH_DAMAGE) {
          damage.delete(chunk);
          crushes.push({ voxel: hit.voxel, speed });
        }
      }
    }
  }
  for (const [chunk, total] of damage) (total < 50 ? damage.delete(chunk) : damage.set(chunk, total * 0.5));
  const tf = ms();
  if (resting.length) physics.freeze(resting, Float32Array.from(poses));
  if (physics.loose > LOOSE_CAP) {
    const far = voxels
      .map((v, k) => ({ v, d: (data[12 * k] + 4) ** 2 + (data[12 * k + 1] + 23) ** 2, k }))
      .filter(({ k, v }) => data[12 * k + 2] > PARKED_BELOW && city.state[v] === 1)
      .sort((a, b) => b.d - a.d)
      .slice(0, physics.loose - LOOSE_CAP);
    capped += far.length;
    physics.remove(far.map(({ v }) => v));
  }
  freezeMs.push(ms() - tf);
  const strikes = crushes.filter(({ voxel }) => city.state[voxel] === 0);
  let fell = 0;
  if (strikes.length) {
    const hits: Hit[] = strikes.map(({ voxel, speed }) => ({
      at: Array.from(city.position.subarray(3 * voxel, 3 * voxel + 3)),
      radius: 0.35 + Math.min(0.6, speed / 40),
      push: speed * 0.2,
      core: 0.25,
    }));
    fell = blasts(city, structure, physics, hits).falling.length;
  }
  return { read: t1 - t0, work: ms() - t1, strikes: strikes.length, fell };
}

const stepMs: number[] = [];
const freezeMs: number[] = [];
const readMs: number[] = [];
const workMs: number[] = [];
const boomMs: number[] = [];
let strikesTotal = 0;
let fellTotal = 0;
async function run(seconds: number, from: number): Promise<void> {
  for (let k = 0; k < seconds * 60; k++) {
    const t = ms();
    physics.step();
    stepMs.push(ms() - t);
    if (k % 6 === 5) {
      const r = await readDebris(from + (k + 1) / 60);
      readMs.push(r.read);
      workMs.push(r.work);
      strikesTotal += r.strikes;
      fellTotal += r.fell;
    }
  }
}

const boom = (x: number, y: number, z: number, radius: number, push: number) => {
  const t = ms();
  const r = blast(city, structure, physics, [x, y, z], radius, push, 0.35);
  boomMs.push(ms() - t);
  console.log(`charge at (${x}, ${y}, ${z}): ${r.gone.length} gone, ${r.loose.length} loose, ${r.falling.length} falling, ${(ms() - t).toFixed(1)} ms`);
};

await run(1, 0);
console.log(`${city.count.toLocaleString()} voxels in ${structure.chunks.length.toLocaleString()} chunks`);
boom(0, -6.8, 3, 3.2, 14);
await run(12, 1);
boom(6, -6.8, 9, 3.2, 14);
await run(12, 13);
console.log(`loose ${physics.loose.toLocaleString()}, rubble ${physics.rubble.size.toLocaleString()}, joints ${physics.solver.jointCount.toLocaleString()}, strikes ${strikesTotal}, felled by strikes ${fellTotal}, capped ${capped}`);
stats('charge (blast)', boomMs);
stats('step CPU', stepMs);
stats('readback wait', readMs);
stats('readback work', workMs);
stats('  of which freeze', freezeMs);
physics.destroy();
process.exit(0);
