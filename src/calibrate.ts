// Tuning the game to the device it runs on, once, while it loads: how many loose voxels the
// physics can carry, and how the scene is drawn, so a collapse holds 60 fps.
//
// Physics: a pile of loose voxels packed against each other in a scratch solver, stepped and
// timed on the GPU (its own timestamps where it has them: a round trip also waits for whatever
// else the GPU is doing, and read 20 times the step). The pile is dense (some 18 contacts a
// voxel); rubble has far fewer (about 2 a loose voxel, measured in a collapse), so the time per
// contact, at CONTACTS_PER_LOOSE a loose voxel, sets how many loose voxels the game allows so
// a step fits PHYSICS_MS; a slow GPU also gets fewer solver iterations. Drawing: the real scene with a smoke cloud in view, drawn at
// high, then medium, then low quality until a frame fits RENDER_MS, and at a lower resolution
// if even low doesn't. Both leave room for each other inside a 60 fps frame. The result is kept
// per device (the GPU and the screen) and scene (orbit's clouds cost far more a pixel than the
// city), so later loads of it skip it.

import { GpuSolver3D, gpuParams3D, REF_UP } from 'three-avbd/advanced';
import { Rigid } from 'three-avbd/advanced';
import { Solver } from 'three-avbd/advanced';
import { VOXEL } from './world.ts';

export type Quality = 'low' | 'medium' | 'high';

export interface DeviceProfile {
  version: number;
  /** The device it was measured on (GPU and screen). */
  key: string;
  quality: Quality;
  /** Share of the quality's resolution (1: all of it). */
  resolution: number;
  /** Loose voxels at once, at most, and the solver's iterations. */
  looseCap: number;
  iterations: number;
  /** Share of the particles' live budget. */
  particles: number;
  /** What was measured: a physics step (ms, per 1000 loose voxels of rubble) and a heavy frame at the chosen quality (ms). */
  stepMsPer1000: number;
  frameMs: number;
}

const VERSION = 3;
const STORE = 'city.profile';
/** Budgets inside a 60 fps frame (16.7 ms): drawing a heavy frame, and one physics step. */
const RENDER_MS = 10;
const PHYSICS_MS = 4.5;
/** Contacts a loose voxel brings, for the budget (rubble measured about 2: a margin). */
const CONTACTS_PER_LOOSE = 3;
/** Loose voxels, fewest and most, whatever the measurement. */
const CAP_MIN = 1200;
const CAP_MAX = 8000;

/** This device: its GPU and its screen (a different window size draws a different number of pixels), in `scene`. */
export function deviceKey(adapter: string, scene: string): string {
  const w = Math.round(screen.width * devicePixelRatio);
  const h = Math.round(screen.height * devicePixelRatio);
  return `${adapter}|${Math.max(w, h)}x${Math.min(w, h)}|${scene}`;
}

/** The profiles kept, by key. */
function kept(): Record<string, DeviceProfile> {
  try {
    const all = JSON.parse(localStorage.getItem(STORE) ?? '{}') as Record<string, DeviceProfile>;
    return all && typeof all === 'object' && !('version' in all) ? all : {};
  } catch {
    return {};
  }
}

export function savedProfile(key: string): DeviceProfile | null {
  const p = kept()[key];
  return p && p.version === VERSION ? p : null;
}

export function saveProfile(p: DeviceProfile): void {
  try {
    localStorage.setItem(STORE, JSON.stringify({ ...kept(), [p.key]: p }));
  } catch {
    // storage unavailable: it will be measured again next time
  }
}

export function forgetProfile(): void {
  try {
    localStorage.removeItem(STORE);
  } catch {
    // nothing kept
  }
}

const median = (xs: number[]) => [...xs].sort((a, b) => a - b)[xs.length >> 1];

/**
 * Time a physics step of a pile of `side`³ loose voxels resting on each other and the ground
 * (ms, median of the settled steps). `onStep(share)` as it goes.
 */
export async function measurePhysics(device: GPUDevice, onStep: (share: number) => void, side = 13): Promise<{ stepMs: number; bodies: number; contacts: number }> {
  const ref = new Solver();
  new Rigid(ref, [4000, 4000, 20], 0, 0.8, [0, 0, -10]);
  const size = [VOXEL, VOXEL, VOXEL];
  for (let z = 0; z < side - 1; z++) {
    for (let y = 0; y < side; y++) {
      for (let x = 0; x < side; x++) new Rigid(ref, size, 1.8, 0.7, [(x - side / 2) * VOXEL, (y - side / 2) * VOXEL, (z + 0.5) * VOXEL]);
    }
  }
  const bodies = ref.bodies.length - 1;
  const solver = new GpuSolver3D(device, ref);
  Object.assign(solver.params, gpuParams3D(), { dt: 1 / 60, iterations: 6, gravity: -9.81, up: REF_UP });
  const wall: number[] = [];
  const gpu: number[] = [];
  const STEPS = 24;
  let contacts = 1;
  try {
    for (let k = 0; k < STEPS; k++) {
      // The colouring settles over the first steps (as in the game, a readback adapts it)
      if (k === 3 || k === 8) solver.adapt(await solver.readCounters());
      const profiled = new Promise<number | null>((resolve) => {
        solver.profileNextStep((p) => resolve(p.total));
        setTimeout(() => resolve(null), 500);
      });
      const t = performance.now();
      solver.step();
      await device.queue.onSubmittedWorkDone();
      wall.push(performance.now() - t);
      const ms = await profiled;
      if (ms !== null && k >= 10) gpu.push(ms);
      onStep((k + 1) / STEPS);
    }
    contacts = Math.max(1, (await solver.readCounters()).contacts);
  } finally {
    solver.destroy();
  }
  return { stepMs: gpu.length >= 4 ? median(gpu) : median(wall.slice(10)), bodies, contacts };
}

/** Time `frames` frames drawn by `draw` (ms each, median), after a few to warm up. */
export async function measureFrames(device: GPUDevice, draw: () => void, frames = 16): Promise<number> {
  const times: number[] = [];
  for (let k = 0; k < frames + 4; k++) {
    const t = performance.now();
    draw();
    await device.queue.onSubmittedWorkDone();
    if (k >= 4) times.push(performance.now() - t);
  }
  return median(times);
}

/**
 * The profile for this device: the physics from `stepMs` for `bodies` loose voxels, the drawing
 * from `frame(quality, resolution)`, which sets the scene up that way and returns a heavy
 * frame's time (tried best first). `onTry(share)` as it goes.
 */
export async function chooseProfile(
  key: string,
  physics: { stepMs: number; bodies: number; contacts: number },
  frame: (quality: Quality, resolution: number) => Promise<number>,
  onTry: (share: number) => void,
): Promise<DeviceProfile> {
  // Physics: loose voxels so a step fits the budget; a slow GPU trades iterations for bodies
  const perContact = Math.max(physics.stepMs, 0.05) / physics.contacts;
  const perBody = perContact * CONTACTS_PER_LOOSE;
  let iterations = 6;
  let looseCap = PHYSICS_MS / perBody;
  if (looseCap < 2500) {
    iterations = 4;
    looseCap *= 6 / 4;
  }
  looseCap = Math.round(Math.min(CAP_MAX, Math.max(CAP_MIN, looseCap)));

  // Drawing: the best quality whose heavy frame fits, then lower resolutions of low
  const tries: [Quality, number][] = [
    ['high', 1],
    ['medium', 1],
    ['low', 1],
    ['low', 0.8],
    ['low', 0.65],
  ];
  let chosen: [Quality, number] = tries[tries.length - 1];
  let frameMs = Infinity;
  for (const [k, t] of tries.entries()) {
    frameMs = await frame(t[0], t[1]);
    onTry((k + 1) / tries.length);
    if (frameMs <= RENDER_MS) {
      chosen = t;
      break;
    }
  }
  const particles = chosen[0] === 'high' ? 1 : chosen[0] === 'medium' ? 0.7 : 0.45;
  return {
    version: VERSION,
    key,
    quality: chosen[0],
    resolution: chosen[1],
    looseCap,
    iterations,
    particles,
    stepMsPer1000: +((perBody * 1000).toFixed(3)),
    frameMs: +frameMs.toFixed(2),
  };
}
