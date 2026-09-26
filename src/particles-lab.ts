// A test bench for the city's particles (./particles.ts): a ground, a few blocks, a sun and a
// sky, drawn through the same pipeline as the city (scene pass with a normal target, GTAO,
// ACES), with keys and buttons for each kind of burst and a live count and frame time.
// `window.lab` drives it from the console: `lab.pause = true; lab.big(); lab.advance(1)` for a
// frame one second after a big blast, `await lab.bench()` for the particles' cost.

import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { ao } from 'three/addons/tsl/display/GTAONode.js';
import { mrt, normalView, output, pass, vec4 } from 'three/tsl';
import * as THREE from 'three/webgpu';
import { type ParticleQuality, Particles } from './particles.ts';

const canvas = document.getElementById('view') as HTMLCanvasElement;
const stats = document.getElementById('stats')!;
const renderer = new THREE.WebGPURenderer({ canvas, antialias: false });
// `?dpr=1` renders at CSS pixels (for timing at a known resolution)
const dpr = Number(new URLSearchParams(location.search).get('dpr'));
renderer.setPixelRatio(dpr > 0 ? dpr : Math.min(window.devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFShadowMap;
await renderer.init();

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x9cc6ea);
const camera = new THREE.PerspectiveCamera(50, 1, 0.5, 3000);
camera.up.set(0, 0, 1);
camera.position.set(70, -85, 32);
const controls = new OrbitControls(camera, canvas);
controls.target.set(0, 0, 14);
controls.enableDamping = true;

const sun = new THREE.Vector3(0.5, -0.35, 0.8).normalize();
const light = new THREE.DirectionalLight(0xfff0d8, 3.2);
light.position.copy(sun).multiplyScalar(150);
light.castShadow = true;
light.shadow.mapSize.set(2048, 2048);
Object.assign(light.shadow.camera, { left: -90, right: 90, top: 90, bottom: -90, near: 1, far: 400 });
scene.add(light, light.target);
scene.add(new THREE.HemisphereLight(0xdde9f5, 0xb7a58a, 1.1));

const ground = new THREE.Mesh(new THREE.PlaneGeometry(1200, 1200), new THREE.MeshStandardNodeMaterial({ color: 0x8c8a7c, roughness: 0.95 }));
ground.receiveShadow = true;
scene.add(ground);
// Blocks standing in for buildings, and a low wall near the blasts
const concrete = new THREE.MeshStandardNodeMaterial({ color: 0xb9b3a8, roughness: 0.85 });
const brick = new THREE.MeshStandardNodeMaterial({ color: 0x9a5a44, roughness: 0.9 });
for (const [x, y, w, d, h, m] of [
  [-35, 25, 16, 14, 60, concrete],
  [-10, 45, 12, 12, 38, brick],
  [30, 38, 18, 12, 24, concrete],
  [45, -10, 10, 16, 45, brick],
  [-45, -25, 14, 10, 18, concrete],
  [8, 12, 20, 1.5, 4, brick],
] as const) {
  const box = new THREE.Mesh(new THREE.BoxGeometry(w, d, h), m);
  box.position.set(x, y, h / 2);
  box.castShadow = box.receiveShadow = true;
  scene.add(box);
}

const particles = new Particles(renderer);
scene.add(particles.object, particles.flash);

// The city's pipeline: the scene with a view-normal target, ambient occlusion over it
const pipeline = new THREE.RenderPipeline(renderer);
const scenePass = pass(scene, camera);
scenePass.setMRT(mrt({ output, normal: normalView }));
const colour = scenePass.getTextureNode('output');
const occlusion = ao(scenePass.getTextureNode('depth'), scenePass.getTextureNode('normal'), camera);
occlusion.resolutionScale = 0.5;
const withAO = vec4(colour.rgb.mul(occlusion.getTextureNode().r), colour.a);
pipeline.outputNode = withAO;

const wind = new THREE.Vector3(2.5, 1, 0);
const concreteDust = new THREE.Color(0.36, 0.33, 0.29);
const brickDust = new THREE.Color(0.42, 0.24, 0.17);

const actions = {
  small: () => particles.explosion([0, -6, 1], 2),
  big: () => particles.explosion([0, 0, 3], 7),
  /** A 10 × 10 × 20 section of a building crumbling (in the open): a point per voxel. */
  dust: () => {
    const points: number[] = [];
    for (let z = 0; z < 20; z++) for (let y = 0; y < 10; y++) for (let x = 0; x < 10; x++) points.push(-20 + x, -40 + y, 8 + z);
    particles.dust(points, 1, concreteDust);
  },
  impacts: () => {
    for (let i = 0; i < 8; i++) particles.impact([20 * Math.random() - 10, 20 * Math.random() - 25, 0], 6 + 14 * Math.random(), i % 2 ? brickDust : concreteDust);
  },
  smoulder: () => particles.smoulder([8 * Math.random() - 4, 8 * Math.random() - 4, 0.5], 20),
  hide: (): void => {
    particles.object.visible = !particles.object.visible;
  },
  /** Ambient occlusion on and off, to see what the particles do to it. */
  occlusion: (): void => {
    pipeline.outputNode = pipeline.outputNode === withAO ? colour : withAO;
    pipeline.needsUpdate = true;
  },
  paused: (): void => {
    lab.pause = !lab.pause;
  },
};
const keys: Record<string, keyof typeof actions> = { '1': 'small', '2': 'big', '3': 'dust', '4': 'impacts', '5': 'smoulder', v: 'hide', p: 'paused', o: 'occlusion' };
window.addEventListener('keydown', (e) => keys[e.key.toLowerCase()] && actions[keys[e.key.toLowerCase()]]());
for (const button of document.querySelectorAll<HTMLButtonElement>('button[data-action]')) button.addEventListener('click', () => actions[button.dataset.action as keyof typeof actions]());
const quality = document.getElementById('quality') as HTMLSelectElement;
quality.addEventListener('change', () => particles.setQuality(quality.value as ParticleQuality));

function resize(): void {
  renderer.setSize(window.innerWidth, window.innerHeight, false);
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);
resize();

let frameMs = 0;
let last = performance.now();
/** Three's per-frame node updates: the scene pass draws once per node frame. */
const nodeFrame = (renderer as unknown as { _nodes: { nodeFrame: { update(): void } } })._nodes.nodeFrame;
const device = (renderer.backend as unknown as { device: GPUDevice }).device;

/** Console hooks: pause, then step the simulation a fixed time and draw one frame; bench. */
const lab = {
  particles,
  camera,
  controls,
  pause: false,
  ...actions,
  benchmarking: false,
  /**
   * Milliseconds per frame (wall clock), drawn `frames` times back to back with the
   * simulation paused, the GPU drained after every ten (best of `repeats`), with the
   * particles drawn and hidden; then the time of a simulation step. The GPU is the
   * bottleneck only when it's slower than the CPU's submission, so a small difference means
   * "cheaper than the CPU side". (Three's render timestamps miss the particle draws here.)
   */
  async bench(frames = 60, repeats = 3): Promise<object> {
    lab.benchmarking = true;
    controls.update();
    camera.updateMatrixWorld();
    const run = async () => {
      let best = Infinity;
      for (let r = 0; r < repeats; r++) {
        await device.queue.onSubmittedWorkDone();
        const t0 = performance.now();
        for (let i = 0; i < frames; i++) {
          nodeFrame.update();
          pipeline.render();
          if (i % 10 === 9) await device.queue.onSubmittedWorkDone();
        }
        await device.queue.onSubmittedWorkDone();
        best = Math.min(best, (performance.now() - t0) / frames);
      }
      return best;
    };
    const shown = await run();
    particles.object.visible = false;
    const hidden = await run();
    particles.object.visible = true;
    // The simulation alone: a second of steps (this advances it)
    await device.queue.onSubmittedWorkDone();
    const t0 = performance.now();
    for (let i = 0; i < 60; i++) particles.update(1 / 60, camera, sun, wind);
    await device.queue.onSubmittedWorkDone();
    const step = (performance.now() - t0) / 60;
    lab.benchmarking = false;
    const size = renderer.getDrawingBufferSize(new THREE.Vector2());
    return { live: particles.live, size: `${size.x}x${size.y}`, shown, hidden, particleDraw: shown - hidden, step };
  },
  advance(seconds: number): void {
    controls.update();
    camera.updateMatrixWorld();
    for (let t = 0; t < seconds - 1e-6; t += 1 / 60) particles.update(1 / 60, camera, sun, wind);
    nodeFrame.update();
    pipeline.render();
  },
};
(window as unknown as { lab: typeof lab }).lab = lab;

renderer.setAnimationLoop(() => {
  if (lab.benchmarking) return;
  const now = performance.now();
  const dt = (now - last) / 1000;
  last = now;
  frameMs += (dt * 1000 - frameMs) * 0.1;
  controls.update();
  if (!lab.pause) particles.update(dt, camera, sun, wind);
  pipeline.render();
  stats.textContent = `live ≈ ${particles.live}\nframe ${frameMs.toFixed(1)} ms${particles.object.visible ? '' : '\n(particles hidden)'}`;
});
