// A bench for the city sky (sky.ts): a flat ground with a few tower blocks for scale, lit by a
// sun that matches the sky, drawn through the same pipeline as the city (MRT normals, GTAO,
// ACES). Keys: arrows move the sun (up/down: elevation, left/right: azimuth), Q cycles the
// quality, [ and ] change the cloud cover, 1-5 jump to set views, H hides the sky (to time it),
// P pauses the wind. `?w=1920&h=1080` renders at a fixed size (to time a 1080p frame), and
// `?q=low` starts at another quality.

import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { ao } from 'three/addons/tsl/display/GTAONode.js';
import { mrt, normalView, output, pass, renderOutput, vec4 } from 'three/tsl';
import * as THREE from 'three/webgpu';
import { CitySky, type SkyQuality } from './sky.ts';

const canvas = document.querySelector<HTMLCanvasElement>('#view')!;
const readout = document.querySelector<HTMLDivElement>('#readout')!;
const query = new URLSearchParams(location.search);
const fixed = query.has('w') ? { width: Number(query.get('w')), height: Number(query.get('h') ?? 1080) } : null;

const renderer = new THREE.WebGPURenderer({ canvas, antialias: false, trackTimestamp: true });
renderer.setPixelRatio(fixed ? 1 : window.devicePixelRatio);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1;
await renderer.init();
const timestamps = renderer.hasFeature('timestamp-query');

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(65, 1, 0.1, 4000);
camera.up.set(0, 0, 1);
camera.position.set(0, 0, 1.7);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.target.set(0, 1, 1.9);

// The sun: elevation and azimuth (degrees, azimuth from +x towards +y)
const sunAngles = { elevation: 8, azimuth: 60 };
const sunDirection = (out: THREE.Vector3) => {
  const [e, a] = [THREE.MathUtils.degToRad(sunAngles.elevation), THREE.MathUtils.degToRad(sunAngles.azimuth)];
  return out.set(Math.cos(e) * Math.cos(a), Math.cos(e) * Math.sin(a), Math.sin(e));
};
const sky = new CitySky(sunDirection(new THREE.Vector3()));
scene.add(sky.object);

const light = new THREE.DirectionalLight();
light.castShadow = true;
light.shadow.mapSize.set(2048, 2048);
const shadowCam = light.shadow.camera;
shadowCam.left = shadowCam.bottom = -400;
shadowCam.right = shadowCam.top = 400;
shadowCam.near = 1;
shadowCam.far = 2000;
light.shadow.bias = -0.0005;
scene.add(light, light.target);
const hemisphere = new THREE.HemisphereLight(0xffffff, 0x6b6258, 0.8);
hemisphere.up.set(0, 0, 1);
hemisphere.position.set(0, 0, 1);
scene.add(hemisphere);
const fog = new THREE.Fog(0xffffff, 300, 3800);
scene.fog = fog;

// Ground and tower blocks, for scale
const ground = new THREE.Mesh(new THREE.PlaneGeometry(8000, 8000), new THREE.MeshStandardNodeMaterial({ color: 0x8c877f, roughness: 0.95 }));
ground.receiveShadow = true;
scene.add(ground);
const blockMaterial = new THREE.MeshStandardNodeMaterial({ color: 0xc9c2b6, roughness: 0.8 });
let seed = 7;
const random = () => ((seed = (seed * 16807) % 2147483647) - 1) / 2147483646;
for (let i = 0; i < 40; i++) {
  const [w, d, h] = [20 + random() * 30, 20 + random() * 30, 15 + random() ** 2 * 140];
  const a = random() * 2 * Math.PI;
  const r = 160 + random() * 300;
  const block = new THREE.Mesh(new THREE.BoxGeometry(w, d, h), blockMaterial);
  block.position.set(r * Math.cos(a), r * Math.sin(a), h / 2);
  block.castShadow = block.receiveShadow = true;
  scene.add(block);
}

// The city's pipeline: MRT normals for GTAO, occlusion applied, then tone mapping
const post = new THREE.RenderPipeline(renderer);
const scenePass = pass(scene, camera, { samples: 0 });
scenePass.setMRT(mrt({ output, normal: normalView }));
const colour = scenePass.getTextureNode('output');
const occlusion = ao(scenePass.getTextureNode('depth'), scenePass.getTextureNode('normal'), camera);
occlusion.resolutionScale = 0.5;
post.outputColorTransform = false;
post.outputNode = renderOutput(vec4(colour.rgb.mul(occlusion.getTextureNode().r), colour.a));

const qualities: SkyQuality[] = ['low', 'medium', 'high'];
let quality: SkyQuality = (query.get('q') as SkyQuality) ?? 'medium';
sky.setQuality(quality);

/** Look from the player's spot at `pitch` degrees up, `turn` degrees round from the sun. */
function view(pitch: number, turn: number): void {
  const a = THREE.MathUtils.degToRad(sunAngles.azimuth + turn);
  const p = THREE.MathUtils.degToRad(pitch);
  camera.position.set(0, 0, 1.7);
  controls.target.copy(camera.position).add(new THREE.Vector3(Math.cos(p) * Math.cos(a), Math.cos(p) * Math.sin(a), Math.sin(p)).multiplyScalar(0.5));
  controls.update();
}
const VIEWS: Record<string, [number, number]> = { '1': [4, 0], '2': [35, 20], '3': [80, 90], '4': [8, 180], '5': [15, -50] };

let paused = false;
let time = 0;
window.addEventListener('keydown', (e) => {
  if (e.key === 'ArrowUp') sunAngles.elevation = Math.min(sunAngles.elevation + 2, 90);
  else if (e.key === 'ArrowDown') sunAngles.elevation = Math.max(sunAngles.elevation - 2, -4);
  else if (e.key === 'ArrowLeft') sunAngles.azimuth -= 10;
  else if (e.key === 'ArrowRight') sunAngles.azimuth += 10;
  else if (e.key === 'q') sky.setQuality((quality = qualities[(qualities.indexOf(quality) + 1) % 3]));
  else if (e.key === '[') sky.coverage -= 0.05;
  else if (e.key === ']') sky.coverage += 0.05;
  else if (e.key === 'h') sky.object.visible = !sky.object.visible;
  else if (e.key === 'p') paused = !paused;
  else if (VIEWS[e.key]) view(...VIEWS[e.key]);
  else return;
  e.preventDefault();
});

function resize(): void {
  const [w, h] = fixed ? [fixed.width, fixed.height] : [window.innerWidth, window.innerHeight];
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);
resize();

// Frame timing: wall-clock between frames and, where the GPU can time itself, the GPU's own
// over the last 90 frames (other tabs share the GPU, so the minimum is the steadiest figure)
const gpuSamples: number[] = [];
const gpuStats = () => {
  const sorted = [...gpuSamples].sort((a, b) => a - b);
  return { min: sorted[0] ?? 0, median: sorted[sorted.length >> 1] ?? 0 };
};
let frameMs = 16;
let last = performance.now();
const clock = new THREE.Timer();

renderer.setAnimationLoop(() => {
  const now = performance.now();
  frameMs += (now - last - frameMs) * 0.05;
  last = now;
  clock.update();
  if (!paused) time += clock.getDelta();
  controls.update();

  sunDirection(sky.sun);
  sky.update(camera, time);
  light.color.copy(sky.sunColor);
  light.intensity = sky.sunIntensity();
  light.target.position.set(camera.position.x, camera.position.y, 0);
  light.position.copy(light.target.position).addScaledVector(sky.sun, 1000);
  hemisphere.color.copy(sky.horizon).multiplyScalar(1 / Math.max(sky.horizon.r, sky.horizon.g, sky.horizon.b));
  fog.color.copy(sky.horizon);

  post.render();
  if (timestamps)
    void renderer.resolveTimestampsAsync(THREE.TimestampQuery.RENDER).then((ms) => {
      if (!ms) return;
      gpuSamples.push(ms);
      if (gpuSamples.length > 90) gpuSamples.shift();
    });

  const size = renderer.getDrawingBufferSize(new THREE.Vector2());
  readout.textContent = [
    `${size.x}×${size.y}  quality ${quality}${sky.object.visible ? '' : ' (sky hidden)'}`,
    `sun ${sunAngles.elevation}° up, ${sunAngles.azimuth}°  cover ${sky.coverage.toFixed(2)}`,
    `${(1000 / frameMs).toFixed(0)} fps  ${frameMs.toFixed(1)} ms` + (timestamps ? `  GPU ${gpuStats().median.toFixed(2)} ms (min ${gpuStats().min.toFixed(2)})` : ''),
    '↑↓←→ sun · Q quality · [ ] cover · 1-5 views · H sky · P wind',
  ].join('\n');
});

Object.assign(window, {
  lab: {
    renderer,
    scene,
    sky,
    camera,
    controls,
    view,
    sunAngles,
    setQuality: (q: SkyQuality) => sky.setQuality((quality = q)),
    gpuStats,
    frameMs: () => frameMs,
  },
});
