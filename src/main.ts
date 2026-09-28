// Voxel City: walk a city built of 1 m voxels and take it apart. Each voxel is a body in the
// AVBD GPU solver (physics.ts), fixed until a tool knocks it loose (destruction.ts); what
// loses its footing falls in pieces that shatter where they land. The player walks on the CPU
// (player.ts) against the fixed voxels and the debris last read back from the GPU.

import * as THREE from 'three/webgpu';
import { Sounds } from './audio.ts';
import { blast, type BlastResult, blasts, type Hit } from './destruction.ts';
import { Hud, type Settings, type Tool } from './hud.ts';
import { bodyCapacity, CityPhysics, PARKED_BELOW } from './physics.ts';
import { type Input, Player } from './player.ts';
import { CityRenderer } from './render.ts';
import { Chips } from './chips.ts';
import { Impacts } from './impacts.ts';
import { skyline } from './skyline.ts';
import { Particles } from './particles.ts';
import { CitySky } from './sky.ts';
import { SpaceSky } from './space.ts';
import { buildStation, HUB } from './station.ts';
import { buzz, preventZoom, TouchControls, wantsTouch } from './touch.ts';
import { Structure } from './structure.ts';
import { ViewModel } from './viewmodel.ts';
import { buildCity, type City, emptyCity, isGlass, Mat, raycast, VOXEL, voxelAt } from './world.ts';
import { buildTrack, gridSlot, KERB, RUNOFF, TRACK_WIDTH, trackField, type TrackLine } from './track.ts';
import { CAR_MASS, CAR_SIZE, CLEARANCE, Car, type Drive, sweep } from './car.ts';
import { CAP_MIN, chooseProfile, type DeviceProfile, deviceKey, forgetProfile, measureFrames, measurePhysics, type Quality, saveProfile, savedProfile } from './calibrate.ts';
import { Flames } from './flames.ts';
import { SuitJets } from './jets.ts';
import { SkidMarks } from './skids.ts';
import type { Satellites } from './satellites.ts';
import { Loader } from './loader.ts';

/** What a tool does where it hits: blast radius (m), share of it turned to dust, push (m/s), reach (m), seconds between uses, held fires again. */
interface ToolSpec extends Tool {
  radius: number;
  core: number;
  push: number;
  reach: number;
  cooldown: number;
  auto: boolean;
  /** Goes off with fire and smoke. */
  explosive?: boolean;
  /** Throws a ball instead of hitting: radius (m), density, speed (m/s). */
  ball?: [number, number, number];
  /** A beam that burns away what it touches while held (it can cut a tower through). */
  laser?: boolean;
}

const TOOLS: ToolSpec[] = [
  { name: 'Hammer', icon: `<svg viewBox="0 0 24 24"><path d="M4 20l9-9"/><path d="M11 5l3-2 7 7-2 3z"/><path d="M12 8l4 4"/></svg>`, hint: 'Knock a fist-sized hole, up close', radius: 0.4, core: 0.6, push: 3, reach: 3.5, cooldown: 0.32, auto: true },
  { name: 'Laser', icon: `<svg viewBox="0 0 24 24"><path d="M3 14h7"/><rect x="10" y="11" width="8" height="6" rx="1"/><path d="M18 14h3"/><path d="M4 10l2 1M4 18l2-1"/></svg>`, hint: 'Hold to cut through anything: cut a tower through and it topples', radius: 0.34, core: 1, push: 0, reach: 250, cooldown: 0.05, auto: true, laser: true },
  { name: 'Rocket', icon: `<svg viewBox="0 0 24 24"><path d="M5 19l3-3"/><path d="M9 15l-3-3 7-7c2-2 5-2 6-2 0 1 0 4-2 6z"/><path d="M14 10l-4 4"/><path d="M7 12l-3 1 2-4 3 0"/><path d="M12 17l-1 3 4-2 0-3"/></svg>`, hint: 'Blow a hole through a wall', radius: 1.7, core: 0.45, push: 12, explosive: true, reach: 600, cooldown: 0.7, auto: false },
  { name: 'Charge', icon: `<svg viewBox="0 0 24 24"><rect x="4" y="9" width="12" height="11" rx="1.5"/><path d="M8 9V6h4v3"/><path d="M12 6c0-2 2-3 4-3"/><path d="M18 2l1 2 2 1-2 1-1 2-1-2-2-1 2-1z"/></svg>`, hint: 'Demolition charge: takes out most of a storey', radius: 3.2, core: 0.4, push: 14, explosive: true, reach: 600, cooldown: 1.2, auto: false },
  { name: 'Wrecking ball', icon: `<svg viewBox="0 0 24 24"><circle cx="12" cy="14" r="7"/><path d="M12 7V2"/><circle cx="12" cy="2.5" r="0"/></svg>`, hint: 'Throw a 2 t ball', radius: 0, core: 0, push: 0, reach: 0, cooldown: 0.5, auto: false, ball: [0.55, 3, 32] },
];

/** Hooks for the sky and the smoke and dust (sky.ts, particles.ts), when present. */
interface Effects {
  explosion(at: ArrayLike<number>, radius: number): void;
  dust(points: ArrayLike<number>, amount: number, tint: THREE.Color): void;
  impact(at: ArrayLike<number>, speed: number, tint: THREE.Color): void;
  sparks(at: ArrayLike<number>, count: number, speed: number): void;
  exhaust(at: ArrayLike<number>, amount: number): void;
  tyreSmoke(at: ArrayLike<number>, amount: number): void;
  engineSmoke(at: ArrayLike<number>, damage: number): void;
  update(dt: number, camera: THREE.Camera): void;
}
const NO_EFFECTS: Effects = { explosion() {}, dust() {}, impact() {}, sparks() {}, exhaust() {}, tyreSmoke() {}, engineSmoke() {}, update() {} };

/** Which world: the city block (the default), the race track (?scene=track) or the space station (?scene=space). */
const SCENE: 'city' | 'track' | 'space' = (['track', 'space'] as const).find((s) => s === new URLSearchParams(location.search).get('scene')) ?? 'city';
if (SCENE !== 'city') {
  const name = SCENE === 'track' ? 'Voxel Circuit' : 'Voxel Orbit';
  document.title = name;
  const title = document.querySelector('#loader h1');
  if (title) title.textContent = name;
}
/** The track's middle line (the track scene). */
let line: TrackLine | null = null;
/** The race car (the track), whether the player's in it, its body in the solver, the boost left (0..1). */
let car: Car | null = null;
let driving = false;
let carSlot = -1;
let boostLeft = 1;
/** Its engine's sound (once the audio's started). */
let engine: ReturnType<Sounds['engine']> = null;
/** The chase camera: its yaw (following the car's heading), the mouse's look round and up (eased back when let go). */
const chase = { yaw: 0, orbit: 0, lift: 0, idle: 0 };
/** Build the scene's world (and, at the track, its line). */
function buildWorld(): City {
  if (SCENE === 'city') return buildCity();
  if (SCENE === 'space') return buildStation();
  const built = buildTrack();
  line = built.line;
  return built.city;
}

const canvas = document.querySelector('#view') as HTMLCanvasElement;
/** Phones and tablets get thumb sticks and buttons (touch.ts) and start a notch lower in quality. */
const touchDevice = wantsTouch();
preventZoom();
const settings: Settings = { sensitivity: 1, fov: 80, invertY: false, quality: touchDevice ? 'medium' : 'high', graphics: 'auto', volume: 0.7, hour: 18, smoothing: false, dof: true, sunRays: true };
try {
  const saved = JSON.parse(localStorage.getItem('city.settings') ?? '{}') as Partial<Settings>;
  // Settings saved before the device tuning had a fixed quality: they start on Auto
  if (!saved.graphics) delete saved.quality;
  Object.assign(settings, saved);
} catch {
  // no saved settings
}
/** This device's tuning (calibrate.ts), once measured or recalled. */
let profile: DeviceProfile | null = null;
const sounds = new Sounds();
const view = new CityRenderer(canvas);
const hand = new ViewModel();
view.camera.add(hand.object);
view.scene.add(view.camera);
const testing = new URLSearchParams(location.search).has('test');
let playing = testing;
const hud = new Hud(
  TOOLS,
  settings,
  () => play(),
  () => applySettings(),
  () => {
    forgetProfile();
    location.reload();
  },
);
hud.showMenu(!testing);
hud.onReset = () => void rebuild().then(() => hud.toast('Reset'));
/** The thumb sticks and buttons, on touch devices (created with the input below). */
let touch: TouchControls | null = null;
hud.setLoading('Building the city…');

function applySettings(): void {
  try {
    localStorage.setItem('city.settings', JSON.stringify(settings));
  } catch {
    // storage unavailable
  }
  view.camera.fov = settings.fov;
  view.camera.updateProjectionMatrix();
  sounds.setVolume(settings.volume);
  view.sunRays.value = settings.sunRays && SCENE !== 'space' ? 1 : 0;
  // Auto: the device's own quality and resolution (before it is measured, the default)
  const auto = settings.graphics === 'auto';
  if (!auto) settings.quality = settings.graphics as Quality;
  else if (profile) settings.quality = profile.quality;
  const base = auto && profile ? profile.resolution : 1;
  if (view.quality !== settings.quality || view.baseResolution !== base) {
    view.baseResolution = base;
    view.setQuality(settings.quality);
  }
  sky?.setQuality(settings.quality);
  smoke?.setQuality(settings.quality);
  smokeLevel = settings.quality;
  setSun(settings.hour);
}

/** Capture the mouse (called straight from a click: browsers only allow it then); on touch, just play. */
function play(): void {
  sounds.resume();
  if (touch) {
    touchPlaying = true;
    playing = true;
    hud.showMenu(false);
    touch.visible = true;
    return;
  }
  const lock = canvas.requestPointerLock() as Promise<void> | undefined;
  lock?.catch(() => hud.toast('The browser would not capture the mouse: click the view to try again'));
}
/** The mouse is captured: only then does it turn the view (a free cursor would wander off). */
let captured = false;
/** Playing on a touch screen (no pointer to lock: the pause button stops). */
let touchPlaying = false;
document.addEventListener('pointerlockchange', () => {
  captured = document.pointerLockElement === canvas;
  playing = captured || testing || touchPlaying;
  hud.showMenu(!playing);
  canvas.style.cursor = captured ? 'none' : '';
  held = false;
  aiming = false;
  keys.clear();
  hud.closeWheel();
});
// A click on the view while the mouse is free captures it (and doesn't fire)
canvas.addEventListener('mousedown', (e) => {
  if (!captured) {
    e.stopPropagation();
    play();
  }
});

// Input
const keys = new Set<string>();
let held = false;
let aiming = false;
/** The tool in hand: the rocket to start with (the most fun first thing to fire). */
let tool = 2;
let wantFire = false;
/** Keys the game takes over while playing (the browser's own uses would get in the way). */
const GAME_KEYS = new Set(['Space', 'Tab', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'AltLeft', 'AltRight', 'Slash']);
addEventListener('keydown', (e) => {
  if (!playing) return;
  if (GAME_KEYS.has(e.code)) e.preventDefault();
  if (e.repeat) return;
  keys.add(e.code);
  // Tools switch by the wheel, a click or the tool wheel, never the digit keys (some browsers'
  // extensions take those before the page sees them)
  // Hold Tab: the tool wheel (the mouse aims it instead of turning the view)
  if (e.code === 'Tab' && !hud.wheelOpen) hud.openWheel();
  if (e.code === 'KeyF') pressF();
  if (e.code === 'KeyR') void rebuild();
  if (e.code === 'KeyX') {
    timeScale = timeScale === 1 ? SLOW : 1;
    hud.toast(timeScale === 1 ? 'Normal speed' : 'Slow motion');
  }
  if (e.code === 'KeyT') {
    const next = SKIES.find((s) => s.hour > settings.hour + 0.01) ?? SKIES[0];
    settings.hour = next.hour;
    applySettings();
    hud.toast(`${next.name.replace(/_\d$/, '').replace('_', ' ')} · ${Math.floor(settings.hour)}:${String(Math.round((settings.hour % 1) * 60)).padStart(2, '0')}`);
  }
});
addEventListener('keyup', (e) => {
  keys.delete(e.code);
  if (e.code === 'Tab') {
    const chosen = hud.closeWheel();
    if (chosen >= 0) selectTool(chosen, true);
  }
});
addEventListener('blur', () => {
  keys.clear();
  hud.closeWheel();
});
addEventListener('mousemove', (e) => {
  // Browsers sometimes report one huge jump as the pointer locks: ignore it
  if (Math.abs(e.movementX) > 400 || Math.abs(e.movementY) > 400) return;
  if (hud.wheelOpen && (captured || testing)) {
    hud.aimWheel(e.movementX, e.movementY);
    return;
  }
  if (!captured) return;
  look.x += e.movementX;
  look.y += e.movementY;
  hand.look(e.movementX, e.movementY);
});
/** Mouse movement since the last frame (pixels), turned into a turn of the view once a frame. */
const look = { x: 0, y: 0, sx: 0, sy: 0 };
/** F: into the car or out of it (at the track, near it), else flying or walking. */
function pressF(): void {
  if (SCENE === 'space') {
    hud.toast('No walking in orbit');
    return;
  }
  if (car && (driving || player.position.distanceTo(car.position) < GET_IN)) setDriving(!driving);
  else toggleFly();
}
/** How near the car (m) F gets in. */
const GET_IN = 8;
/** Near enough to get in, last frame (the prompt shows as the player comes up to it). */
let byCar = false;
/** Into the car (the chase view behind it) or out of it (standing by its door, on foot). */
function setDriving(on: boolean): void {
  if (!car) return;
  driving = on;
  hud.setDriving(on);
  hand.object.visible = !on;
  touch?.setFlying(on || player.flying);
  touch?.setDriving(on);
  if (on) {
    chase.yaw = car.heading;
    chase.orbit = chase.lift = 0;
    return;
  }
  const f = car.forward;
  player.position.set(car.position.x - f.y * 2.3, car.position.y + f.x * 2.3, 0);
  player.velocity.set(0, 0, 0);
  player.flying = false;
  player.yaw = car.heading;
  player.pitch = 0;
  car.wantGuns = false;
  rocketDue = false;
  byCar = true;
  hud.toast('On foot · F by the car to get back in');
}
function toggleFly(): void {
  player.flying = !player.flying;
  player.velocity.set(0, 0, 0);
  hud.toast(player.flying ? (touch ? 'Flying' : 'Flying · Shift fast · Alt slow') : 'Walking');
  touch?.setFlying(player.flying);
}
function turn(dt: number): void {
  if (driving) {
    const thumb = touch && playing ? touch.takeLook() : { x: 0, y: 0 };
    const [mx, my] = [look.x * 0.0017 + thumb.x * TOUCH_LOOK, look.y * 0.0017 + thumb.y * TOUCH_LOOK];
    look.x = look.y = 0;
    chase.orbit -= mx * settings.sensitivity;
    chase.lift = Math.max(-0.15, Math.min(0.9, chase.lift - my * settings.sensitivity * (settings.invertY ? -1 : 1)));
    if (mx || my) chase.idle = 0;
    return;
  }
  if (touch && playing) {
    // Dragging to look: the view follows the thumb (radians per pixel), a little faster for
    // quick flicks so a turn doesn't take a whole swipe
    const { x, y } = touch.takeLook();
    const speed = Math.hypot(x, y) / Math.max(dt, 1e-3);
    const k = TOUCH_LOOK * settings.sensitivity * (aiming ? 0.5 : 1) * (1 + Math.min(1, speed / 2500) * TOUCH_FLICK);
    player.yaw -= x * k;
    player.pitch = Math.max(-1.55, Math.min(1.55, player.pitch - y * k * (settings.invertY ? -1 : 1)));
    if (x || y) hand.look(x, y);
  }
  // Optional smoothing: a short exponential average (off: the movement as it came)
  const a = settings.smoothing ? 1 - Math.exp(-dt / 0.025) : 1;
  look.sx += (look.x - look.sx) * a;
  look.sy += (look.y - look.sy) * a;
  const [mx, my] = settings.smoothing ? [look.sx, look.sy] : [look.x, look.y];
  look.x = look.y = 0;
  const k = 0.0017 * settings.sensitivity * (aiming ? 0.45 : 1);
  player.yaw -= mx * k;
  player.pitch = Math.max(-1.55, Math.min(1.55, player.pitch - my * k * (settings.invertY ? -1 : 1)));
}
addEventListener('mousedown', (e) => {
  // With the tool wheel open a click picks the tool it points at (right click: never mind)
  if (hud.wheelOpen) {
    const chosen = hud.closeWheel();
    if (e.button === 0 && chosen >= 0) selectTool(chosen, true);
    return;
  }
  if (!captured) return;
  if (e.button === 0) {
    held = true;
    wantFire = true;
  }
  if (e.button === 2) aiming = true;
});
addEventListener('mouseup', (e) => {
  if (e.button === 0) held = false;
  if (e.button === 2) aiming = false;
});
addEventListener('contextmenu', (e) => e.preventDefault());
addEventListener(
  'wheel',
  (e) => {
    if (!playing) return;
    // Up/down or left/right (a trackpad's sideways swipe), whichever way it mostly goes
    const d = Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.deltaY;
    carousel(d * (e.deltaMode === 1 ? 33 : e.deltaMode === 2 ? 400 : 1));
  },
  { passive: true },
);
/**
 * The tool carousel, driven by scrolling like a picker wheel: the strip's position follows the
 * scroll exactly (either way, at once), the tool changes as it passes halfway to the next, and
 * when scrolling stops it springs to the nearest tool (a short scroll springs back, scrolling
 * back undoes a switch). Once a scroll has changed the tool, scrolling that is slowing down
 * (each event no bigger than the last: a trackpad's momentum after the fingers lift, or a
 * finger easing off) can't carry it past halfway to another tool, and once it is plainly dying
 * away (three shrinking in a row) it eases the strip into the new tool: a swipe can't run on
 * past the next tool. Anything else (a fresh swipe, a turn back, a wheel click) moves it at
 * once. A wheel click (one event of more than half a tool) lands exactly on the next tool.
 */
const SCROLL_PER_TOOL = 120;
/** The quiet (ms) that ends a scroll. */
const SETTLE = 120;
const wheel = { pos: 0, arrived: false, speed: 0, way: 0, at: 0, settle: 0, falls: 0 };
function carousel(delta: number): void {
  if (delta === 0) return;
  const now = performance.now();
  const w = wheel;
  if (now - w.at > SETTLE) {
    // A new scroll, from the tool in hand
    w.pos = tool;
    w.arrived = false;
    w.speed = 0;
    w.way = 0;
    w.falls = 0;
  }
  // Slowing down: no bigger than the last event, in quick succession, the same way (a wheel
  // click never is: its one event is most of a tool)
  const way = Math.sign(delta);
  const falling = way === w.way && now - w.at < 50 && Math.abs(delta) <= w.speed && Math.abs(delta) < SCROLL_PER_TOOL * 0.5;
  w.falls = falling ? w.falls + 1 : 0;
  w.speed = Math.abs(delta);
  w.way = way;
  w.at = now;
  clearTimeout(w.settle);
  w.settle = window.setTimeout(() => {
    w.at = 0;
    hud.release();
  }, SETTLE);
  if (w.arrived && w.falls >= 3) {
    // Dying away after a switch: eased into the new tool, not carried past it
    const off = w.pos - tool;
    w.pos = tool + Math.sign(off) * Math.max(0, Math.abs(off) - Math.abs(delta) / SCROLL_PER_TOOL);
    hud.nudge(w.pos - tool);
    return;
  }
  const n = TOOLS.length;
  w.pos = Math.max(-0.35, Math.min(n - 1 + 0.35, w.pos + delta / SCROLL_PER_TOOL));
  if (w.arrived && falling) w.pos = Math.max(tool - 0.45, Math.min(tool + 0.45, w.pos));
  const nearest = Math.max(0, Math.min(n - 1, Math.round(w.pos)));
  if (nearest !== tool) {
    selectTool(nearest, true);
    w.arrived = true;
    if (Math.abs(delta) / SCROLL_PER_TOOL >= 0.5) w.pos = tool;
  }
  hud.nudge(w.pos - tool);
}
/** Take tool `i` in hand; `announce`: the player chose it (it says so on screen and clicks). */
function selectTool(i: number, announce = false): void {
  if (announce && i !== tool) {
    sounds.click();
    buzz(8);
  }
  tool = i;
  hud.select(i, announce);
  hand.select(i);
  touch?.setTool(TOOLS[i].icon);
}
hud.onPick = (i) => selectTool(i, true);
hud.onSwipe = (tools) => carousel(tools * SCROLL_PER_TOOL);
selectTool(TOOLS.findIndex((t) => t.name === 'Rocket'));

function input(): Input {
  const k = (...codes: string[]) => (codes.some((c) => keys.has(c)) ? 1 : 0);
  const clamp = (v: number) => Math.max(-1, Math.min(1, v));
  // The touch sticks and buttons add to the keys
  const stick = touch?.move ?? { forward: 0, right: 0 };
  const up = touch?.up ? 1 : 0;
  const down = touch?.down ? 1 : 0;
  return {
    forward: clamp(k('KeyW', 'ArrowUp') - k('KeyS', 'ArrowDown') + stick.forward),
    right: clamp(k('KeyD', 'ArrowRight') - k('KeyA', 'ArrowLeft') + stick.right),
    jump: keys.has('Space') || up === 1,
    sprint: k('ShiftLeft', 'ShiftRight') === 1 || touch?.sprint === true,
    crouch: k('KeyC') === 1 || down === 1,
    rise: clamp(k('Space', 'KeyE') - k('KeyQ', 'KeyC') + up - down),
    slow: k('AltLeft', 'AltRight') === 1,
  };
}

/**
 * Touch controls (phones and tablets, touch.ts): the stick moves, a thumb dragged on the right
 * looks, the fire button uses the tool in hand, the tool bar swipes.
 */
/**
 * Look speed for a dragged thumb (radians per pixel, before sensitivity: 100 px turns about
 * 14 degrees, a landscape phone's width about 110), and the extra for quick flicks.
 */
const TOUCH_LOOK = 0.0024;
const TOUCH_FLICK = 0.25;
/** On touch, a gentler flight: speed (m/s) and boost (the stick pushed past its ring). */
const TOUCH_FLY_SPEED = 4.5;
const TOUCH_BOOST = 2.2;
touch = touchDevice
  ? new TouchControls({
      fire: (down) => {
        held = down;
        if (down) wantFire = true;
      },
      aim: (on) => (aiming = on),
      toggleFly: () => pressF(),
      pause: () => {
        touchPlaying = false;
        playing = captured || testing;
        held = false;
        aiming = false;
        touch!.visible = false;
        hud.showMenu(true);
      },
    })
  : null;
if (touch) {
  hud.setTouch();
  touch.visible = testing;
  touch.setTool(TOOLS[tool].icon);
}

// The world
// Built while the page loads (start): empty until then
let city = emptyCity();
let structure = new Structure(city);
const player = new Player(city);
touch?.setFlying(player.flying);
if (touch) {
  player.flySpeed = TOUCH_FLY_SPEED;
  player.boost = TOUCH_BOOST;
}
let physics: CityPhysics;
let effects: Effects = NO_EFFECTS;
const dustTint: Record<number, THREE.Color> = {
  [Mat.Brick]: new THREE.Color(0.5, 0.42, 0.36),
  [Mat.Glass]: new THREE.Color(0.7, 0.72, 0.72),
  [Mat.LitGlass]: new THREE.Color(0.7, 0.72, 0.72),
};
const concreteDust = new THREE.Color(0.62, 0.6, 0.56);

/** Start on the pavement off the tower's rounded corner, looking up at it: the shopfront and the blade sign in view, the tower rising overhead. */
function spawn(): void {
  player.velocity.set(0, 0, 0);
  if (SCENE === 'space') {
    // Floating off the station's corner, the truss and its wings across the view, the Earth below
    player.zeroG = true;
    player.flying = true;
    player.position.set(HUB[0] - 34, HUB[1] - 30, HUB[2] + 3);
    player.yaw = Math.atan2(HUB[1] - player.position.y, HUB[0] - player.position.x);
    player.pitch = -0.2;
    return;
  }
  if (SCENE === 'track' && line) {
    // On the grid behind the start line, looking down the main straight at the wall across it
    const { position, heading } = gridSlot(line);
    player.position.set(position[0], position[1], 0);
    player.yaw = heading;
    player.pitch = 0.05;
    return;
  }
  player.position.set(-27, -12, 0);
  player.yaw = 0.3;
  player.pitch = 0.38;
}

/** The stages of starting up, as the loading bar shows them (weights: their share of the bar). */
const loader = new Loader([
  { name: 'Starting the GPU and loading materials', weight: 4 },
  { name: SCENE === 'track' ? 'Building the track' : SCENE === 'space' ? 'Building the station' : 'Building the building', weight: 2 },
  { name: 'Starting the physics', weight: 2 },
  { name: 'Lighting the sky', weight: 1 },
  { name: 'Compiling shaders', weight: 3 },
  { name: 'Tuning for this device', weight: 3 },
]);

async function start(): Promise<void> {
  await loader.stage(0);
  const device = await view.init((share) => loader.progress(share));

  await loader.stage(1);
  city = buildWorld();
  structure = new Structure(city);
  (player as unknown as { city: typeof city }).city = city;
  chips.city = city;
  shards.city = city;

  await loader.stage(2);
  const capacity = bodyCapacity(city);
  const buffer = view.attachVoxels(capacity, city);
  physics = new CityPhysics(device, city, buffer);
  physics.onCarry = (bodies, parent, offsets) => view.setLinks(bodies, parent, offsets);
  view.setPaints(city, physics.voxelOf);
  view.attachBalls(physics.spares);
  if (SCENE === 'track' && line) {
    // Open country: the air clearer than over the city, so the far side of the circuit shows
    (view.scene.fog as THREE.FogExp2).density = 0.0011;
    const extent = 240;
    view.setTrack(trackField(line, 1024, extent), 1024, extent, TRACK_WIDTH, KERB, RUNOFF, line.points[2 * line.start], [-160, -82, -95, -60]);
  } else if (SCENE === 'space') view.setSpace();
  else {
    view.setStreets(city);
    view.scene.add(skyline(city));
  }
  effects = startEffects();
  if (SCENE === 'space') {
    // Vacuum: chips and smoke fly on as thrown, nothing falls, nothing to land on, sound's muffled
    chips.vacuum = shards.vacuum = true;
    sounds.muffled = true;
    suitJets = new SuitJets(VENTS.length);
    view.scene.add(suitJets.object);
    if (smoke) {
      smoke.gravity.value = 0;
      smoke.air.value = 0;
      smoke.groundHeight = -1e5;
    }
  }
  spawn();
  if (SCENE === 'track' && line) {
    car = new Car();
    flames = new Flames(car.exhausts().at.length);
    skids = new SkidMarks(2);
    view.scene.add(car.object, car.bits.object, flames.object, skids.object);
    placeCar();
    setDriving(true);
  }

  await loader.stage(3);
  await startSky();
  view.camera.position.set(player.position.x, player.position.y, 1.7);

  await loader.stage(4);
  // The scene's and the smoke's materials compile without blocking the page, then a frame
  // makes the shadow map (the sun shafts need it)
  await view.renderer.compileAsync(view.scene, view.camera);
  loader.progress(0.5);
  await view.renderer.compileAsync(view.smokeScene, view.camera);
  loader.progress(0.7);
  view.warm();
  applySettings();
  // A small blast far below the ground, unseen, so the fire, smoke and dust pipelines compile
  // now (the first real one used to drop a second of frames on them); it is cleared after
  effects.explosion([0, 0, -400], 0.3);
  effects.dust(new Float32Array([0, 0, -400]), 0.2, concreteDust);
  effects.impact([0, 0, -400], 10, concreteDust);
  for (let k = 0; k < 3; k++) {
    effects.update(1 / 60, view.camera);
    view.render();
  }
  smoke?.clear();

  await loader.stage(5);
  const key = deviceKey(view.adapterInfo, SCENE);
  profile = new URLSearchParams(location.search).has('retune') ? null : savedProfile(key);
  if (profile) loader.detail('Tuned for this device already');
  else {
    profile = await tune(device, key);
    saveProfile(profile);
  }
  applyProfile(profile);

  await loader.finish();
  hud.setLoading(null);
  if (playing) hud.showMenu(false);
  // The tuning ran the game on a clock of its own: the real one starts now
  last = performance.now();
  accumulator = 0;
  requestAnimationFrame(frame);
}

/**
 * Measure this device (calibrate.ts): a physics step of a pile of loose voxels, then heavy
 * frames (a blast's smoke and dust in front of the building, the game running) at each quality
 * until one fits.
 */
async function tune(device: GPUDevice, key: string): Promise<DeviceProfile> {
  loader.detail('Timing the physics…');
  const physicsTime = await measurePhysics(device, (share) => loader.progress(share * 0.35));
  loader.detail('Timing the drawing…');
  // A charge's smoke and a collapse's dust in front of the building, seen from the street
  spawn();
  // (In front of the spawn point, wherever the scene puts it)
  const ahead = new THREE.Vector3(Math.cos(player.yaw), Math.sin(player.yaw), 0).multiplyScalar(18).add(player.position);
  const off = SCENE === 'city' ? new THREE.Vector3(0, -7.5, 0) : ahead;
  effects.explosion([off.x, off.y, 5], 4.5);
  const dust: number[] = [];
  for (let k = 0; k < 400; k++) dust.push(off.x + (Math.random() - 0.5) * 16, off.y - Math.random() * 4, Math.random() * 6);
  effects.dust(Float32Array.from(dust), 0.6, concreteDust);
  const step = () => tick(last + 1000 / 60);
  for (let k = 0; k < 40; k++) step();
  const chosen = await chooseProfile(
    key,
    physicsTime,
    async (quality, resolution) => {
      loader.detail(`Timing the drawing: ${quality}${resolution < 1 ? `, ${Math.round(resolution * 100)}% resolution` : ''}…`);
      settings.quality = quality;
      view.baseResolution = resolution;
      view.setQuality(quality);
      sky.setQuality(quality);
      smoke?.setQuality(quality);
      return measureFrames(device, step);
    },
    (share) => loader.progress(0.35 + share * 0.65),
  );
  smoke?.clear();
  spawn();
  return chosen;
}

/** Use the device's tuning: the physics' limits, the particles' budget, and (under Auto) the drawing. */
function applyProfile(p: DeviceProfile): void {
  fullLooseCap = p.looseCap;
  setLoad(loadStep);
  physics.solver.params.iterations = p.iterations;
  if (smoke) smoke.budget = p.particles;
  applySettings();
  hud.setTuned(`This device: ${p.quality} quality${p.resolution < 1 ? ` at ${Math.round(p.resolution * 100)}% resolution` : ''}, up to ${p.looseCap.toLocaleString()} loose voxels`);
}

/**
 * The scene as it started, without reloading the page (the shaders stay compiled): a new world
 * and physics, nothing in flight or pending, the smoke and chips cleared, the player (and the
 * car) back where they began.
 */
async function rebuild(): Promise<void> {
  alarms.clear();
  for (const r of rockets.splice(0)) view.scene.remove(r.mesh);
  pending.length = charges.length = blows.length = crushes.length = 0;
  chips.clear();
  shards.clear();
  smoke?.clear();
  skids?.clear();
  dustiness = 0;
  physics.destroy();
  city = buildWorld();
  structure = new Structure(city);
  (player as unknown as { city: typeof city }).city = city;
  chips.city = city;
  shards.city = city;
  damage = new Map();
  const buffer = view.attachVoxels(bodyCapacity(city), city);
  const device = (view.renderer.backend as unknown as { device: GPUDevice }).device;
  physics = new CityPhysics(device, city, buffer);
  physics.onCarry = (bodies, parent, offsets) => view.setLinks(bodies, parent, offsets);
  if (profile) physics.solver.params.iterations = profile.iterations;
  view.setPaints(city, physics.voxelOf);
  view.attachBalls(physics.spares);
  player.debrisCount = 0;
  spawn();
  if (car) {
    placeCar();
    setDriving(true);
  }
}

/** The car back on the grid, at rest, with its body in the (new) solver. */
function placeCar(): void {
  if (!car || !line) return;
  const { position, heading } = gridSlot(line);
  car.place(position[0], position[1], heading);
  car.repair();
  carSlot = physics.vehicle(CAR_SIZE, [position[0], position[1], CLEARANCE + CAR_SIZE[2] / 2]);
  boostLeft = 1;
  chase.yaw = heading;
}

/** The car's controls: keys and the thumb stick (throttle up, steer across), or coasting to a stop with no one in it. */
function driveInput(): Drive {
  // No one in it: it brakes to a stop (full brakes, not a crawl on the handbrake)
  if (!driving || !playing) {
    const along = car ? car.speed : 0;
    return { throttle: along > 0.3 ? -1 : along < -0.3 ? 1 : 0, steer: 0, handbrake: false, boost: false };
  }
  const k = (...codes: string[]) => (codes.some((c) => keys.has(c)) ? 1 : 0);
  const clamp = (v: number) => Math.max(-1, Math.min(1, v));
  const stick = touch?.move ?? { forward: 0, right: 0 };
  return {
    throttle: clamp(k('KeyW', 'ArrowUp') - k('KeyS', 'ArrowDown') + stick.forward),
    steer: clamp(k('KeyA', 'ArrowLeft') - k('KeyD', 'ArrowRight') - stick.right),
    handbrake: keys.has('Space') || touch?.up === true,
    boost: (keys.has('KeyB') || touch?.sprint === true || touch?.down === true) && boostLeft > 0,
  };
}

/**
 * One world step of the car: its handling in parts of no more than a third of a metre (a fast
 * car can't pass through a half-metre wall between checks), each part checked against what
 * still stands: broken through if the car carries enough momentum, else it stops there and
 * bounces back. Then its body in the solver is set moving from where it was to where it is.
 */
function driveStep(dt: number): void {
  const c = car!;
  const d = driveInput();
  boostLeft = Math.max(0, Math.min(1, boostLeft + (d.boost ? -dt * 0.22 : dt * 0.08)));
  c.begin();
  const [x0, y0, h0] = [c.position.x, c.position.y, c.heading];
  const parts = Math.max(1, Math.ceil((c.velocity.length() * dt) / 0.33));
  // Already into something (wedged, or a wall left standing round it)? A move that goes no further in is free:
  // it can always drive out, or along, never deeper
  let inside = sweep(city, c.position.x, c.position.y, c.heading).length;
  for (let k = 0; k < parts; k++) {
    c.step(dt / parts, d);
    const hits = sweep(city, c.position.x, c.position.y, c.heading);
    if (hits.length && hits.length <= inside) {
      inside = hits.length;
      continue;
    }
    if (hits.length && !crash(hits)) {
      c.undo(0.2);
      break;
    }
  }
  if (carSlot >= 0) {
    const turn = Math.atan2(Math.sin(c.heading - h0), Math.cos(c.heading - h0));
    physics.drive(carSlot, [x0, y0, CLEARANCE + CAR_SIZE[2] / 2], [0, 0, Math.sin(h0 / 2), Math.cos(h0 / 2)], [(c.position.x - x0) / dt, (c.position.y - y0) / dt, 0], [0, 0, turn / dt]);
  }
}

/**
 * The car runs into `hits` (fixed voxels): it breaks through if it keeps enough speed after
 * shoving their mass (its momentum shared with what it hits), blowing them out ahead of it
 * across the width of its nose, and slows by that much; else it's stopped (false).
 */
function crash(hits: number[]): boolean {
  const c = car!;
  const v = c.velocity.length();
  let mass = 0;
  for (const h of hits) mass += isGlass(city.material[h]) ? 6 : city.material[h] === Mat.Leaf ? 2 : 28;
  const kept = CAR_MASS / (CAR_MASS + 0.7 * mass);
  if (v * kept < 3.5) {
    if (v > 2) {
      sounds.hammer(2);
      shake = Math.max(shake, Math.min(0.8, v / 20));
    }
    // Stopped dead: the harder it hit, the more it's hurt (a nudge at walking pace, none), where it hit
    hurtCar(Math.max(0, v - 5) * STOP_DAMAGE, struck(hits));
    return false;
  }
  // Where it's going into things (nose, tail or side: the middle of what it struck), and across its way there
  const f = new THREE.Vector2(c.velocity.x, c.velocity.y).normalize();
  const nose = struck(hits).setZ(CLEARANCE + 0.55);
  const at = (h: number) => Array.from(city.position.subarray(3 * h, 3 * h + 3));
  const near = (a: number[], b: number[], r: number) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2 < r * r;
  const hitsAt: Hit[] = [];
  for (const o of [-0.7, 0, 0.7]) {
    const p = [nose.x - f.y * o, nose.y + f.x * o, nose.z];
    if (hits.some((h) => near(at(h), p, 1.3))) hitsAt.push({ at: p, radius: 0.95, push: v * 0.5, core: 0.12 });
  }
  // What it touched beyond those (a corner clipping a post) goes too
  for (const h of hits.slice(0, 40)) if (!hitsAt.some((b) => near(at(h), b.at, 0.95))) hitsAt.push({ at: at(h), radius: 0.3, push: v * 0.4, core: 0 });
  const result = blasts(city, structure, physics, hitsAt);
  unsettled = true;
  afterBlast(result, hitsAt);
  chips.burst(result.gone, nose.toArray(), 2, v * 0.6 + 2);
  chips.burst(result.loose, nose.toArray(), 0.6, v * 0.5 + 2);
  const tint = dustTint[city.material[hits[0]]] ?? concreteDust;
  effects.impact(nose.toArray(), Math.min(20, v), tint);
  effects.sparks(nose.toArray(), 10, 4 + v * 0.2);
  const broken = [...result.gone, ...result.loose];
  const points = new Float32Array(broken.length * 3);
  broken.forEach((u, k) => points.set(city.position.subarray(3 * u, 3 * u + 3), 3 * k));
  if (broken.length) effects.dust(points, 0.25, tint);
  if (result.falling.length > 50) sounds.collapse(result.falling.length, 6);
  sounds.wreck(3, Math.min(40, v));
  shake = Math.max(shake, Math.min(1.1, 0.25 + v / 45));
  c.velocity.multiplyScalar(kept);
  // Through, but hurt by as much speed as what it hit took off it, where it hit
  hurtCar(v * (1 - kept) * THROUGH_DAMAGE, struck(hits));
  return true;
}

/** Where the car met what it ran into: the middle of the voxels it struck (world). */
function struck(hits: number[]): THREE.Vector3 {
  const at = new THREE.Vector3();
  for (const h of hits) at.add(new THREE.Vector3(city.position[3 * h], city.position[3 * h + 1], city.position[3 * h + 2]));
  return at.divideScalar(hits.length);
}

/** Damage (a share of a new car's health) a metre a second of speed lost stopping dead, and ploughing through. */
const STOP_DAMAGE = 0.022;
const THROUGH_DAMAGE = 0.02;

/**
 * The car hurt by `amount` (a share of its health) by a blow at `at`: paint and sparks off it, and at nothing
 * left, wrecked.
 */
function hurtCar(amount: number, at: THREE.Vector3): void {
  const c = car;
  if (!c || c.health <= 0 || amount < 0.01) return;
  c.hurt(amount, at);
  effects.sparks(at.toArray(), Math.round(4 + amount * 40), 6);
  effects.impact(at.toArray(), 4 + amount * 30, paintFlakes);
  if (driving) hud.flash(Math.min(0.5, amount * 1.5));
  if (c.health <= 0) wreckCar();
}
const paintFlakes = new THREE.Color(0.5, 0.58, 0.68);

/**
 * The car wrecked: it goes up (a charge's blast where it stood, wrecking what's round it), whoever's driving is
 * thrown clear on foot, and a new car waits on the grid.
 */
function wreckCar(): void {
  const c = car!;
  const at = new THREE.Vector3(c.position.x, c.position.y, CLEARANCE + 0.6);
  const f = c.forward;
  if (driving) {
    setDriving(false);
    player.velocity.set(-f.y * 5, f.x * 5, 7);
  }
  detonate(at, TOOLS[3], player.position.distanceTo(at));
  placeCar();
  hud.toast('Wrecked · a new car waits on the grid');
}

/** A damaged engine smokes, darker the worse it is, and past half its health gone it's on fire. */
let engineSmokeDue = 0;
function engineSmoke(dt: number): void {
  const c = car!;
  if (c.health > 0.55) return;
  const damage = (0.55 - c.health) / 0.55;
  engineSmokeDue -= dt * (0.5 + 2 * damage);
  if (engineSmokeDue > 0) return;
  engineSmokeDue = 0.06;
  effects.engineSmoke(c.engine().toArray(), damage);
}

/**
 * The car's weapons. Rocket launchers (left button): a rocket's blast, each launcher reloading
 * for a while after its shot, so two quick ones then a wait. Machine guns (right button): a
 * quick crack each, alternating barrels, knocking a fist-sized bite out of what they hit, their
 * tracers flying fast.
 */
const BULLET: ToolSpec = { name: 'Machine gun', icon: '', hint: '', radius: 0.32, core: 0.55, push: 5, reach: 400, cooldown: 0.075, auto: true };
const CAR_ROCKET: ToolSpec = { name: 'Rocket', icon: '', hint: '', radius: 1.8, core: 0.45, push: 13, explosive: true, reach: 600, cooldown: 0.3, auto: true };
const BULLET_SPEED = 420;
const CAR_ROCKET_SPEED = 85;
/** Seconds a launcher takes to reload after its shot. */
const RELOAD = 1.6;
let gunCooldown = 0;
let barrel = 0;
/** When each launcher is loaded again (s, the world's clock), and which fires next. */
const loaded = [0, 0];
let launcher = 0;
/** Seconds since each weapon last fired (they fold away after a while). */
let gunsIdle = 0;
let rocketsIdle = 0;
/** A click's rocket, waiting for the launchers to swing up. */
let rocketDue = false;
/** The camera's eye and aim this frame (the weapons fire where it looks). */
const aimFrom = new THREE.Vector3();
const aimDir = new THREE.Vector3(1, 0, 0);
/** Where the crosshair is on something (and where), for a shot from `from`. */
function aimed(spec: ToolSpec): { to: THREE.Vector3; hit: boolean } {
  const hit = raycast(city, aimFrom.toArray(), aimDir.toArray(), spec.reach);
  const to = aimFrom.clone().addScaledVector(aimDir, hit ? hit.t : spec.reach * 0.7);
  if (hit) to.addScaledVector(new THREE.Vector3(...hit.normal), -0.2);
  // (The ground counts: a shot into the road goes off there)
  return { to, hit: !!hit };
}
function fireGun(): void {
  const muzzle = car!.muzzles().guns[barrel];
  barrel = 1 - barrel;
  const { to, hit } = aimed(BULLET);
  rockets.push({ from: muzzle, to, t: 0, time: muzzle.distanceTo(to) / BULLET_SPEED, mesh: tracerMesh(), spec: hit ? BULLET : null, quiet: true });
  sounds.gun(3);
  shake = Math.max(shake, 0.08);
  effects.sparks(muzzle.toArray(), 3, 5);
  impacts.hit(muzzle.toArray(), aimDir.toArray(), 0.22, 0.05, hotRing, 0.12);
}
function fireRocket(): void {
  const muzzle = car!.muzzles().rockets[launcher];
  loaded[launcher] = simTime + RELOAD;
  launcher = 1 - launcher;
  const { to, hit } = aimed(CAR_ROCKET);
  rockets.push({ from: muzzle, to, t: 0, time: muzzle.distanceTo(to) / CAR_ROCKET_SPEED, mesh: rocketMesh(0.8), spec: hit ? CAR_ROCKET : null });
  sounds.zap();
  shake = Math.max(shake, 0.22);
  effects.impact(muzzle.toArray(), 4, smokeTint);
}
/** A machine gun's tracer: a short bright streak. */
function tracerMesh(): THREE.Object3D {
  const m = new THREE.Mesh(new THREE.BoxGeometry(0.03, 0.03, 1.4), new THREE.MeshBasicMaterial({ color: new THREE.Color(5, 3.2, 1.2) }));
  view.scene.add(m);
  return m;
}
/**
 * In orbit, the suit's jets (jets.ts): plumes from the vents on its backpack against the push (so they fire as you
 * set off, stop or turn, bigger the harder you push), and their rush of gas for as long as you steer. The vents are
 * at the pack's corners, behind and beside the eye, so the gas streams into view from the suit, its vent out of
 * sight (m: forward, out to the side, down). Only those facing the way the gas goes fire (none blow across you).
 */
const VENTS = [
  [-0.1, 0.3, -0.3],
  [-0.1, -0.3, -0.3],
  [-0.1, 0.3, -0.6],
  [-0.1, -0.3, -0.6],
];
let suitJets: SuitJets | null = null;
function jets(dt: number, steering: boolean): void {
  const push = player.thrust.length();
  const amount = push < 0.5 ? 0 : Math.min(push / 15, 1);
  // (Heard only while you steer: the suit braking itself to a stop after is silent)
  if (steering && amount > 0) sounds.jet(amount);
  const away = player.thrust.clone().divideScalar(-Math.max(push, 1e-6));
  const eye = player.eye(new THREE.Vector3());
  const [c, s] = [Math.cos(player.yaw), Math.sin(player.yaw)];
  const ahead = Math.abs(away.x * c + away.y * s);
  const vents = VENTS.map(([f, side, down]) => new THREE.Vector3(eye.x + c * f + s * side, eye.y + s * f - c * side, eye.z + down));
  const amounts = VENTS.map(([, side, down]) => {
    // The way the vent faces off the pack: out to its side, and up at the shoulders or down at the hips
    const facing = new THREE.Vector3(s * side, -c * side, down > -0.45 ? 0.25 : -0.25).normalize();
    return amount * Math.max(ahead, THREE.MathUtils.smoothstep(facing.dot(away), -0.2, 0.3));
  });
  suitJets?.update(dt, eye, player.velocity, vents, amounts, away);
}

/** The race car's afterfire (flames.ts) and the marks its tyres leave (skids.ts). */
let flames: Flames | null = null;
let skids: SkidMarks | null = null;

/** The afterfire at a gear change, and the exhaust's wisps (from each pipe in turn, more under throttle). */
let upshifts = 0;
let smokeDue = 0;
function exhaust(dt: number, throttle: number): void {
  const c = car!;
  const { at, back } = c.exhausts();
  if (c.shifts !== upshifts) {
    // The afterfire: a flame out of every pipe, a few sparks, a pop and a flare of the bloom (the car and the
    // camera carry on smoothly)
    upshifts = c.shifts;
    flames?.fire();
    for (const p of at) effects.sparks(p.toArray(), 2, 3.5);
    sounds.pop(driving ? 4 : eye.distanceTo(c.position));
    view.glow = Math.max(0.6, (view as unknown as { bloomPass: { strength: { value: number } } }).bloomPass.strength.value);
  }
  flames?.update(dt, at, back);
  // Under throttle only (clear at idle), a pipe at a time, thinning out at speed where it's left behind at once
  smokeDue -= dt * throttle * (Math.abs(c.speed) < 45 ? 1 : 0.4);
  if (smokeDue <= 0) {
    smokeDue = 0.07;
    effects.exhaust(at[Math.floor(Math.random() * at.length)].toArray(), throttle * 0.5);
  }
}

/**
 * The rear tyres sliding (drifting, or locked by the handbrake): black marks laid on the track, white smoke
 * rolling out from under them, and their screech; the harder they slide, the more of each.
 */
let tyreSmokeDue = 0;
function tyres(dt: number): void {
  const c = car!;
  const at = c.rearTyres();
  skids?.update(at, c.skid, simTime);
  if (c.skid < 0.05) return;
  sounds.skid(c.skid);
  tyreSmokeDue -= dt * (0.4 + c.skid);
  if (tyreSmokeDue > 0) return;
  tyreSmokeDue = 0.05;
  for (const p of at) effects.tyreSmoke(p.toArray(), c.skid);
}

/** The chase camera behind the car: its eye and the way it looks (into `eye`, `dir`). */
function chaseView(dt: number): void {
  const c = car!;
  const p = c.object.position;
  const heading = c.object.rotation.z;
  const turnBy = Math.atan2(Math.sin(heading - chase.yaw), Math.cos(heading - chase.yaw));
  chase.yaw += turnBy * Math.min(1, dt * 4.5);
  chase.idle += dt;
  if (chase.idle > 1.2) {
    chase.orbit *= Math.exp(-dt * 2.5);
    chase.lift *= Math.exp(-dt * 2.5);
  }
  const speed = Math.abs(c.speed);
  const yaw = chase.yaw + chase.orbit;
  // A gear change nudges the view back a touch (the car surging after the lost beat of drive)
  const dist = 6.4 + Math.min(3, speed * 0.035);
  const pitch = 0.19 + chase.lift;
  eye.set(p.x - Math.cos(yaw) * dist * Math.cos(pitch), p.y - Math.sin(yaw) * dist * Math.cos(pitch), 1.1 + dist * Math.sin(pitch));
  eye.z = Math.max(0.5, eye.z);
  // Something standing between it and the car: in front of that instead
  const from = new THREE.Vector3(p.x, p.y, 1.3);
  const back = eye.clone().sub(from);
  const reach = back.length();
  back.divideScalar(reach);
  const blocked = raycast(city, from.toArray(), back.toArray(), reach);
  if (blocked && blocked.voxel >= 0) eye.copy(from).addScaledVector(back, Math.max(1.5, blocked.t - 0.4));
  // Looking well ahead (the crosshair near the horizon, where the weapons reach), over the car
  dir.set(p.x + Math.cos(yaw) * 40 - eye.x, p.y + Math.sin(yaw) * 40 - eye.y, 1.5 - eye.z).normalize();
}

/**
 * Smoke, dust and fire (particles.ts): fireballs light the buildings around them (one light,
 * added once, so no material recompiles), the smoke is lit by the sky's sun and horizon, and
 * blown by a steady breeze.
 */
function startEffects(): Effects {
  const particles = new Particles(view.renderer, undefined, view.sceneDepth);
  view.smokeScene.add(particles.object);
  view.scene.add(particles.flash);
  smoke = particles;
  const wind = SCENE === 'space' ? new THREE.Vector3() : new THREE.Vector3(1.6, 0.8, 0);
  const sun = new THREE.Color();
  return {
    explosion: (at, radius) => {
      particles.explosion(at, radius);
      if (radius >= 5) particles.smoulder([at[0], at[1], Math.max(0.5, at[2] - 1)], 20);
    },
    dust: (points, amount, tint) => particles.dust(points, amount, tint),
    impact: (at, speed, tint) => particles.impact(at, speed, tint),
    sparks: (at, count, speed) => particles.sparks(at, count, speed),
    exhaust: (at, amount) => particles.exhaust(at, amount),
    tyreSmoke: (at, amount) => particles.tyreSmoke(at, amount),
    engineSmoke: (at, damage) => particles.engineSmoke(at, damage),
    update: (dt, camera) => {
      if (sky) {
        sun.copy(sky.sunColor).multiplyScalar(0.45 * sky.sunIntensity());
        particles.sunColour.value.copy(sun);
        particles.skyAmbient.value.copy(sky.horizon).multiplyScalar(0.75);
      }
      particles.update(dt, camera, view.sunDirection, wind);
    },
  };
}

// The sky: its sun lights the city, its horizon colours the fog, and its colours the reflections
/** What the frame needs of the sky: the city's (sky.ts) or orbit's (space.ts). */
interface Sky {
  readonly object: THREE.Object3D;
  readonly sun: THREE.Vector3;
  readonly horizon: THREE.Color;
  readonly sunColor: THREE.Color;
  sunIntensity(): number;
  update(camera: THREE.Camera, time: number): void;
  setQuality(quality: Settings['quality']): void;
}
let sky: Sky;
/**
 * The smoke and dust, and the quality it's drawn at: the setting's, a step lower while a
 * thick cloud slows frames. The smoke has its own reduced-resolution pass and a budget of
 * live particles (particles.ts), so a collapse now costs at most a refresh here and there; if
 * frames are still slow for a while, the physics sheds loose voxels (the resolution stays full),
 * and takes them back once they are quick again.
 */
let smoke: Particles | null = null;
let smokeLevel: Settings['quality'] = 'high';
let frameMs = 16;
let slowFor = 0;
let quickFor = 0;
/**
 * The display's own frame interval (ms): the shortest of the last second or so of frames. Frames come no quicker
 * than it, so on a 60 Hz screen (a phone's, most monitors) every frame takes 16.7 ms however little it costs: slow
 * is dropping frames (longer than that), and quick is measured by the work itself (below), not the frames.
 */
const intervals = new Float32Array(90).fill(1000 / 60);
let intervalAt = 0;
/** How long a frame's work takes (ms): from the frame starting to the GPU finishing it (sampled, smoothed). */
let workMs = 8;
let timingWork = false;
function timeWork(device: GPUDevice, started: number): void {
  if (timingWork) return;
  timingWork = true;
  void device.queue.onSubmittedWorkDone().then(() => {
    workMs += (performance.now() - started - workMs) * 0.2;
    timingWork = false;
  });
}
/**
 * The physics' load steps, as shares of the device's loose voxels (profile.looseCap): if frames are still slow, the
 * debris is thinned a step (over the cap, the farthest settles as rubble or goes in a puff of chips) rather than the
 * resolution lowered, and allowed back once they are quick again.
 */
const LOADS = [1, 0.7, 0.45];
let loadStep = 0;
/** The device's loose voxels at most (its profile's), before the load steps. */
let fullLooseCap = 8000;
function setLoad(step: number): void {
  loadStep = step;
  looseCap = Math.max(CAP_MIN, Math.round(fullLooseCap * LOADS[step]));
}
function adaptSmoke(dt: number): void {
  if (!smoke) return;
  frameMs += (dt * 1000 - frameMs) * 0.12;
  intervals[intervalAt++ % intervals.length] = dt * 1000;
  const display = Math.max(1000 / 240, Math.min(...intervals));
  // Slow: below 60 fps, or dropping the display's frames; quick: time to spare (on a 60 Hz display the frames
  // can't show it, the work can)
  const slow = Math.max(15, display * 1.4);
  const slower = Math.max(12.5, display * 1.2);
  const quick = display < 12 ? frameMs < 9.5 : workMs < 10 && frameMs < display * 1.1;
  const lower: Record<Settings['quality'], Settings['quality']> = { high: 'medium', medium: 'low', low: 'low' };
  const want = frameMs > slow + 0.5 ? 'low' : frameMs > slower ? lower[settings.quality] : quick ? settings.quality : smokeLevel;
  if (want !== smokeLevel) {
    smokeLevel = want;
    smoke.setQuality(want);
  }
  slowFor = frameMs > slow ? slowFor + dt : 0;
  quickFor = quick ? quickFor + dt : 0;
  // (A second of slow frames, not a moment's: a collapse's brief dip shouldn't cost it its debris)
  if (slowFor > 1 && loadStep < LOADS.length - 1) {
    setLoad(loadStep + 1);
    slowFor = 0;
  } else if (quickFor > 4 && loadStep > 0) {
    setLoad(loadStep - 1);
    quickFor = 0;
  }
}
let sunHour = -1;
/** The sun at `hour`: rising in the east, highest at 12:30, setting in the west (the city faces south-west). */
/**
 * The skies: a Polyhaven HDRI for each time of day (public/hdri; its sun and colours in
 * index.json, from scripts/hdri-sun.ts), lighting the world and giving the sun its direction
 * (turned so it moves round through the day) and strength; the sky itself is drawn (sky.ts)
 * with clouds from the player's camera. `exposure` evens out the times of day.
 */
const SKIES: { hour: number; name: string; exposure: number; elevation?: number; sun?: number }[] = [
  { hour: 7.5, name: 'morning', exposure: 1 },
  { hour: 12, name: 'noon', exposure: 1 },
  { hour: 14.5, name: 'afternoon', exposure: 1 },
  { hour: 17, name: 'late_afternoon', exposure: 1 },
  { hour: 18.8, name: 'sunset', exposure: 1 },
  // After sunset the HDRI's brightest patch is the afterglow, not a sun: the sun goes below the
  // horizon (the sky darkens) and what light it gives is scaled down
  { hour: 19.6, name: 'dusk_2', exposure: 0.9, elevation: -3, sun: 0.35 },
  { hour: 21.5, name: 'night', exposure: 0.6, elevation: -25, sun: 0 },
];
interface SkyInfo {
  file: string;
  sun: number[];
  sunColour: number[];
  sunIrradiance: number;
  skyColour: number[];
}
let skyIndex: Record<string, SkyInfo> | null = null;
/** The sun's strength per unit of the HDRI's radiance (the sky light a little more: glass should show the sky). */
const LIGHT_SCALE = 0.2;
let sunStrength = 3;
function setSun(hour: number): void {
  if (!sky || Math.abs(hour - sunHour) < 1e-3) return;
  sunHour = hour;
  if (SCENE === 'space') {
    // In orbit the "time" turns the sun round the station, 35 degrees up (the Earth below lit)
    const azimuth = (hour / 24) * Math.PI * 2 + 2.2;
    const elevation = 0.6;
    view.sunDirection.set(Math.cos(elevation) * Math.cos(azimuth), Math.cos(elevation) * Math.sin(azimuth), Math.sin(elevation)).normalize();
    sky.sun.copy(view.sunDirection);
    sky.update(view.camera, performance.now() / 1000);
    sunStrength = sky.sunIntensity();
    view.renderer.toneMappingExposure = 1;
    view.setSpace();
    return;
  }
  const day = (hour - 6.3) / 12.8;
  const azimuth = Math.PI * (0.15 - day * 1.1);
  const preset = SKIES.reduce((best, s) => (Math.abs(s.hour - hour) < Math.abs(best.hour - hour) ? s : best));
  const info = skyIndex?.[preset.name];
  if (info) {
    const [x, y] = info.sun;
    const turn = azimuth - Math.atan2(y, x);
    const elevation = preset.elevation === undefined ? Math.asin(Math.max(-1, Math.min(1, info.sun[2]))) : (preset.elevation * Math.PI) / 180;
    const flat = Math.cos(elevation) / Math.max(1e-6, Math.hypot(x, y));
    view.sunDirection.set((x * Math.cos(turn) - y * Math.sin(turn)) * flat, (x * Math.sin(turn) + y * Math.cos(turn)) * flat, Math.sin(elevation)).normalize();
    sunStrength = info.sunIrradiance * LIGHT_SCALE * (preset.sun ?? 1);
    sky.sun.copy(view.sunDirection);
    sky.update(view.camera, performance.now() / 1000);
    view.renderer.toneMappingExposure = preset.exposure;
    view
      .loadSky(info.file)
      .then((t) => {
        if (sunHour === hour) view.setSkyLight(t, turn, LIGHT_SCALE * 1.75);
      })
      .catch(() => view.setEnvironment(sky.horizon, sky.sunColor, sky.sunIntensity()));
  } else {
    const elevation = Math.sin(Math.PI * day) * 0.95;
    view.sunDirection.set(Math.cos(elevation) * Math.cos(azimuth), Math.cos(elevation) * Math.sin(azimuth), Math.sin(elevation)).normalize();
    sky.sun.copy(view.sunDirection);
    sky.update(view.camera, performance.now() / 1000);
    sunStrength = sky.sunIntensity();
    view.setEnvironment(sky.horizon, sky.sunColor, sky.sunIntensity());
  }
}
async function startSky(): Promise<void> {
  skyIndex = await fetch('hdri/index.json')
    .then((r) => (r.ok ? (r.json() as Promise<Record<string, SkyInfo>>) : null))
    .catch(() => null);
  if (SCENE === 'space') {
    const space = new SpaceSky(new THREE.Vector3(...HUB));
    satellites = space.satellites;
    // Its clouds are drawn on a quad on the camera, once their noise volumes are made
    view.camera.add(space.clouds.object);
    await space.clouds.load(view.renderer);
    sky = space;
  } else sky = new CitySky(view.sunDirection);
  view.scene.add(sky.object);
  sky.update(view.camera, 0);
  setSun(settings.hour);
}
const chips = new Chips(city);
view.scene.add(chips.object);
const shards = Chips.glass(city);
view.scene.add(shards.object);
const impacts = new Impacts();
view.scene.add(impacts.object);
const dustRing = new THREE.Color(1.1, 1.0, 0.9);
const hotRing = new THREE.Color(2.2, 1.4, 0.8);

/**
 * Car alarms: a car that's hit starts honking, and keeps on until it's smashed enough that
 * it wouldn't (a sixth of its body gone, glass aside; after ALARM_LEAST at least) or half a
 * minute has passed. Loudness
 * follows the player's distance.
 */
/** An alarm honks at least this long (s) before a wrecked car's cuts out. */
const ALARM_LEAST = 2.2;
const alarms = {
  on: new Map<number, { alarm: NonNullable<ReturnType<Sounds['alarm']>>; until: number; at: THREE.Vector3; started: number }>(),
  /** Cars already silenced for good. */
  dead: new Set<number>(),
  broken(i: number): boolean {
    const b = city.buildings[i];
    let body = 0;
    let gone = 0;
    for (const v of b.cells) {
      if (v < 0 || city.material[v] === Mat.Glass) continue;
      body++;
      if (city.state[v] !== 0) gone++;
    }
    return gone > body / 6;
  },
  hit(result: BlastResult): void {
    const cars = new Set<number>();
    for (const list of [result.gone, result.loose, result.falling]) for (const v of list) if (city.buildings[city.building[v]].kind === 'car') cars.add(city.building[v]);
    for (const p of result.panes) {
      const i = city.building[p.voxels[0]];
      if (city.buildings[i].kind === 'car') cars.add(i);
    }
    const now = performance.now() / 1000;
    for (const i of cars) {
      if (this.dead.has(i)) continue;
      let on = this.on.get(i);
      if (!on) {
        // The first blow always sets it off
        const alarm = sounds.alarm();
        if (!alarm) continue;
        const b = city.buildings[i];
        const at = new THREE.Vector3((b.x0 + b.w / 2) * VOXEL, (b.y0 + b.d / 2) * VOXEL, 0.8);
        on = { alarm, until: now + 30, at, started: now };
        this.on.set(i, on);
      }
      // Smashed enough, it cuts out (after a couple of seconds of honking at least)
      if (this.broken(i)) on.until = Math.min(on.until, Math.max(now + 0.05, on.started + ALARM_LEAST));
    }
  },
  update(): void {
    const now = performance.now() / 1000;
    for (const [i, a] of this.on) {
      if (now > a.until) {
        a.alarm.stop();
        this.on.delete(i);
        this.dead.add(i);
        continue;
      }
      a.alarm.level(1 / (1 + player.position.distanceTo(a.at) / 12));
    }
  },
  clear(): void {
    for (const a of this.on.values()) a.alarm.stop();
    this.on.clear();
    this.dead.clear();
  },
};

/**
 * Windows a blast (or the frame giving way) shattered: shards blown away from the blast that
 * broke them, fast close to it, or spilling out of the facade; a puff of dust out of a window
 * whose storey gave way; and the sound, once for all of them.
 */
function afterBlast(result: BlastResult, hits: { at: ArrayLike<number> }[]): void {
  alarms.hit(result);
  if (!result.panes.length) return;
  let nearest = Infinity;
  let high = 0;
  const puffs: number[] = [];
  // A few thousand shards at most, however many windows go
  const total = result.panes.reduce((n, p) => n + p.voxels.length, 0);
  const each = Math.min(1.2, 2500 / Math.max(1, total));
  for (const { pane, voxels, by } of result.panes) {
    const { centre, normal } = city.panes[pane];
    let from: ArrayLike<number>;
    let speed: number;
    if (by >= 0) {
      from = hits[by].at;
      speed = Math.max(3, 16 - 1.2 * Math.hypot(centre[0] - from[0], centre[1] - from[1], centre[2] - from[2]));
    } else {
      from = [centre[0] - normal[0] * 2, centre[1] - normal[1] * 2, centre[2]];
      speed = 3;
      puffs.push(centre[0] + normal[0] * 0.6, centre[1] + normal[1] * 0.6, centre[2]);
    }
    shards.burst(voxels, from, each, speed, 0xd4e4ea);
    nearest = Math.min(nearest, player.position.distanceTo(new THREE.Vector3(centre[0], centre[1], centre[2])));
    high = Math.max(high, centre[2]);
  }
  // A window blown in by what gives way behind it puffs dust; a whole piece's windows going at
  // once (cut free, it falls without them) share a little, not a cloud over all of it
  if (puffs.length) effects.dust(puffs, 6 * Math.min(1, 8 / (puffs.length / 3)), concreteDust);
  sounds.glass(result.panes.length, nearest, Math.min(2, high / 12));
}

// Using a tool
let cooldown = 0;
const eye = new THREE.Vector3();
const dir = new THREE.Vector3();
let shake = 0;
/** The block being destroyed (its middle, m) and how far out from it stays in focus (m). */
const BLOCK_CENTRE = new THREE.Vector2(0, 0);
const BLOCK_REACH = SCENE === 'track' ? 150 : SCENE === 'space' ? 120 : 22;
/** Depth of field: the distance wanted in focus, the eased one, and the bokeh's size (eased). */
let focusWant = 30;
let focusAt = 30;
let dofAmount = 0;
/** The view's dip on landing (m, eased back). */
let dip = 0;
/** Dust in the air after blasts (thickens the sun shafts), settling over half a minute. */
let dustiness = 0;

function useTool(): void {
  const t = TOOLS[tool];
  hand.use(t.cooldown);
  player.eye(eye);
  player.look(dir);
  if (t.ball) {
    const [r, density, speed] = t.ball;
    const from = eye.clone().addScaledVector(dir, 2.2);
    const v = dir.clone().multiplyScalar(speed).add(player.velocity);
    physics.throwBall(from.toArray(), v.toArray(), r, density);
    planBall(from, v, r);
    shake = Math.max(shake, 0.35);
    sounds.swing();
    return;
  }
  if (t.name === 'Hammer') {
    // The blow lands as the swing comes down, where the player is looking then
    sounds.swing();
    blows.push(performance.now() / 1000 + ViewModel.STRIKE);
    return;
  }
  if (t.laser) {
    burn(t);
    return;
  }
  const hit = raycast(city, eye.toArray(), dir.toArray(), t.reach);
  // (In orbit a rocket fired at nothing flies off into space, gone at the end of its reach: it may yet meet a satellite)
  const intoSpace = !hit && t.name === 'Rocket' && city.weightless;
  if (!hit && !intoSpace) return;
  const at = eye.clone().addScaledVector(dir, hit ? hit.t : t.reach);
  // Centre the blast just inside what was hit
  if (hit) at.addScaledVector(new THREE.Vector3(...hit.normal), -Math.min(0.5, t.radius * 0.3));
  if (t.name === 'Rocket') {
    // A projectile: it flies there, then goes off
    const muzzle = eye.clone().add(new THREE.Vector3(0, 0, -0.15)).addScaledVector(dir, 0.8);
    rockets.push({ from: muzzle, to: at, t: 0, time: muzzle.distanceTo(at) / ROCKET_SPEED, mesh: rocketMesh(), spec: intoSpace ? null : undefined });
    sounds.zap();
    shake = Math.max(shake, 0.3);
    return;
  }
  if (!hit) return;
  if (t.name === 'Charge') {
    // The detonator's button clicks in; the charge goes off a beat later
    sounds.click();
    charges.push({ due: performance.now() / 1000 + DETONATE_DELAY, at, distance: hit.t });
    return;
  }
  detonate(at, t, hit.t);
}

/** Seconds from pressing the detonator to the charge going off (a beat, for the reaction). */
const DETONATE_DELAY = 0.2;
/** Charges set off, due to go off (s, performance clock). */
const charges: { due: number; at: THREE.Vector3; distance: number }[] = [];

/** Hammer blows due (s, performance clock): the swing lands a moment after it starts. */
const blows: number[] = [];
function land(): void {
  const t = TOOLS[0];
  player.eye(eye);
  player.look(dir);
  const hit = raycast(city, eye.toArray(), dir.toArray(), t.reach);
  if (!hit || hit.voxel < 0) return;
  const at = eye.clone().addScaledVector(dir, hit.t).addScaledVector(new THREE.Vector3(...hit.normal), -Math.min(0.5, t.radius * 0.3));
  detonate(at, t, hit.t, hit.normal);
}

/** The most the laser burns through in one tick (m): whatever it hits, out the far side. */
const LASER_DEPTH = 24;
/** At most this many beams a tick, filling the sweep since the last (a flick cuts a slot, not dots). */
const LASER_FILL = 12;
/** The last tick's aim while the trigger is held: the sweep since then is burned too. */
let lastBeam: THREE.Vector3 | null = null;

/** How far a ray from `o` along unit `d` runs from `t0` to where it leaves object `i`'s bounds (m). */
function throughObject(i: number, o: THREE.Vector3, d: THREE.Vector3, t0: number): number {
  const b = city.buildings[i];
  const lo = [b.x0 * VOXEL, b.y0 * VOXEL, 0];
  const hi = [(b.x0 + b.w) * VOXEL, (b.y0 + b.d) * VOXEL, b.h * VOXEL];
  const [oa, da] = [o.toArray(), d.toArray()];
  let out = Infinity;
  for (let a = 0; a < 3; a++) {
    if (Math.abs(da[a]) < 1e-9) continue;
    out = Math.min(out, ((da[a] > 0 ? hi[a] : lo[a]) - oa[a]) / da[a]);
  }
  return Math.max(0, out - t0);
}

/**
 * The laser's burn this tick: what the beam touches melts away, through to the far side of
 * what it hits, along the whole sweep since the last tick (its way: a cut tips what falls).
 */
function burn(t: ToolSpec): void {
  player.eye(eye);
  player.look(dir);
  const hit = raycast(city, eye.toArray(), dir.toArray(), t.reach);
  const from = lastBeam;
  lastBeam = dir.clone();
  if (!hit || hit.voxel < 0) return;
  const way = Math.hypot(dir.x, dir.y) > 0.2 ? [dir.x / Math.hypot(dir.x, dir.y), dir.y / Math.hypot(dir.x, dir.y)] : undefined;
  // Beams across the sweep, close enough at the range it burns that their slots join
  const step = t.radius * 1.4;
  const n = from ? Math.min(LASER_FILL, Math.max(1, Math.ceil((from.angleTo(dir) * hit.t) / step))) : 1;
  const hits: Hit[] = [];
  const ray = new THREE.Vector3();
  for (let k = 1; k <= n; k++) {
    ray.copy(from ?? dir).lerp(dir, k / n).normalize();
    const h = k === n ? hit : raycast(city, eye.toArray(), ray.toArray(), t.reach);
    if (!h || h.voxel < 0) continue;
    // It burns a slot through what it hits, not a dimple: swept across a building, it cuts
    // through the walls, the columns inside and the core behind them, out the far side
    const depth = Math.min(LASER_DEPTH, throughObject(city.building[h.voxel], eye, ray, h.t) + 0.3);
    for (let d = 0.1; d <= depth; d += step) hits.push({ at: eye.clone().addScaledVector(ray, h.t + d).toArray(), radius: t.radius, push: 0, core: t.core, topple: way });
  }
  const result = blasts(city, structure, physics, hits);
  unsettled = true;
  afterBlast(result, hits);
  view.heat(cutFaces(result.gone), simTime);
  const molten = eye.clone().addScaledVector(dir, hit.t - 0.05).toArray();
  effects.sparks(molten, 5, 5);
  if (Math.random() < 0.35) effects.impact(molten, 2, smokeTint);
  if (result.falling.length > 50) {
    sounds.collapse(result.falling.length, hit.t);
    shake = Math.max(shake, 0.5);
  }
}

/** The bodies of the fixed voxels next to `gone` ones (not glass): a cut's faces, which glow. */
function cutFaces(gone: number[]): number[] {
  const out: number[] = [];
  for (const v of gone) {
    const b = city.buildings[city.building[v]];
    const c = city.cell[v];
    const [x, y, z] = [c % b.w, Math.floor(c / b.w) % b.d, Math.floor(c / (b.w * b.d))];
    for (const [dx, dy, dz] of NEIGHBOURS) {
      const [nx, ny, nz] = [x + dx, y + dy, z + dz];
      if (nx < 0 || ny < 0 || nz < 0 || nx >= b.w || ny >= b.d || nz >= b.h) continue;
      const u = b.cells[nx + b.w * (ny + b.d * nz)];
      if (u >= 0 && city.state[u] === 0 && !isGlass(city.material[u])) out.push(physics.body[u]);
    }
  }
  return out;
}
const NEIGHBOURS = [
  [1, 0, 0],
  [-1, 0, 0],
  [0, 1, 0],
  [0, -1, 0],
  [0, 0, 1],
  [0, 0, -1],
];

/** The laser's beam, drawn each frame while it fires, with its hum and the burn's crackle. */
let lasing = false;
const muzzleAt = new THREE.Vector3();
function laserFrame(now: number): void {
  const t = TOOLS[tool];
  const on = playing && held && !!t.laser;
  if (on !== lasing) {
    sounds.laser(on);
    lasing = on;
    hand.firing = on;
  }
  if (!on) {
    impacts.beam(null);
    lastBeam = null;
    return;
  }
  player.eye(eye);
  player.look(dir);
  const hit = raycast(city, eye.toArray(), dir.toArray(), t.reach);
  // A satellite before whatever else the beam meets: it cuts through it, and it's wrecked
  const sat = satellites?.hit(eye, dir, hit ? hit.t : t.reach);
  const end = eye.clone().addScaledVector(dir, sat ? sat.distance : hit ? hit.t : t.reach);
  if (sat) {
    satellites!.smash(sat.craft, eye);
    wreck(end);
  }
  impacts.beam(hand.muzzle(muzzleAt), end, !!hit || !!sat, now);
  sounds.laserBurn(hit && hit.voxel >= 0 ? 1 : 0);
  shake = Math.max(shake, 0.16);
}

/** In orbit, the satellites passing under the station (space.ts): rockets, the laser and blasts break them. */
let satellites: Satellites | null = null;

/** A satellite broken at `at` by a hit that didn't blow up (the laser): sparks, a flash and a bang. */
function wreck(at: THREE.Vector3): void {
  effects.sparks(at.toArray(), 30, 8);
  effects.explosion(at.toArray(), 1.2);
  sounds.blast(3, player.position.distanceTo(at));
}

/** Blasts due at a time (of the world's clock): where a thrown ball will smash through what it meets. */
const pending: { time: number; at: THREE.Vector3; radius: number; push: number }[] = [];
/**
 * The world's clock (s) and how fast it runs: the player moves in real time either way, so
 * in slow motion they can fly round a collapse as it happens.
 */
let simTime = 0;
let timeScale = 1;
const SLOW = 0.12;

/**
 * Follow a thrown ball's arc through the city: where it meets fixed voxels, knock them out a
 * moment before it arrives (it can't break what's fixed by touch alone), losing speed through
 * each wall until it's too slow to break any more.
 */
function planBall(from: THREE.Vector3, v: THREE.Vector3, r: number): void {
  const p = from.clone();
  const vel = v.clone();
  const now = simTime;
  const dt = 1 / 120;
  for (let t = 0; t < 4 && vel.length() > 10; t += dt) {
    if (!city.weightless) vel.z -= 9.81 * dt;
    const step = vel.clone().multiplyScalar(dt);
    const hit = raycast(city, p.toArray(), step.clone().normalize().toArray(), step.length() + r);
    p.add(step);
    if (!hit || hit.voxel < 0) continue;
    pending.push({ time: now + t - 0.03, at: p.clone().addScaledVector(vel.clone().normalize(), r), radius: r * 1.7, push: vel.length() * 0.35 });
    vel.multiplyScalar(0.6);
    p.addScaledVector(vel.clone().normalize(), r * 1.5);
  }
}

// Rockets in flight
const ROCKET_SPEED = 70;
const rockets: { from: THREE.Vector3; to: THREE.Vector3; t: number; time: number; mesh: THREE.Object3D; spec?: ToolSpec | null; quiet?: boolean }[] = [];
function rocketMesh(scale = 1): THREE.Object3D {
  const g = new THREE.Group();
  g.scale.setScalar(scale);
  const body = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, 0.6, 10), new THREE.MeshStandardMaterial({ color: 0x5a5f55, roughness: 0.6 }));
  body.rotation.x = Math.PI / 2;
  const flame = new THREE.Mesh(new THREE.SphereGeometry(0.12, 10, 8), new THREE.MeshBasicMaterial({ color: new THREE.Color(4, 2.2, 0.8) }));
  flame.position.z = 0.35;
  flame.scale.set(1, 1, 2.2);
  // (No light of its own: adding and removing lights recompiles every material; the bloom makes it glow)
  g.add(body, flame);
  view.scene.add(g);
  return g;
}
function flyRockets(dt: number): void {
  for (let k = rockets.length - 1; k >= 0; k--) {
    const r = rockets[k];
    r.t += dt;
    const f = Math.min(1, r.t / r.time);
    const was = r.mesh.position.clone();
    r.mesh.position.lerpVectors(r.from, r.to, f);
    r.mesh.lookAt(r.from);
    if (!r.quiet) effects.impact(r.mesh.position.toArray(), 3, smokeTint);
    // A satellite in its way (they move, so it's looked for as the rocket flies): it goes off there
    const step = r.mesh.position.clone().sub(was);
    const flown = step.length();
    const sat = !r.quiet && flown > 0 ? satellites?.hit(was, step.divideScalar(flown), flown) : null;
    if (sat) r.to.copy(was).addScaledVector(step, sat.distance);
    if (f >= 1 || sat) {
      view.scene.remove(r.mesh);
      if (r.spec !== null || sat) detonate(r.to, r.spec ?? TOOLS[2], player.position.distanceTo(r.to));
      rockets.splice(k, 1);
    }
  }
  for (let k = pending.length - 1; k >= 0; k--) {
    if (pending[k].time > simTime) continue;
    const { at, radius, push } = pending.splice(k, 1)[0];
    const result = blast(city, structure, physics, at.toArray(), radius, push, 0.35);
    unsettled = true;
    afterBlast(result, [{ at: at.toArray() }]);
    chips.burst(result.gone, at.toArray(), 5, push * 0.8);
    chips.burst(result.loose, at.toArray(), 1, push * 0.6);
    // The ball's blow: a boom and a crunch, a shockwave off the wall, sparks off the steel,
    // a cloud of the wall's dust, and a jolt felt from well off
    const distance = player.position.distanceTo(at);
    sounds.wreck(distance, push / 0.35);
    shake = Math.max(shake, Math.min(0.9, 14 / (distance + 10)));
    const back = at.clone().sub(player.position).normalize().negate();
    impacts.hit(at.toArray(), back.toArray(), radius * 3.2, 0.55, dustRing, radius * 0.9);
    effects.sparks(at.toArray(), 26, 9);
    effects.impact(at.toArray(), 14, dustTint[city.material[result.gone[0] ?? 0]] ?? concreteDust);
    const broken = [...result.gone, ...result.loose];
    const points = new Float32Array(broken.length * 3);
    broken.forEach((v, i) => points.set(city.position.subarray(3 * v, 3 * v + 3), 3 * i));
    if (broken.length) effects.dust(points, 0.8, dustTint[city.material[broken[0]]] ?? concreteDust);
  }
}
const smokeTint = new THREE.Color(0.5, 0.5, 0.5);

/** Tool `t`'s blast at `at`, `distance` m from the player: the damage, and its sound, flash, dust. */
function detonate(at: THREE.Vector3, t: ToolSpec, distance: number, normal?: ArrayLike<number>): void {
  satellites?.blast(at, t.radius);
  // The car caught in it (its own rockets too, fired too close)
  if (car) {
    const reach = t.radius + 3;
    const off = at.distanceTo(car.position);
    if (off < reach) hurtCar(0.45 * (1 - off / reach), at);
  }
  const result = blast(city, structure, physics, at.toArray(), t.radius, t.push, t.core);
  unsettled = true;
  afterBlast(result, [{ at: at.toArray() }]);
  chips.burst(result.gone, at.toArray(), t.explosive ? 1 : 3, t.push * 0.9 + 3);
  chips.burst(result.loose, at.toArray(), 0.6, t.push * 0.6 + 2);
  if (t.explosive) {
    effects.explosion(at.toArray(), t.radius * 1.4);
    dustiness = Math.min(1.2, dustiness + t.radius / 5 + result.falling.length / 40000);
    sounds.blast(t.radius * 2, distance);
    hud.flash(Math.min(1, (t.radius * 6) / (distance + 5)));
    shake = Math.max(shake, Math.min(1.5, (t.radius * 6) / (distance + 6) + 0.25));
    view.glow = 1.2;
  } else if (t.name === 'Hammer') {
    // The blow: a thud with a crack and a ring of steel, a jolt, sparks off the head, a ring of
    // dust racing out over the face it struck, and a puff of it
    const broke = result.gone.length + result.loose.length;
    sounds.sledge(distance, broke);
    shake = Math.max(shake, 0.55);
    hand.hitstop();
    effects.sparks(at.toArray(), 14, 6);
    impacts.hit(at.toArray(), normal ?? [0, 0, 1], 1.3, 0.3, broke ? dustRing : hotRing, 0.18);
    effects.impact(at.toArray(), 9, dustTint[city.material[result.gone[0] ?? result.loose[0] ?? 0]] ?? concreteDust);
  } else sounds.zap();
  // Dust where voxels broke off and where pieces came loose
  const broken = [...result.gone, ...result.loose];
  const points = new Float32Array(broken.length * 3);
  broken.forEach((v, k) => points.set(city.position.subarray(3 * v, 3 * v + 3), 3 * k));
  const tint = dustTint[city.material[broken[0]]] ?? concreteDust;
  if (broken.length) effects.dust(points, t.explosive ? 0.3 : 0.15, tint);
  if (result.falling.length > 50) {
    const sample = result.falling.filter((_, k) => k % 4 === 0);
    const fp = new Float32Array(sample.length * 3);
    sample.forEach((v, k) => fp.set(city.position.subarray(3 * v, 3 * v + 3), 3 * k));
    effects.dust(fp, 0.5, dustTint[city.material[sample[0]]] ?? concreteDust);
    sounds.collapse(result.falling.length, distance);
  }
}

// Reading the debris back (for the player to stand on, dust where it lands, and what it smashes) a few times a second
/**
 * Debris faster than this (m/s) breaks what it runs into once the damage to the chunk it hits
 * (speed² per voxel, summed over a readback, halving each readback) passes CRUSH_DAMAGE: a
 * slab coming down smashes what it lands on, a stray brick does not (each brick's strike
 * knocked out more bricks, whose strikes knocked out more: a runaway).
 */
const CRUSH_SPEED = 7;
/** A section's velocity jumping by this much (m/s) between readbacks is an impact: it breaks. */
const SECTION_JOLT = 2.5;
const CRUSH_DAMAGE = 2500;
/** Loose bodies at once, at most: past it the ones farthest from the player crumble to dust. */
let looseCap = 8000;
let damage = new Map<number, number>();
const crushes: { voxel: number; speed: number }[] = [];
/** In orbit, debris farther than this (m) from the station is let go of. */
const DRIFT_OFF = 150;
/** Debris slower than this (m/s) for this long (s) comes to rest. */
const REST_SPEED = 0.25;
const REST_SECONDS = 1.5;
let stillFor = new Float32Array(0);
let lastRead = 0;
let reading = false;
/**
 * Something changed under the rubble (a blast, rubble freezing, debris removed): check it's
 * still held, at every readback for a while (what a blast knocks loose takes a moment to start
 * falling away from under a heap), and every few seconds regardless.
 */
let unsettled = false;
let settleUntil = 0;
let lastSettle = 0;
const SETTLE_WATCH = 2.5;
let lastImpact = new Float32Array(0);
let frames = 0;
async function readDebris(): Promise<void> {
  if (reading || (physics.loose === 0 && physics.rubble.size === 0)) return;
  reading = true;
  try {
    const counters = await physics.solver.readCounters();
    physics.solver.adapt(counters);
    const { voxels, data, sections, sectionData } = await physics.readLoose();
    const out: number[] = [];
    const { x, y } = player.position;
    const now = performance.now() / 1000;
    if (lastImpact.length !== city.count) lastImpact = new Float32Array(city.count);
    let puffs = 0;
    if (stillFor.length !== city.count) stillFor = new Float32Array(city.count);
    const since = (now - lastRead) * timeScale;
    lastRead = now;
    const resting: number[] = [];
    const poses: number[] = [];
    /** Loose voxels lying still: rubble on them is held. */
    const still: number[] = [];
    for (const [k, v] of voxels.entries()) {
      const o = 12 * k;
      const [px, py, pz] = [data[o], data[o + 1], data[o + 2]];
      if (pz < PARKED_BELOW) continue;
      if (Math.abs(px - x) < 12 && Math.abs(py - y) < 12) out.push(px, py, pz);
      const speed = Math.hypot(data[o + 4], data[o + 5], data[o + 6]);
      // Debris that has lain still a while holds rubble (just knocked loose, it may be about to fall)
      if (speed < 1 && stillFor[v] > 1) still.push(px, py, pz);
      // At rest a while: frozen where it lies (so a settled heap costs nothing)
      stillFor[v] = speed < REST_SPEED ? stillFor[v] + since : 0;
      // (In orbit nothing comes to rest on anything: it drifts on, until it's far enough to let go of)
      if (!city.weightless && stillFor[v] > REST_SECONDS && resting.length < 4000) {
        resting.push(v);
        poses.push(px, py, pz, data[o + 8], data[o + 9], data[o + 10], data[o + 11]);
      }
      // Debris flying into what still stands breaks it (by its speed squared), so collapses cascade
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
      // Debris coming down hard near the ground kicks up dust (a few puffs a readback, one per voxel a while)
      if (!city.weightless && speed > 6 && pz < 2.5 && puffs < 40 && now - lastImpact[v] > 3) {
        lastImpact[v] = now;
        puffs++;
        effects.impact([px, py, Math.max(0.2, pz - 0.4)], speed, dustTint[city.material[v]] ?? concreteDust);
      }
    }
    for (const [chunk, total] of damage) (total < 50 ? damage.delete(chunk) : damage.set(chunk, total * 0.5));
    if (resting.length) {
      physics.freeze(resting, Float32Array.from(poses));
      unsettled = true;
    }
    // Past the cap, the loose voxels farthest from the player go (in a puff of chips)
    if (physics.loose > looseCap) {
      // Over the budget: first the slow ones low down settle as rubble where they are (they
      // were about to anyway), farthest first; only if that isn't enough do the farthest go
      const candidates = voxels
        .map((v, k) => ({ v, d: (data[12 * k] - x) ** 2 + (data[12 * k + 1] - y) ** 2, k }))
        .filter(({ k, v }) => data[12 * k + 2] > PARKED_BELOW && city.state[v] === 1)
        .sort((a, b) => b.d - a.d);
      const settleNow = (city.weightless ? [] : candidates)
        .filter(({ k }) => data[12 * k + 2] < 2.5 && Math.hypot(data[12 * k + 4], data[12 * k + 5], data[12 * k + 6]) < 1.5)
        .slice(0, physics.loose - looseCap);
      if (settleNow.length) {
        const settlePoses = new Float32Array(settleNow.length * 7);
        settleNow.forEach(({ k }, i) => settlePoses.set([data[12 * k], data[12 * k + 1], data[12 * k + 2], data[12 * k + 8], data[12 * k + 9], data[12 * k + 10], data[12 * k + 11]], 7 * i));
        physics.freeze(
          settleNow.map(({ v }) => v),
          settlePoses,
        );
        unsettled = true;
      }
      const far = candidates.filter(({ v }) => city.state[v] === 1).slice(0, Math.max(0, physics.loose - looseCap));
      const gone = far.map(({ v }) => v);
      for (const { v, k } of far) city.position.set(data.subarray(12 * k, 12 * k + 3), 3 * v);
      chips.burst(gone.slice(0, 400), [x, y, 0], 1, 2);
      if (gone.length) physics.remove(gone);
      unsettled = true;
    }
    // In orbit, what's drifted far off goes (it's not coming back), loose voxels and whole pieces alike
    if (city.weightless) {
      const off = (d: Float32Array, o: number) => Math.hypot(d[o] - HUB[0], d[o + 1] - HUB[1], d[o + 2] - HUB[2]) > DRIFT_OFF;
      const far = voxels.filter((v, k) => city.state[v] === 1 && off(data, 12 * k));
      if (far.length) physics.remove(far);
      sections.forEach((slot, k) => {
        if (off(sectionData, 12 * k)) physics.discard(slot);
      });
    }
    // Rubble at rest near the player is solid to them too
    for (const v of physics.rubble) {
      const o = 7 * v;
      if (Math.abs(physics.rest[o] - x) < 12 && Math.abs(physics.rest[o + 1] - y) < 12) out.push(physics.rest[o], physics.rest[o + 1], physics.rest[o + 2]);
    }
    player.debris = Float32Array.from(out);
    player.debrisCount = out.length / 3;
    // Sections falling whole break when they hit something (a jolt in their velocity) or come
    // to rest: a hard hit throws more of them loose and smashes what they came down on, so a
    // tower's storeys pancake down through the ones below
    const crushAt: Hit[] = [];
    for (const [k, slot] of sections.entries()) {
      const section = physics.sections.get(slot);
      if (!section) continue;
      const o = 12 * k;
      const vel = [sectionData[o + 4], sectionData[o + 5], sectionData[o + 6]];
      const jolt = Math.hypot(vel[0] - section.velocity[0], vel[1] - section.velocity[1], vel[2] - section.velocity[2]);
      const speed = Math.hypot(vel[0], vel[1], vel[2]);
      const age = now - section.born;
      section.velocity = [vel[0], vel[1], vel[2]];
      const pose = [sectionData[o], sectionData[o + 1], sectionData[o + 2], sectionData[o + 8], sectionData[o + 9], sectionData[o + 10], sectionData[o + 11]];
      section.pose = pose;
      // It breaks only where breaking holds: hitting the ground or a floor still standing
      // (sections knocking each other in the air, stacked as they fall, stay whole), or at rest
      // (In orbit a piece never rests: it drifts, whole, until it hits something)
      const resting = !city.weightless && age > 1 && speed < 0.5;
      let hit = false;
      if (age > 0.2 && jolt > SECTION_JOLT) {
        if (city.weightless) hit = true;
        const low = physics.underside(slot, pose, 16);
        for (let i = 0; !hit && i < low.length; i += 3) hit = low[i + 2] < 1.2 || voxelAt(city, low[i], low[i + 1], low[i + 2] - 0.3) >= 0;
      }
      if (!hit && !resting && sectionData[o + 2] > PARKED_BELOW) continue;
      // Resting on another section still whole: it waits for that one to break first (else
      // its rubble would lie on a box about to vanish, and all wake at once)
      if (resting && !hit) {
        const low = physics.underside(slot, pose, 16);
        let onSection = false;
        for (let i = 0; !onSection && i < low.length; i += 3) onSection = physics.inSection(low[i], low[i + 1], low[i + 2] - 0.3, slot);
        if (onSection) continue;
      }
      // At rest high up on other wreckage (not the ground, not a floor still standing), it
      // spills: half of it tumbles off, so falling storeys don't stand in a stack
      let high = false;
      if (resting && !hit) {
        const low = physics.underside(slot, pose, 16);
        high = true;
        for (let i = 0; high && i < low.length; i += 3) if (low[i + 2] < 2 || voxelAt(city, low[i], low[i + 1], low[i + 2] - 0.3) >= 0) high = false;
      }
      // (In orbit it all flies apart: nothing to lay rubble down on)
      const share = city.weightless ? 1 : hit ? Math.min(0.12, 0.03 + jolt / 100) : high ? 0.3 : 0.02;
      const low = physics.shatter(slot, pose, vel, share);
      unsettled = true;
      const at = new THREE.Vector3(pose[0], pose[1], pose[2]);
      const distance = player.position.distanceTo(at);
      if (low.length) effects.dust(low, 2.5, concreteDust);
      if (hit && section.voxels.length > 200) {
        sounds.collapse(section.voxels.length, distance);
        shake = Math.max(shake, Math.min(0.8, section.voxels.length / 6000 / (1 + distance / 30)));
      }
      if (hit && jolt > CRUSH_SPEED)
        for (let i = 0; i < low.length; i += 9) crushAt.push({ at: [low[i], low[i + 1], low[i + 2] - 0.3], radius: 0.6 + Math.min(0.9, jolt / 20), push: jolt * 0.12, core: 0.35 });
    }
    if (crushAt.length) {
      afterBlast(blasts(city, structure, physics, crushAt), crushAt);
      unsettled = true;
    }
    // The strikes of this readback break what they hit in one go (one pass over what stands)
    const strikes = crushes.splice(0).filter(({ voxel }) => city.state[voxel] === 0);
    if (strikes.length) {
      const hits: Hit[] = strikes.map(({ voxel, speed }) => ({
        at: Array.from(city.position.subarray(3 * voxel, 3 * voxel + 3)),
        radius: 0.35 + Math.min(0.6, speed / 40),
        push: speed * 0.2,
        core: 0.25,
      }));
      const result = blasts(city, structure, physics, hits);
      unsettled = true;
      afterBlast(result, hits);
      result.hits.forEach((hit, k) => chips.burst([...hit.gone, ...hit.loose], hits[k].at, 2, strikes[k].speed * 0.3));
      const broken = [...result.gone, ...result.loose, ...result.falling.filter((_, k) => k % 6 === 0)];
      const points = new Float32Array(broken.length * 3);
      broken.forEach((v, k) => points.set(city.position.subarray(3 * v, 3 * v + 3), 3 * k));
      if (broken.length) effects.dust(points, 0.5, dustTint[city.material[broken[0]]] ?? concreteDust);
      const near = new THREE.Vector3(...hits[0].at);
      if (result.falling.length > 50) sounds.collapse(result.falling.length, player.position.distanceTo(near));
      else sounds.hammer(player.position.distanceTo(near) * 0.3);
    }
    // Rubble whose support went (a floor fell away under a heap) falls again
    if (unsettled) settleUntil = now + SETTLE_WATCH;
    unsettled = false;
    if (!city.weightless && ((now < settleUntil && now - lastSettle > 0.3) || now - lastSettle > 3)) {
      lastSettle = now;
      physics.settle(still);
    }
  } finally {
    reading = false;
  }
}

// The loop
let last = performance.now();
let accumulator = 0;
let fps = 60;
let stepMs = 0;
function frame(now: number): void {
  tick(now);
  requestAnimationFrame(frame);
}
/** One frame of the game at time `now` (ms): input, physics, effects, the render. */
function tick(now: number): void {
  const began = performance.now();
  // (Never negative: a clock that jumps back, a frame from a timer and one from the display
  // out of order, would stall the physics until it caught up)
  const dt = Math.max(0, Math.min((now - last) / 1000, 0.1));
  last = now;
  fps += (1 / Math.max(dt, 1e-3) - fps) * 0.05;
  if (playing) {
    turn(dt);
    cooldown -= dt;
    if (driving && car) {
      // Each weapon comes up while its button's held, fires once it's up, and folds away after a while
      // (a click's rocket goes as soon as the launchers are up)
      gunCooldown -= dt;
      if (wantFire) rocketDue = true;
      if (held || rocketDue) {
        car.wantRockets = true;
        rocketsIdle = 0;
        if (car.rockets > 0.95 && cooldown <= 0 && simTime >= loaded[launcher]) {
          fireRocket();
          cooldown = CAR_ROCKET.cooldown;
          rocketDue = false;
        }
      } else if ((rocketsIdle += dt) > 3) car.wantRockets = false;
      if (aiming) {
        car.wantGuns = true;
        gunsIdle = 0;
        if (car.guns > 0.95 && gunCooldown <= 0) {
          fireGun();
          gunCooldown = BULLET.cooldown;
        }
      } else if ((gunsIdle += dt) > 2.5) car.wantGuns = false;
    } else {
      const stride = Math.floor(player.walk / Math.PI);
      const move = input();
      player.update(dt, move);
      if (player.zeroG) jets(dt, move.forward !== 0 || move.right !== 0 || move.rise !== 0);
      // Footsteps at each half stride; a thud and a dip on landing
      if (player.onGround && Math.floor(player.walk / Math.PI) !== stride) sounds.step(keys.has('ShiftLeft'));
      if (player.landed > 4) {
        sounds.land(player.landed);
        dip = Math.min(0.25, player.landed * 0.02);
      }
      const t = TOOLS[tool];
      if ((wantFire || (held && t.auto)) && cooldown <= 0) {
        useTool();
        cooldown = t.cooldown;
      }
    }
    wantFire = false;
  }
  for (let k = blows.length - 1; k >= 0; k--) {
    if (blows[k] > now / 1000) continue;
    blows.splice(k, 1);
    land();
  }
  for (let k = charges.length - 1; k >= 0; k--) {
    if (charges[k].due > now / 1000) continue;
    const [due] = charges.splice(k, 1);
    detonate(due.at, TOOLS[3], due.distance);
  }
  // Physics at 60 Hz of the world's clock, at most two steps a frame (slowing rather than a spiral)
  const sim = dt * timeScale;
  simTime += sim;
  accumulator += sim;
  let steps = 0;
  const t0 = performance.now();
  while (accumulator >= 1 / 60 && steps < 2) {
    if (car) driveStep(1 / 60);
    physics.step();
    accumulator -= 1 / 60;
    steps++;
  }
  if (steps === 2) accumulator = 0;
  if (steps) stepMs += ((performance.now() - t0) / steps - stepMs) * 0.1;
  if (++frames % 6 === 0) void readDebris();

  // Camera: the player's eye (or behind the car), shaken by blasts
  if (car) {
    car.draw(Math.min(1, accumulator * 60));
    if (driving) player.position.set(car.position.x, car.position.y, 0);
  }
  if (driving && car) chaseView(dt);
  else {
    player.eye(eye);
    player.look(dir);
  }
  aimFrom.copy(eye);
  aimDir.copy(dir);
  const cam = view.camera;
  // Shake as trauma: it eases off, and the view moves by its square (a knock barely stirs it, a
  // blast next to you throws it about), wandering smoothly rather than jittering
  shake = Math.min(1.6, shake) * Math.exp(-dt * 3.2);
  const trauma = shake * shake;
  const t = now / 1000;
  const wander = (k: number) => Math.sin(t * 37 + k * 1.7) * 0.5 + Math.sin(t * 23.3 + k * 4.1) * 0.35 + Math.sin(t * 61 + k) * 0.15;
  dip *= Math.exp(-dt * 10);
  cam.position.copy(eye).add(new THREE.Vector3(wander(1) * trauma * 0.09, wander(2) * trauma * 0.09, wander(3) * trauma * 0.09 - dip));
  const sprinting = playing && keys.has('ShiftLeft') && Math.hypot(player.velocity.x, player.velocity.y) > 6;
  const carSpeed = driving && car ? Math.abs(car.speed) : 0;
  const fov = driving ? settings.fov + Math.min(16, carSpeed * 0.18) + (keys.has('KeyB') && boostLeft > 0 ? 5 : 0) : aiming ? settings.fov * 0.45 : settings.fov + (sprinting ? 6 : 0);
  if (Math.abs(cam.fov - fov) > 0.05) {
    cam.fov += (fov - cam.fov) * Math.min(1, dt * 8);
    cam.updateProjectionMatrix();
  }
  cam.lookAt(eye.clone().add(dir));
  cam.rotateX(wander(4) * trauma * 0.025);
  cam.rotateY(wander(5) * trauma * 0.02);
  cam.rotateZ(wander(6) * trauma * 0.035);
  // Depth of field: focus eased to what's under the crosshair; sharp over a wide range, and
  // shallow with bokeh while aiming down the sights
  if (frames % 3 === 0) {
    const f = raycast(city, eye.toArray(), dir.toArray(), 600);
    focusWant = f ? Math.max(1.5, f.t) : 600;
  }
  focusAt += (focusWant - focusAt) * Math.min(1, dt * 5);
  // (Not in orbit: the stars and the Earth are as sharp as the station)
  const dofOn = settings.dof && view.quality === 'high' && SCENE !== 'space';
  // (In the car the right button fires the machine guns: no zoom)
  const zoomed = aiming && !driving;
  dofAmount += ((dofOn ? (zoomed ? 12 : 8) : 0) - dofAmount) * Math.min(1, dt * 6);
  // In focus: everything up to the far side of the block being destroyed (or what's aimed at,
  // if that's further); the city beyond softening gradually. Aiming down the sights focuses on
  // what's under the crosshair, the rest behind it going soft sooner.
  // (At the track: what's within reach of the player, wherever that is)
  const block = (SCENE === 'city' ? Math.hypot(eye.x - BLOCK_CENTRE.x, eye.y - BLOCK_CENTRE.y) : 0) + BLOCK_REACH;
  view.setFocus(zoomed ? focusAt + 2 : Math.max(block, Math.min(focusAt, 80)), zoomed ? 25 : 90, dofAmount);
  const pace = Math.hypot(player.velocity.x, player.velocity.y);
  hand.update(dt, player.walk, player.onGround ? Math.min(1, pace / 9) : 0, sprinting);
  view.glow = Math.max(0.35, (view as unknown as { bloomPass: { strength: { value: number } } }).bloomPass.strength.value - dt * 1.5);
  const aim = raycast(city, eye.toArray(), dir.toArray(), driving ? CAR_ROCKET.reach : TOOLS[tool].reach || 3);
  if (driving && car) hud.speed(car.speed * 3.6, boostLeft, car.gear, loaded.map((t) => Math.max(0, Math.min(1, 1 - (t - simTime) / RELOAD))), car.health);
  if (car) {
    exhaust(sim, driving && playing ? Math.max(0, driveInput().throttle) : 0);
    tyres(sim);
    engineSmoke(sim);
    car.bits.update(sim);
  }
  // Walking up to the car: say how to get in
  const near = !!car && !driving && player.position.distanceTo(car.position) < GET_IN;
  if (near && !byCar) hud.toast(touch ? 'Tap the fly button to get in' : 'F to get in');
  byCar = near;
  if (car) {
    engine ??= sounds.engine();
    const throttle = driving && playing ? Math.max(0, driveInput().throttle) : 0;
    engine?.update(car.revs, throttle, car.slip, driving ? 3 : eye.distanceTo(car.position));
  }
  hud.aim(!!aim && aim.voxel >= 0);
  dustiness *= Math.exp(-sim / 25);
  view.dust = dustiness;
  flyRockets(sim);
  laserFrame(now / 1000);
  alarms.update();
  impacts.update(dt);
  adaptSmoke(dt);
  chips.update(sim);
  shards.update(sim);
  effects.update(sim, cam);
  sky.update(cam, now / 1000);
  view.heatClock.value = simTime;
  view.update(player.position, sky.sunColor, sunStrength, sky.horizon);
  view.render();
  timeWork((view.renderer.backend as unknown as { device: GPUDevice }).device, began);
  if (frames % 15 === 0) {
    hud.stats(`${fps.toFixed(0)} fps\n${city.count.toLocaleString()} voxels · ${physics.loose.toLocaleString()} loose\n${[player.flying ? 'flying' : '', timeScale < 1 ? 'slow motion' : ''].filter(Boolean).join(' · ')}`);
  }
}

// ?test plays without capturing the mouse (automation, recording: scripts drive the view
// through the handle); the mouse only turns the view once captured
Object.assign(window, {
  city: {
    player,
    view,
    keys,
    fire: () => useTool(),
    hold: (on: boolean) => {
      held = on;
    },
    /** Rockets, shells and tracers in flight. */
    get flying() {
      return rockets;
    },
    /** The right button (aim on foot, the machine guns in the car). */
    aim: (on: boolean) => {
      aiming = on;
    },
    boom: (x: number, y: number, z: number, tool = 3) => detonate(new THREE.Vector3(x, y, z), TOOLS[tool], player.position.distanceTo(new THREE.Vector3(x, y, z))),
    /** Run one frame of `dt` seconds by hand (a hidden page gets no animation frames). */
    tick: (dt: number) => tick(last + dt * 1000),
    set timeScale(s: number) {
      timeScale = s;
    },
    select: selectTool,
    get tool() {
      return tool;
    },
    get physics() {
      return physics;
    },
    /** The race car (the track), and getting in or out. */
    get car() {
      return car;
    },
    drive: (on: boolean) => setDriving(on),
    get line() {
      return line;
    },
    /** The chase camera's look round (orbit, lift: rad). */
    chase,
    get world() {
      return city;
    },
    get structure() {
      return structure;
    },
    settings,
    apply: applySettings,
    get smoke() {
      return smoke;
    },
    /** In orbit, the satellites under the station. */
    get satellites() {
      return satellites;
    },
    /** A fresh city (after a warm-up), and the sound (to record it: tap()). */
    rebuild,
    sounds,
    hand,
  },
});

addEventListener('resize', () => view.resize(innerWidth, innerHeight));
// Leaving the page (each scene is a page of its own): its GPU memory goes now. iOS Safari keeps a page it leaves, for
// Back, buffers and all, and the next scene, starting alongside it, ran out of memory ("range ... out of bounds").
// A page it brings back from there has no GPU left, so it loads afresh.
addEventListener('pagehide', () => (view.renderer.backend as unknown as { device?: GPUDevice }).device?.destroy());
addEventListener('pageshow', (e) => {
  if (e.persisted) location.reload();
});
view.resize(innerWidth, innerHeight);
start().catch((e: unknown) => {
  console.error(e);
  loader.fail(e);
});
