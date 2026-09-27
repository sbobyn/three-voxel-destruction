// The city's heads-up display: crosshair, the tool bar, a status line, the title screen that
// asks for a click (the pointer is locked to look around), and the pause menu with settings.
// Plain DOM over the canvas.
//
// Tools are switched three ways, all shown on screen: the mouse wheel, a click on a slot while
// the cursor is free, and a radial wheel held open with Tab (move the mouse towards a tool,
// release or click). Not the digit keys: some browser extensions take them before the page
// sees them. Each switch says so: the slot bounces, the tool's name comes up, and it clicks.
//
// The tool bar is a carousel: the tool in hand in the middle, its neighbours smaller either
// side. Scrolling slides the strip under the finger (nudge) and it snaps to the next tool, or
// back; at the ends it gives a little and springs back. A mouse with a rolling wheel, arrows
// at the sides and, until the first switch, a peek towards the next tool say it scrolls.

export interface Tool {
  name: string;
  hint: string;
  icon: string;
}

export interface Settings {
  sensitivity: number;
  fov: number;
  invertY: boolean;
  /** The quality the scene is drawn at (the chosen one, or the device's own under Auto). */
  quality: 'low' | 'medium' | 'high';
  /** What the player chose: the device's own tuning (calibrate.ts), or a quality. */
  graphics: 'auto' | 'low' | 'medium' | 'high';
  volume: number;
  /** Time of day (hours, 6.5 to 21.5: the nearest sky preset lights the world). */
  hour: number;
  /** Smooth the mouse a little (a couple of frames). */
  smoothing: boolean;
  /** Depth of field (high quality only), and rays of sunlight round the buildings. */
  dof: boolean;
  sunRays: boolean;
}

const CSS = `
.hud { position: fixed; inset: 0; z-index: 1; pointer-events: none; user-select: none; }
.hud .cross { position: absolute; left: 50%; top: 50%; width: 22px; height: 22px; transform: translate(-50%, -50%); }
.hud .cross::before, .hud .cross::after { content: ''; position: absolute; background: rgba(255,255,255,0.9); box-shadow: 0 0 2px rgba(0,0,0,0.8); }
.hud .cross::before { left: 10px; top: 2px; width: 2px; height: 18px; clip-path: polygon(0 0,100% 0,100% 38%,0 38%,0 62%,100% 62%,100% 100%,0 100%); }
.hud .cross::after { top: 10px; left: 2px; height: 2px; width: 18px; clip-path: polygon(0 0,38% 0,38% 100%,0 100%,0 0,62% 0,100% 0,100% 100%,62% 100%); }
.hud .cross.hit::before, .hud .cross.hit::after { background: #ffb347; }
.hud .dock { position: absolute; left: 50%; bottom: 16px; transform: translateX(-50%); width: min(500px, calc(100vw - 16px)); display: flex; flex-direction: column; align-items: center; gap: 6px; }
.hud .carousel { display: flex; align-items: center; width: 100%; gap: 2px; }
.hud .rail { position: relative; flex: 1; height: 78px; -webkit-mask-image: linear-gradient(90deg, transparent, #000 16%, #000 84%, transparent); mask-image: linear-gradient(90deg, transparent, #000 16%, #000 84%, transparent); }
.hud .bar { position: absolute; left: 0; bottom: 4px; display: flex; gap: 10px; align-items: flex-end; will-change: transform; }
.hud .slot { position: relative; flex: none; width: clamp(58px, 16vw, 84px); padding: 10px 4px 7px; box-sizing: border-box; border-radius: 10px; background: rgba(14,16,20,0.55); border: 1px solid rgba(255,255,255,0.12); text-align: center; backdrop-filter: blur(6px); transition: transform 220ms cubic-bezier(.2,.9,.3,1.2), opacity 220ms, border-color 140ms, background 140ms; transform-origin: 50% 100%; pointer-events: auto; cursor: pointer; }
.hud .slot:hover { border-color: rgba(255,179,71,0.6); }
.hud .slot.on { border-color: #ffb347; background: rgba(40,28,14,0.75); }
.hud .chev { width: 18px; flex: none; text-align: center; font-size: 20px; line-height: 1; opacity: 0.55; text-shadow: 0 1px 3px #000; transition: opacity 200ms; }
.hud .chev.l { animation: chevl 2.2s ease-in-out infinite; }
.hud .chev.r { animation: chevr 2.2s ease-in-out infinite; }
.hud .chev.end { opacity: 0; }
@keyframes chevl { 0%, 60%, 100% { transform: translateX(0); } 75% { transform: translateX(-3px); } }
@keyframes chevr { 0%, 60%, 100% { transform: translateX(0); } 75% { transform: translateX(3px); } }
.hud .how .mouse .roll { transform-box: fill-box; transform-origin: center; animation: roll 1.5s ease-in-out infinite; }
@keyframes roll { 0% { transform: translateY(-1.5px); opacity: 0; } 25% { opacity: 1; } 70% { transform: translateY(2.5px); opacity: 1; } 100% { transform: translateY(3px); opacity: 0; } }
.hud .slot.pop { animation: pop 280ms ease-out; }
@keyframes pop { 0% { transform: translateY(-6px) scale(1); } 40% { transform: translateY(-10px) scale(1.08); } 100% { transform: translateY(-6px) scale(1); } }
.hud .slot .i { height: 28px; display: flex; justify-content: center; }
.hud .slot .i svg { width: 26px; height: 26px; fill: none; stroke: #f3efe6; stroke-width: 1.7; stroke-linecap: round; stroke-linejoin: round; }
.hud .slot.on .i svg { stroke: #ffb347; }
.hud .slot .n { font-size: 11px; font-weight: 600; letter-spacing: 0.02em; margin-top: 3px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
@media (max-width: 480px) { .hud .slot .n { font-size: 9.5px; letter-spacing: 0; } .hud .how { font-size: 10.5px; gap: 10px; } }
.hud .how { display: flex; flex-wrap: wrap; justify-content: center; gap: 14px; align-items: center; font-size: 11.5px; opacity: 0.7; text-shadow: 0 1px 3px #000; }
.hud .how b { display: inline-block; min-width: 16px; padding: 0 5px; margin-right: 4px; border-radius: 4px; background: rgba(255,255,255,0.16); font-weight: 600; text-align: center; }
.hud .how svg { width: 13px; height: 13px; vertical-align: -2px; margin-right: 4px; fill: none; stroke: currentColor; stroke-width: 1.8; }
.hud .dock.learn .rail { animation: learn 1.6s ease-in-out infinite; }
.hud .dock.learn .how { opacity: 1; }
@keyframes learn { 0%, 100% { filter: drop-shadow(0 0 0 rgba(255,179,71,0)); } 50% { filter: drop-shadow(0 0 10px rgba(255,179,71,0.55)); } }
.hud .tip { position: absolute; left: 50%; bottom: 142px; transform: translateX(-50%); font-size: 12px; opacity: 0.75; text-shadow: 0 1px 3px #000; white-space: nowrap; }
.hud .picked { position: absolute; left: 50%; bottom: 164px; transform: translate(-50%, 6px); text-align: center; opacity: 0; transition: opacity 180ms, transform 180ms; text-shadow: 0 2px 8px rgba(0,0,0,0.8); }
.hud .picked.show { opacity: 1; transform: translate(-50%, 0); }
.hud .picked .t { font-size: 22px; font-weight: 700; letter-spacing: 0.01em; }
.hud .wheel { position: absolute; left: 50%; top: 50%; width: 340px; height: 340px; transform: translate(-50%, -50%) scale(0.92); opacity: 0; transition: opacity 110ms, transform 110ms; pointer-events: none; }
.hud .wheel.open { opacity: 1; transform: translate(-50%, -50%) scale(1); pointer-events: auto; }
.hud .wheel .ring { position: absolute; inset: 0; border-radius: 50%; background: radial-gradient(circle, rgba(10,12,16,0.35) 0 28%, rgba(14,16,20,0.72) 29% 100%); border: 1px solid rgba(255,255,255,0.12); backdrop-filter: blur(6px); }
.hud .wheel .item { position: absolute; width: 84px; height: 70px; margin: -35px 0 0 -42px; display: flex; flex-direction: column; align-items: center; justify-content: center; border-radius: 12px; cursor: pointer; transition: background 100ms, transform 100ms; }
.hud .wheel .item svg { width: 30px; height: 30px; fill: none; stroke: #f3efe6; stroke-width: 1.7; stroke-linecap: round; stroke-linejoin: round; }
.hud .wheel .item span { font-size: 11px; font-weight: 600; margin-top: 4px; }
.hud .wheel .item.on { background: rgba(255,179,71,0.18); transform: scale(1.12); }
.hud .wheel .item.on svg { stroke: #ffb347; }
.hud .wheel .centre { position: absolute; left: 50%; top: 50%; width: 150px; transform: translate(-50%, -50%); text-align: center; font-size: 11px; opacity: 0.85; }
.hud .wheel .centre .t { font-size: 15px; font-weight: 700; margin-bottom: 3px; }
.hud .wheel .pointer { position: absolute; left: 50%; top: 50%; width: 6px; height: 6px; margin: -3px 0 0 -3px; border-radius: 50%; background: #ffb347; }
.hud .rail { pointer-events: auto; touch-action: none; }
.hud .how .swipe .glide { transform-box: fill-box; animation: glide 1.8s ease-in-out infinite; }
@keyframes glide { 0%, 100% { transform: translateX(-3px); } 50% { transform: translateX(3px); } }
/* Touch: minimal. Icons only on the tool bar (the name comes up on a switch), no debug stats,
   no hint lines once the player has switched tools once */
body.touch .hud .keys, body.touch .hud .stats, body.touch .hud .tip { display: none; }
body.touch .hud .dock { width: clamp(200px, calc(100vw - 560px), 300px); bottom: calc(env(safe-area-inset-bottom) + 12px); gap: 4px; }
body.touch .hud .rail { height: 60px; }
body.touch .hud .slot { width: 50px; padding: 8px 2px 6px; border-radius: 12px; }
body.touch .hud .slot .n { display: none; }
body.touch .hud .slot .i svg { width: 24px; height: 24px; }
body.touch .hud .bar { gap: 8px; }
body.touch .hud .chev { width: 12px; font-size: 16px; opacity: 0.35; }
body.touch .hud .dock:not(.learn) .how { display: none; }
body.touch .hud .picked { bottom: calc(env(safe-area-inset-bottom) + 92px); }
body.touch .hud .picked .t { font-size: 17px; }
body.touch .menu .desk, body:not(.touch) .menu .tap { display: none; }
@media (orientation: portrait) {
  body.touch .hud .dock { top: calc(env(safe-area-inset-top) + 58px); bottom: auto; width: min(300px, calc(100vw - 32px)); }
  body.touch .hud .picked { top: calc(env(safe-area-inset-top) + 132px); bottom: auto; }
}
.hud .stats { position: absolute; right: 14px; top: 12px; font: 600 12px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; text-align: right; text-shadow: 0 1px 3px #000; opacity: 0.85; white-space: pre; }
.hud .flash { position: absolute; inset: 0; background: radial-gradient(circle, rgba(255,210,150,0.35), rgba(255,140,60,0) 70%); opacity: 0; transition: opacity 400ms ease-out; }
.hud .speedo { position: absolute; left: 50%; bottom: 34px; transform: translateX(-50%); text-align: center; text-shadow: 0 2px 6px #000; display: none; }
.hud .speedo .v { font: 800 44px/1 ui-monospace, SFMono-Regular, Menlo, monospace; letter-spacing: -0.02em; }
.hud .speedo .u { font-size: 11px; opacity: 0.7; letter-spacing: 0.12em; }
.hud .speedo .gear { position: absolute; left: -34px; top: 4px; font: 800 20px/1 ui-monospace, SFMono-Regular, Menlo, monospace; opacity: 0.8; }
.hud .speedo .pods { display: flex; gap: 6px; justify-content: center; margin-top: 6px; font-size: 10px; letter-spacing: 0.1em; opacity: 0.85; align-items: center; }
.hud .speedo .pod { width: 26px; height: 6px; border-radius: 3px; background: rgba(255,255,255,0.15); overflow: hidden; }
.hud .speedo .pod i { display: block; height: 100%; background: #ff5a3d; }
.hud .speedo .pod.ready i { background: #ffd23d; }
.hud .speedo .boost { height: 4px; width: 120px; margin: 8px auto 0; border-radius: 2px; background: rgba(255,255,255,0.15); overflow: hidden; }
.hud .speedo .boost i { display: block; height: 100%; background: #ff8a3d; }
.hud.driving .dock, .hud.driving .tip, .hud.driving .picked { display: none; }
.hud.driving .speedo { display: block; }
.hud .keys { position: absolute; left: 14px; top: 12px; font-size: 11.5px; line-height: 1.6; opacity: 0.7; text-shadow: 0 1px 3px #000; }
.hud .keys b { display: inline-block; min-width: 18px; padding: 0 4px; border-radius: 4px; background: rgba(255,255,255,0.14); text-align: center; margin-right: 4px; font-weight: 600; }
.menu { position: fixed; inset: 0; z-index: 10; overflow-y: auto; display: flex; align-items: safe center; justify-content: center; padding: 12px 0; box-sizing: border-box; background: radial-gradient(ellipse at center, rgba(10,12,16,0.55), rgba(6,7,10,0.85)); backdrop-filter: blur(4px); }
.menu[hidden] { display: none; }
.menu .card { width: min(460px, 92vw); padding: 26px 28px; border-radius: 16px; background: rgba(22,24,30,0.92); border: 1px solid rgba(255,255,255,0.1); box-shadow: 0 20px 60px rgba(0,0,0,0.5); }
.menu h1 { margin: 0 0 4px; font-size: 30px; letter-spacing: -0.01em; }
.menu p { margin: 0 0 18px; opacity: 0.7; font-size: 13.5px; line-height: 1.5; }
.menu button.play { width: 100%; padding: 13px; border: 0; border-radius: 10px; background: #ff8a3d; color: #1a0f06; font: 700 15px inherit; font-family: inherit; cursor: pointer; }
.menu button.play:hover { background: #ff9d57; }
.menu .buttons { display: flex; gap: 8px; }
.menu .row { display: flex; align-items: center; justify-content: space-between; gap: 14px; margin: 12px 0; font-size: 13px; }
.menu .row input[type=range] { width: 200px; accent-color: #ff8a3d; }
.menu .row select { background: #2a2d35; color: inherit; border: 1px solid rgba(255,255,255,0.15); border-radius: 6px; padding: 4px 8px; font: inherit; }
.menu .tune { font-size: 12px; opacity: 0.75; margin-top: -4px; }
.menu .retune { background: none; border: 1px solid rgba(255,255,255,0.2); color: inherit; border-radius: 6px; padding: 3px 10px; font: inherit; cursor: pointer; }
.menu .retune:hover { border-color: #ff8a3d; }
.topbar { position: fixed; top: 10px; left: 50%; transform: translateX(-50%); z-index: 11; display: flex; align-items: center; gap: 2px; padding: 4px; border-radius: 12px; background: rgba(14,16,20,0.6); border: 1px solid rgba(255,255,255,0.12); backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px); font-size: 13px; font-weight: 600; user-select: none; }
.topbar label { display: flex; align-items: center; gap: 6px; padding-left: 10px; color: rgba(255,255,255,0.55); }
.topbar select { appearance: none; -webkit-appearance: none; padding: 6px 28px 6px 10px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.14); color: #fff; font: inherit; cursor: pointer; background: rgba(255,255,255,0.06) url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M1 1l4 4 4-4' fill='none' stroke='%23ffffff' stroke-opacity='0.7' stroke-width='1.5'/%3E%3C/svg%3E") no-repeat right 10px center; transition: border-color 120ms; }
.topbar select:hover, .topbar select:focus-visible { border-color: #ff8a3d; outline: none; }
.topbar option { color: #fff; background: #16181e; }
.topbar button { padding: 6px 12px; border-radius: 8px; border: 0; background: none; color: rgba(255,255,255,0.72); font: inherit; cursor: pointer; white-space: nowrap; transition: background 120ms, color 120ms; }
.topbar button:hover { color: #fff; background: rgba(255,255,255,0.08); }
.topbar .sep { width: 1px; height: 18px; margin: 0 4px; background: rgba(255,255,255,0.15); }
.topbar kbd { margin-left: 6px; padding: 0 5px; border-radius: 4px; border: 1px solid rgba(255,255,255,0.25); font: inherit; font-size: 11px; opacity: 0.7; }
/* Touch: at the top left, level with the fly and pause buttons in the top right corner */
body.touch .topbar { top: calc(env(safe-area-inset-top) + 12px); left: calc(env(safe-area-inset-left) + 12px); transform: none; font-size: 12px; }
body.touch .topbar button { padding: 6px 9px; }
body.touch .topbar kbd { display: none; }
.menu .grid { display: grid; grid-template-columns: auto 1fr; gap: 4px 14px; margin: 14px 0 18px; font-size: 12.5px; opacity: 0.8; }
.menu .grid b { font-weight: 600; opacity: 0.9; }
.hud .toast { position: absolute; left: 50%; top: 58%; transform: translateX(-50%); padding: 6px 14px; border-radius: 8px; background: rgba(14,16,20,0.6); font-size: 13px; font-weight: 600; opacity: 0; transition: opacity 300ms; }
.loading { position: fixed; left: 50%; top: 58%; transform: translateX(-50%); font-size: 13px; opacity: 0.7; }
`;

/** The key help on foot, and in the car. */
const ON_FOOT = [
  ['WASD', 'move'],
  ['Mouse', 'look · L use · R aim'],
  ['Space / Q', 'up / down (fly), jump (walk)'],
  ['Shift', 'boost · Alt creep'],
  ['Wheel', 'switch tool · hold Tab: tool wheel'],
  ['F', 'fly / walk'],
  ['X', 'slow motion'],
  ['Esc', 'menu'],
];
const DRIVING = [
  ['W / S', 'throttle · brake, reverse'],
  ['A / D', 'steer'],
  ['Space', 'handbrake'],
  ['Shift', 'boost'],
  ['Mouse L', 'machine guns'],
  ['Mouse R', 'rockets'],
  ['F', 'get out'],
  ['X', 'slow motion'],
  ['Esc', 'menu'],
];

export class Hud {
  private readonly root = document.createElement('div');
  private readonly cross = document.createElement('div');
  private readonly bar = document.createElement('div');
  private readonly dock = document.createElement('div');
  private readonly rail = document.createElement('div');
  private readonly chevrons = [document.createElement('div'), document.createElement('div')];
  /** How far the strip is slid off the tool in hand (tools; the scroll in progress). */
  private drag = 0;
  private peekTimer = 0;
  private readonly tip = document.createElement('div');
  private readonly picked = document.createElement('div');
  private pickedTimer = 0;
  private readonly wheel = document.createElement('div');
  private readonly wheelItems: HTMLElement[] = [];
  private readonly wheelCentre = document.createElement('div');
  private readonly wheelPointer = document.createElement('div');
  /** The tool wheel's aim: where the mouse has moved since it opened (px), and the tool under it. */
  private readonly wheelAim = { x: 0, y: 0 };
  private wheelOn = -1;
  private current = 0;
  /** Called by the menu's Reset button: the scene as it started. */
  onReset: () => void = () => {};
  /** Called when a slot or a wheel item is clicked or tapped. */
  onPick: (i: number) => void = () => {};
  /** Called as a finger or the mouse drags the carousel sideways (tools, towards the next one positive). */
  onSwipe: (tools: number) => void = () => {};
  private readonly how = document.createElement('div');
  private readonly statsLine = document.createElement('div');
  private readonly flashEl = document.createElement('div');
  readonly menu = document.createElement('div');
  private readonly loading = document.createElement('div');
  private readonly slots: HTMLElement[] = [];
  private readonly toastEl = document.createElement('div');
  private toastTimer = 0;
  private readonly keysEl = document.createElement('div');
  private readonly speedo = document.createElement('div');

  readonly tools: Tool[];
  readonly settings: Settings;

  constructor(tools: Tool[], settings: Settings, onPlay: () => void, onChange: () => void, onRetune: () => void = () => {}) {
    this.tools = tools;
    this.settings = settings;
    const style = document.createElement('style');
    style.textContent = CSS;
    document.head.append(style);
    this.root.className = 'hud';
    this.cross.className = 'cross';
    this.bar.className = 'bar';
    this.tip.className = 'tip';
    this.statsLine.className = 'stats';
    this.flashEl.className = 'flash';
    const keys = this.keysEl;
    keys.className = 'keys';
    this.setKeys(ON_FOOT);
    this.speedo.className = 'speedo';
    this.speedo.innerHTML = `<div class="gear">1</div><div class="v">0</div><div class="u">KM/H</div><div class="boost"><i></i></div><div class="pods"><span>RKT</span><div class="pod"><i></i></div><div class="pod"><i></i></div></div>`;
    for (const [i, t] of tools.entries()) {
      const slot = document.createElement('div');
      slot.className = 'slot';
      slot.title = t.name;
      slot.innerHTML = `<div class="i">${t.icon}</div><div class="n">${t.name}</div>`;
      this.bar.append(slot);
      this.slots[i] = slot;
    }
    // The carousel: arrows either side of the rail the strip slides in
    this.rail.className = 'rail';
    this.rail.append(this.bar);
    const [left, right] = this.chevrons;
    left.className = 'chev l';
    left.textContent = '‹';
    right.className = 'chev r';
    right.textContent = '›';
    const carousel = document.createElement('div');
    carousel.className = 'carousel';
    carousel.append(left, this.rail, right);
    // How to switch, under the bar: a mouse whose wheel rolls, pulsing until the first switch
    const mouse = `<svg class="mouse" viewBox="0 0 24 24"><rect x="7" y="3" width="10" height="18" rx="5"/><path class="roll" d="M12 7v3"/></svg>`;
    const how = this.how;
    how.className = 'how';
    how.innerHTML = `<span>${mouse}Scroll to switch</span><span><b>Tab</b>hold for tool wheel</span>`;
    this.dock.className = 'dock';
    this.dock.append(carousel, how);
    // Tap a slot to take it; drag the strip sideways (finger or mouse) to slide through them
    let start: { id: number; x: number; last: number; slid: boolean } | null = null;
    this.rail.addEventListener('pointerdown', (e) => {
      if (start) return;
      e.preventDefault();
      e.stopPropagation();
      start = { id: e.pointerId, x: e.clientX, last: e.clientX, slid: false };
      try {
        this.rail.setPointerCapture(e.pointerId);
      } catch {
        // the pointer lifted already
      }
    });
    this.rail.addEventListener('pointermove', (e) => {
      if (!start || e.pointerId !== start.id) return;
      if (!start.slid && Math.abs(e.clientX - start.x) < 8) return;
      start.slid = true;
      const n = this.tools.length;
      const pitch = n > 1 ? this.slots[1].offsetLeft - this.slots[0].offsetLeft : 80;
      this.onSwipe(-(e.clientX - start.last) / pitch);
      start.last = e.clientX;
    });
    const lift = (e: PointerEvent) => {
      if (!start || e.pointerId !== start.id) return;
      const tapped = !start.slid;
      start = null;
      if (!tapped || e.type !== 'pointerup') return;
      const hit = this.slots.findIndex((slot) => {
        const r = slot.getBoundingClientRect();
        return e.clientX >= r.left && e.clientX <= r.right && e.clientY >= r.top - 12 && e.clientY <= r.bottom;
      });
      if (hit >= 0) this.onPick(hit);
    };
    for (const type of ['pointerup', 'pointercancel', 'lostpointercapture'] as const) this.rail.addEventListener(type, lift);
    addEventListener('resize', () => this.place(false));
    this.picked.className = 'picked';
    // The tool wheel: the tools round a ring, the one the mouse points at lit, its use in the middle
    this.wheel.className = 'wheel';
    const ring = document.createElement('div');
    ring.className = 'ring';
    this.wheel.append(ring);
    for (const [i, t] of tools.entries()) {
      const item = document.createElement('div');
      item.className = 'item';
      const a = this.wheelAngle(i);
      item.style.left = `${170 + Math.sin(a) * 118}px`;
      item.style.top = `${170 - Math.cos(a) * 118}px`;
      item.innerHTML = `${t.icon}<span>${t.name}</span>`;
      item.addEventListener('mouseenter', () => this.highlight(i));
      item.addEventListener('mousedown', (e) => {
        e.stopPropagation();
        e.preventDefault();
        this.onPick(i);
        this.closeWheel();
      });
      this.wheel.append(item);
      this.wheelItems[i] = item;
    }
    this.wheelCentre.className = 'centre';
    this.wheelPointer.className = 'pointer';
    this.wheel.append(this.wheelCentre, this.wheelPointer);
    this.toastEl.className = 'toast';
    this.root.append(this.flashEl, this.cross, this.dock, this.tip, this.picked, this.wheel, this.statsLine, keys, this.speedo, this.toastEl);
    let learn = true;
    try {
      learn = !localStorage.getItem('city.switched');
    } catch {
      // storage unavailable: teach every time
    }
    if (learn) {
      this.dock.classList.add('learn');
      // A peek towards the next tool now and then, until the player switches
      this.peekTimer = window.setInterval(() => {
        if (this.drag !== 0 || this.menu.hidden === false) return;
        const way = this.current < this.tools.length - 1 ? 1 : -1;
        this.drag = 0.3 * way;
        this.place(true);
        window.setTimeout(() => {
          this.drag = 0;
          this.place(true);
        }, 420);
      }, 2600);
    }
    requestAnimationFrame(() => this.place(false));
    document.body.append(this.root);

    this.menu.className = 'menu';
    // Which scene, and links to each (the page's other parameters kept)
    const scene = new URLSearchParams(location.search).get('scene');
    const track = scene === 'track';
    const space = scene === 'space';
    const link = (scene: string | null) => {
      const q = new URLSearchParams(location.search);
      if (scene) q.set('scene', scene);
      else q.delete('scene');
      const text = q.toString();
      return text ? `?${text}` : location.pathname;
    };
    this.menu.innerHTML = `<div class="card">
      <h1>${track ? 'Voxel Circuit' : space ? 'Voxel Orbit' : 'Voxel City'}</h1>
      <p>${
        track
          ? 'A race car on a circuit where every wall, stand and tyre is voxels in the AVBD solver, on your GPU. Drive through them, open up the machine guns and rockets, or get out and take the place apart on foot.'
          : space
            ? 'A space station in orbit, every truss, module and solar wing voxels in the AVBD solver, on your GPU, with no gravity. Cut a wing loose and it drifts off; blow a module apart and it scatters into the dark.'
            : "Every one of the city's voxels is a body in the AVBD solver, running on your GPU. Knock a hole in a tower and whatever it held up comes down."
      }</p>
      <div class="grid tap">
        <b>Left thumb</b><span>move · push past the ring to sprint</span>
        <b>Right thumb</b><span>drag anywhere to look</span>
        <b>Fire</b><span>use the tool in hand · drag from it to aim while firing</span>
        <b>Aim</b><span>tap to zoom in, tap again to zoom out</span>
        <b>Arrows</b><span>up and down (flying) · jump and crouch (walking)</span>
        <b>Tools</b><span>swipe or tap the bar</span>
      </div>
      <div class="grid desk">
        <b>Mouse</b><span>look · left click uses the tool · right click aims</span>
        <b>WASD</b><span>move where you look · Space/E up · Q/C down · Shift fast · Alt slow</span>
        <b>Tools</b><span>scroll to switch, or hold Tab for the tool wheel</span>
        <b>F</b><span>fly or walk (walking: Space jump, C crouch, Shift sprint)</span>
        <b>X</b><span>slow motion (the player moves at full speed: fly round a collapse)</span>
        <b>T · R</b><span>${space ? 'turn the sun' : 'time of day'} · rebuild the ${track ? 'track' : space ? 'station' : 'city'}</span>
        ${track ? '<b>In the car</b><span>W/S throttle and brake · A/D steer · Space handbrake · Shift boost · left button: the machine guns · right button: the rockets · F gets out (and back in)</span>' : ''}
      </div>
      <div class="row"><span>Look sensitivity</span><input type="range" name="sensitivity" min="0.2" max="3" step="0.05"></div>
      <div class="row"><span>Field of view</span><input type="range" name="fov" min="60" max="110" step="1"></div>
      <div class="row"><span>Volume</span><input type="range" name="volume" min="0" max="1" step="0.05"></div>
      <div class="row"><span>Time of day</span><input type="range" name="hour" min="6.5" max="21.5" step="0.1"></div>
      <div class="row"><span>Invert look</span><input type="checkbox" name="invertY"></div>
      <div class="row"><span>Mouse smoothing</span><input type="checkbox" name="smoothing"></div>
      <div class="row"><span>Depth of field</span><input type="checkbox" name="dof"></div>
      <div class="row"><span>Sun rays</span><input type="checkbox" name="sunRays"></div>
      <div class="row"><span>Graphics</span><select name="graphics"><option value="auto">Auto (tuned for this device)</option><option value="low">Low</option><option value="medium">Medium</option><option value="high">High</option></select></div>
      <div class="row tune"><span class="tuned"></span><button class="retune">Re-tune</button></div>
      <div class="buttons"><button class="play">Play</button></div>
    </div>`;
    document.body.append(this.menu);
    this.loading.className = 'loading';
    this.menu.append(this.loading);
    const input = (name: string) => this.menu.querySelector(`[name=${name}]`) as HTMLInputElement;
    for (const name of ['sensitivity', 'fov', 'volume', 'hour'] as const) {
      input(name).value = String(settings[name]);
      input(name).addEventListener('input', () => {
        settings[name] = Number(input(name).value);
        onChange();
      });
    }
    for (const name of ['invertY', 'smoothing', 'dof', 'sunRays'] as const) {
      input(name).checked = settings[name];
      input(name).addEventListener('change', () => {
        settings[name] = input(name).checked;
        onChange();
      });
    }
    (this.menu.querySelector('.retune') as HTMLButtonElement).addEventListener('click', () => onRetune());
    const quality = this.menu.querySelector('select') as HTMLSelectElement;
    quality.value = settings.graphics;
    quality.addEventListener('change', () => {
      settings.graphics = quality.value as Settings['graphics'];
      onChange();
    });
    (this.menu.querySelector('.play') as HTMLButtonElement).addEventListener('click', onPlay);
    // The bar across the top: which scene (a choice of them), and the scene as it started. Above the menu too
    // (with the pointer locked in play it can't be clicked: there it shows where you are, and R resets)
    const bar = document.createElement('nav');
    bar.className = 'topbar';
    const current = track ? 'track' : space ? 'space' : '';
    const option = (value: string, name: string) => `<option value="${value}"${value === current ? ' selected' : ''}>${name}</option>`;
    bar.innerHTML = `<label>Scene<select>${option('', 'City')}${option('track', 'Race track')}${option('space', 'Space station')}</select></label><span class="sep"></span><button class="reset" title="The scene as it started">Reset<kbd>R</kbd></button>`;
    const choice = bar.querySelector('select') as HTMLSelectElement;
    choice.addEventListener('change', () => location.assign(link(choice.value || null)));
    (bar.querySelector('.reset') as HTMLButtonElement).addEventListener('click', () => this.onReset());
    document.body.append(bar);
  }

  /** What the device tuning chose, shown under the Graphics setting. */
  setTuned(text: string): void {
    (this.menu.querySelector('.tuned') as HTMLElement).textContent = text;
  }

  setLoading(text: string | null): void {
    this.loading.textContent = text ?? '';
    (this.menu.querySelector('.play') as HTMLButtonElement).disabled = !!text;
  }

  /**
   * Show tool `i` as the one in hand. `announce`: the player switched to it (not the start),
   * so the slot bounces and its name and use come up for a moment.
   */
  select(i: number, announce = false): void {
    this.current = i;
    this.drag = 0;
    this.slots.forEach((s, k) => s.classList.toggle('on', k === i));
    this.tip.textContent = this.tools[i].hint;
    this.place(true);
    if (!announce) return;
    const slot = this.slots[i];
    slot.classList.remove('pop');
    void slot.offsetWidth;
    slot.classList.add('pop');
    this.picked.innerHTML = `<div class="t">${this.tools[i].name}</div>`;
    this.picked.classList.add('show');
    clearTimeout(this.pickedTimer);
    this.pickedTimer = window.setTimeout(() => this.picked.classList.remove('show'), 900);
    if (this.dock.classList.contains('learn')) {
      this.dock.classList.remove('learn');
      clearInterval(this.peekTimer);
      try {
        localStorage.setItem('city.switched', '1');
      } catch {
        // storage unavailable: it will pulse again next time
      }
    }
  }

  /**
   * Slide the strip `offset` tools off the tool in hand (the scroll in progress: its sign the
   * way it goes), following the finger. Past the first or last tool it gives only a little.
   */
  nudge(offset: number): void {
    this.drag = offset;
    this.place(false);
  }

  /** Let go of a slide that didn't reach the next tool: the strip springs back. */
  release(): void {
    if (this.drag === 0) return;
    this.drag = 0;
    this.place(true);
  }

  /**
   * Lay out the carousel: the strip shifted so the slide position (tool in hand plus drag,
   * rubber-banded at the ends) sits mid-rail, each slot smaller and fainter with its distance.
   * `spring`: ease there (a snap), else follow at once (a finger).
   */
  private place(spring: boolean): void {
    const n = this.tools.length;
    let pos = this.current + this.drag;
    const give = 0.22;
    if (pos < 0) pos = -give * Math.tanh(-pos / give);
    if (pos > n - 1) pos = n - 1 + give * Math.tanh((pos - (n - 1)) / give);
    const first = this.slots[0];
    const pitch = n > 1 ? this.slots[1].offsetLeft - first.offsetLeft : 0;
    const centre = first.offsetLeft + first.offsetWidth / 2;
    this.bar.style.transition = spring ? 'transform 300ms cubic-bezier(.2,.9,.25,1.12)' : 'transform 70ms linear';
    this.bar.style.transform = `translateX(${this.rail.clientWidth / 2 - (centre + pos * pitch)}px)`;
    this.slots.forEach((slot, k) => {
      const d = Math.min(Math.abs(k - pos), 2.2);
      const lift = k === this.current ? -6 : 0;
      slot.style.transform = `translateY(${lift}px) scale(${1 - d * 0.1})`;
      slot.style.opacity = String(1 - d * 0.26);
    });
    this.chevrons[0].classList.toggle('end', this.current === 0);
    this.chevrons[1].classList.toggle('end', this.current === n - 1);
  }

  /** Touch controls: say how to switch tools with a finger, and show the touch help in the menu. */
  setTouch(): void {
    const swipe = `<svg class="swipe" viewBox="0 0 24 24"><g class="glide"><path d="M4 12h16"/><path d="M8 8l-4 4 4 4"/><path d="M16 8l4 4-4 4"/></g></svg>`;
    this.how.innerHTML = `<span>${swipe}Swipe or tap to switch</span>`;
  }

  /** The angle of tool i round the wheel (radians clockwise from the top). */
  private wheelAngle(i: number): number {
    return (i / this.tools.length) * 2 * Math.PI;
  }

  private highlight(i: number): void {
    this.wheelOn = i;
    this.wheelItems.forEach((item, k) => item.classList.toggle('on', k === i));
    const t = this.tools[i];
    this.wheelCentre.innerHTML = `<div class="t">${t.name}</div>${t.hint}`;
  }

  get wheelOpen(): boolean {
    return this.wheel.classList.contains('open');
  }

  /** Open the tool wheel on the tool in hand. */
  openWheel(): void {
    this.wheelAim.x = 0;
    this.wheelAim.y = 0;
    this.wheelPointer.style.transform = '';
    this.highlight(this.current);
    this.wheel.classList.add('open');
  }

  /**
   * Mouse movement while the wheel is open (px): the aim moves with it, held within the ring,
   * and past a small dead zone the tool in its direction lights up.
   */
  aimWheel(dx: number, dy: number): void {
    const aim = this.wheelAim;
    aim.x += dx;
    aim.y += dy;
    const r = Math.hypot(aim.x, aim.y);
    if (r > 110) {
      aim.x *= 110 / r;
      aim.y *= 110 / r;
    }
    this.wheelPointer.style.transform = `translate(${aim.x}px, ${aim.y}px)`;
    if (r < 24) return;
    const angle = Math.atan2(aim.x, -aim.y);
    const n = this.tools.length;
    this.highlight((Math.round((angle / (2 * Math.PI)) * n) + n) % n);
  }

  /** Close the tool wheel: the tool it points at, or -1 if it was closed already. */
  closeWheel(): number {
    if (!this.wheelOpen) return -1;
    this.wheel.classList.remove('open');
    return this.wheelOn;
  }

  /** A short message mid-screen (mode, speed), fading after a moment. */
  toast(text: string): void {
    this.toastEl.textContent = text;
    this.toastEl.style.opacity = '1';
    clearTimeout(this.toastTimer);
    this.toastTimer = window.setTimeout(() => (this.toastEl.style.opacity = '0'), 1400);
  }

  aim(onSomething: boolean): void {
    this.cross.classList.toggle('hit', onSomething);
  }

  stats(text: string): void {
    this.statsLine.textContent = text;
  }

  /** The key help, top left: [key, what] rows. */
  setKeys(rows: string[][]): void {
    this.keysEl.innerHTML = rows.map(([k, v]) => `<b>${k}</b>${v}`).join('<br>');
  }

  /** Driving: the speedometer and the car's keys instead of the tool bar (the track). */
  setDriving(on: boolean): void {
    this.root.classList.toggle('driving', on);
    this.setKeys(on ? DRIVING : ON_FOOT);
  }

  /** The speedometer: km/h, the boost left (0..1), the gear, and each rocket launcher's reload (0..1, 1 loaded). */
  speed(kmh: number, boost: number, gear: number, launchers: number[]): void {
    (this.speedo.querySelector('.v') as HTMLElement).textContent = String(Math.round(Math.abs(kmh)));
    (this.speedo.querySelector('.gear') as HTMLElement).textContent = kmh < -1 ? 'R' : String(gear);
    (this.speedo.querySelector('.boost i') as HTMLElement).style.width = `${Math.round(boost * 100)}%`;
    this.speedo.querySelectorAll('.pod').forEach((pod, k) => {
      const f = launchers[k] ?? 1;
      (pod.firstElementChild as HTMLElement).style.width = `${Math.round(f * 100)}%`;
      pod.classList.toggle('ready', f >= 1);
    });
  }

  /** A warm flash over the screen, stronger for nearer, bigger blasts (0..1). */
  flash(strength: number): void {
    this.flashEl.style.transition = 'none';
    this.flashEl.style.opacity = String(Math.min(1, strength));
    void this.flashEl.offsetWidth;
    this.flashEl.style.transition = 'opacity 500ms ease-out';
    this.flashEl.style.opacity = '0';
  }

  showMenu(on: boolean): void {
    this.menu.hidden = !on;
    this.root.style.opacity = on ? '0.35' : '1';
  }
}
