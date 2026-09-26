// Touch controls for phones and tablets, laid out and behaving as mobile shooters do (Fortnite,
// COD Mobile): a floating stick on the left to move (push past its ring to sprint, or boost
// when flying), and on the right, drag anywhere to look, the view following the thumb as a
// mouse would (a stick that turns at a rate felt sluggish to aim with). The fire button shows
// the tool in hand; dragging from it keeps turning the view, so you aim while firing. Around it
// in an arc: aim (tap to toggle), jump or up, crouch or down. Small icon buttons at the top
// right fly or walk and pause. Minimal: the stick rests as a faint ring, and nothing else is on
// screen. Pointer events, each finger followed by its id, so moving, looking and firing work
// together. Laid out for landscape and portrait by CSS (orientation queries).

export interface TouchActions {
  /** The fire button went down (true) or up (false). */
  fire(down: boolean): void;
  aim(on: boolean): void;
  toggleFly(): void;
  pause(): void;
}

const CSS = `
body.touch, body.touch * { -webkit-user-select: none; user-select: none; -webkit-touch-callout: none; -webkit-tap-highlight-color: transparent; }
body.touch { touch-action: none; overscroll-behavior: none; }
.touch-ui { position: fixed; inset: 0; z-index: 0; touch-action: none; }
.touch-ui[hidden] { display: none; }
.touch-ui .zone { position: absolute; touch-action: none; }
.touch-ui .zone.move { left: 0; bottom: 0; width: 42%; height: 72%; }
.touch-ui .zone.look { right: 0; top: 0; width: 58%; height: 100%; }
.touch-ui .stick { position: absolute; width: 112px; height: 112px; margin: -56px 0 0 -56px; border-radius: 50%; border: 1.5px solid rgba(255,255,255,0.22); background: radial-gradient(circle, rgba(0,0,0,0) 55%, rgba(0,0,0,0.18)); pointer-events: none; transition: opacity 180ms; opacity: 0.55; }
.touch-ui .stick.rest { margin: 0; left: calc(env(safe-area-inset-left) + 44px); bottom: calc(env(safe-area-inset-bottom) + 34px); }
.touch-ui .stick.on { opacity: 1; transition: none; }
.touch-ui .stick .knob { position: absolute; left: 50%; top: 50%; width: 44px; height: 44px; margin: -22px 0 0 -22px; border-radius: 50%; background: rgba(243,239,230,0.9); box-shadow: 0 2px 8px rgba(0,0,0,0.35); transition: background 120ms, box-shadow 120ms; }
.touch-ui .stick.rest .knob { width: 14px; height: 14px; margin: -7px 0 0 -7px; background: rgba(243,239,230,0.55); box-shadow: none; }
.touch-ui .stick.sprint .knob { background: #ffb347; box-shadow: 0 0 14px rgba(255,160,60,0.8); }
.touch-ui .stick .run { position: absolute; left: 50%; top: -22px; transform: translateX(-50%); font: 700 12px system-ui, sans-serif; color: #ffb347; opacity: 0; transition: opacity 120ms; text-shadow: 0 1px 3px #000; }
.touch-ui .stick.on .run { opacity: 0.6; }
.touch-ui .stick.sprint .run { opacity: 1; }
.touch-ui .btn { position: absolute; display: flex; align-items: center; justify-content: center; border-radius: 50%; background: rgba(12,14,18,0.38); border: 1.5px solid rgba(255,255,255,0.3); color: #f3efe6; touch-action: none; transition: transform 80ms, background 120ms, border-color 120ms; }
.touch-ui .btn svg { fill: none; stroke: currentColor; stroke-width: 1.9; stroke-linecap: round; stroke-linejoin: round; }
.touch-ui .btn.down { transform: scale(0.9); background: rgba(255,138,61,0.5); border-color: #ffb347; }
.touch-ui .btn.on { background: rgba(255,138,61,0.42); border-color: #ffb347; color: #fff; }
.touch-ui .fire { width: 80px; height: 80px; background: rgba(255,138,61,0.26); border: 2px solid rgba(255,179,71,0.85); }
.touch-ui .fire svg { width: 34px; height: 34px; }
.touch-ui .act { width: 52px; height: 52px; }
.touch-ui .act svg { width: 22px; height: 22px; }
.touch-ui .top { width: 38px; height: 38px; border-color: rgba(255,255,255,0.2); background: rgba(12,14,18,0.3); }
.touch-ui .top svg { width: 18px; height: 18px; }
/* Landscape: the stick bottom left; the fire button bottom right, its actions in an arc */
.touch-ui .fire { right: calc(env(safe-area-inset-right) + 56px); bottom: calc(env(safe-area-inset-bottom) + 68px); }
.touch-ui .aim { right: calc(env(safe-area-inset-right) + 150px); bottom: calc(env(safe-area-inset-bottom) + 58px); }
.touch-ui .up { right: calc(env(safe-area-inset-right) + 142px); bottom: calc(env(safe-area-inset-bottom) + 146px); }
.touch-ui .dn { right: calc(env(safe-area-inset-right) + 60px); bottom: calc(env(safe-area-inset-bottom) + 170px); }
.touch-ui .fly { right: calc(env(safe-area-inset-right) + 60px); top: calc(env(safe-area-inset-top) + 12px); }
.touch-ui .pause { right: calc(env(safe-area-inset-right) + 14px); top: calc(env(safe-area-inset-top) + 12px); }
@media (orientation: portrait) {
  .touch-ui .zone.move { width: 50%; height: 50%; }
  .touch-ui .zone.look { width: 100%; height: 100%; }
  .touch-ui .stick.rest { left: 28px; bottom: calc(env(safe-area-inset-bottom) + 56px); }
  .touch-ui .fire { right: 36px; bottom: calc(env(safe-area-inset-bottom) + 84px); }
  .touch-ui .aim { right: 130px; bottom: calc(env(safe-area-inset-bottom) + 70px); }
  .touch-ui .up { right: 122px; bottom: calc(env(safe-area-inset-bottom) + 160px); }
  .touch-ui .dn { right: 40px; bottom: calc(env(safe-area-inset-bottom) + 186px); }
}
`;

const ICONS = {
  up: `<svg viewBox="0 0 24 24"><path d="M6 14l6-6 6 6"/><path d="M6 19l6-6 6 6" opacity="0.45"/></svg>`,
  down: `<svg viewBox="0 0 24 24"><path d="M6 10l6 6 6-6"/><path d="M6 5l6 6 6-6" opacity="0.45"/></svg>`,
  aim: `<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="6.5"/><circle cx="12" cy="12" r="1.2"/><path d="M12 2.5v4M12 17.5v4M2.5 12h4M17.5 12h4"/></svg>`,
  pause: `<svg viewBox="0 0 24 24"><path d="M9 6v12M15 6v12"/></svg>`,
  fly: `<svg viewBox="0 0 24 24"><path d="M3 13c4-1 6-5 9-9 0 5 2 7 9 8-6 1-8 3-9 8-2-4-5-6-9-7z"/></svg>`,
  walk: `<svg viewBox="0 0 24 24"><circle cx="13" cy="4.5" r="1.8"/><path d="M11 21l2-6-3-3 1-4 4 3 3 1"/><path d="M9 8l-3 4"/><path d="M13 15l3 6"/></svg>`,
};

/** Follow this finger even when it leaves the element (it may already be gone: then never mind). */
function capture(el: HTMLElement, id: number): void {
  try {
    el.setPointerCapture(id);
  } catch {
    // the pointer lifted already
  }
}

/** Short buzz where the device has one (Android; iOS browsers don't). */
export function buzz(ms: number): void {
  try {
    navigator.vibrate?.(ms);
  } catch {
    // no vibration
  }
}

/**
 * The floating movement stick in `zone`: its value (-1..1 each way, y down), shaped for fine
 * control near the centre, and `sprint` while the thumb is pushed past the ring.
 */
class Stick {
  readonly value = { x: 0, y: 0 };
  sprint = false;
  readonly el = document.createElement('div');
  private readonly knob = document.createElement('div');
  private finger = -1;
  private readonly centre = { x: 0, y: 0 };
  private readonly zone: HTMLElement;
  private readonly radius = 44;

  constructor(zone: HTMLElement) {
    this.zone = zone;
    this.el.className = 'stick rest';
    this.knob.className = 'knob';
    const run = document.createElement('div');
    run.className = 'run';
    run.textContent = '»';
    this.el.append(run, this.knob);
    zone.parentElement!.append(this.el);
    zone.addEventListener('pointerdown', (e) => this.down(e));
    zone.addEventListener('pointermove', (e) => this.move(e));
    for (const type of ['pointerup', 'pointercancel', 'lostpointercapture'] as const) zone.addEventListener(type, (e) => this.up(e));
  }

  private down(e: PointerEvent): void {
    if (this.finger >= 0) return;
    e.preventDefault();
    this.finger = e.pointerId;
    capture(this.zone, e.pointerId);
    // The stick comes to the thumb (kept whole on screen)
    const m = 60;
    this.centre.x = Math.min(Math.max(e.clientX, m), innerWidth - m);
    this.centre.y = Math.min(Math.max(e.clientY, m), innerHeight - m);
    this.el.classList.remove('rest');
    this.el.classList.add('on');
    this.el.style.left = `${this.centre.x}px`;
    this.el.style.top = `${this.centre.y}px`;
    this.move(e);
  }

  private move(e: PointerEvent): void {
    if (e.pointerId !== this.finger) return;
    e.preventDefault();
    let dx = e.clientX - this.centre.x;
    let dy = e.clientY - this.centre.y;
    const d = Math.hypot(dx, dy);
    // Past the ring: sprint (the knob stays on the ring)
    const sprint = d > this.radius * 1.45 && -dy > Math.abs(dx) * 0.6;
    if (sprint !== this.sprint) {
      this.sprint = sprint;
      this.el.classList.toggle('sprint', sprint);
      if (sprint) buzz(6);
    }
    // A thumb dragged far off drags the stick along (it stays under the thumb)
    if (d > this.radius * 2.2) {
      const pull = (d - this.radius * 2.2) / d;
      this.centre.x += dx * pull;
      this.centre.y += dy * pull;
      dx -= dx * pull;
      dy -= dy * pull;
      this.el.style.left = `${this.centre.x}px`;
      this.el.style.top = `${this.centre.y}px`;
    }
    const r = Math.min(Math.hypot(dx, dy), this.radius);
    const k = r / Math.max(Math.hypot(dx, dy), 1e-6);
    this.knob.style.transform = `translate(${dx * k}px, ${dy * k}px)`;
    // A small dead zone, then a curve: small tilts creep, full tilt goes at full speed
    const t = r / this.radius;
    const shaped = t < 0.12 ? 0 : ((t - 0.12) / 0.88) ** 1.8;
    const len = Math.max(Math.hypot(dx, dy), 1e-6);
    this.value.x = (dx / len) * shaped;
    this.value.y = (dy / len) * shaped;
  }

  private up(e: PointerEvent): void {
    if (e.pointerId !== this.finger) return;
    this.finger = -1;
    this.value.x = this.value.y = 0;
    this.sprint = false;
    this.knob.style.transform = '';
    this.el.classList.remove('on', 'sprint');
    this.el.classList.add('rest');
    this.el.style.left = this.el.style.top = '';
  }

  /** Let go (the game paused). */
  reset(): void {
    if (this.finger >= 0) this.up({ pointerId: this.finger } as PointerEvent);
  }
}

export class TouchControls {
  readonly root = document.createElement('div');
  private readonly stick: Stick;
  private readonly fireButton: HTMLElement;
  private readonly aimButton: HTMLElement;
  private readonly flyButton: HTMLElement;
  private upHeld = false;
  private downHeld = false;
  private aiming = false;
  /** Look movement since it was last taken (px), from any finger dragging to look. */
  private readonly lookDelta = { x: 0, y: 0 };
  private readonly lookers = new Map<number, { x: number; y: number }>();
  private readonly actions: TouchActions;

  constructor(actions: TouchActions) {
    this.actions = actions;
    const style = document.createElement('style');
    style.textContent = CSS;
    document.head.append(style);
    document.body.classList.add('touch');
    this.root.className = 'touch-ui';
    const zone = (kind: string) => {
      const z = document.createElement('div');
      z.className = `zone ${kind}`;
      this.root.append(z);
      return z;
    };
    // The look area first (under), the move area over its left, the buttons on top
    this.drag(zone('look'));
    this.stick = new Stick(zone('move'));
    this.fireButton = this.button('fire', '', (down) => {
      actions.fire(down);
      if (down) buzz(10);
    });
    this.drag(this.fireButton);
    this.aimButton = this.button('act aim', ICONS.aim, (down) => down && this.setAim(!this.aiming));
    this.button('act up', ICONS.up, (down) => (this.upHeld = down));
    this.button('act dn', ICONS.down, (down) => (this.downHeld = down));
    this.flyButton = this.button('top fly', ICONS.walk, (down) => down && actions.toggleFly());
    this.button('top pause', ICONS.pause, (down) => down && actions.pause());
    document.body.append(this.root);
  }

  /** Dragging a finger on `el` turns the view (it adds to lookDelta). */
  private drag(el: HTMLElement): void {
    el.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      this.lookers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      capture(el, e.pointerId);
    });
    el.addEventListener('pointermove', (e) => {
      const at = this.lookers.get(e.pointerId);
      if (!at) return;
      e.preventDefault();
      this.lookDelta.x += e.clientX - at.x;
      this.lookDelta.y += e.clientY - at.y;
      at.x = e.clientX;
      at.y = e.clientY;
    });
    for (const type of ['pointerup', 'pointercancel', 'lostpointercapture'] as const) el.addEventListener(type, (e) => this.lookers.delete(e.pointerId));
  }

  /** A button: `hold` gets true when a finger goes down on it and false when it comes up. */
  private button(kind: string, content: string, hold: (down: boolean) => void): HTMLElement {
    const b = document.createElement('div');
    b.className = `btn ${kind}`;
    b.innerHTML = content;
    let finger = -1;
    b.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (finger >= 0) return;
      finger = e.pointerId;
      capture(b, e.pointerId);
      b.classList.add('down');
      hold(true);
    });
    const release = (e: PointerEvent) => {
      if (e.pointerId !== finger) return;
      finger = -1;
      b.classList.remove('down');
      hold(false);
    };
    for (const type of ['pointerup', 'pointercancel', 'lostpointercapture'] as const) b.addEventListener(type, release);
    this.root.append(b);
    return b;
  }

  private setAim(on: boolean): void {
    this.aiming = on;
    this.aimButton.classList.toggle('on', on);
    this.actions.aim(on);
  }

  /** Movement: forward and right (-1..1). */
  get move(): { forward: number; right: number } {
    return { forward: -this.stick.value.y, right: this.stick.value.x };
  }

  /** The thumb pushed past the stick's ring, forward: sprint (boost when flying). */
  get sprint(): boolean {
    return this.stick.sprint;
  }

  /** Look movement since the last call (px, y down), then cleared. */
  takeLook(): { x: number; y: number } {
    const d = { x: this.lookDelta.x, y: this.lookDelta.y };
    this.lookDelta.x = this.lookDelta.y = 0;
    return d;
  }

  /** Up and down held (flying), or jump and crouch (walking). */
  get up(): boolean {
    return this.upHeld;
  }

  get down(): boolean {
    return this.downHeld;
  }

  /** Show the tool in hand on the fire button. */
  setTool(icon: string): void {
    this.fireButton.innerHTML = icon;
  }

  /** Flying or walking: the toggle shows what it switches to. */
  setFlying(flying: boolean): void {
    this.flyButton.innerHTML = flying ? ICONS.walk : ICONS.fly;
  }

  set visible(on: boolean) {
    this.root.hidden = !on;
    if (!on) {
      this.stick.reset();
      this.lookers.clear();
      this.lookDelta.x = this.lookDelta.y = 0;
      this.upHeld = this.downHeld = false;
      if (this.aiming) this.setAim(false);
    }
  }
}

/**
 * Keep the page from zooming, and put it back if it has: iOS ignores the viewport's
 * user-scalable, and a double tap or a pinch outside the controls zoomed the view in for good.
 * touch-action (city.html) stops most of it; this stops Safari's pinch gestures, double taps
 * and two-finger moves, a desktop trackpad's pinch (ctrl + wheel), and, should the page be
 * zoomed anyway, resets the viewport (setting it again makes iOS return to its scale).
 */
export function preventZoom(): void {
  const stop = (e: Event) => e.preventDefault();
  for (const type of ['gesturestart', 'gesturechange', 'gestureend', 'dblclick']) document.addEventListener(type, stop, { passive: false });
  document.addEventListener(
    'touchmove',
    (e) => {
      if (e.touches.length > 1 && !(e.target instanceof Element && e.target.closest('.touch-ui'))) e.preventDefault();
    },
    { passive: false },
  );
  // Two taps in quick succession on the same spot: the second would zoom
  let lastTap = 0;
  document.addEventListener(
    'touchend',
    (e) => {
      const now = performance.now();
      if (now - lastTap < 350 && e.touches.length === 0 && !(e.target instanceof Element && e.target.closest('.menu, button, input, select, a'))) e.preventDefault();
      lastTap = now;
    },
    { passive: false },
  );
  addEventListener('wheel', (e) => e.ctrlKey && e.preventDefault(), { passive: false });
  // Zoomed after all: set the viewport again to bring it back
  const meta = document.querySelector('meta[name=viewport]');
  const reset = () => {
    const scale = window.visualViewport?.scale ?? 1;
    if (!meta || Math.abs(scale - 1) < 0.01) return;
    const content = meta.getAttribute('content') ?? '';
    meta.setAttribute('content', `${content.replace(/,?\s*maximum-scale=[^,]*/, '')}, maximum-scale=1.0001`);
    requestAnimationFrame(() => meta.setAttribute('content', content));
    window.scrollTo(0, 0);
  };
  window.visualViewport?.addEventListener('resize', reset);
  addEventListener('orientationchange', () => setTimeout(reset, 300));
}

/** Whether this is a touch device (a coarse pointer and no fine one), or `?touch` asks for the controls. */
export function wantsTouch(): boolean {
  if (new URLSearchParams(location.search).has('touch')) return true;
  return matchMedia('(pointer: coarse)').matches && !matchMedia('(any-pointer: fine)').matches;
}
