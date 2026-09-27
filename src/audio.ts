// Sounds, synthesised (no samples to load): a blast is a burst of noise through a closing
// low-pass with a sub-bass thump; the hammer a short knock; the blaster a falling chirp; a
// collapse a long low rumble. Quieter and duller with distance.

export class Sounds {
  private ctx: AudioContext | null = null;
  private master: GainNode | null = null;
  private noise: AudioBuffer | null = null;
  volume = 0.7;

  /** Start the audio (browsers only allow it after a click). */
  resume(): void {
    if (!this.ctx) {
      this.ctx = new AudioContext();
      this.master = this.ctx.createGain();
      this.master.connect(this.ctx.destination);
      const n = this.ctx.sampleRate * 3;
      this.noise = this.ctx.createBuffer(1, n, this.ctx.sampleRate);
      const d = this.noise.getChannelData(0);
      for (let i = 0; i < n; i++) d[i] = Math.random() * 2 - 1;
    }
    void this.ctx.resume();
    this.master!.gain.value = this.volume;
  }

  /** Everything played, as a stream too (for recording the game with its sound). */
  tap(): MediaStream | null {
    if (!this.ctx || !this.master) return null;
    const out = this.ctx.createMediaStreamDestination();
    this.master.connect(out);
    return out.stream;
  }

  setVolume(v: number): void {
    this.volume = v;
    if (this.master) this.master.gain.value = v;
  }

  /** Noise through a low-pass falling from `from` to `to` Hz over `seconds`, at `gain`. */
  private rumble(gain: number, from: number, to: number, seconds: number, delay = 0): void {
    const { ctx } = this;
    if (!ctx || !this.master) return;
    const t = ctx.currentTime + delay;
    const src = ctx.createBufferSource();
    src.buffer = this.noise;
    src.playbackRate.value = 0.6 + Math.random() * 0.2;
    const lp = ctx.createBiquadFilter();
    lp.type = 'lowpass';
    lp.frequency.setValueAtTime(from, t);
    lp.frequency.exponentialRampToValueAtTime(Math.max(to, 20), t + seconds);
    const g = ctx.createGain();
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(gain, t + 0.012);
    g.gain.exponentialRampToValueAtTime(0.0001, t + seconds);
    src.connect(lp).connect(g).connect(this.master);
    src.start(t, Math.random() * 1.5, seconds + 0.1);
  }

  private tone(gain: number, from: number, to: number, seconds: number, type: OscillatorType = 'sine', delay = 0): void {
    const { ctx } = this;
    if (!ctx || !this.master) return;
    const t = ctx.currentTime + delay;
    const osc = ctx.createOscillator();
    osc.type = type;
    osc.frequency.setValueAtTime(from, t);
    osc.frequency.exponentialRampToValueAtTime(to, t + seconds);
    const g = ctx.createGain();
    g.gain.setValueAtTime(gain, t);
    g.gain.exponentialRampToValueAtTime(0.0001, t + seconds);
    osc.connect(g).connect(this.master);
    osc.start(t);
    osc.stop(t + seconds + 0.05);
  }

  /** A blast of `radius` m heard from `distance` m. */
  blast(radius: number, distance: number): void {
    const near = 1 / (1 + distance / 40);
    const size = Math.min(1, radius / 7);
    this.rumble(0.9 * near * (0.4 + size), 4000 * near + 300, 60, 1.2 + 2 * size);
    this.tone(0.8 * near * (0.3 + size), 90, 28, 0.6 + size);
    this.rumble(0.35 * near * size, 900, 40, 3 + 2 * size, 0.15);
  }

  hammer(distance: number): void {
    const near = 1 / (1 + distance / 10);
    this.rumble(0.6 * near, 2200, 200, 0.18);
    this.tone(0.4 * near, 160, 60, 0.12, 'triangle');
  }

  /** Switching tools: a soft mechanical click. */
  click(): void {
    this.tone(0.07, 900, 600, 0.05, 'triangle');
    this.rumble(0.05, 5000, 1500, 0.03);
  }

  zap(): void {
    this.tone(0.12, 1800, 220, 0.12, 'sawtooth');
    this.rumble(0.12, 6000, 800, 0.08);
  }

  /** Noise through a high-pass (from `cut` Hz), sharp attack, gone over `seconds`. */
  private hiss(gain: number, cut: number, seconds: number, delay = 0): void {
    const { ctx } = this;
    if (!ctx || !this.master) return;
    const t = ctx.currentTime + delay;
    const src = ctx.createBufferSource();
    src.buffer = this.noise;
    src.playbackRate.value = 0.9 + Math.random() * 0.3;
    const hp = ctx.createBiquadFilter();
    hp.type = 'highpass';
    hp.frequency.value = cut;
    const g = ctx.createGain();
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(gain, t + 0.004);
    g.gain.exponentialRampToValueAtTime(0.0001, t + seconds);
    src.connect(hp).connect(g).connect(this.master);
    src.start(t, Math.random() * 2, seconds + 0.1);
  }

  /**
   * Windows shattering (`panes` of them) heard from `distance` m: the crack of the break, the
   * crash of the glass giving way, then shards raining down and tinkling on the pavement, the
   * further off the later (they fall farther) and duller.
   */
  glass(panes: number, distance: number, fall = 1): void {
    const near = 1 / (1 + distance / 30);
    const size = Math.min(1, panes / 12);
    const dull = 1 / (1 + distance / 80);
    this.hiss(0.5 * near * (0.5 + size), 2500, 0.09);
    this.rumble(0.35 * near * (0.4 + size), 7000 * dull + 1500, 900, 0.5 + 0.6 * size);
    this.hiss(0.18 * near * (0.3 + size), 4500, 0.9 + size, 0.25 + 0.3 * fall);
    const tinkles = Math.round(6 + 28 * size);
    for (let k = 0; k < tinkles; k++) {
      const at = 0.3 + Math.random() ** 0.7 * (0.9 + 0.8 * fall);
      const f = (2600 + Math.random() * 6500) * (0.6 + 0.4 * dull);
      this.tone((0.02 + Math.random() * 0.05) * near, f, f * (0.97 + Math.random() * 0.02), 0.04 + Math.random() * 0.12, Math.random() < 0.5 ? 'sine' : 'triangle', at);
    }
  }

  /** Noise through a band-pass sweeping `from` to `to` Hz: a whoosh. */
  private swoosh(gain: number, from: number, to: number, seconds: number, delay = 0): void {
    const { ctx } = this;
    if (!ctx || !this.master) return;
    const t = ctx.currentTime + delay;
    const src = ctx.createBufferSource();
    src.buffer = this.noise;
    const bp = ctx.createBiquadFilter();
    bp.type = 'bandpass';
    bp.Q.value = 1.4;
    bp.frequency.setValueAtTime(from, t);
    bp.frequency.exponentialRampToValueAtTime(to, t + seconds);
    const g = ctx.createGain();
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(gain, t + seconds * 0.6);
    g.gain.exponentialRampToValueAtTime(0.0001, t + seconds);
    src.connect(bp).connect(g).connect(this.master);
    src.start(t, Math.random() * 2, seconds + 0.1);
  }

  /** A heavy tool swung through the air. */
  swing(): void {
    this.swoosh(0.22, 350, 1400, 0.22);
  }

  /**
   * A sledgehammer landing, `distance` m off: the thud of the blow, the crack of what it
   * breaks, a ring of the steel head, and the rubble trickling after.
   */
  sledge(distance: number, broke: number): void {
    const near = 1 / (1 + distance / 12);
    const size = Math.min(1, broke / 60);
    this.tone(0.9 * near, 120, 42, 0.28);
    this.tone(0.35 * near, 240, 70, 0.12, 'triangle');
    this.hiss(0.4 * near, 1800, 0.06);
    this.rumble(0.7 * near * (0.5 + size), 5000, 250, 0.35);
    this.tone(0.06 * near, 2150, 2100, 0.35, 'triangle');
    this.tone(0.035 * near, 3420, 3380, 0.25, 'sine');
    this.rumble(0.25 * near * size, 1800, 150, 0.7, 0.09);
  }

  /**
   * The wrecking ball hitting at `speed` m/s, `distance` m off: a deep boom through the
   * structure, the crunch of the wall, a clang of the steel ball, the wall coming down after.
   */
  wreck(distance: number, speed: number): void {
    const near = 1 / (1 + distance / 30);
    const force = Math.min(1, speed / 30);
    this.tone(1.0 * near * (0.5 + force), 75, 28, 0.7);
    this.rumble(0.9 * near * (0.5 + force), 3000, 90, 0.9);
    this.hiss(0.35 * near * force, 1500, 0.08);
    this.tone(0.1 * near, 380, 372, 0.9, 'triangle');
    this.tone(0.05 * near, 1130, 1120, 0.6, 'triangle');
    this.rumble(0.45 * near * force, 1200, 60, 1.8, 0.15);
  }

  /** The laser: a hum while it fires (start with true, stop with false), crackling where it burns. */
  /**
   * The race car's engine, running while it's there: `update` it each frame with the revs
   * (0 idle to 1 the limiter: car.ts climbs them through its gears), the throttle (0..1), how
   * fast it's sliding sideways (m/s) and how far off the listener is (m). Two detuned saws and
   * one an octave down, through a low-pass that opens with the throttle. The tyres screech
   * (band-passed noise) as it slides.
   */
  engine(): { update(revs: number, throttle: number, slide: number, distance: number): void; stop(): void } | null {
    const { ctx } = this;
    if (!ctx || !this.master) return null;
    const t = ctx.currentTime;
    const out = ctx.createGain();
    out.gain.value = 0;
    out.connect(this.master);
    const lp = ctx.createBiquadFilter();
    lp.type = 'lowpass';
    lp.Q.value = 3;
    lp.connect(out);
    const voices = [1, 1.007, 0.5].map((k, i) => {
      const o = ctx.createOscillator();
      o.type = i === 2 ? 'square' : 'sawtooth';
      const g = ctx.createGain();
      g.gain.value = i === 2 ? 0.5 : 0.35;
      o.connect(g).connect(lp);
      o.start(t);
      return { o, k };
    });
    const hiss = ctx.createBufferSource();
    hiss.buffer = this.noise;
    hiss.loop = true;
    const band = ctx.createBiquadFilter();
    band.type = 'bandpass';
    band.frequency.value = 1900;
    band.Q.value = 4;
    const screech = ctx.createGain();
    screech.gain.value = 0;
    hiss.connect(band).connect(screech).connect(this.master);
    hiss.start(t);
    return {
      update: (revs, throttle, slide, distance) => {
        const now = ctx.currentTime;
        const f = 38 + revs * 170;
        for (const { o, k } of voices) o.frequency.setTargetAtTime(f * k, now, 0.03);
        lp.frequency.setTargetAtTime(350 + throttle * 2600 + revs * 600, now, 0.05);
        const near = 1 / (1 + distance / 15);
        out.gain.setTargetAtTime((0.05 + throttle * 0.08 + revs * 0.03) * near, now, 0.05);
        screech.gain.setTargetAtTime(Math.min(0.12, Math.max(0, slide - 2.5) * 0.015) * near, now, 0.05);
      },
      stop: () => {
        out.gain.setTargetAtTime(0, ctx.currentTime, 0.1);
        screech.gain.setTargetAtTime(0, ctx.currentTime, 0.1);
        for (const { o } of voices) o.stop(ctx.currentTime + 0.5);
        hiss.stop(ctx.currentTime + 0.5);
      },
    };
  }

  /** A machine gun's shot heard from `distance` m: a short hard crack. */
  gun(distance: number): void {
    const near = 1 / (1 + distance / 20);
    this.rumble(0.22 * near, 4200, 500, 0.07);
    this.tone(0.06 * near, 900, 180, 0.04, 'square');
  }

  /** An exhaust's afterfire pop at a gear change, from `distance` m. */
  pop(distance: number): void {
    const near = 1 / (1 + distance / 12);
    this.rumble(0.2 * near, 1600, 120, 0.14);
    this.tone(0.07 * near, 260, 50, 0.09, 'square');
    this.rumble(0.12 * near, 1300, 100, 0.1, 0.07);
  }

  laser(on: boolean): void {
    const { ctx } = this;
    if (!ctx || !this.master) return;
    if (on && !this.beam) {
      const t = ctx.currentTime;
      const out = ctx.createGain();
      out.gain.setValueAtTime(0.0001, t);
      out.gain.exponentialRampToValueAtTime(0.16, t + 0.05);
      out.connect(this.master);
      const hum = ctx.createOscillator();
      hum.type = 'sawtooth';
      hum.frequency.value = 96;
      const lp = ctx.createBiquadFilter();
      lp.type = 'lowpass';
      lp.frequency.value = 900;
      hum.connect(lp).connect(out);
      const whine = ctx.createOscillator();
      whine.type = 'sine';
      whine.frequency.value = 1720;
      const wg = ctx.createGain();
      wg.gain.value = 0.18;
      whine.connect(wg).connect(out);
      // Wobble: a slow vibrato on the whine
      const lfo = ctx.createOscillator();
      lfo.frequency.value = 7;
      const depth = ctx.createGain();
      depth.gain.value = 25;
      lfo.connect(depth).connect(whine.frequency);
      const sizzle = ctx.createBufferSource();
      sizzle.buffer = this.noise;
      sizzle.loop = true;
      const hp = ctx.createBiquadFilter();
      hp.type = 'highpass';
      hp.frequency.value = 3000;
      const burn = ctx.createGain();
      burn.gain.value = 0;
      sizzle.connect(hp).connect(burn).connect(out);
      for (const n of [hum, whine, lfo, sizzle]) n.start(t);
      this.tone(0.12, 2600, 900, 0.12, 'square');
      this.beam = { out, nodes: [hum, whine, lfo, sizzle], burn };
    } else if (!on && this.beam) {
      const t = ctx.currentTime;
      const { out, nodes } = this.beam;
      out.gain.cancelScheduledValues(t);
      out.gain.setValueAtTime(out.gain.value, t);
      out.gain.exponentialRampToValueAtTime(0.0001, t + 0.08);
      for (const n of nodes) n.stop(t + 0.1);
      this.tone(0.06, 700, 180, 0.15, 'sawtooth');
      this.beam = null;
    }
  }
  private beam: { out: GainNode; nodes: AudioScheduledSourceNode[]; burn: GainNode } | null = null;

  /** How much the laser is burning something (0..1): the crackle of the beam's sizzle. */
  laserBurn(amount: number): void {
    if (!this.ctx || !this.beam) return;
    this.beam.burn.gain.setTargetAtTime(amount * (0.35 + Math.random() * 0.25), this.ctx.currentTime, 0.02);
  }

  /**
   * A car alarm: the horn honking over and over (two brassy tones a third apart, the car's own
   * horn, a little overdriven), in bursts of honks with a breath between. Returns its volume
   * control (0..1, set from the listener's distance) and a stop.
   */
  alarm(): { level: (v: number) => void; stop: () => void } | null {
    const { ctx } = this;
    if (!ctx || !this.master) return null;
    const t = ctx.currentTime;
    const out = ctx.createGain();
    out.gain.value = 0;
    out.connect(this.master);
    const tones = [392, 494].map((f) => {
      const o = ctx.createOscillator();
      o.type = 'sawtooth';
      o.frequency.value = f * (0.995 + Math.random() * 0.01);
      return o;
    });
    const drive = ctx.createWaveShaper();
    const curve = new Float32Array(256);
    for (let i = 0; i < 256; i++) {
      const x = (i / 255) * 2 - 1;
      curve[i] = Math.tanh(x * 2.5);
    }
    drive.curve = curve;
    const tone = ctx.createBiquadFilter();
    tone.type = 'lowpass';
    tone.frequency.value = 2400;
    const body = ctx.createBiquadFilter();
    body.type = 'peaking';
    body.frequency.value = 900;
    body.gain.value = 6;
    // The honks: scheduled on/off, three honks then a pause, round and round
    const gate = ctx.createGain();
    gate.gain.value = 0;
    for (const o of tones) o.connect(drive);
    drive.connect(tone).connect(body).connect(gate).connect(out);
    for (const o of tones) o.start(t);
    const HONK = 0.34;
    const GAP = 0.2;
    let at = t + 0.02;
    let running = true;
    const schedule = () => {
      if (!running) return;
      // Queue a few seconds ahead
      while (at < ctx.currentTime + 3) {
        for (let k = 0; k < 3; k++) {
          gate.gain.setTargetAtTime(1, at, 0.008);
          gate.gain.setTargetAtTime(0, at + HONK, 0.012);
          at += HONK + GAP;
        }
        at += 0.35;
      }
    };
    schedule();
    const timer = setInterval(schedule, 1000);
    let stopped = false;
    return {
      level: (v) => {
        if (!stopped) out.gain.setTargetAtTime(0.16 * v, ctx.currentTime, 0.05);
      },
      stop: () => {
        if (stopped) return;
        stopped = true;
        running = false;
        clearInterval(timer);
        const now = ctx.currentTime;
        out.gain.cancelScheduledValues(now);
        out.gain.setValueAtTime(out.gain.value, now);
        out.gain.exponentialRampToValueAtTime(0.0001, now + 0.15);
        for (const o of tones) o.stop(now + 0.2);
      },
    };
  }

  /** Something big came down (voxel count). */
  collapse(voxels: number, distance: number): void {
    const near = 1 / (1 + distance / 60);
    const size = Math.min(1, voxels / 3000);
    this.rumble(0.6 * near * (0.3 + size), 500, 40, 2 + 4 * size, 0.3);
  }

  /** A footstep on concrete (a little different each time). */
  step(run: boolean): void {
    this.rumble(run ? 0.12 : 0.08, 1400 + Math.random() * 600, 180, 0.07);
  }

  /** Landing from a fall at `speed` m/s. */
  land(speed: number): void {
    this.rumble(Math.min(0.5, speed * 0.04), 900, 60, 0.2);
  }
}
