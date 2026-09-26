// The loading screen (its markup is in city.html, so it shows before any script runs): a bar
// that fills through the stages of starting up, each named, the current one pulsing, and a
// plain message if the device can't run the game. Stages have weights (their share of the
// bar); `progress` moves within the current one.

export interface Stage {
  name: string;
  weight: number;
}

export class Loader {
  private readonly root = document.querySelector('#loader') as HTMLElement;
  private readonly fill = this.root.querySelector('.fill') as HTMLElement;
  private readonly phase = this.root.querySelector('.phase') as HTMLElement;
  private readonly pct = this.root.querySelector('.pct') as HTMLElement;
  private readonly error = this.root.querySelector('.error') as HTMLElement;
  private readonly items: HTMLElement[] = [];
  private readonly total: number;
  private current = -1;
  private readonly stages: Stage[];

  constructor(stages: Stage[]) {
    this.stages = stages;
    this.total = stages.reduce((a, s) => a + s.weight, 0);
    const list = this.root.querySelector('.steps') as HTMLElement;
    for (const s of stages) {
      const li = document.createElement('li');
      li.textContent = s.name;
      list.append(li);
      this.items.push(li);
    }
  }

  /** Begin stage `i` (the ones before it are done), and let the page paint. */
  async stage(i: number, detail?: string): Promise<void> {
    this.current = i;
    this.items.forEach((li, k) => {
      li.classList.toggle('done', k < i);
      li.classList.toggle('now', k === i);
    });
    this.phase.textContent = detail ?? `${this.stages[i].name}…`;
    this.progress(0);
    await paint();
  }

  /** How far through the current stage (0..1). */
  progress(share: number): void {
    const before = this.stages.slice(0, this.current).reduce((a, s) => a + s.weight, 0);
    const done = (before + this.stages[this.current].weight * Math.min(1, Math.max(0, share))) / this.total;
    this.fill.style.width = `${(done * 100).toFixed(1)}%`;
    this.pct.textContent = `${Math.floor(done * 100)}%`;
  }

  /** A word on what the current stage is doing now. */
  detail(text: string): void {
    this.phase.textContent = text;
  }

  /** All done: fill the bar and fade out. */
  async finish(): Promise<void> {
    this.items.forEach((li) => {
      li.classList.remove('now');
      li.classList.add('done');
    });
    this.fill.style.width = '100%';
    this.pct.textContent = '100%';
    this.phase.textContent = 'Ready';
    await new Promise((r) => setTimeout(r, 250));
    this.root.classList.add('done');
    setTimeout(() => this.root.remove(), 600);
  }

  /** Starting failed: say why, plainly, and what to try. */
  fail(error: unknown): void {
    const message = error instanceof Error ? error.message : String(error);
    const gpu = /webgpu|adapter|gpu|device/i.test(message);
    const insecure = /secure page/i.test(message);
    this.root.classList.add('failed');
    this.phase.textContent = 'Could not start';
    this.error.innerHTML = insecure
      ? `Browsers only allow WebGPU on secure pages. Open this page over https:// (or on localhost on the computer running it).<small></small>`
      : gpu
      ? `This browser or device can't run Voxel City: it needs WebGPU. Try the latest Chrome or Edge, or Safari on iOS 26 or macOS 26 (on older iOS, turn on WebGPU in Settings › Safari › Advanced › Feature Flags).<small></small>`
      : `Something went wrong while starting. Reloading the page may help.<small></small>`;
    (this.error.querySelector('small') as HTMLElement).textContent = message;
  }
}

/** Wait until the page has painted (two frames: the change is on screen), or a moment if it can't (a hidden tab never paints). */
export function paint(): Promise<void> {
  return new Promise((resolve) => {
    const timer = setTimeout(resolve, 60);
    requestAnimationFrame(() =>
      requestAnimationFrame(() => {
        clearTimeout(timer);
        resolve();
      }),
    );
  });
}
