// From web-gpu-gems (src/clouds/model.ts at 51e1988), the volumetric cloud layer of its Clouds lab and planet.
// Cloud model shared by the GPU graph (graph.ts) and the CPU reference (reference.ts): shell geometry, altitude,
// density. Units are kilometres. The camera sits at the origin of a local frame whose +y is up; the planet's
// centre is at (0, −(R + h), 0). Formulas are written for float32 on the GPU (see sphere()).

export type V3 = readonly [number, number, number];
export const PERIODS = { base: 24, detail: 2.5, weather: 120 } as const;

export interface CloudParams {
  planetRadius: number; bottom: number; top: number;
  coverage: number; density: number; erosion: number;
  albedo: number; anisotropy: number;
  /** Second, backward lobe: phase = (1 − lobeBlend)·HG(g) + lobeBlend·HG(backAnisotropy). lobeBlend 0 is one lobe. */
  backAnisotropy: number; lobeBlend: number;
  /** Wrenninge's multiple-scattering approximation: octave n scales extinction toward the sun by aⁿ, scattering by bⁿ
   * and both lobes' anisotropy by cⁿ. One octave is plain single scattering. */
  octaves: number; octaveA: number; octaveB: number; octaveC: number;
  /** Schneider's powder term: sun light is scaled by 1 − e^(−2d), darkening edges that face the sun. */
  powder: boolean;
  sun: V3; sunColour: V3; ambient: V3;
  viewSteps: number; lightSteps: number; lightDistance: number;
  energyConserving: boolean; earlyExit: boolean;
}
export const DEFAULTS: CloudParams = {
  planetRadius: 6360, bottom: 1.5, top: 4,
  coverage: 0.35, density: 20, erosion: 0.35,
  albedo: 0.99, anisotropy: 0.6,
  backAnisotropy: -0.3, lobeBlend: 0,
  octaves: 4, octaveA: 0.5, octaveB: 0.5, octaveC: 0.5, powder: false,
  sun: [0, 1, 0], sunColour: [1, 0.98, 0.94], ambient: [0.16, 0.2, 0.27],
  viewSteps: 128, lightSteps: 6, lightDistance: 8,
  energyConserving: true, earlyExit: false,
};

export interface Camera { altitude: number; offset: V3 }
/** Camera position relative to the planet's centre, modulo each noise period (double precision on the CPU). */
export function noiseOrigins(camera: Camera, radius: number) {
  const world = [camera.offset[0], radius + camera.altitude + camera.offset[1], camera.offset[2]];
  const mod = (p: number) => world.map(v => ((v % p) + p) % p) as unknown as V3;
  return { base: mod(PERIODS.base), detail: mod(PERIODS.detail), weather: mod(PERIODS.weather) };
}

/**
 * Ray from the camera (altitude h) against a sphere at altitude hs, direction with vertical component dy.
 * c = |o|² − Rs² is formed as (h − hs)(h + hs + 2R) and the roots use the stable quadratic, avoiding
 * cancellation at small altitudes. The float32 GPU additionally compensates the discriminant near the orbital
 * limb; this independent CPU reference uses doubles. Returns ordered roots, or null for a miss.
 */
export function sphere(h: number, hs: number, radius: number, dy: number): [number, number] | null {
  const b = (radius + h) * dy, c = (h - hs) * (h + hs + 2 * radius), disc = b * b - c;
  if (disc < 0) return null;
  const q = b >= 0 ? -(b + Math.sqrt(disc)) : -(b - Math.sqrt(disc));
  if (q === 0) return [0, 0];
  const t1 = q, t2 = c / q;
  return t1 < t2 ? [t1, t2] : [t2, t1];
}

/** The ray's path through the cloud shell as up to two segments (a limb ray can leave and re-enter). */
export interface Segments { a0: number; lenA: number; b0: number; lenB: number; hitsGround: boolean }
export function shellSegments(h: number, p: CloudParams, dy: number): Segments {
  const none = { a0: 0, lenA: 0, b0: 0, lenB: 0 };
  const ground = sphere(h, 0, p.planetRadius, dy);
  const hitsGround = ground !== null && ground[0] > 0;
  const outer = sphere(h, p.top, p.planetRadius, dy);
  if (!outer || outer[1] <= 0) return { ...none, hitsGround };
  const inner = sphere(h, p.bottom, p.planetRadius, dy);
  const lo = Math.max(outer[0], 0), hi = outer[1];
  const lenA = Math.max(0, (inner ? Math.min(hi, inner[0]) : hi) - lo);
  if (!inner || hitsGround) return { a0: lo, lenA, b0: 0, lenB: 0, hitsGround };
  const b0 = Math.max(inner[1], lo);
  return { a0: lo, lenA, b0, lenB: Math.max(0, hi - b0), hitsGround };
}
/** Distance along the ray of view sample k of n, spread over both segments. */
export function sampleDistance(s: Segments, k: number, n: number) {
  const along = (k + 0.5) * (s.lenA + s.lenB) / n;
  return along < s.lenA ? s.a0 + along : s.b0 + along - s.lenA;
}

/** Altitude of o + t·dir, where o is at altitude h and b = (R + h)·dy, without forming |p| − R directly. */
export function altitudeAt(h: number, radius: number, b: number, t: number) {
  const numerator = h * (h + 2 * radius) + 2 * t * b + t * t;
  return numerator / (Math.sqrt((radius + h) ** 2 + 2 * t * b + t * t) + radius);
}
/** Distance from altitude a along a direction with (P·L) = bl to the shell top, or 0 if blocked by the ground. */
export function lightPath(a: number, bl: number, p: CloudParams) {
  const R = p.planetRadius;
  if (bl < 0 && bl * bl - a * (a + 2 * R) > 0) return 0;
  const c = (a - p.top) * (a + p.top + 2 * R), disc = Math.max(0, bl * bl - c);
  const exit = bl > 0 ? -c / (bl + Math.sqrt(disc)) : -bl + Math.sqrt(disc);
  return Math.min(Math.max(exit, 0), p.lightDistance);
}

const saturate = (v: number) => Math.min(1, Math.max(0, v));
const smoothstep = (e0: number, e1: number, x: number) => { const t = saturate((x - e0) / (e1 - e0)); return t * t * (3 - 2 * t); };
/**
 * Extinction (per km) from the three texture samples and altitude. Nubis-style: the weather sample varies
 * coverage, the base shape is Perlin–Worley eroded by its Worley fBm, a height profile rounds the layer,
 * coverage cuts the shape, and detail Worley erodes the edges (billowy low, wispy high).
 */
export function extinction(weather: number, base: readonly number[], detail: readonly number[], altitude: number, p: CloudParams) {
  const cover = saturate(p.coverage + 0.8 * (weather - 0.5));
  const shapeFbm = base[1]! * 0.625 + base[2]! * 0.25 + base[3]! * 0.125;
  const hf = saturate((altitude - p.bottom) / (p.top - p.bottom));
  const profile = smoothstep(0, 0.15, hf) * (1 - smoothstep(0.5, 1, hf));
  const shape = (base[0]! - shapeFbm + 1) / (2 - shapeFbm) * profile;
  const covered = saturate((shape - (1 - cover)) / Math.max(cover, 1e-3)) * cover;
  const fbm = detail[0]! * 0.625 + detail[1]! * 0.25 + detail[2]! * 0.125;
  const erode = (fbm + (1 - 2 * fbm) * saturate(hf * 4)) * p.erosion;
  return saturate((covered - erode) / (1 - erode)) * p.density;
}
/** Henyey–Greenstein phase function, normalized over the sphere. */
export const henyeyGreenstein = (g: number, cosine: number) => (1 - g * g) / (4 * Math.PI * (1 + g * g - 2 * g * cosine) ** 1.5);
/** One or two HG lobes, with both anisotropies scaled by `scale` (an octave's cⁿ). */
export const phase = (p: CloudParams, cosine: number, scale = 1) =>
  (1 - p.lobeBlend) * henyeyGreenstein(p.anisotropy * scale, cosine) + p.lobeBlend * henyeyGreenstein(p.backAnisotropy * scale, cosine);
/**
 * Sun light scattered toward the viewer per unit sun colour, before σs: Σₙ bⁿ · T(aⁿ·depth) · phase(cⁿ), where T is Beer's
 * law, optionally times the powder term. `depth` is the optical depth toward the sun; `lit` false means no path (shadowed).
 */
export function sunScatter(p: CloudParams, cosine: number, depth: number, lit: boolean) {
  if (!lit) return 0;
  const powder = p.powder ? 1 - Math.exp(-2 * depth) : 1;
  let sum = 0;
  for (let n = 0, a = 1, b = 1, c = 1; n < p.octaves; n++, a *= p.octaveA, b *= p.octaveB, c *= p.octaveC) sum += b * Math.exp(-a * depth) * phase(p, cosine, c);
  return sum * powder;
}
