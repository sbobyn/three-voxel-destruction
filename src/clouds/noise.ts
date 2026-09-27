// From web-gpu-gems (src/clouds/noise.ts at 51e1988), the volumetric cloud layer of its Clouds lab and planet.
// CPU twin of the cloud noise. The GPU compute pass (noise-gpu.ts) evaluates the same formulas; the lab
// checks the two against each other before the GPU's bytes are used for rendering and for the CPU reference.

export const BASE_SIZE = 128;
export const DETAIL_SIZE = 32;

/** lowbias32 (Chris Wellons). Unsigned 32-bit arithmetic, identical in WGSL, GLSL and here. */
export function lowbias32(x: number) {
  x = (x ^ (x >>> 16)) >>> 0; x = Math.imul(x, 0x7feb352d) >>> 0;
  x = (x ^ (x >>> 15)) >>> 0; x = Math.imul(x, 0x846ca68b) >>> 0;
  return (x ^ (x >>> 16)) >>> 0;
}
export const hash3 = (x: number, y: number, z: number, seed: number) =>
  lowbias32((x + lowbias32((y + lowbias32((z + seed) >>> 0)) >>> 0)) >>> 0);
/** Top 24 bits as a float in [0, 1); exact in float32. */
export const unit = (h: number) => (h >>> 8) / 16777216;

const fade = (t: number) => t * t * t * (t * (t * 6 - 15) + 10);
const wrap = (k: number, c: number) => ((k % c) + c) % c;
/** Ken Perlin's improved-noise gradient selection from the low four bits. */
function grad(h: number, x: number, y: number, z: number) {
  const b = h & 15;
  const u = b < 8 ? x : y;
  const v = b < 4 ? y : b === 12 || b === 14 ? x : z;
  return ((b & 1) === 0 ? u : -u) + ((b & 2) === 0 ? v : -v);
}

/** Gradient noise with `c` cells per unit tile, at q in cell units; period c in every axis. Roughly [-1, 1]. */
export function perlin(qx: number, qy: number, qz: number, c: number, seed: number) {
  const ix = Math.floor(qx), iy = Math.floor(qy), iz = Math.floor(qz);
  const fx = qx - ix, fy = qy - iy, fz = qz - iz;
  const corner = (dx: number, dy: number, dz: number) =>
    grad(hash3(wrap(ix + dx, c), wrap(iy + dy, c), wrap(iz + dz, c), seed), fx - dx, fy - dy, fz - dz);
  const u = fade(fx), v = fade(fy), w = fade(fz);
  const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
  return lerp(
    lerp(lerp(corner(0, 0, 0), corner(1, 0, 0), u), lerp(corner(0, 1, 0), corner(1, 1, 0), u), v),
    lerp(lerp(corner(0, 0, 1), corner(1, 0, 1), u), lerp(corner(0, 1, 1), corner(1, 1, 1), u), v), w);
}

/** 1 − distance to the nearest feature point (one per cell), in cell units, clamped to [0, 1]; period c. */
export function worley(qx: number, qy: number, qz: number, c: number, seed: number) {
  const ix = Math.floor(qx), iy = Math.floor(qy), iz = Math.floor(qz);
  let nearest = 1;
  for (let dz = -1; dz <= 1; dz++) for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
    const kx = ix + dx, ky = iy + dy, kz = iz + dz;
    const h1 = hash3(wrap(kx, c), wrap(ky, c), wrap(kz, c), seed), h2 = lowbias32(h1), h3 = lowbias32(h2);
    const ex = kx + unit(h1) - qx, ey = ky + unit(h2) - qy, ez = kz + unit(h3) - qz;
    nearest = Math.min(nearest, Math.sqrt(ex * ex + ey * ey + ez * ez));
  }
  return 1 - nearest;
}

const saturate = (v: number) => Math.min(1, Math.max(0, v));
// Contrast constants: the fBm sums cluster near their middle, so each is stretched to span [0, 1]. Worley fBm's
// 1st–99th percentiles are about 0.22–0.76; Perlin fBm's (normalized by 1.875) about ±0.37.
const WORLEY_LOW = 0.22, WORLEY_GAIN = 1.85, PERLIN_GAIN = 1.35;

/** Three Worley octaves (c, 2c, 4c) at a point u in tile units [0, 1), stretched to [0, 1]. */
export function worleyFbm(ux: number, uy: number, uz: number, c: number, seed: number) {
  const at = (k: number, s: number) => worley(ux * k, uy * k, uz * k, k, s);
  return saturate((at(c, seed) * 0.625 + at(2 * c, seed + 1) * 0.25 + at(4 * c, seed + 2) * 0.125 - WORLEY_LOW) * WORLEY_GAIN);
}
/** Four Perlin octaves from c cells, normalized and stretched to [0, 1]. */
export function perlinFbm(ux: number, uy: number, uz: number, c: number, seed: number) {
  let sum = 0;
  for (let o = 0, k = c, a = 1; o < 4; o++, k *= 2, a *= 0.5) sum += a * perlin(ux * k, uy * k, uz * k, k, seed + o);
  return saturate(0.5 + PERLIN_GAIN * sum / 1.875);
}
/** Base shape texel: R Perlin–Worley, G/B/A Worley fBm at 4, 8 and 16 cells. */
export function baseTexel(i: number, j: number, k: number, size = BASE_SIZE): [number, number, number, number] {
  const ux = (i + 0.5) / size, uy = (j + 0.5) / size, uz = (k + 0.5) / size;
  const w = worleyFbm(ux, uy, uz, 4, 10);
  const p = perlinFbm(ux, uy, uz, 4, 100);
  // Perlin–Worley: billowy Worley cells modulated by Perlin, contrast-stretched around 0.5.
  return [saturate((p * 0.6 + w * 0.4 - 0.5) * 1.6 + 0.5), worleyFbm(ux, uy, uz, 4, 20), worleyFbm(ux, uy, uz, 8, 30), worleyFbm(ux, uy, uz, 16, 40)];
}
/** Detail texel: Worley fBm at 2, 4 and 8 cells; alpha unused (1). */
export function detailTexel(i: number, j: number, k: number, size = DETAIL_SIZE): [number, number, number, number] {
  const ux = (i + 0.5) / size, uy = (j + 0.5) / size, uz = (k + 0.5) / size;
  return [worleyFbm(ux, uy, uz, 2, 50), worleyFbm(ux, uy, uz, 4, 60), worleyFbm(ux, uy, uz, 8, 70), 1];
}
/** Byte quantization shared with the GPU pass: floor(v·255 + 0.5). */
export const toByte = (v: number) => Math.floor(saturate(v) * 255 + 0.5);
