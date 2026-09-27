// From web-gpu-gems (src/clouds/noise-gpu.ts at 51e1988), the volumetric cloud layer of its Clouds lab and planet.
// GPU twin of noise.ts: a compute pass evaluates every texel, packs RGBA8 into one uint, and the bytes are read back
// once. The lab samples those bytes as a 3D texture; the CPU reference reconstructs the same bytes.
import * as THREE from 'three/webgpu';
import { float, floor, Fn, instancedArray, instanceIndex, int, min, select, sqrt, uint, vec3 } from 'three/tsl';
import { BASE_SIZE, DETAIL_SIZE } from './noise.ts';

type N<T extends string = 'float'> = THREE.Node<T>;

const lowbias32 = Fn(([x0]: [N<'uint'>]) => {
  const x = uint(x0).toVar();
  x.assign(x.bitXor(x.shiftRight(uint(16)))); x.assign(x.mul(uint(0x7feb352d)));
  x.assign(x.bitXor(x.shiftRight(uint(15)))); x.assign(x.mul(uint(0x846ca68b)));
  x.assign(x.bitXor(x.shiftRight(uint(16))));
  return x;
}).setLayout({ name: 'cloudLowbias32', type: 'uint', inputs: [{ name: 'x', type: 'uint' }] });

const hash3 = (x: N<'int'>, y: N<'int'>, z: N<'int'>, seed: number) =>
  lowbias32(uint(x).add(lowbias32(uint(y).add(lowbias32(uint(z).add(uint(seed)))))));
const unit = (h: N<'uint'>) => float(h.shiftRight(uint(8))).mul(1 / 16777216);
// Cells stay ≥ −1 inside a tile, so (k + c) % c equals the CPU's non-negative modulo there and avoids
// GLSL's undefined % on negative ints.
const wrap = (k: N<'int'>, c: N<'int'>) => k.add(c).mod(c);
const fade = (t: N) => t.mul(t).mul(t).mul(t.mul(t.mul(6).sub(15)).add(10));

function grad(h: N<'uint'>, x: N, y: N, z: N) {
  const b = h.bitAnd(uint(15));
  const u = select(b.lessThan(uint(8)), x, y);
  const v = select(b.lessThan(uint(4)), y, select(b.equal(uint(12)).or(b.equal(uint(14))), x, z));
  return select(b.bitAnd(uint(1)).equal(uint(0)), u, u.negate()).add(select(b.bitAnd(uint(2)).equal(uint(0)), v, v.negate()));
}

const perlinFns = new Map<number, ReturnType<typeof makePerlin>>();
const perlinAt = (seed: number) => perlinFns.get(seed) ?? perlinFns.set(seed, makePerlin(seed)).get(seed)!;
function makePerlin(seed: number) { return Fn(([q, c]: [N<'vec3'>, N<'int'>]) => {
  const i = floor(q), f = q.sub(i);
  const ix = int(i.x), iy = int(i.y), iz = int(i.z);
  const corner = (dx: number, dy: number, dz: number) =>
    grad(hash3(wrap(ix.add(dx), c), wrap(iy.add(dy), c), wrap(iz.add(dz), c), seed), f.x.sub(dx), f.y.sub(dy), f.z.sub(dz));
  const u = fade(f.x), v = fade(f.y), w = fade(f.z);
  const lerp = (a: N, b: N, t: N) => a.add(b.sub(a).mul(t));
  return lerp(
    lerp(lerp(corner(0, 0, 0), corner(1, 0, 0), u), lerp(corner(0, 1, 0), corner(1, 1, 0), u), v),
    lerp(lerp(corner(0, 0, 1), corner(1, 0, 1), u), lerp(corner(0, 1, 1), corner(1, 1, 1), u), v), w);
}).setLayout({ name: `cloudPerlin${seed}`, type: 'float', inputs: [{ name: 'q', type: 'vec3' }, { name: 'c', type: 'int' }] }); }

// One shader function per seed, created once so repeated octaves reuse it.
const worleyFns = new Map<number, ReturnType<typeof makeWorley>>();
const worleyAt = (seed: number) => worleyFns.get(seed) ?? worleyFns.set(seed, makeWorley(seed)).get(seed)!;
function makeWorley(seed: number) { return Fn(([q, c]: [N<'vec3'>, N<'int'>]) => {
  const i = floor(q), ix = int(i.x), iy = int(i.y), iz = int(i.z);
  const nearest = float(1).toVar();
  for (let dz = -1; dz <= 1; dz++) for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
    const kx = ix.add(dx), ky = iy.add(dy), kz = iz.add(dz);
    const h1 = hash3(wrap(kx, c), wrap(ky, c), wrap(kz, c), seed), h2 = lowbias32(h1), h3 = lowbias32(h2);
    const e = vec3(float(kx).add(unit(h1)).sub(q.x), float(ky).add(unit(h2)).sub(q.y), float(kz).add(unit(h3)).sub(q.z));
    nearest.assign(min(nearest, sqrt(e.x.mul(e.x).add(e.y.mul(e.y)).add(e.z.mul(e.z)))));
  }
  return float(1).sub(nearest);
}).setLayout({ name: `cloudWorley${seed}`, type: 'float', inputs: [{ name: 'q', type: 'vec3' }, { name: 'c', type: 'int' }] }); }

// Same contrast constants as noise.ts.
const WORLEY_LOW = 0.22, WORLEY_GAIN = 1.85, PERLIN_GAIN = 1.35;
function worleyFbm(u: N<'vec3'>, c: number, seed: number) {
  const at = (k: number, s: number) => worleyAt(s)(u.mul(k), int(k));
  return at(c, seed).mul(0.625).add(at(2 * c, seed + 1).mul(0.25)).add(at(4 * c, seed + 2).mul(0.125)).sub(WORLEY_LOW).mul(WORLEY_GAIN).clamp(0, 1);
}
function perlinFbm(u: N<'vec3'>, c: number, seed: number) {
  let sum: N = float(0);
  for (let o = 0, k = c, a = 1; o < 4; o++, k *= 2, a *= 0.5) sum = sum.add(perlinAt(seed + o)(u.mul(k), int(k)).mul(a));
  return sum.mul(PERLIN_GAIN).div(1.875).add(0.5).clamp(0, 1);
}
const toByte = (v: N) => uint(floor(v.clamp(0, 1).mul(255).add(0.5)));
const pack = (r: N, g: N, b: N, a: N) =>
  toByte(r).bitOr(toByte(g).shiftLeft(uint(8))).bitOr(toByte(b).shiftLeft(uint(16))).bitOr(toByte(a).shiftLeft(uint(24)));

function texelKernel(size: number, texel: (u: N<'vec3'>) => [N, N, N, N]) {
  const out = instancedArray(size ** 3, 'uint');
  const kernel = Fn(() => {
    const index = instanceIndex, s = uint(size);
    const i = index.mod(s), j = index.div(s).mod(s), k = index.div(s.mul(s));
    const u = vec3(float(i), float(j), float(k)).add(0.5).div(size);
    out.element(index).assign(pack(...texel(u)));
  })().compute(size ** 3);
  return { out, kernel };
}
const baseTexel = (u: N<'vec3'>): [N, N, N, N] => {
  const w = worleyFbm(u, 4, 10), p = perlinFbm(u, 4, 100);
  return [p.mul(0.6).add(w.mul(0.4)).sub(0.5).mul(1.6).add(0.5), worleyFbm(u, 4, 20), worleyFbm(u, 8, 30), worleyFbm(u, 16, 40)];
};
const detailTexel = (u: N<'vec3'>): [N, N, N, N] => [worleyFbm(u, 2, 50), worleyFbm(u, 4, 60), worleyFbm(u, 8, 70), float(1)];

export function makeVolumeTexture(bytes: Uint8Array, size: number) {
  const texture = new THREE.Data3DTexture(bytes, size, size, size);
  texture.format = THREE.RGBAFormat; texture.type = THREE.UnsignedByteType;
  texture.minFilter = texture.magFilter = THREE.LinearFilter;
  texture.wrapS = texture.wrapT = texture.wrapR = THREE.RepeatWrapping;
  texture.colorSpace = THREE.NoColorSpace; texture.unpackAlignment = 1; texture.generateMipmaps = false;
  texture.needsUpdate = true;
  return texture;
}

/** Runs both noise kernels and returns the packed bytes (RGBA8, x fastest). */
export async function generateNoise(renderer: THREE.WebGPURenderer) {
  const result: Record<'base' | 'detail', Uint8Array> = { base: new Uint8Array(), detail: new Uint8Array() };
  for (const [name, size, texel] of [['base', BASE_SIZE, baseTexel], ['detail', DETAIL_SIZE, detailTexel]] as const) {
    const { out, kernel } = texelKernel(size, texel);
    await renderer.computeAsync(kernel);
    result[name] = new Uint8Array((await renderer.getArrayBufferAsync(out.value)).slice(0));
    kernel.dispose();
  }
  return result;
}
