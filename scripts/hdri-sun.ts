// The sun in each sky HDRI (public/hdri): its direction (world: x east, y north, z up, from the
// image's equirectangular mapping with +z up and the image's centre column facing +y), its
// colour and its irradiance, and the sky's mean colour above the horizon, written to
// public/hdri/index.json for the game (render.ts / sky.ts). Run after changing the HDRIs.
//
//   node --experimental-transform-types --no-warnings scripts/hdri-sun.ts

import { readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { HDRLoader } from 'three/examples/jsm/loaders/HDRLoader.js';

const dir = 'public/hdri';
const out: Record<string, unknown> = {};
for (const file of readdirSync(dir).filter((f) => f.endsWith('.hdr')).sort()) {
  const buf = readFileSync(`${dir}/${file}`);
  const img = new HDRLoader().parse(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength)) as { width: number; height: number; data: Uint16Array | Float32Array; type: number };
  const { width, height } = img;
  const float = img.data instanceof Float32Array;
  // Half floats to numbers
  const half = (h: number) => {
    const s = (h & 0x8000) >> 15;
    const e = (h & 0x7c00) >> 10;
    const f = h & 0x03ff;
    const v = e === 0 ? f / 1024 * 2 ** -14 : e === 31 ? Infinity : (1 + f / 1024) * 2 ** (e - 15);
    return s ? -v : v;
  };
  const at = (x: number, y: number, c: number) => (float ? (img.data as Float32Array)[4 * (y * width + x) + c] : half((img.data as Uint16Array)[4 * (y * width + x) + c]));
  // Direction of pixel (x, y): longitude across, latitude down; the centre column faces +y
  const dirOf = (x: number, y: number) => {
    const lon = ((x + 0.5) / width - 0.5) * 2 * Math.PI;
    const lat = (0.5 - (y + 0.5) / height) * Math.PI;
    return [Math.sin(lon) * Math.cos(lat), Math.cos(lon) * Math.cos(lat), Math.sin(lat)];
  };
  // The sun: the pixels within a stop of the brightest, weighted (after sunset that is the
  // brightest of the afterglow, a dim light from the horizon); its irradiance is the sum of
  // their radiance times solid angle
  let peak = 0;
  for (let y = 0; y < height / 2; y++) for (let x = 0; x < width; x++) peak = Math.max(peak, at(x, y, 0) + at(x, y, 1) + at(x, y, 2));
  const sum = [0, 0, 0];
  const colour = [0, 0, 0];
  let weight = 0;
  const sky = [0, 0, 0];
  let skyWeight = 0;
  for (let y = 0; y < height; y++) {
    const lat = (0.5 - (y + 0.5) / height) * Math.PI;
    const solid = ((2 * Math.PI) / width) * (Math.PI / height) * Math.cos(lat);
    for (let x = 0; x < width; x++) {
      const [r, g, b] = [at(x, y, 0), at(x, y, 1), at(x, y, 2)];
      const lum = r + g + b;
      if (lum > peak * 0.5 && lat > 0) {
        const d = dirOf(x, y);
        for (let a = 0; a < 3; a++) sum[a] += d[a] * lum;
        colour[0] += r * solid;
        colour[1] += g * solid;
        colour[2] += b * solid;
        weight += lum;
      } else if (lat > 0) {
        sky[0] += r * solid;
        sky[1] += g * solid;
        sky[2] += b * solid;
        skyWeight += solid;
      }
    }
  }
  const len = Math.hypot(...sum) || 1;
  const direction = sum.map((v) => v / len);
  const irradiance = colour[0] + colour[1] + colour[2];
  const tint = colour.map((c) => c / Math.max(1e-6, Math.max(...colour)));
  const name = file.replace('_1k.hdr', '').replace('qwantani_', '').replace('_puresky', '');
  const entry = {
    file,
    sun: direction.map((v) => +v.toFixed(4)),
    elevation: +((Math.asin(direction[2]) * 180) / Math.PI).toFixed(1),
    sunColour: tint.map((v) => +v.toFixed(3)),
    // Irradiance of the sun's disc (W/m² per unit radiance scale), and the sky's mean radiance
    sunIrradiance: +irradiance.toFixed(3),
    skyColour: sky.map((c) => +(c / skyWeight).toFixed(4)),
  };
  out[name] = entry;
  console.log(name.padEnd(16), 'elevation', entry.elevation, 'sun', direction.map((v) => v.toFixed(2)).join(','), 'irr', irradiance.toFixed(2), 'sky', sky.map((c) => (c / skyWeight).toFixed(3)).join(','));
}
writeFileSync(`${dir}/index.json`, JSON.stringify(out, null, 2) + '\n');
