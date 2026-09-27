// The Moon and the planets in orbit's sky (space.ts): lit spheres drawn by the sky's own shader where their discs
// fall, no textures. Each is at a fixed place among the stars (so they wheel round with them as the station orbits,
// rising over the Earth's limb), bigger than life so each reads as what it is: the Moon, its seas and craters and
// its phase; Jupiter's bands; Saturn and its rings; red Mars; bright Venus. Placed for the sun as the scene has it
// (towards 0.55, 0.61, 0.57): the Moon gibbous, well away from it, and Venus a morning star near it.

import * as THREE from 'three/webgpu';
import { abs, dot, float, fwidth, max, mix, mx_noise_float, mx_worley_noise_float, normalize, select, smoothstep, sqrt, vec2, vec3 } from 'three/tsl';

type N<T extends string = 'float'> = THREE.Node<T>;

interface Body {
  /** Where it is among the stars (a direction, the stars' frame) and its disc's angular radius (degrees). */
  at: [number, number, number];
  radius: number;
  surface: (n: N<'vec3'>) => N<'vec3'>;
  /** Saturn's: the ring plane's normal, and the rings' inner and outer radius (planet radii). */
  rings?: { normal: [number, number, number]; inner: number; outer: number };
  /** Seen nearly full, whatever the sun (the outer planets: from near the Earth we only ever see their day side). */
  full?: boolean;
}

const BODIES: Body[] = [
  {
    // The Moon: grey highlands, dark seas, and craters (the rims of Worley cells)
    at: [-0.932, 0.094, 0.343],
    radius: 1.8,
    surface: (n) => {
      const seas = smoothstep(0.05, 0.35, mx_noise_float(n.mul(1.7)));
      const craters = smoothstep(0.1, 0.02, mx_worley_noise_float(n.mul(9)));
      return vec3(0.62, 0.6, 0.57).mul(float(1).sub(seas.mul(0.4))).add(craters.mul(0.1));
    },
  },
  {
    // Jupiter: cream and tan bands, wavering
    at: [-0.5, 0.7, 0.35],
    radius: 0.6,
    full: true,
    surface: (n) => {
      const band = mx_noise_float(vec3(n.z.mul(14), n.x.mul(2), n.y.mul(2))).mul(0.5).add(0.5);
      return mix(vec3(0.9, 0.82, 0.68), vec3(0.66, 0.48, 0.34), smoothstep(0.35, 0.65, band));
    },
  },
  {
    // Saturn: pale gold, faintly banded, and its rings (the Cassini division a dark gap in them)
    at: [-0.284, -0.831, 0.479],
    radius: 0.45,
    surface: (n) => mix(vec3(0.92, 0.84, 0.62), vec3(0.8, 0.7, 0.5), smoothstep(-0.3, 0.3, mx_noise_float(vec3(n.z.mul(10), 0, 0)))),
    rings: { normal: [0.25, 0.3, 0.92], inner: 1.3, outer: 2.3 },
    full: true,
  },
  { at: [0.3, -0.8, 0.3], radius: 0.2, full: true, surface: (n) => vec3(0.8, 0.36, 0.2).mul(mx_noise_float(n.mul(4)).mul(0.2).add(0.9)) },
  { at: [0.86, 0.266, 0.435], radius: 0.25, full: true, surface: () => vec3(1, 0.96, 0.86) },
];

/**
 * The bodies seen along `dir` (unit, the stars' frame), lit from `sun`: their light, and how much of the sky behind
 * they cover (the stars there hidden).
 */
export function skyBodies(dir: N<'vec3'>, sun: N<'vec3'>): { colour: N<'vec3'>; cover: N } {
  let colour: N<'vec3'> = vec3(0);
  let cover: N = float(0);
  for (const body of BODIES) {
    const c = new THREE.Vector3(...body.at).normalize();
    const t1 = new THREE.Vector3(0, 0, 1).cross(c).normalize();
    const t2 = c.clone().cross(t1);
    const [C, T1, T2] = [c, t1, t2].map((v) => vec3(v.x, v.y, v.z));
    // Where the ray falls on the plane through the body square to it, in its radii (small, so flat is exact enough)
    const facing = dot(dir, C);
    const q = vec2(dot(dir, T1), dot(dir, T2)).div(facing.max(1e-4).mul(Math.tan(THREE.MathUtils.degToRad(body.radius))));
    const r = q.length();
    const disc = smoothstep(1, float(1).sub(fwidth(r).mul(1.5)), r).mul(select(facing.greaterThan(0), float(1), float(0)));
    // The near hemisphere's normal there, and its light: day side lit, the terminator soft
    const z = sqrt(max(float(1).sub(r.mul(r)), 0));
    const n = T1.mul(q.x).add(T2.mul(q.y)).sub(C.mul(z));
    const from = body.full ? normalize(C.mul(-0.9).add(sun.mul(0.3))) : sun;
    const lit = smoothstep(-0.05, 0.25, dot(n, from));
    // (Under the bloom's threshold: seen as a surface, not a glow)
    let light: N<'vec3'> = body.surface(n).mul(lit).mul(0.9);
    let covered: N = disc;
    if (body.rings) {
      // Where the ray meets the ring plane (s along it from the plane through the planet: < 0 is nearer), and
      // how far out that is: bands of ring, and in front of the planet or hidden behind it
      const m = new THREE.Vector3(...body.rings.normal).normalize();
      const s = q.x.mul(t1.dot(m)).add(q.y.mul(t2.dot(m))).negate().div(c.dot(m));
      const rho = vec3(q.x, q.y, s).length();
      const { inner, outer } = body.rings;
      const band = smoothstep(inner, inner + 0.1, rho).mul(smoothstep(outer, outer - 0.1, rho)).mul(float(1).sub(smoothstep(0.05, 0, abs(rho.sub(1.95))).mul(0.8)));
      const ring = band.mul(mx_noise_float(vec3(rho.mul(30), 0, 0)).mul(0.25).add(0.7)).mul(select(facing.greaterThan(0), float(1), float(0)));
      const seen = ring.mul(select(s.lessThan(0).or(r.greaterThan(1)), float(1), float(0)));
      const ringLight = vec3(0.88, 0.8, 0.62).mul(abs(dot(vec3(m.x, m.y, m.z), sun)).mul(0.6).add(0.3)).mul(0.8);
      light = mix(light.mul(disc), ringLight, seen);
      covered = max(disc, seen);
    } else light = light.mul(disc);
    colour = colour.add(light);
    cover = max(cover, covered);
  }
  return { colour, cover };
}
