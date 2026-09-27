// Orbit's sky, in place of the city's (sky.ts): black, with stars and the sun's disc, and the
// Earth below. The Earth is a sphere far under the station, scaled with it: 5 km across at 350 m
// down keeps the ISS's ratio of height to radius, so its horizon dips as it would from 400 km
// (some 20 degrees) and its curve shows. NASA's Blue Marble on it (public/space, public
// domain), lit by the sun with a soft terminator and a night side, clouds drifting over it, a
// thin blue atmosphere glowing round its limb.

import * as THREE from 'three/webgpu';
import { abs, cameraPosition, clamp, cross, dot, float, floor, fract, hash, length, max, mix, mx_noise_float, mx_worley_noise_float, normalize, positionWorld, pow, reflect, select, smoothstep, texture, uniform, uv, vec2, vec3, vec4 } from 'three/tsl';

/** The Earth's radius (m, scaled), and how far under the station its top is. */
export const EARTH_RADIUS = 5000;
const ALTITUDE = 350;

/**
 * A texture read bicubic (B-spline, from four bilinear reads): smooth where it's magnified,
 * where plain bilinear shows each texel's square. `size`: the texture's texels.
 */
function bicubic(map: THREE.Texture, at: THREE.Node<'vec2'>, size: [number, number]): THREE.Node<'vec4'> {
  const texels = vec2(size[0], size[1]);
  const st = at.mul(texels).sub(0.5);
  const f = fract(st);
  const i = st.sub(f);
  // Cubic B-spline weights for the four texels along each axis
  const cubic = (v: THREE.Node<'float'>) => {
    const n = vec4(1, 2, 3, 4).sub(v);
    const c = n.mul(n).mul(n);
    const x = c.x;
    const y = c.y.sub(c.x.mul(4));
    const z = c.z.sub(c.y.mul(4)).add(c.x.mul(6));
    const w = float(6).sub(x).sub(y).sub(z);
    return vec4(x, y, z, w).div(6);
  };
  const cx = cubic(f.x);
  const cy = cubic(f.y);
  const corners = vec4(i.x, i.x, i.y, i.y).add(vec4(-0.5, 1.5, -0.5, 1.5));
  const sums = vec4(cx.x.add(cx.y), cx.z.add(cx.w), cy.x.add(cy.y), cy.z.add(cy.w));
  const offset = corners.add(vec4(cx.y, cx.w, cy.y, cy.w).div(sums)).div(vec4(texels.x, texels.x, texels.y, texels.y));
  const s0 = texture(map, vec2(offset.x, offset.z));
  const s1 = texture(map, vec2(offset.y, offset.z));
  const s2 = texture(map, vec2(offset.x, offset.w));
  const s3 = texture(map, vec2(offset.y, offset.w));
  const sx = sums.x.div(sums.x.add(sums.y));
  const sy = sums.z.div(sums.z.add(sums.w));
  return mix(mix(s3, s2, sx), mix(s1, s0, sx), sy) as unknown as THREE.Node<'vec4'>;
}

export class SpaceSky {
  readonly object = new THREE.Group();
  /** Towards the sun (unit), as the city's sky has it: main.ts sets it. */
  readonly sun = new THREE.Vector3(0.5, -0.4, 0.6).normalize();
  /** The colours main.ts lights with: black air, a white sun. */
  readonly horizon = new THREE.Color(0.01, 0.015, 0.03);
  readonly sunColor = new THREE.Color(1, 0.98, 0.95);
  private readonly sunUniform = uniform(new THREE.Vector3());
  private readonly clock = uniform(0);
  private readonly stars: THREE.Mesh;

  /** `centre`: the point the Earth is under (the station's middle). */
  constructor(centre: THREE.Vector3) {
    const loader = new THREE.TextureLoader();
    const day = loader.load('space/earth.webp');
    day.colorSpace = THREE.SRGBColorSpace;
    day.anisotropy = 8;
    day.wrapS = THREE.RepeatWrapping;
    const clouds = loader.load('space/clouds.webp');
    clouds.wrapS = THREE.RepeatWrapping;
    clouds.anisotropy = 8;

    // The sky: stars in three sizes, hashed from the direction, and the sun's disc with its glare
    const skyMat = new THREE.MeshBasicNodeMaterial({ side: THREE.BackSide, depthWrite: false, fog: false });
    const dir = normalize(positionWorld.sub(cameraPosition));
    let stars = float(0) as unknown as THREE.Node<'float'>;
    for (const [scale, keep, bright] of [
      [180, 0.992, 0.9],
      [420, 0.985, 0.55],
      [900, 0.98, 0.35],
    ] as const) {
      const p = dir.mul(scale);
      const cell = floor(p);
      const h = hash(cell.dot(vec3(1, 57, 113)));
      const spot = length(fract(p).sub(0.5));
      stars = stars.add(smoothstep(0.16, 0.0, spot).mul(smoothstep(keep, 1.0, h)).mul(bright * 30)) as unknown as THREE.Node<'float'>;
    }
    const toSun = max(dot(dir, this.sunUniform), 0);
    const disc = smoothstep(0.99985, 0.9999, toSun).mul(14).add(pow(toSun, 1500).mul(1.5)).add(pow(toSun, 120).mul(0.04));
    skyMat.colorNode = vec3(0.85, 0.9, 1).mul(stars).add(vec3(1, 0.97, 0.92).mul(disc)) as unknown as THREE.Node<'color'>;
    this.stars = new THREE.Mesh(new THREE.SphereGeometry(5500, 48, 24), skyMat);
    this.stars.renderOrder = -2;
    this.stars.frustumCulled = false;

    // The Earth: day map and clouds by the sphere's uv, lit by the sun; the atmosphere's haze
    // towards the limb, blue on the day side
    const earthCentre = new THREE.Vector3(centre.x, centre.y, centre.z - EARTH_RADIUS - ALTITUDE);
    const earthMat = new THREE.MeshBasicNodeMaterial({ fog: false });
    const n = normalize(positionWorld.sub(vec3(earthCentre.x, earthCentre.y, earthCentre.z)));
    const view = normalize(cameraPosition.sub(positionWorld));
    const light = dot(n, this.sunUniform);
    const lit = smoothstep(-0.08, 0.25, light);
    // Up close the maps are coarse (a texel some 8 m across for the land, 15 m for the clouds,
    // a degree or so of view straight down), and sampled plainly they show as blocks. So both
    // are read bicubic (smooth, no texel corners), and the cloud map only says where there's
    // cloud: its edges and texture come from noise down to a few metres, each finer octave
    // fading out with distance before it would shimmer
    const far = length(positionWorld.sub(cameraPosition));
    const ground = bicubic(day, uv(), [4096, 2048]).rgb;
    const cover = bicubic(clouds, uv().add(vec2(this.clock.mul(0.0015), 0)), [2048, 1024]).r;
    // The cloud's thickness at a point q (m): the cover map says how much sky is cloud there;
    // what it's made of are heaps of rounded puffs, as cumulus seen from above: cells of
    // domes (the distance to each cell's middle, rounded off) in two sizes, merging into
    // banks where the cover is thick, over broad billows. Finer sizes fade with distance
    // before they'd shimmer.
    const fade = (from: number, to: number) => smoothstep(from, to, far);
    const dome = (q: THREE.Node<'vec3'>, size: number) => {
      const d = mx_worley_noise_float(q.xy.mul(1 / size)).div(0.8);
      return float(1).sub(d.mul(d)).max(0).sqrt();
    };
    const billow = (q: THREE.Node<'vec3'>, size: number) => float(1).sub(abs(mx_noise_float(q.mul(1 / size))).mul(2));
    const thickness = (q: THREE.Node<'vec3'>) =>
      cover
        .mul(1.5)
        .add(billow(q, 480).mul(0.22))
        // Puffs of different heights: the big ones swell and shrink across the field
        // (At the Earth's scale here a metre is over a kilometre: cumulus a kilometre or two
        // across are a metre or two, fields of popcorn, as from the real station)
        .add(dome(q, 16).mul(billow(q, 90).mul(0.25).add(0.5)).mul(fade(3200, 1800)).add(fade(1800, 3200).mul(0.3)))
        .add(dome(q, 5.5).mul(0.22).mul(fade(1100, 500)).add(fade(500, 1100).mul(0.12)))
        // And lumps on the lumps, up close
        .add(dome(q, 1.8).mul(0.09).mul(fade(420, 200)).add(fade(200, 420).mul(0.05)))
        .sub(0.97);
    const p = positionWorld;
    const density = thickness(p);
    const cloud = smoothstep(0.0, 0.12, density);
    // Relief: the thickness as height (its slope from two nearby points along the surface), lit
    // by the sun: each puff's dome bright on its sunward side, blue-grey on the other
    const east = normalize(cross(n, vec3(0, 1, 0)));
    const north = cross(east, n);
    const step = float(0.5);
    const dx = thickness(p.add(east.mul(step))).sub(density);
    const dy = thickness(p.add(north.mul(step))).sub(density);
    const bump = normalize(n.mul(0.05).sub(east.mul(dx)).sub(north.mul(dy)));
    const towardSun = normalize(this.sunUniform.sub(n.mul(dot(this.sunUniform, n))));
    const lighting = clamp(dot(bump, this.sunUniform).mul(1.05).add(0.1), 0, 1.15);
    // Thin edges translucent and dimmer, thick middles bright
    const tone = mix(vec3(0.26, 0.32, 0.45), vec3(1.05, 1.02, 0.98), clamp(lighting, 0, 1)).mul(mix(float(0.8), float(1.04), smoothstep(0.0, 0.35, density)));
    const shadowOnSea = smoothstep(0.0, 0.12, thickness(p.sub(towardSun.mul(30)))).mul(0.62);
    const land = ground.mul(float(1).sub(shadowOnSea)).mul(billow(p, 60).mul(0.06).add(1));
    // The sun's glint off open sea (the map's blue over its red), under no cloud nor its shadow
    const sea = smoothstep(0.02, 0.1, ground.b.sub(ground.r)).mul(float(1).sub(cloud)).mul(float(1).sub(shadowOnSea.mul(2)).max(0));
    const mirrored = reflect(view.negate(), n);
    const glint = pow(max(dot(mirrored, this.sunUniform), 0), 180).mul(2.5).add(pow(max(dot(mirrored, this.sunUniform), 0), 18).mul(0.12));
    const surface = mix(land.mul(1.2), tone, cloud.mul(0.97)).add(vec3(1, 0.95, 0.85).mul(glint.mul(sea)));
    const rim = pow(float(1).sub(clamp(dot(n, view), 0, 1)), 3);
    const haze = vec3(0.35, 0.6, 1).mul(rim.mul(0.9).add(0.08));
    // Day: the surface and its haze; night: nearly black, a faint blue at the rim
    const dayColour = surface.mul(light.max(0).mul(0.85).add(0.15)).mul(1.4).add(haze.mul(0.25));
    const nightColour = vec3(0.004, 0.006, 0.012).add(vec3(0.02, 0.04, 0.09).mul(rim));
    earthMat.colorNode = mix(nightColour, dayColour, lit) as unknown as THREE.Node<'color'>;
    const earth = new THREE.Mesh(new THREE.SphereGeometry(EARTH_RADIUS, 192, 96), earthMat);
    earth.position.copy(earthCentre);
    // The poles along y (the sphere's own axis), so the station is over the tropics, not a pole
    // (where the map's meridians meet and smear): turned about them to put the Atlantic off
    // West Africa underneath (the map's 10 degrees west), tipped a little north
    earth.rotation.set(-0.25, -1.4, 0, 'XYZ');
    earth.renderOrder = -1;

    // The atmosphere: a shell round the Earth, its glow worked out along each view ray from how
    // near the ray passes the Earth's middle. Past the limb it fades smoothly to nothing at the
    // top of the air; over the planet it's a haze thickening towards the limb, meeting the
    // glow there. Brighter where the sun is on it.
    const top = EARTH_RADIUS * 1.04;
    const airMat = new THREE.MeshBasicNodeMaterial({ transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, fog: false });
    const c = vec3(earthCentre.x, earthCentre.y, earthCentre.z);
    const ray = normalize(positionWorld.sub(cameraPosition));
    const toCentre = c.sub(cameraPosition);
    const along = dot(toCentre, ray);
    const nearest = cameraPosition.add(ray.mul(along));
    const miss = length(nearest.sub(c));
    const outside = clamp(miss.sub(EARTH_RADIUS).div(top - EARTH_RADIUS), 0, 1);
    const glow = pow(float(1).sub(outside), 2.2);
    const inside = clamp(float(EARTH_RADIUS).sub(miss).div(EARTH_RADIUS * 0.22), 0, 1);
    const veil = pow(float(1).sub(inside), 4).mul(0.55);
    const thick = select(miss.greaterThan(EARTH_RADIUS), glow, veil);
    const sunlit = clamp(dot(normalize(nearest.sub(c)), this.sunUniform).mul(1.4).add(0.3), 0, 1);
    airMat.colorNode = vec3(0.28, 0.52, 1).mul(thick.mul(sunlit).mul(0.9)) as unknown as THREE.Node<'color'>;
    const air = new THREE.Mesh(new THREE.SphereGeometry(top, 160, 80), airMat);
    air.position.copy(earthCentre);
    air.renderOrder = 0;

    this.object.add(this.stars, earth, air);
    this.sunUniform.value.copy(this.sun);
  }

  /** How bright the sun is (in orbit, always full). */
  sunIntensity(): number {
    return 3.2;
  }

  /** The stars and the sun go with the camera; the clouds drift. */
  update(camera: THREE.Camera, time: number): void {
    this.stars.position.copy(camera.position);
    this.sunUniform.value.copy(this.sun);
    this.clock.value = time;
  }

  setQuality(_quality: string): void {}
}
