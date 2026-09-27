// Orbit's sky, in place of the city's (sky.ts): black, with stars and the sun's disc, and the
// Earth below. The Earth is a sphere far under the station, scaled with it: 5 km across at 350 m
// down keeps the ISS's ratio of height to radius, so its horizon dips as it would from 400 km
// (some 20 degrees) and its curve shows. NASA's Blue Marble on it (public/space, public
// domain), lit by the sun with a soft terminator and a night side, clouds drifting over it, a
// thin blue atmosphere glowing round its limb.

import * as THREE from 'three/webgpu';
import { OrbitClouds } from './orbit-clouds.ts';
import { cameraPosition, clamp, dot, float, floor, fract, hash, length, max, mix, normalize, positionWorld, pow, reflect, select, smoothstep, texture, uniform, uv, vec2, vec3, vec4 } from 'three/tsl';

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

/** The clouds' sun and sky light, against the Earth's surface as the Earth shader lights it. */
const CLOUD_LIGHT = 1.4;

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
  /** The clouds over the Earth (on a quad on the camera: main.ts adds it there and loads them). */
  readonly clouds: OrbitClouds;

  /** `centre`: the point the Earth is under (the station's middle). */
  constructor(centre: THREE.Vector3) {
    const loader = new THREE.TextureLoader();
    const day = loader.load('space/earth.webp');
    day.colorSpace = THREE.SRGBColorSpace;
    day.anisotropy = 8;
    day.wrapS = THREE.RepeatWrapping;

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
    // The land and sea, read bicubic (magnified straight down, plain bilinear shows each texel's square), and
    // the sun's glint off open sea (the map's blue over its red). The clouds are a layer of their own, marched
    // over this (orbit-clouds.ts).
    const ground = bicubic(day, uv(), [4096, 2048]).rgb;
    const sea = smoothstep(0.02, 0.1, ground.b.sub(ground.r));
    const mirrored = reflect(view.negate(), n);
    const glint = pow(max(dot(mirrored, this.sunUniform), 0), 180).mul(2.5).add(pow(max(dot(mirrored, this.sunUniform), 0), 18).mul(0.12));
    const surface = ground.mul(1.2).add(vec3(1, 0.95, 0.85).mul(glint.mul(sea)));
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

    this.clouds = new OrbitClouds(earthCentre, EARTH_RADIUS);
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
    if (this.clouds.ready) this.clouds.update(camera as THREE.PerspectiveCamera, this.sun, this.sunColor, CLOUD_LIGHT);
  }

  setQuality(_quality: string): void {}
}
