// Orbit's sky, in place of the city's (sky.ts): black, with stars and the sun's disc, and the
// Earth below. The Earth is a sphere far under the station, scaled with it: 5 km across at 350 m
// down keeps the ISS's ratio of height to radius, so its horizon dips as it would from 400 km
// (some 20 degrees) and its curve shows. NASA's Blue Marble on it (public/space, public
// domain), lit by the sun with a soft terminator and a night side, clouds drifting over it, a
// thin blue atmosphere glowing round its limb.

import * as THREE from 'three/webgpu';
import { OrbitClouds } from './orbit-clouds.ts';
import { skyBodies } from './planets.ts';
import { Satellites } from './satellites.ts';
import { cameraPosition, clamp, dot, fwidth, modelPosition, mx_noise_float, positionLocal, float, floor, fract, length, max, mix, mx_cell_noise_float, normalize, positionWorld, pow, reflect, select, smoothstep, texture, uniform, uv, vec2, vec3, vec4 } from 'three/tsl';

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

/**
 * A lap of the orbit (s): the real station's takes 92 minutes; this one's quicker, so the ground slides by under
 * it and the stars wheel. The axis it turns the Earth about: the ground moves towards -y.
 */
const ORBIT = 20 * 60;
const ORBIT_AXIS = new THREE.Vector3(1, 0, 0);
/** The cirrus's height above the Earth (scene units: about 11 km at the clouds' scale, over the cumulus's 7). */
const CIRRUS_HEIGHT = 60;

/**
 * High cirrus: a thin shell over the cloud layer, streaks of ice drawn out along the winds, in patches, lit by the
 * sun (dark on the night side), faded at the grazing edge where it would pile into a white band. Its noise is in
 * its own (turning) frame, so it goes round with the Earth.
 */
function cirrusShell(sun: THREE.UniformNode<'vec3', THREE.Vector3>): THREE.Mesh {
  const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthWrite: false, fog: false });
  const p = positionLocal;
  const streaks = mx_noise_float(p.mul(vec3(0.02, 0.006, 0.006))).mul(0.6).add(mx_noise_float(p.mul(vec3(0.05, 0.016, 0.016))).mul(0.3)).add(mx_noise_float(p.mul(0.2)).mul(0.1));
  const patches = smoothstep(0.25, 0.55, mx_noise_float(p.mul(0.0022)).add(mx_noise_float(p.mul(0.007)).mul(0.3)));
  const n = normalize(positionWorld.sub(modelPosition));
  const view = normalize(cameraPosition.sub(positionWorld));
  const edge = smoothstep(0.02, 0.25, dot(n, view));
  const lit = clamp(dot(n, sun).mul(1.2).add(0.1), 0, 1);
  material.colorNode = vec3(1.0, 1.0, 1.02).mul(lit.mul(1.1).add(0.03)) as unknown as THREE.Node<'color'>;
  material.opacityNode = smoothstep(0.05, 0.7, streaks).mul(patches).mul(edge).mul(0.35) as unknown as THREE.Node<'float'>;
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(EARTH_RADIUS + CIRRUS_HEIGHT, 160, 80), material);
  mesh.renderOrder = 1;
  return mesh;
}

/**
 * The station's sway (rad): a slow roll and pitch of a fifth of a degree or so, each from two periods (s) that
 * never quite line up. Seen from on board it's everything outside that sways, about the station's middle.
 */
const SWAY = 0.004;
const SWAY_PERIODS = [11, 17, 13, 19];

/** The clouds' sun and sky light, against the Earth's surface as the Earth shader lights it. */
const CLOUD_LIGHT = 0.8;

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
  /** Satellites passing under the station. */
  private readonly satellites: Satellites;
  /** The high, thin cirrus, a shell over the clouds. */
  private readonly cirrus: THREE.Mesh;
  /** The Earth, its air and its cirrus, swayed about the station's middle (`pivot`). */
  private readonly outside = new THREE.Group();
  private readonly pivot: THREE.Vector3;
  private readonly sway = new THREE.Quaternion();
  private readonly earthCentre: THREE.Vector3;
  private readonly earth!: THREE.Mesh;
  private readonly earthRest!: THREE.Quaternion;
  /** How far round the orbit has turned the Earth and the sky, and back again (for the stars). */
  private readonly spin = new THREE.Quaternion();
  private readonly unspin = uniform(new THREE.Matrix3());
  /** The clock when the orbit started (the first update). */
  private started = -1;

  /** `centre`: the point the Earth is under (the station's middle). */
  constructor(centre: THREE.Vector3) {
    this.pivot = centre.clone();
    const loader = new THREE.TextureLoader();
    const day = loader.load('space/earth.webp');
    day.colorSpace = THREE.SRGBColorSpace;
    day.anisotropy = 8;
    day.wrapS = THREE.RepeatWrapping;

    // The sky: stars in two sizes, hashed from the direction, and the sun's disc with its glare
    const skyMat = new THREE.MeshBasicNodeMaterial({ side: THREE.BackSide, depthWrite: false, fog: false });
    const dir = normalize(positionWorld.sub(cameraPosition));
    // The stars stay put in the sky the station orbits through: in its frame they wheel round with the Earth
    const fixedDir = this.unspin.mul(dir);
    let stars = float(0) as unknown as THREE.Node<'float'>;
    // Each star at least about a pixel across (the pixel's size in cells, from the screen-space derivative), its
    // brightness spread to keep its light the same: a star smaller than a pixel flickered on and off as the sky
    // turned it across pixels. (The faintest, finest layer went: it was nearly all under a pixel.)
    for (const [scale, keep, bright] of [
      [180, 0.992, 0.9],
      [420, 0.986, 0.5],
    ] as const) {
      const p = fixedDir.mul(scale);
      const cell = floor(p);
      // (A hash of the whole cell: one of its sum, x + 57y + 113z, is the same all along lines of cells, and drew
      // the stars in dotted streaks)
      const h = mx_cell_noise_float(cell);
      const spot = length(fract(p).sub(0.5));
      const w = fwidth(p);
      const pixel = max(w.x, max(w.y, w.z));
      const radius = max(float(0.12), pixel.mul(1.1));
      const spread = float(0.12).div(radius).pow(2);
      stars = stars.add(smoothstep(radius, 0.0, spot).mul(smoothstep(keep, 1.0, h)).mul(spread).mul(bright * 30)) as unknown as THREE.Node<'float'>;
    }
    const toSun = max(dot(dir, this.sunUniform), 0);
    const disc = smoothstep(0.99985, 0.9999, toSun).mul(14).add(pow(toSun, 1500).mul(1.5)).add(pow(toSun, 120).mul(0.04));
    // The Moon and the planets among the stars (planets.ts), hiding those behind them. (Lit by the sun as it is at
    // the start: kept in the stars' frame, the Moon's phase would turn over with each lap of the orbit.)
    const bodies = skyBodies(fixedDir, this.sunUniform);
    skyMat.colorNode = vec3(0.85, 0.9, 1)
      .mul(stars)
      .mul(float(1).sub(bodies.cover))
      .add(bodies.colour)
      .add(vec3(1, 0.97, 0.92).mul(disc)) as unknown as THREE.Node<'color'>;
    this.stars = new THREE.Mesh(new THREE.SphereGeometry(5500, 48, 24), skyMat);
    this.stars.renderOrder = -2;
    this.stars.frustumCulled = false;

    // The Earth: day map and clouds by the sphere's uv, lit by the sun; the atmosphere's haze
    // towards the limb, blue on the day side
    const earthCentre = new THREE.Vector3(centre.x, centre.y, centre.z - EARTH_RADIUS - ALTITUDE);
    const earthMat = new THREE.MeshBasicNodeMaterial({ fog: false });
    const n = normalize(positionWorld.sub(modelPosition));
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
    this.earth = earth;
    this.earthRest = earth.quaternion.clone();

    // The atmosphere: a shell round the Earth, its glow worked out along each view ray from how
    // near the ray passes the Earth's middle. Past the limb it fades smoothly to nothing at the
    // top of the air; over the planet it's a haze thickening towards the limb, meeting the
    // glow there. Brighter where the sun is on it.
    const top = EARTH_RADIUS * 1.04;
    const airMat = new THREE.MeshBasicNodeMaterial({ transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, fog: false });
    const c = modelPosition;
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

    this.clouds = new OrbitClouds(EARTH_RADIUS);
    this.cirrus = cirrusShell(this.sunUniform);
    this.cirrus.position.copy(earthCentre);
    this.earthCentre = earthCentre;
    this.satellites = new Satellites(centre, EARTH_RADIUS + ALTITUDE);
    this.outside.add(earth, air, this.cirrus, this.satellites.object);
    this.object.add(this.stars, this.outside);
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
    // The orbit: the Earth (and its clouds) and the stars turning together about the orbit's axis, a lap every
    // ORBIT seconds. (The sun is kept where it is, as in a dawn-to-dusk orbit: the station never goes into the dark.)
    if (this.started < 0) this.started = time;
    const t = time - this.started;
    this.satellites.update(t);
    this.spin.setFromAxisAngle(ORBIT_AXIS, (t / ORBIT) * Math.PI * 2);
    this.earth.quaternion.copy(this.spin).multiply(this.earthRest);
    this.cirrus.quaternion.copy(this.spin);
    // The sway, the outside turned about the station's middle (and the stars and clouds with it)
    const wave = (a: number, b: number, phase: number) => 0.6 * Math.sin((2 * Math.PI * t) / a + phase) + 0.4 * Math.sin((2 * Math.PI * t) / b + 2 * phase);
    const [a, b, c, d] = SWAY_PERIODS;
    this.sway.setFromEuler(new THREE.Euler(SWAY * wave(a, b, 0), SWAY * wave(c, d, 1), 0));
    this.outside.quaternion.copy(this.sway);
    this.outside.position.copy(this.pivot).sub(this.pivot.clone().applyQuaternion(this.sway));
    const turn = this.sway.clone().multiply(this.spin);
    this.unspin.value.setFromMatrix4(new THREE.Matrix4().makeRotationFromQuaternion(turn.clone().invert()));
    if (this.clouds.ready) {
      const centre = this.earthCentre.clone().sub(this.pivot).applyQuaternion(this.sway).add(this.pivot);
      this.clouds.update(camera as THREE.PerspectiveCamera, centre, turn, this.sun, this.sunColor, CLOUD_LIGHT);
    }
  }

  setQuality(_quality: string): void {}
}
