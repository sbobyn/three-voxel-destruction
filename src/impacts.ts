// Impacts you can see: a shockwave ring racing out over the surface that was struck, and a
// flash at the point of the blow (the sledgehammer, the wrecking ball); and the laser's beam
// with the glow where it burns. A few pooled meshes, additive and bright enough for the bloom
// to catch, drawn with no lights of their own (adding lights recompiles every material).

import * as THREE from 'three/webgpu';

const RINGS = 16;

interface Ring {
  mesh: THREE.Mesh;
  flash: THREE.Mesh;
  material: THREE.MeshBasicMaterial;
  flashMaterial: THREE.MeshBasicMaterial;
  age: number;
  life: number;
  size: number;
}

const additive = (colour: THREE.Color) =>
  new THREE.MeshBasicMaterial({ color: colour, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide, fog: false, toneMapped: true });

export class Impacts {
  readonly object = new THREE.Group();
  private readonly rings: Ring[] = [];
  private next = 0;
  private readonly q = new THREE.Quaternion();
  private readonly z = new THREE.Vector3(0, 0, 1);
  private readonly y = new THREE.Vector3(0, 1, 0);
  /** The laser: a white-hot core in a red glow, and the glow where it lands. */
  private readonly core: THREE.Mesh;
  private readonly halo: THREE.Mesh;
  private readonly spot: THREE.Mesh;
  private readonly coreMaterial: THREE.MeshBasicMaterial;
  private readonly haloMaterial: THREE.MeshBasicMaterial;
  private readonly spotMaterial: THREE.MeshBasicMaterial;

  constructor() {
    const ring = new THREE.RingGeometry(0.82, 1, 48);
    const flash = new THREE.IcosahedronGeometry(1, 2);
    for (let k = 0; k < RINGS; k++) {
      const material = additive(new THREE.Color(1, 1, 1));
      const flashMaterial = additive(new THREE.Color(1, 1, 1));
      const mesh = new THREE.Mesh(ring, material);
      const f = new THREE.Mesh(flash, flashMaterial);
      mesh.visible = f.visible = false;
      mesh.frustumCulled = f.frustumCulled = false;
      this.object.add(mesh, f);
      this.rings.push({ mesh, flash: f, material, flashMaterial, age: 1, life: 1, size: 1 });
    }
    // The beam: unit-length cylinders along y, stretched between the muzzle and the burn
    const beam = new THREE.CylinderGeometry(1, 1, 1, 10, 1, true);
    beam.translate(0, 0.5, 0);
    this.coreMaterial = additive(new THREE.Color(6, 2.2, 1.8));
    this.haloMaterial = additive(new THREE.Color(3.2, 0.12, 0.06));
    this.spotMaterial = additive(new THREE.Color(8, 2.4, 0.9));
    this.core = new THREE.Mesh(beam, this.coreMaterial);
    this.halo = new THREE.Mesh(beam, this.haloMaterial);
    this.spot = new THREE.Mesh(flash, this.spotMaterial);
    for (const m of [this.core, this.halo, this.spot]) {
      m.visible = false;
      m.frustumCulled = false;
      this.object.add(m);
    }
  }

  /**
   * A blow at `at` on a surface facing `normal`: a ring out to `size` m over `life` s, and a
   * flash of `flash` m, both `colour` (linear, may be over 1 to bloom).
   */
  hit(at: ArrayLike<number>, normal: ArrayLike<number>, size: number, life: number, colour: THREE.Color, flash = size * 0.25): void {
    const r = this.rings[this.next];
    this.next = (this.next + 1) % RINGS;
    r.age = 0;
    r.life = life;
    r.size = size;
    r.material.color.copy(colour);
    r.flashMaterial.color.copy(colour).multiplyScalar(2.5);
    const n = new THREE.Vector3(normal[0], normal[1], normal[2]);
    if (n.lengthSq() < 1e-6) n.set(0, 0, 1);
    n.normalize();
    // Just off the surface, facing out of it
    r.mesh.position.set(at[0] + n.x * 0.06, at[1] + n.y * 0.06, at[2] + n.z * 0.06);
    r.mesh.quaternion.copy(this.q.setFromUnitVectors(this.z, n));
    r.flash.position.set(at[0], at[1], at[2]);
    r.flash.scale.setScalar(flash);
    r.mesh.visible = r.flash.visible = true;
  }

  /**
   * The laser's beam this frame, from `from` to `to` (world m), or hidden (null). `burning`:
   * whether it lands on something (the glow there).
   */
  beam(from: THREE.Vector3 | null, to?: THREE.Vector3, burning = false, time = 0): void {
    const on = !!from && !!to;
    this.core.visible = this.halo.visible = on;
    this.spot.visible = on && burning;
    if (!on) return;
    const d = to!.clone().sub(from!);
    const length = d.length();
    this.q.setFromUnitVectors(this.y, d.divideScalar(Math.max(length, 1e-6)));
    // Flickering a little, as a beam through dust does
    const flicker = 0.85 + 0.15 * Math.sin(time * 90) * Math.sin(time * 37);
    for (const [m, w] of [
      [this.core, 0.012],
      [this.halo, 0.045],
    ] as [THREE.Mesh, number][]) {
      m.position.copy(from!);
      m.quaternion.copy(this.q);
      m.scale.set(w * flicker, length, w * flicker);
    }
    this.spot.position.copy(to!);
    this.spot.scale.setScalar(0.16 + 0.08 * Math.random());
  }

  update(dt: number): void {
    for (const r of this.rings) {
      if (!r.mesh.visible) continue;
      r.age += dt;
      const f = r.age / r.life;
      if (f >= 1) {
        r.mesh.visible = r.flash.visible = false;
        continue;
      }
      // Fast out, easing; fading as it goes
      const e = 1 - (1 - f) ** 3;
      r.mesh.scale.setScalar(0.15 + r.size * e);
      r.material.opacity = (1 - f) ** 2 * 0.8;
      const ff = Math.min(1, r.age / (r.life * 0.25));
      r.flashMaterial.opacity = (1 - ff) ** 2;
      if (ff >= 1) r.flash.visible = false;
    }
  }
}
