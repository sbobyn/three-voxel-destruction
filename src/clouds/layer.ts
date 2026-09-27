// From web-gpu-gems (src/clouds/layer.ts at 51e1988), the volumetric cloud layer of its Clouds lab and planet.
// TSL cloud march. Mirrors model.ts and reference.ts formula for formula so the CPU reference can check it.
import * as THREE from 'three/webgpu';
import { bitcast, Break, dot, exp, float, floor, Fn, fract, If, int, Loop, max, min, mix, normalize, screenCoordinate, struct, pow, select, smoothstep, sqrt, texture3D, uint, uniform, uv, vec3, vec4 } from 'three/tsl';
import { DEFAULTS, PERIODS } from './model.ts';

type N<T extends string = 'float'> = THREE.Node<T>;
const v3 = (a: readonly number[]) => new THREE.Vector3(a[0], a[1], a[2]);

/** All distances are kilometres, with +Y radial up in the camera-local frame. */
export interface CloudLayerConfig {
  base: THREE.Data3DTexture;
  detail: THREE.Data3DTexture;
  worldFrame?: boolean;
  /** Regional coverage and weather noise are independent [0,1] inputs at camera-local offsets. */
  coverage?: (offset: N<'vec3'>) => N;
  weather?: (offset: N<'vec3'>) => N;
  sunTransmittance?: (radius: N, sunZenithCosine: N) => N<'vec3'>;
  /** Linear incident ambient radiance; defaults to the ambient uniform. */
  ambient?: (offset: N<'vec3'>, altitude: N) => N<'vec3'>;
  /** Borrowed by default. Transfer only textures not shared with another layer. */
  textureOwnership?: 'borrowed' | 'owned';
}

export function createCloudLayer(options: CloudLayerConfig) {
  const { base, detail } = options;
  const d = DEFAULTS;
  const u = {
    radius: uniform(d.planetRadius), bottom: uniform(d.bottom), top: uniform(d.top), altitude: uniform(0.2),
    forward: uniform(new THREE.Vector3(0, 0, -1)), right: uniform(new THREE.Vector3(1, 0, 0)), up: uniform(new THREE.Vector3(0, 1, 0)),
    tanHalf: uniform(Math.tan(Math.PI / 6)), aspect: uniform(1),
    sun: uniform(v3(d.sun)), sunColour: uniform(v3(d.sunColour)), ambient: uniform(v3(d.ambient)),
    coverage: uniform(d.coverage), coverageScale: uniform(1), density: uniform(d.density), erosion: uniform(d.erosion),
    albedo: uniform(d.albedo), anisotropy: uniform(d.anisotropy),
    backAnisotropy: uniform(d.backAnisotropy), lobeBlend: uniform(d.lobeBlend),
    octaves: uniform(d.octaves, 'int'), octaveA: uniform(d.octaveA), octaveB: uniform(d.octaveB), octaveC: uniform(d.octaveC),
    powder: uniform(0),
    /** Split view: pixels left of splitX use one lobe, one octave and no powder (the Clouds 1 look). 0 disables it. */
    splitX: uniform(0),
    /** Cost reductions used by the display march only (fastMarch); the checked march never uses them. */
    skip: uniform(0), adaptive: uniform(0), cheapLight: uniform(0), jitter: uniform(0),
    /** Shortest fine step (km); 0 keeps a fixed count. Short rays then take fewer steps instead of oversampling. */
    minStep: uniform(0),
    /** Distance LOD, each 0/1-switched: detail erosion fades to its mean (0.494) between lodStart and lodEnd km (set from
     * the pixel footprint: where a pixel covers half the detail period), and light marches beyond lightLodDistance km
     * use farLightSteps. */
    lod: uniform(0), lodStart: uniform(800), lodEnd: uniform(1600), lightLod: uniform(0), lightLodDistance: uniform(100), farLightSteps: uniform(3, 'int'),
    /** Host-controlled sub-pixel and step jitter; no history or clock is owned here. */
    uvJitter: uniform(new THREE.Vector2()), frameNoise: uniform(0),
    /** Step growth with distance (km of step per km along the ray); 0 disables it. */
    stepPerKm: uniform(0),
    viewSteps: uniform(d.viewSteps, 'int'), lightSteps: uniform(d.lightSteps, 'int'), lightDistance: uniform(d.lightDistance),
    energy: uniform(1), early: uniform(0),
    originBase: uniform(new THREE.Vector3()), originDetail: uniform(new THREE.Vector3()), originWeather: uniform(new THREE.Vector3()),
    noiseFrame: uniform(new THREE.Matrix3()),
  };
  const R = u.radius;

  /** Pixel ray; the quad's UV y runs downward, so camera-up uses 1 − 2v. */
  const rayDirection = (at: N<'vec2'> = uv()) => normalize(u.forward
    .add(u.right.mul(at.x.mul(2).sub(1).mul(u.tanHalf).mul(u.aspect)))
    .add(u.up.mul(float(1).sub(at.y.mul(2)).mul(u.tanHalf))));

  /** Roots of the ray against the sphere at altitude hs (x ≤ y) and whether it hits (z). See model.sphere(). */
  const sphere = (h: N, hs: N, dy: N) => Fn(() => {
    const b = R.add(h).mul(dy).toVar(), c = h.sub(hs).mul(h.add(hs).add(R.mul(2))).toVar();
    // Near the orbital limb b² and c nearly cancel. Split operands into 12-bit
    // significands so their high products are exact, retaining the low terms
    // until after the large subtraction. Derive both parts from bits: an
    // algebraic splitter can be folded back to the original float32 product.
    // r185's bitcast returns a TSL proxy; its declarations omit the node extensions.
    const bitsOf = (v: N) => bitcast(v, 'uint') as unknown as N<'uint'>;
    const floatOf = (v: N<'uint'>) => bitcast(v, 'float') as unknown as N;
    const high = (v: N) => floatOf(bitsOf(v).bitAnd(uint(0xfffff000)));
    const low = (v: N) => {
      const bits = bitsOf(v), exponent = bits.bitAnd(uint(0xff800000));
      return floatOf(exponent.bitOr(bits.bitAnd(uint(0xfff)))).sub(floatOf(exponent));
    };
    const product = (x: N, y: N) => {
      const xh = high(x).toVar(), yh = high(y).toVar(), xl = low(x).toVar(), yl = low(y).toVar();
      return { hi: xh.mul(yh).toVar(), lo: xh.mul(yl).add(xl.mul(yh)).add(xl.mul(yl)).toVar() };
    };
    const bp = product(R.add(h), dy), bh = high(b).toVar(), bDifference = bp.hi.sub(bh).toVar();
    // Split the differences too: shader reassociation must not add the low
    // product back to the large high product before subtracting its neighbour.
    const bl = high(bDifference).add(low(bDifference).add(bp.lo)).toVar();
    const cp = product(h.sub(hs), h.add(hs).add(R.mul(2)));
    const difference = bh.mul(bh).sub(cp.hi).toVar();
    const disc = high(difference).add(low(difference).add(bh.mul(bl).mul(2)).add(bl.mul(bl)).sub(cp.lo)).toVar(), s = sqrt(max(disc, 0)).toVar();
    const q = select(b.greaterThanEqual(0), bh.add(s).add(bl).negate(), s.sub(bh).sub(bl));
    // At the limb, direct subtraction is well conditioned and avoids rounding
    // c/q again. Retain the stable quotient where a near-zero root would cancel.
    const direct = select(b.greaterThanEqual(0), s.sub(bh).sub(bl), bh.add(s).add(bl).negate());
    const t2 = select(s.lessThan(b.abs().mul(0.5)), direct, select(q.equal(0), float(0), c.div(q)));
    return vec3(min(q, t2), max(q, t2), select(disc.greaterThanEqual(0), float(1), float(0)));
  })();
  const altitudeAt = (h: N, b: N, t: N) => {
    const rise = t.mul(b).mul(2).add(t.mul(t));
    return h.mul(h.add(R.mul(2))).add(rise).div(sqrt(R.add(h).mul(R.add(h)).add(rise)).add(R));
  };
  const lightPath = (a: N, bl: N) => {
    const blocked = bl.lessThan(0).and(bl.mul(bl).sub(a.mul(a.add(R.mul(2)))).greaterThan(0));
    const c = a.sub(u.top).mul(a.add(u.top).add(R.mul(2))), s = sqrt(max(bl.mul(bl).sub(c), 0));
    const exit = select(bl.greaterThan(0), c.negate().div(bl.add(s)), bl.negate().add(s));
    return select(blocked, float(0), min(max(exit, 0), u.lightDistance));
  };
  /**
   * Hardware trilinear filtering, or (exact) the same reconstruction done in float32: eight fetches at texel
   * centres, where the hardware's weights are exactly 0 or 1, blended with full-precision weights. GPUs may
   * quantize filter weights (to 8 bits on many), so only the exact path can match the CPU reference tightly;
   * the lab checks both and reports the difference.
   */
  const sample = (tex: THREE.Data3DTexture, origin: N<'vec3'>, q: N<'vec3'>, period: number, exact: boolean) => {
    const coordinate = origin.add(options.worldFrame ? u.noiseFrame.mul(q) : q).div(period);
    if (!exact) return texture3D(tex, coordinate).level(float(0));
    const size = tex.image.width, c = coordinate.mul(size).sub(0.5), i0 = floor(c), f = c.sub(i0);
    let sum: N<'vec4'> = vec4(0);
    for (let dz = 0; dz < 2; dz++) for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) {
      const w = (dx ? f.x : float(1).sub(f.x)).mul(dy ? f.y : float(1).sub(f.y)).mul(dz ? f.z : float(1).sub(f.z));
      sum = sum.add(texture3D(tex, i0.add(vec3(dx, dy, dz)).add(0.5).div(size)).level(float(0)).mul(w));
    }
    return sum;
  };
  /** The base shape after coverage, in [0, 1]: everything but the detail erosion, which can only lower it. */
  const coverage = (q: N<'vec3'>, a: N, exact = false) => {
    const weather = options.weather ? options.weather(q) : sample(base, u.originWeather, q, PERIODS.weather, exact).r;
    const b = sample(base, u.originBase, q, PERIODS.base, exact);
    const regional=options.coverage?options.coverage(q):u.coverage;
    const cover = regional.add(weather.sub(0.5).mul(0.8)).clamp(0, 1).mul(u.coverageScale).clamp(0, 1);
    const shapeFbm = b.g.mul(0.625).add(b.b.mul(0.25)).add(b.a.mul(0.125));
    const hf = a.sub(u.bottom).div(u.top.sub(u.bottom)).clamp(0, 1);
    const profile = smoothstep(0, 0.15, hf).mul(float(1).sub(smoothstep(0.5, 1, hf)));
    const shape = b.r.sub(shapeFbm).add(1).div(float(2).sub(shapeFbm)).mul(profile);
    return shape.sub(float(1).sub(cover)).div(max(cover, 1e-3)).clamp(0, 1).mul(cover);
  };
  /** Extinction from a coverage value: detail erosion, then the density scale. Zero wherever coverage is zero. */
  const erodeWith = (covered: N, q: N<'vec3'>, a: N, exact = false) => {
    const dt = sample(detail, u.originDetail, q, PERIODS.detail, exact);
    const hf = a.sub(u.bottom).div(u.top.sub(u.bottom)).clamp(0, 1);
    const fbm = dt.r.mul(0.625).add(dt.g.mul(0.25)).add(dt.b.mul(0.125));
    const erode = fbm.add(float(1).sub(fbm.mul(2)).mul(hf.mul(4).clamp(0, 1))).mul(u.erosion);
    return covered.sub(erode).div(float(1).sub(erode)).clamp(0, 1).mul(u.density);
  };
  const extinction = (q: N<'vec3'>, a: N, exact = false) => erodeWith(coverage(q, a, exact), q, a, exact);
  /** erodeWith with the detail fBm replaced by its mean: no detail fetch, for distant cloud. The mean is measured from
   * the detail texels in tests/clouds-noise.test.ts. */
  const DETAIL_MEAN = 0.494;
  const erodeFar = (covered: N, a: N) => {
    const hf = a.sub(u.bottom).div(u.top.sub(u.bottom)).clamp(0, 1);
    const erode = float(DETAIL_MEAN).add(float(1 - 2 * DETAIL_MEAN).mul(hf.mul(4).clamp(0, 1))).mul(u.erosion);
    return covered.sub(erode).div(float(1).sub(erode)).clamp(0, 1).mul(u.density);
  };
  /** Sun colour reaching a sample at altitude a, where bl = P·L (so cos sun zenith = bl / (R + a)). */
  const sunColourAt = (a: N, bl: N) => options.sunTransmittance
    ? u.sunColour.mul(options.sunTransmittance(R.add(a), bl.div(R.add(a))))
    : u.sunColour;
  const ambientAt = (q: N<'vec3'>, a: N) => options.ambient ? options.ambient(q, a) : u.ambient;
  const henyeyGreenstein = (g: N, cosine: N) =>
    float(1).sub(g.mul(g)).div(pow(float(1).add(g.mul(g)).sub(g.mul(2).mul(cosine)), 1.5).mul(4 * Math.PI));
  /** model.sunScatter: Σₙ bⁿ·e^(−aⁿ·depth)·phase(cⁿ), times the powder term when enabled. */
  const sunScatter = (cosine: N, depth: N, lit: N<'bool'>, baseline: N<'bool'>) => {
    const blend = select(baseline, float(0), u.lobeBlend), count = select(baseline, int(1), u.octaves);
    const sum = float(0).toVar(), a = float(1).toVar(), b = float(1).toVar(), c = float(1).toVar();
    Loop({ start: int(0), end: count, type: 'int', condition: '<', name: 'n' } as never, (() => {
      const lobes = float(1).sub(blend).mul(henyeyGreenstein(u.anisotropy.mul(c), cosine)).add(blend.mul(henyeyGreenstein(u.backAnisotropy.mul(c), cosine)));
      sum.addAssign(b.mul(exp(a.mul(depth).negate())).mul(lobes));
      a.mulAssign(u.octaveA); b.mulAssign(u.octaveB); c.mulAssign(u.octaveC);
    }) as never);
    const powder = select(u.powder.greaterThan(0.5).and(baseline.not()), float(1).sub(exp(depth.mul(-2))), float(1));
    return select(lit, sum.mul(powder), float(0));
  };

  /** The ray's path through the shell as two segments, as model.shellSegments. */
  const segments = (dir: N<'vec3'>, limit?: N) => {
    const h = u.altitude, ground = sphere(h, float(0), dir.y), outer = sphere(h, u.top, dir.y), inner = sphere(h, u.bottom, dir.y);
    const hitsGround = ground.z.greaterThan(0.5).and(ground.x.greaterThan(0));
    const outerOk = outer.z.greaterThan(0.5).and(outer.y.greaterThan(0)), innerOk = inner.z.greaterThan(0.5);
    const lo = max(outer.x, 0), hi = outer.y;
    const lenA = select(outerOk, max(select(innerOk, min(hi, inner.x), hi).sub(lo), 0), float(0));
    const b0 = max(inner.y, lo);
    const lenB = select(outerOk.and(innerOk).and(hitsGround.not()), max(hi.sub(b0), 0), float(0));
    // An optional limit (the distance to opaque scene geometry) ends the path early.
    if (!limit) return { lo, lenA, b0, lenB };
    return { lo, lenA: min(lenA, max(limit.sub(lo), 0)), b0, lenB: min(lenB, max(limit.sub(b0), 0)) };
  };
  /** Premultiplied single-scattered cloud radiance (rgb) and transmittance (a) along the pixel's ray, or along given rays. */
  const march = (exact = false, given?: N<'vec3'>) => Fn(() => {
    const dir = given ?? rayDirection(), h = u.altitude, L = u.sun;
    const { lo, lenA, b0, lenB } = segments(dir);
    const dt = lenA.add(lenB).div(float(u.viewSteps));
    const b = R.add(h).mul(dir.y), cosine = dot(dir, L);
    const baseline = given ? float(0).greaterThan(1) : uv().x.lessThan(u.splitX);
    const radiance = vec3(0).toVar(), T = float(1).toVar();
    If(dt.greaterThan(0), () => {
      Loop({ start: int(0), end: u.viewSteps, type: 'int', condition: '<' }, ({ i }) => {
        If(u.early.greaterThan(0.5).and(T.lessThan(0.01)), () => { Break(); });
        const along = float(i).add(0.5).mul(dt);
        const t = select(along.lessThan(lenA), lo.add(along), b0.add(along).sub(lenA)).toVar();
        const a = altitudeAt(h, b, t).toVar(), q = dir.mul(t).toVar();
        const sigma = extinction(q, a, exact).toVar();
        If(sigma.greaterThan(0), () => {
          const bl = R.add(h).mul(L.y).add(t.mul(cosine));
          const length = lightPath(a, bl), ds = length.div(float(u.lightSteps));
          const depth = float(0).toVar();
          // A distinct name keeps the inner counter from shadowing the view loop's i (three's typings omit `name`).
          Loop({ start: int(0), end: u.lightSteps, type: 'int', condition: '<', name: 'j' } as never, (({ j }: { j: N<'int'> }) => {
            const s = float(j).add(0.5).mul(ds);
            depth.addAssign(extinction(q.add(L.mul(s)), altitudeAt(a, bl, s), exact).mul(ds));
          }) as never);
          const sun = sunScatter(cosine, depth, length.greaterThan(0), baseline), stepT = exp(sigma.mul(dt).negate());
          // Energy-conserving (Hillaire 2015): integrate S·e^(−σx) over the step instead of S·dt.
          const weight = select(u.energy.greaterThan(0.5), float(1).sub(stepT).div(sigma), dt);
          radiance.addAssign(sunColourAt(a, bl).mul(sun).add(ambientAt(q, a)).mul(T.mul(u.albedo).mul(sigma).mul(weight)));
          T.mulAssign(stepT);
        });
      });
    });
    return vec4(radiance, T);
  })();

  /**
   * The display march, with optional cost reductions. With all of them off it samples the checked march's
   * grid, up to float accumulation of the position, so the GPU-vs-CPU check covers that configuration. The reductions:
   * - skip: where the base shape is empty the extinction is exactly zero, so the detail fetch and light march are skipped;
   * - adaptive: 4× steps through empty space; on a hit, step back one coarse step and continue finely; after four
   *   empty fine samples, go coarse again (can step over thin cloud);
   * - cheapLight: the sun march uses the base shape without detail erosion (darker in sunlight, an approximation);
   * - jitter: each pixel starts at its own offset within the first step (interleaved gradient noise), trading banding for noise;
   * - minStep: a floor on the fine step, so a 3 km ray does not take as many samples as a 100 km one;
   * - stepPerKm: the step grows with distance, so cloud hundreds of kilometres away is sampled coarsely;
   * - lod: detail erosion fades to its mean where a pixel is wider than half the detail period (no detail fetch there);
   * - lightLod: fewer light-march steps beyond a distance.
   * Early exit is the checked march's own switch.
   */
  const fastMarchBody = (limit?: (dir: N<'vec3'>) => N) => {
    const dir = rayDirection(uv().add(u.uvJitter)), h = u.altitude, L = u.sun;
    const { lo, lenA, b0, lenB } = segments(dir, limit?.(dir));
    const total = lenA.add(lenB), fine = max(total.div(float(u.viewSteps)), u.minStep), K = 4;
    const b = R.add(h).mul(dir.y), cosine = dot(dir, L);
    const baseline = uv().x.lessThan(u.splitX);
    // Interleaved gradient noise, shifted each frame (by the golden ratio) when accumulating over time.
    const noise = fract(fract(fract(screenCoordinate.x.mul(0.06711056).add(screenCoordinate.y.mul(0.00583715))).mul(52.9829189)).add(u.frameNoise));
    const s = select(u.jitter.greaterThan(0.5), noise, float(0.5)).mul(fine).toVar();
    const coarse = select(u.adaptive.greaterThan(0.5), float(1), float(0)).toVar(), empties = int(0).toVar();
    // Position of the previous sample, starting one step before the path so the first step back stops at 0. A step back
    // never returns behind it, so every iteration advances by at least `fine` ≥ total / viewSteps and the path is always
    // finished within viewSteps iterations; the loop allows twice that.
    const last = fine.negate().toVar();
    const radiance = vec3(0).toVar(), T = float(1).toVar();
    // Cloud depth for reprojection: the distance of each step weighted by the light it removes, T·(1 − e^(−σΔ)).
    const depthSum = float(0).toVar(), weightSum = float(0).toVar();
    If(total.greaterThan(0), () => {
      Loop({ start: int(0), end: u.viewSteps.mul(2), type: 'int', condition: '<' }, () => {
        If(s.greaterThanEqual(total), () => { Break(); });
        If(u.early.greaterThan(0.5).and(T.lessThan(0.01)), () => { Break(); });
        const t = select(s.lessThan(lenA), lo.add(s), b0.add(s).sub(lenA)).toVar();
        // Distance growth stops at an eighth of the layer's thickness, so coarse steps (4×) cannot jump the whole layer.
        const step = max(fine, min(t.mul(u.stepPerKm), u.top.sub(u.bottom).div(8))).toVar();
        const a = altitudeAt(h, b, t).toVar(), q = dir.mul(t).toVar();
        const covered = coverage(q, a).toVar();
        If(coarse.greaterThan(0.5), () => {
          // A hit while coarse: resume finely one step past the last empty coarse sample.
          If(covered.greaterThan(0), () => { s.assign(max(s.sub(step.mul(K - 1)), last.add(fine))); coarse.assign(0); empties.assign(0); })
            .Else(() => { last.assign(s); s.addAssign(step.mul(K)); });
        }).Else(() => {
          const sigma = float(0).toVar();
          const lod = select(u.lod.greaterThan(0.5), smoothstep(u.lodStart, u.lodEnd, t), float(0)).toVar();
          If(covered.greaterThan(0).or(u.skip.lessThan(0.5)), () => {
            If(lod.greaterThan(0.999), () => { sigma.assign(erodeFar(covered, a)); })
              .ElseIf(lod.greaterThan(0), () => { sigma.assign(mix(erodeWith(covered, q, a), erodeFar(covered, a), lod)); })
              .Else(() => { sigma.assign(erodeWith(covered, q, a)); });
          });
          If(sigma.greaterThan(0), () => {
            const bl = R.add(h).mul(L.y).add(t.mul(cosine));
            // Convert each branch: a select between two int uniforms does not take float() around it in WGSL.
            const far = u.lightLod.greaterThan(0.5).and(t.greaterThan(u.lightLodDistance));
            const lightCount = select(far, u.farLightSteps, u.lightSteps);
            const length = lightPath(a, bl), ds = length.div(select(far, float(u.farLightSteps), float(u.lightSteps)));
            const depth = float(0).toVar();
            Loop({ start: int(0), end: lightCount, type: 'int', condition: '<', name: 'j' } as never, (({ j }: { j: N<'int'> }) => {
              const sl = float(j).add(0.5).mul(ds), ql = q.add(L.mul(sl)), al = altitudeAt(a, bl, sl);
              const c = coverage(ql, al).toVar();
              If(u.cheapLight.greaterThan(0.5), () => { depth.addAssign(c.mul(u.density).mul(ds)); })
                .ElseIf(lod.greaterThan(0.999).and(c.greaterThan(0).or(u.skip.lessThan(0.5))), () => { depth.addAssign(erodeFar(c, al).mul(ds)); })
                .ElseIf(c.greaterThan(0).or(u.skip.lessThan(0.5)), () => { depth.addAssign(erodeWith(c, ql, al).mul(ds)); });
            }) as never);
            const sun = sunScatter(cosine, depth, length.greaterThan(0), baseline), stepT = exp(sigma.mul(step).negate());
            const weight = select(u.energy.greaterThan(0.5), float(1).sub(stepT).div(sigma), step);
            radiance.addAssign(sunColourAt(a, bl).mul(sun).add(ambientAt(q, a)).mul(T.mul(u.albedo).mul(sigma).mul(weight)));
            const removed = T.mul(float(1).sub(stepT));
            depthSum.addAssign(removed.mul(t)); weightSum.addAssign(removed);
            T.mulAssign(stepT);
          });
          empties.assign(select(covered.greaterThan(0), int(0), empties.add(1)));
          If(u.adaptive.greaterThan(0.5).and(empties.greaterThanEqual(4)), () => { coarse.assign(1); });
          last.assign(s); s.addAssign(step);
        });
      });
    });
    // Without cloud, the middle of the shell (or far away) keeps reprojection close to a pure rotation.
    const depth = select(weightSum.greaterThan(1e-4), depthSum.div(max(weightSum, 1e-4)), select(total.greaterThan(0), lo.add(total.mul(0.5)), float(1e4)));
    return { colour: vec4(radiance, T), depth };
  };
  const fastMarch = Fn(() => fastMarchBody().colour)();
  /** The display march stopped at a per-pixel distance (km along the ray) given as a function of the ray direction. */
  const fastMarchTo = (limit: (dir: N<'vec3'>) => N) => Fn(() => fastMarchBody(limit).colour)();
  /** The same march returning colour and cloud depth (km) together, evaluated once; for a two-attachment target. */
  const MarchOut = struct({ colour: 'vec4', depth: 'float' }, 'CloudMarchOut');
  // toVar (missing from three's StructNode typings) stores the call once so both members read the same result.
  const withDepth = (limit?: (dir: N<'vec3'>) => N) => {
    const out = (Fn(() => { const r = fastMarchBody(limit); return MarchOut(r.colour, r.depth); })() as unknown as { toVar(): unknown }).toVar() as { get(name: string): THREE.Node };
    return { colour: out.get('colour') as N<'vec4'>, depth: out.get('depth') as N };
  };
  const fastMarchWithDepth = withDepth();
  /** fastMarchWithDepth stopped at a per-pixel distance, as fastMarchTo. */
  const fastMarchWithDepthTo = (limit: (dir: N<'vec3'>) => N) => withDepth(limit);

  let disposed = false;
  const dispose = () => {
    if (disposed) return;
    disposed = true;
    if (options.textureOwnership === 'owned') {
      for (const texture of new Set([base, detail])) texture.dispose();
    }
  };
  return { uniforms: u, textures: { base, detail }, march, fastMarch, fastMarchTo,
    fastMarchWithDepth, fastMarchWithDepthTo, depth: fastMarchWithDepth.depth,
    rayDirection, sphere, extinction, dispose };
}
export type CloudLayer = ReturnType<typeof createCloudLayer>;
