// The city round the block: towers and mid-rises on the lots beyond the streets, for scale and
// a skyline behind the tower. Drawn only (one instanced mesh of boxes, a facade shader with
// floors, windows and lights left on): nothing here is in the physics, so it costs next to
// nothing. The lots south and south-west of the block stay open (parks), where the views of
// the tower are taken from.

import { abs, cameraPosition, float, floor, fract, hash, instanceIndex, normalWorld, positionWorld, select, smoothstep, step, varying, vec3 } from 'three/tsl';
import * as THREE from 'three/webgpu';
import { type City, random } from './world.ts';

/**
 * Lots (block grid cells) left open: the ones the views of the tower are taken from, and a
 * line of parks toward the low evening sun (west-south-west), so it can set over them.
 */
const OPEN = new Set(['0,0', '0,-1', '-1,-1', '1,-1', '-1,0', '-2,-1', '-3,-1', '-3,-2']);
/** How many blocks out the city goes. */
const RINGS = 3;
const STOREY = 3.2;

export function skyline(city: City, seed = 17): THREE.Mesh {
  const rnd = random(seed);
  const block = city.pitch - city.street;
  const boxes: { x: number; y: number; w: number; d: number; h: number; tone: number }[] = [];
  for (let j = -RINGS; j <= RINGS; j++) {
    for (let i = -RINGS; i <= RINGS; i++) {
      if (OPEN.has(`${i},${j}`)) continue;
      const [cx, cy] = [i * city.pitch, j * city.pitch];
      const ring = Math.max(Math.abs(i), Math.abs(j));
      // A plinth of paving under the lot
      boxes.push({ x: cx, y: cy, w: block, d: block, h: 0.12, tone: -1 });
      // Low next to the block (the tower stands out), taller further out, a few landmark
      // towers at the back (north and east)
      const behind = j >= 0 || i >= 1;
      const [lo, hi] = ring === 1 ? [8, 20] : ring === 2 ? [18, behind ? 60 : 36] : [30, behind ? 110 : 60];
      if (rnd() < (ring === 1 ? 0.15 : 0.45)) {
        const sz = 14 + rnd() * 10;
        const h = (lo + rnd() * (hi - lo)) * (ring === 3 && behind && rnd() < 0.35 ? 1.5 : 1);
        boxes.push({ x: cx + (rnd() - 0.5) * (block - sz - 3), y: cy + (rnd() - 0.5) * (block - sz - 3), w: sz, d: sz * (0.8 + rnd() * 0.4), h, tone: rnd() });
      } else {
        // Two to four smaller blocks round the lot
        for (const [qx, qy] of [
          [-1, -1],
          [1, -1],
          [-1, 1],
          [1, 1],
        ]) {
          if (rnd() < 0.3) continue;
          const w = 9 + rnd() * 4;
          const d = 9 + rnd() * 4;
          const h = lo + rnd() * (hi - lo) * 0.7;
          boxes.push({ x: cx + qx * (block / 4 + 0.5), y: cy + qy * (block / 4 + 0.5), w, d, h, tone: rnd() });
        }
      }
    }
  }
  const mesh = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), facade(), boxes.length);
  const m = new THREE.Matrix4();
  const c = new THREE.Color();
  const palette = [0xb9b2a4, 0xa7a9ab, 0x8f8a80, 0xc8bda8, 0x7d8388, 0xd2cabb];
  boxes.forEach((b, k) => {
    m.makeScale(b.w, b.d, b.h).setPosition(b.x, b.y, b.h / 2);
    mesh.setMatrixAt(k, m);
    mesh.setColorAt(k, b.tone < 0 ? c.setHex(0x6f6c66) : c.setHex(palette[Math.floor(b.tone * palette.length)]));
  });
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  mesh.frustumCulled = false;
  return mesh;
}

/**
 * Facades from world position, two kinds by building: glass curtain walls (glass floor to
 * floor, thin spandrels and mullions, reflecting the sky) and concrete blocks with ribbon
 * windows; a few rooms lit, dimly (it's day); roofs plain.
 */
function facade(): THREE.MeshStandardNodeMaterial {
  const material = new THREE.MeshStandardNodeMaterial({ roughness: 0.85, metalness: 0 });
  const p = varying(positionWorld) as unknown as THREE.Node<'vec3'>;
  const n = varying(normalWorld) as unknown as THREE.Node<'vec3'>;
  const curtain = varying(hash(instanceIndex.add(7))).greaterThan(0.55);
  const roof = n.z.greaterThan(0.5);
  const across = select(abs(n.x).greaterThan(0.5), p.y, p.x);
  const up = p.z.div(STOREY);
  const floorId = floor(up);
  const bay = floor(across.div(1.25));
  const inFloor = fract(up);
  const inBay = fract(across.div(1.25));
  // Curtain wall: glass but for a thin band at each floor and slim mullions; ribbon windows:
  // a band of glass along each floor, mullions every other bay
  const curtainGlass = step(0.14, inFloor).mul(step(0.06, inBay));
  const ribbon = step(0.34, inFloor).mul(step(inFloor, 0.84)).mul(step(0.05, fract(across.div(2.5))));
  const lobby = step(p.z, 4.2).mul(step(0.5, p.z));
  const glass = select(curtain, curtainGlass, ribbon).max(lobby).mul(select(roof, float(0), float(1))).mul(step(0.4, p.z));
  const isGlass = glass.greaterThan(0.5);
  const room = hash(floorId.mul(131).add(bay.mul(17)).add(floor(p.x.add(p.y).mul(0.1))));
  const lit = step(0.93, room).mul(glass);
  const wall = vec3(1).mul(hash(floorId.mul(7).add(floor(across.div(6)))).mul(0.06).add(0.94)).mul(select(roof, float(0.62), float(1)));
  const tint = select(curtain, vec3(0.1, 0.14, 0.18), vec3(0.06, 0.07, 0.08));
  const glassColour = tint.mul(hash(bay.add(floorId.mul(3))).mul(0.4).add(0.8));
  material.colorNode = select(isGlass, glassColour, wall) as unknown as THREE.Node<'color'>;
  material.roughnessNode = select(isGlass, select(curtain, float(0.06), float(0.15)), float(0.88));
  material.metalnessNode = select(isGlass, select(curtain, float(0.85), float(0.5)), float(0));
  const far = smoothstep(float(60), float(400), p.sub(cameraPosition).length());
  material.emissiveNode = vec3(1.0, 0.74, 0.46).mul(lit).mul(far.mul(0.3).add(0.35)) as unknown as THREE.Node<'color'>;
  return material;
}
