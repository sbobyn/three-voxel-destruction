// Shading helpers (from three-avbd's demo app, ../app3d/look.ts there).

import { abs, float, max, min, smoothstep } from 'three/tsl';
import type * as THREE from 'three/webgpu';

type Vec3 = THREE.Node<'vec3'>;
type Float = THREE.Node<'float'>;

/**
 * Darkens a box's faces towards its edges (`local`: the vertex in the unit box, `size`: the
 * box's size), so stacked blocks read as blocks.
 */
export function edgeShade(local: Vec3, size: Vec3, width = 0.035): Float {
  const inset = size.mul(0.5).sub(abs(local.mul(size)));
  // On a face one component is ~0; the distance to the nearest edge is the middle one
  const sum = inset.x.add(inset.y).add(inset.z);
  const mid = sum.sub(max(inset.x, max(inset.y, inset.z))).sub(min(inset.x, min(inset.y, inset.z)));
  return smoothstep(float(0), float(width), mid).mul(0.45).add(0.55);
}
