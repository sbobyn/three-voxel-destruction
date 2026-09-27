// The race track (src/track.ts) and its car (src/car.ts): the circuit closes on itself and
// starts on the main straight, only the obstacles put there on purpose stand on the racing
// line, the ground's picture of it measures what it should, and the car drives, steers, slides
// and runs into what stands.

import assert from 'node:assert/strict';
import { describe, test } from 'node:test';
import { CAR_SIZE, Car, sweep } from '../src/car.ts';
import { buildTrack, gridSlot, trackField, trackLine } from '../src/track.ts';
import { State } from '../src/world.ts';

describe('the track', () => {
  const line = trackLine();
  const { city } = buildTrack();

  test('the line closes on itself, a metre between samples, the start on the main straight', () => {
    const n = line.length;
    assert.ok(n > 800 && n < 1200, `lap ${n} m`);
    for (let k = 0; k < n; k++) {
      const j = (k + 1) % n;
      const step = Math.hypot(line.points[2 * j] - line.points[2 * k], line.points[2 * j + 1] - line.points[2 * k + 1]);
      assert.ok(Math.abs(step - 1) < 0.05, `sample ${k} to ${j}: ${step} m`);
    }
    const [sx, sy] = [line.points[2 * line.start], line.points[2 * line.start + 1]];
    assert.ok(Math.abs(sx + 60) < 1 && Math.abs(sy + 90) < 1, `start at ${sx}, ${sy}`);
    assert.ok(Math.abs(line.tangents[2 * line.start] - 1) < 0.01, 'the start faces along +x');
  });

  test('driving the line, the car meets only the obstacles put on it (and meets the wall and the truck)', () => {
    const hit = new Set<number>();
    for (let k = 0; k < line.length; k++) {
      const heading = Math.atan2(line.tangents[2 * k + 1], line.tangents[2 * k]);
      for (const v of sweep(city, line.points[2 * k], line.points[2 * k + 1], heading)) hit.add(city.building[v]);
    }
    const stray = [...hit].filter((i) => city.buildings[i].kind !== 'obstacle');
    assert.deepStrictEqual(stray, [], `in the way on the line: ${stray.map((i) => JSON.stringify({ x: city.buildings[i].x0 / 4, y: city.buildings[i].y0 / 4 }))}`);
    const obstacles = city.buildings.map((b, i) => (b.kind === 'obstacle' ? i : -1)).filter((i) => i >= 0);
    // The wall is placed first; the truck is the widest obstacle after it
    assert.ok(hit.has(obstacles[0]), 'the brick wall is across the line');
    const truck = obstacles.reduce((a, b) => (city.buildings[b].w * city.buildings[b].d > city.buildings[a].w * city.buildings[a].d && b !== obstacles[0] ? b : a), obstacles[1]);
    assert.ok(hit.has(truck), 'the truck is across the line');
  });

  test('the grid slot is on the asphalt, clear of everything', () => {
    const { position, heading } = gridSlot(line);
    assert.deepStrictEqual(sweep(city, position[0], position[1], heading), []);
  });

  test("the ground's picture: distance to the line, + to its left, and the kerbs' corners", () => {
    const extent = 240;
    const size = 512;
    const field = trackField(line, size, extent);
    const texel = (x: number, y: number) => 4 * (Math.floor(((x + extent) / (2 * extent)) * size) + size * Math.floor(((y + extent) / (2 * extent)) * size));
    // On the main straight (along +x at y = -90): 5 m to the left is y = -85
    const k = line.start + 30;
    const [x, y] = [line.points[2 * k], line.points[2 * k + 1]];
    assert.ok(Math.abs(field[texel(x, y)]) < 0.6, `on the line: ${field[texel(x, y)]}`);
    assert.ok(Math.abs(field[texel(x, y + 5)] - 5) < 0.6, `5 m left: ${field[texel(x, y + 5)]}`);
    assert.ok(Math.abs(field[texel(x, y - 5)] + 5) < 0.6, `5 m right: ${field[texel(x, y - 5)]}`);
    assert.ok(field[texel(x, y + 60)] >= 17, 'far off: "far"');
    // Somewhere round the lap is a corner, and the straight isn't one
    assert.strictEqual(field[texel(x, y) + 2], 0);
    assert.ok(Array.from(line.corner).some((c) => c > 0.9));
  });
});

describe('the car', () => {
  const go = { throttle: 1, steer: 0, handbrake: false, boost: false };

  test('full throttle from rest: 100 km/h in about 2 to 3 s, straight ahead', () => {
    const car = new Car();
    car.place(0, 0, 0);
    let t = 0;
    while (car.speed < 100 / 3.6 && t < 10) {
      car.begin();
      car.step(1 / 60, go);
      t += 1 / 60;
    }
    assert.ok(t > 1.5 && t < 3.5, `0-100 km/h in ${t.toFixed(2)} s`);
    assert.ok(Math.abs(car.position.y) < 1e-6 && Math.abs(car.heading) < 1e-6, 'straight on');
  });

  test('steering left turns it left; the handbrake lets it slide', () => {
    const run = (handbrake: boolean) => {
      const car = new Car();
      car.place(0, 0, 0);
      car.velocity.set(25, 0);
      let most = 0;
      for (let k = 0; k < 60; k++) {
        car.begin();
        car.step(1 / 60, { throttle: 0, steer: 1, handbrake, boost: false });
        most = Math.max(most, car.slip);
      }
      return { heading: car.heading, most };
    };
    const grip = run(false);
    const drift = run(true);
    assert.ok(grip.heading > 0.3, `turned ${grip.heading.toFixed(2)} rad left`);
    assert.ok(drift.most > grip.most * 3, `slides ${drift.most.toFixed(2)} m/s on the handbrake, ${grip.most.toFixed(2)} without`);
  });

  test('its box finds the voxels it overlaps, and not ones already knocked loose', () => {
    const { city } = buildTrack();
    const wall = city.buildings.find((b) => b.kind === 'obstacle')!;
    // Centred on the wall, square to it
    const [x, y] = [(wall.x0 + wall.w / 2) * 0.25, (wall.y0 + wall.d / 2) * 0.25];
    const hits = sweep(city, x, y, 0);
    // The car is 2 m wide: 8 voxels across the wall's 2 thickness, from just off the ground to its roof
    assert.ok(hits.length >= 2 * 8 * 4, `${hits.length} voxels under the car`);
    assert.ok(hits.every((v) => Math.abs(city.position[3 * v] - x) <= CAR_SIZE[0] / 2 + 0.13));
    for (const v of hits) city.state[v] = State.Loose;
    assert.deepStrictEqual(sweep(city, x, y, 0), []);
  });
});
