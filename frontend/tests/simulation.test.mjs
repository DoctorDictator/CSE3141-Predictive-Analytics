import test from "node:test";
import assert from "node:assert/strict";
import {
  createWorld,
  generate,
  vehicle,
  overlap,
  tick,
  STEP,
  setLanes,
  changeLane,
  brake,
  reset,
  saveInitial,
  rewind,
  loadScenario,
  leader,
  observations,
  addCar,
  SCENARIOS,
} from "../src/simulation.js";
const advance = (w, seconds) => {
  for (let i = 0; i < seconds / STEP; i++) tick(w);
};
test("seed reproduces every starting property", () =>
  assert.deepEqual(generate(4, 12, 123), generate(4, 12, 123)));
test("1–8 lanes and 2–30 cars start without overlaps", () => {
  for (let lanes = 1; lanes <= 8; lanes++)
    for (let count = 2; count <= 30; count++) {
      const cars = generate(lanes, count, 44);
      assert.equal(cars.length, count);
      assert.equal(new Set(cars.map((c) => c.id)).size, count);
      for (const c of cars) assert.ok(c.lane >= 0 && c.lane < lanes);
      for (let i = 0; i < count; i++)
        for (let j = i + 1; j < count; j++)
          assert.ok(!overlap(cars[i], cars[j]));
    }
});
test("lane compression recalculates geometry and avoids contacts", () => {
  const w = createWorld(8, 30);
  advance(w, 0.2);
  assert.ok(w.history.length > 0);
  setLanes(w, 1);
  assert.equal(w.history.length, 0);
  assert.ok(w.cars.every((c) => c.lane === 0 && c.y === 1.85));
  for (let i = 0; i < 30; i++)
    for (let j = i + 1; j < 30; j++) assert.ok(!overlap(w.cars[i], w.cars[j]));
});
test("longitudinal movement integrates speed and acceleration", () => {
  const w = createWorld();
  w.cars = [vehicle(1, 0, 0, 10, { accel: 2 })];
  advance(w, 1);
  assert.ok(Math.abs(w.cars[0].x - 11) < 0.15);
  assert.ok(Math.abs(w.cars[0].speed - 12) < 0.1);
});
test("manual target speed changes accelerate progressively", () => {
  const w = createWorld();
  w.cars = [vehicle(1, 0, 0, 10, { desired: 20, targetActive: true })];
  tick(w);
  assert.ok(w.cars[0].speed > 10 && w.cars[0].speed < 11);
  advance(w, 2);
  assert.ok(w.cars[0].speed > 14 && w.cars[0].speed < 20);
});
test("braking reduces actual speed without becoming negative", () => {
  const w = createWorld();
  w.cars = [vehicle(1, 0, 0, 10)];
  brake(w, w.cars[0]);
  advance(w, 3);
  assert.equal(w.cars[0].speed, 0);
  assert.ok(w.cars[0].x < 20);
});
test("lane change is continuous, bounded, and finishes at the new lane", () => {
  const w = createWorld();
  w.cars = [vehicle(1, 0, 0, 10)];
  const c = w.cars[0];
  assert.equal(changeLane(w, c, -1), false);
  changeLane(w, c, 1);
  advance(w, 0.5);
  assert.ok(c.y > 1.85 && c.y < 5.55);
  assert.equal(c.changing, true);
  advance(w, 1.2);
  assert.equal(c.lane, 1);
  assert.ok(Math.abs(c.y - 5.55) < 0.001);
  assert.equal(c.changing, false);
});
test("rear-end demo makes geometric contact; no random crash flag", () => {
  const w = createWorld();
  loadScenario(w, "Rear-End Collision Demo");
  assert.equal(w.collisions.length, 0);
  advance(w, 3);
  assert.equal(w.collisions.length, 1);
  assert.ok(w.cars.every((c) => c.crashed && c.speed === 0));
  assert.ok(overlap(...w.cars));
});
test("unsafe lane change contacts during lateral movement", () => {
  const w = createWorld();
  loadScenario(w, "Unsafe Lane Change");
  advance(w, 2);
  assert.equal(w.collisions.length, 1);
  assert.ok(w.cars[0].y > 1.85 && w.cars[0].y < 5.55);
});
test("multiple incident produces sequential contacts with stopped obstacles", () => {
  const w = createWorld();
  loadScenario(w, "Multiple-Vehicle Incident");
  advance(w, 5);
  assert.ok(w.collisions.length >= 2);
  assert.ok(w.collisions[1].time > w.collisions[0].time);
});
test("normal highway preset has no immediate contact", () => {
  const w = createWorld();
  loadScenario(w, SCENARIOS[0]);
  advance(w, 5);
  assert.equal(w.collisions.length, 0);
});
test("reset restores exact initial configuration; repeated run is deterministic", () => {
  const w = createWorld();
  loadScenario(w, "Rear-End Collision Demo");
  const initial = structuredClone(w.cars);
  advance(w, 3);
  const impact = w.collisions[0].time;
  reset(w);
  assert.deepEqual(w.cars, initial);
  assert.equal(w.time, 0);
  advance(w, 3);
  assert.equal(w.collisions[0].time, impact);
});
test("rewind restores pre-impact state and replay reaches same incident", () => {
  const w = createWorld();
  loadScenario(w, "Rear-End Collision Demo");
  advance(w, 3);
  rewind(w);
  assert.equal(w.collisions.length, 0);
  assert.ok(w.cars.every((c) => !c.crashed));
  advance(w, 3);
  assert.equal(w.collisions.length, 1);
});
test("leader pairing uses nearest same-lane front; features use front spacing", () => {
  const w = createWorld();
  w.cars = [
    vehicle(1, 0, 20, 15),
    vehicle(2, 0, 45, 10),
    vehicle(3, 0, 80, 10),
    vehicle(4, 1, 30, 10),
  ];
  assert.equal(leader(w, w.cars[0]).id, 2);
  const p = observations(w).find((p) => p.id === "1");
  assert.equal(p.inputs.spacing_m, 25);
  assert.equal(p.inputs.leader_length, 4.6);
  assert.equal(p.inputs.follower_speed, 15);
  changeLane(w, w.cars[0], 1);
  assert.ok(!observations(w).some((p) => p.id === "1"));
});
test("stopped, collided, leaderless and changing vehicles are not eligible", () => {
  const w = createWorld();
  w.cars = [vehicle(1, 0, 0, 0), vehicle(2, 0, 30, 15)];
  assert.equal(observations(w).length, 0);
  w.cars[0].speed = 15;
  w.cars[1].crashed = true;
  assert.equal(observations(w).length, 0);
});
test("automatic avoidance brakes; disabling avoidance permits a contact", () => {
  const w = createWorld();
  w.mode = "Automatic Traffic";
  w.cars = [vehicle(1, 0, 10, 25), vehicle(2, 0, 40, 25, { desired: 25 })];
  tick(w);
  assert.ok(w.cars[0].actualAccel < 0);
  loadScenario(w, "Rear-End Collision Demo");
  w.mode = "Automatic Traffic";
  w.avoidance = false;
  advance(w, 4);
  assert.ok(w.collisions.length > 0);
});
test("adding vehicles keeps IDs unique and enforces maximum", () => {
  const w = createWorld(3, 2);
  for (let i = 0; i < 10; i++) addCar(w);
  assert.equal(new Set(w.cars.map((c) => c.id)).size, w.cars.length);
  w.cars = generate(3, 30);
  assert.equal(addCar(w), null);
});
