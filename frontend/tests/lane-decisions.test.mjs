import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import {
  createWorld,
  vehicle,
  laneAssessment,
  loadScenario,
  tick,
  STEP,
  requestLaneChange,
  laneObservations,
} from "../src/simulation.js";
import { inferLaneModel } from "../src/services/lane-inference.js";
const advance = (w, s) => {
  for (let i = 0; i < s / STEP; i++) tick(w);
};
const info = JSON.parse(
  fs.readFileSync(new URL("../public/data/lane-info.json", import.meta.url)),
);
test("portable learned model matches actual sklearn held-out outputs", () => {
  for (const row of info.checks) {
    const p = inferLaneModel(info, row.inputs);
    assert.equal(p.eligible, true);
    assert.ok(Math.abs(p.probability - row.probability) < 1e-12);
  }
});
test("no fabricated learned score for unsupported speeds or gaps", () => {
  assert.equal(
    inferLaneModel(info, {
      ego_speed: 100,
      front_speed: 15,
      rear_speed: 15,
      front_gap: 30,
      rear_gap: 30,
    }).eligible,
    false,
  );
});
test('explicit acceleration temporarily overrides automatic cruising',()=>{
  const w=createWorld();w.mode='Automatic Traffic';w.cars=[vehicle(1,0,0,10,{desired:10,accel:2,manualAccelUntil:3})];
  advance(w,1);assert.ok(w.cars[0].speed>11.9);
  advance(w,3);assert.equal(w.cars[0].accel,0);assert.equal(w.cars[0].manualAccelUntil,undefined);assert.ok(w.cars[0].actualAccel<0);
});
test("six-lane blocked-road preset performs bypass and passes obstacle without contact", () => {
  const w = createWorld();
  w.laneModel = info;
  loadScenario(w, "Six-Lane Blockage - Safe Bypass");
  advance(w, 9);
  const c = w.cars.find((c) => c.id === 2),
    obstacle = w.cars.find((c) => c.id === 1);
  assert.equal(w.lanes, 6);
  assert.ok(c.lane !== 2);
  assert.ok(c.x > obstacle.x);
  assert.equal(w.collisions.length, 0);
  assert.ok(w.completedLaneChanges >= 1);
  assert.ok(
    w.events.some(
      (e) =>
        e.text.includes("completed change") &&
        e.text.includes("no simulated contact"),
    ),
  );
});
test("a stopped uncollided follower can change into an empty lane", () => {
  const w = createWorld(3, 2);
  w.mode = "Automatic Traffic";
  w.autoLane = true;
  w.cars = [
    vehicle(1, 1, 50, 0, { desired: 15 }),
    vehicle(2, 1, 60, 0, { desired: 0 }),
  ];
  advance(w, 3);
  assert.notEqual(w.cars[0].lane, 1);
  assert.equal(w.cars[0].crashed, false);
  assert.equal(w.collisions.length, 0);
});
test("all lanes blocked: wait and stop, without an unsafe pointless lane change", () => {
  const w = createWorld();
  loadScenario(w, "Blocked Lane - No Safe Gap");
  advance(w, 10);
  assert.equal(w.collisions.length, 0);
  assert.equal(w.cars.find((c) => c.id === 2).lane, 1);
  assert.equal(w.laneChanges, 0);
  assert.ok(w.cars.find((c) => c.id === 2).speed < 0.1);
});
test("fast target-lane rear vehicle blocks a merge", () => {
  const w = createWorld();
  w.cars = [vehicle(1, 0, 100, 10), vehicle(2, 1, 70, 30)];
  const a = laneAssessment(w, w.cars[0], 1);
  assert.equal(a.clear, false);
  assert.match(a.reason, /Rear vehicle/);
  assert.equal(requestLaneChange(w, w.cars[0], 1), false);
});
test("empty target lane does not hide imminent contact along the lateral path", () => {
  const w = createWorld();
  w.cars = [vehicle(1, 0, 20, 20), vehicle(2, 0, 30, 0)];
  const a = laneAssessment(w, w.cars[0], 1);
  assert.equal(a.empty, true);
  assert.equal(a.clear, false);
  assert.match(a.reason, /path/);
});
test("simultaneous merges reserve the target path", () => {
  const w = createWorld();
  w.cars = [vehicle(1, 0, 50, 12), vehicle(2, 2, 50, 12)];
  assert.equal(requestLaneChange(w, w.cars[0], 1), true);
  assert.equal(requestLaneChange(w, w.cars[1], 1), false);
});
test("already collided cars cannot evade; approaching cars may bypass them", () => {
  const w = createWorld();
  w.mode = "Automatic Traffic";
  w.autoLane = true;
  w.cars = [
    vehicle(1, 1, 70, 0, { crashed: true, desired: 0 }),
    vehicle(2, 1, 10, 12),
  ];
  assert.equal(laneAssessment(w, w.cars[0], 0).clear, false);
  advance(w, 7);
  assert.equal(w.cars[0].lane, 1);
  assert.notEqual(w.cars[1].lane, 1);
  assert.equal(w.collisions.length, 0);
});
test("elevated fitted response vetoes a geometrically clear occupied-lane candidate", () => {
  const w = createWorld();
  w.cars = [
    vehicle(1, 0, 100, 15),
    vehicle(2, 1, 165, 15),
    vehicle(3, 1, 40, 15),
  ];
  const before = laneAssessment(w, w.cars[0], 1);
  assert.equal(before.clear, true);
  w.laneScores = {
    "1:1": {
      eligible: true,
      probability: 0.9,
      model: "Logistic Regression",
      signature: before.signature,
      at: 0,
      wall: Date.now(),
    },
  };
  assert.equal(laneAssessment(w, w.cars[0], 1).clear, false);
  w.laneScores["1:1"].signature = "old-neighbors";
  assert.equal(laneAssessment(w, w.cars[0], 1).clear, true);
  assert.ok(Math.abs(laneObservations(w)[0].inputs.front_gap - 60.4) < 1e-9);
});
