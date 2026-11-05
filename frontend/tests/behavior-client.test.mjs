import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import { createBehaviorClient } from "../src/services/behavior-client.js";
import { inferBehavior } from "../src/services/behavior-inference.js";

const info = JSON.parse(fs.readFileSync(new URL("../public/data/info.json", import.meta.url)));
const inputs = Object.fromEntries(info.features.map((key) => {
  if (key.endsWith("_gap")) return [key, 150];
  if (key === "speed") return [key, 25];
  return [key, 0];
}));
const rows = [{ id: "car-1", inputs }];
const prediction = { ...inferBehavior(info, inputs), id: "car-1" };
const envelope = { task: info.task, version: info.version, model_id: info.model_id };

test("an old or waking API uses the saved model and retries after a delay", async () => {
  let time = 0, available = false, calls = [];
  const client = createBehaviorClient(info, {
    now: () => time,
    request: async (path) => {
      calls.push(path);
      if (!available) throw new Error("API 404");
      return path.endsWith("/info") ? info : { ...envelope, predictions: [prediction] };
    },
  });
  const first = await client.predict(rows);
  assert.match(first.source, /local inference/);
  assert.equal(first.predictions[0].probability, prediction.probability);
  await client.predict(rows);
  assert.equal(calls.length, 1, "No repeated requests during the retry delay");
  available = true; time = 31000;
  const recovered = await client.predict(rows);
  assert.match(recovered.source, /Model connected/);
  assert.equal(recovered.predictions[0].probability, prediction.probability);
  assert.deepEqual(calls, ["/api/model/info", "/api/model/info", "/api/predict-batch"]);
});

test("a different fitted API model cannot replace this website model", async () => {
  const calls = [];
  const client = createBehaviorClient(info, { request: async (path) => {
    calls.push(path);return { ...info, model_id: "0".repeat(64) };
  }});
  const result = await client.predict(rows);
  assert.match(result.source, /local inference/);
  assert.equal(calls.length, 1, "A mismatched model never receives observations");
});

test("a model swap or missing vehicle response during deployment falls back safely", async () => {
  for (const response of [
    { ...envelope, model_id: "0".repeat(64), predictions: [prediction] },
    { ...envelope, predictions: [] },
  ]) {
    const client = createBehaviorClient(info, { request: async (path) => path.endsWith("/info") ? info : response });
    const result = await client.predict(rows);
    assert.match(result.source, /local inference/);
    assert.equal(result.predictions[0].probability, prediction.probability);
  }
});
