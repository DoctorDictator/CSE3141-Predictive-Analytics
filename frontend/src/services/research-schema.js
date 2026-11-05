import { TASK } from "./behavior-inference.js";
const finite = Number.isFinite;
const array = (a, fn) => Array.isArray(a) && a.length > 0 && a.every(fn);
const probability = (v) => finite(v) && v >= 0 && v <= 1;
const algorithms = ["Logistic Regression", "LDA", "QDA"];
const metrics = (m) =>
  m &&
  [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "average_precision",
    "balanced_accuracy",
  ].every((k) => probability(m[k])) &&
  Array.isArray(m.confusion_matrix) &&
  m.confusion_matrix.length === 2 &&
  m.confusion_matrix.every(
    (r) =>
      Array.isArray(r) &&
      r.length === 2 &&
      r.every((v) => Number.isInteger(v) && v >= 0),
  );
export function validateResearch(name, data) {
  let valid = data?.task === TASK && data.version === "behavior-3";
  if (name === "info")
    valid =
      valid &&
      algorithms.includes(data.name) &&
      data.horizon_seconds === 2 &&
      probability(data.decision_threshold) &&
      array(
        data.features,
        (k) =>
          typeof k === "string" &&
          typeof data.units?.[k] === "string" &&
          Array.isArray(data.ranges?.[k]) &&
          data.ranges[k].length === 2 &&
          data.ranges[k].every(finite),
      ) &&
      ["scaled-logit", "scaled-qda"].includes(data.browser_model?.kind);
  else if (name === "metrics")
    valid =
      valid &&
      metrics(data.test) &&
      data.baseline &&
      probability(data.baseline.accuracy) &&
      array(
        data.comparison,
        (r) =>
          algorithms.includes(r.name) &&
          metrics(r.training) &&
          metrics(r.validation) &&
          probability(r.threshold),
      ) &&
      data.comparison.length === 3 &&
      ["train", "validation", "test"].every((k) =>
        finite(data.split_sizes?.[k]),
      );
  else if (name === "analytics")
    valid =
      valid &&
      [
        "rows",
        "vehicles",
        "recording_count",
        "eligible_rows",
        "recorded_crossings",
        "duration_seconds",
      ].every((k) => finite(data[k])) &&
      typeof data.sha256 === "string" &&
      data.class_counts &&
      ["v_vel", "v_acc", "front_gap", "headway", "lane_id", "v_class"].every(
        (k) =>
          array(
            data.histograms?.[k],
            (v) => finite(v.value) && finite(v.count),
          ),
      ) &&
      data.simulator_profile &&
      Array.isArray(data.simulator_profile.speed_bins) &&
      data.simulator_profile.speed_bins.length > 0 &&
      data.simulator_profile.speed_bins.every((b) => finite(b.value) && finite(b.count) && b.count >= 0) &&
      Array.isArray(data.simulator_profile.acceleration_bins) &&
      data.simulator_profile.acceleration_bins.length > 0 &&
      data.simulator_profile.acceleration_bins.every((b) => finite(b.value) && finite(b.count) && b.count >= 0) &&
      probability(data.simulator_profile.truck_probability) &&
      Number.isInteger(data.simulator_profile.lane_count) &&
      data.simulator_profile.lane_count >= 1 &&
      ["car_length", "truck_length", "car_width", "truck_width"].every((k) => finite(data.simulator_profile[k]) && data.simulator_profile[k] > 0) &&
      data.correlation;
  else valid = false;
  if (!valid) throw new Error("Incompatible lane-change behavior data.");
  return data;
}
export function validatePredictions(data) {
  if (
    data?.task !== TASK ||
    !Array.isArray(data.predictions) ||
    !data.predictions.every(
      (p) =>
        typeof p.id === "string" &&
        typeof p.eligible === "boolean" &&
        (!p.eligible ||
          (p.task === TASK &&
            probability(p.probability) &&
            [0, 1].includes(p.prediction))),
    )
  )
    throw new Error("Incompatible behavior predictions.");
  return data;
}
