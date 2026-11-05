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
  let valid = data?.task === TASK && data.version === "behavior-1";
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
      Array.isArray(data.traffic_seeds) &&
      data.correlation;
  else if (name === "replay")
    valid =
      valid &&
      array(
        data.examples,
        (e) =>
          Number.isInteger(e.actual) &&
          [0, 1].includes(e.actual) &&
          [0, 1].includes(e.predicted) &&
          probability(e.probability) &&
          e.inputs &&
          Array.isArray(e.snapshots) &&
          e.snapshots.length === 21 &&
          e.snapshots.every(
            (s) =>
              finite(s.seconds) &&
              array(s.cars, (c) =>
                ["id", "x", "y", "lane", "speed", "length", "width"].every(
                  (k) => finite(c[k]),
                ),
              ),
          ),
      );
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
