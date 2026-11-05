import { TASK } from "./behavior-inference.js";
export const VERSION = "behavior-3";
const modelId = (v) => typeof v === "string" && /^[a-f0-9]{64}$/.test(v);
const finite = Number.isFinite;
const array = (a, fn) => Array.isArray(a) && a.length > 0 && a.every(fn);
const probability = (v) => finite(v) && v >= 0 && v <= 1;
const vector = (v, n) => Array.isArray(v) && v.length === n && v.every(finite);
const matrix = (v, rows, columns) => Array.isArray(v) && v.length === rows && v.every((r) => vector(r, columns));
function fitted(model, n) {
  if (!model || !vector(model.mean, n) || !vector(model.scale, n) || !model.scale.every((v) => v > 0)) return false;
  if (model.kind === "scaled-logit") return finite(model.intercept) && vector(model.coefficients, n);
  return model.kind === "scaled-qda" && matrix(model.means, 2, n) &&
    Array.isArray(model.rotations) && model.rotations.length === 2 && model.rotations.every((r) => matrix(r, n, n)) &&
    matrix(model.variances, 2, n) && model.variances.every((r) => r.every((v) => v > 0)) &&
    vector(model.priors, 2) && model.priors.every((v) => v > 0) && Math.abs(model.priors[0] + model.priors[1] - 1) < 1e-8;
}
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
  let valid = data?.task === TASK && data.version === VERSION && modelId(data.model_id);
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
          data.ranges[k].every(finite) && data.ranges[k][0] <= data.ranges[k][1],
      ) &&
      data.features.length === 19 && new Set(data.features).size === 19 &&
      fitted(data.browser_model, data.features.length);
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
          probability(r.threshold) && probability(r.cv_f1),
      ) &&
      data.comparison.length === 3 &&
      ["train", "validation", "test"].every((k) =>
        Number.isInteger(data.split_sizes?.[k]) && data.split_sizes[k] > 0 &&
        ["0", "1"].every((label) => Number.isInteger(data.class_counts?.[k]?.[label]) && data.class_counts[k][label] >= 0) &&
        data.class_counts[k]["0"] + data.class_counts[k]["1"] === data.split_sizes[k],
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
      data.statistics && Object.keys(data.statistics).length === 19 &&
      Object.values(data.statistics).every((row) => ["mean", "median", "var", "std", "min", "max"].every((k) => finite(row?.[k]))) &&
      data.correlation && Object.keys(data.correlation).length === 20 &&
      Object.values(data.correlation).every((row) => Object.keys(data.correlation).every((k) => finite(row?.[k])));
  else valid = false;
  if (!valid) throw new Error("Incompatible lane-change behavior data.");
  return data;
}
export function validatePredictions(data, info, ids) {
  if (
    data?.task !== TASK ||
    data.version !== VERSION ||
    !modelId(data.model_id) ||
    (info && data.model_id !== info.model_id) ||
    !Array.isArray(data.predictions) ||
    new Set(data.predictions.map((p) => p?.id)).size !== data.predictions.length ||
    (ids && (data.predictions.length !== ids.length || data.predictions.some((p) => !ids.includes(p?.id)))) ||
    !data.predictions.every(
      (p) =>
        p && typeof p.id === "string" &&
        typeof p.eligible === "boolean" &&
        (!p.eligible ||
          (p.task === TASK &&
            p.version === VERSION &&
            p.model_id === data.model_id &&
            probability(p.probability) &&
            [0, 1].includes(p.prediction))),
    )
  )
    throw new Error("Incompatible behavior predictions.");
  return data;
}
