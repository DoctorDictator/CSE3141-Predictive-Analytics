export const TASK = "lane-change-behavior";
export function inferBehavior(info, inputs, { support = true } = {}) {
  if (info?.task !== TASK || !info.browser_model)
    return { eligible: false, reason: "Behavior model unavailable" };
  const model = info.browser_model,
    n = info.features.length;
  if (
    ![model.mean, model.scale].every(
      (a) => Array.isArray(a) && a.length === n && a.every(Number.isFinite),
    )
  )
    return { eligible: false, reason: "Invalid fitted scaler" };
  const x = [];
  for (let i = 0; i < n; i++) {
    const key = info.features[i],
      v = inputs[key],
      range = info.ranges[key];
    if (!Number.isFinite(v) || model.scale[i] <= 0)
      return { eligible: false, reason: `Invalid feature: ${key}` };
    if (support && (v < range[0] || v > range[1]))
      return { eligible: false, reason: `Outside training support: ${key}` };
    x.push((v - model.mean[i]) / model.scale[i]);
  }
  let z;
  if (model.kind === "scaled-logit")
    z =
      model.intercept + x.reduce((s, v, i) => s + v * model.coefficients[i], 0);
  else if (model.kind === "scaled-qda") {
    const logp = model.means.map((mean, c) => {
      const difference = x.map((v, i) => v - mean[i]);
      let norm = 0,
        det = 0;
      for (let j = 0; j < model.variances[c].length; j++) {
        const variance = model.variances[c][j];
        if (!(variance > 0)) return NaN;
        const projected = difference.reduce(
          (s, v, i) => s + v * model.rotations[c][i][j],
          0,
        );
        norm += (projected * projected) / variance;
        det += Math.log(variance);
      }
      return -0.5 * (norm + det) + Math.log(model.priors[c]);
    });
    z = logp[1] - logp[0];
  } else return { eligible: false, reason: "Unsupported fitted model format" };
  if (!Number.isFinite(z))
    return { eligible: false, reason: "Invalid fitted classifier" };
  const probability =
    z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));
  const prediction = Number(probability >= info.decision_threshold);
  return {
    task: TASK,
    eligible: true,
    probability,
    prediction,
    label: prediction ? "Lane change" : "Stay in lane",
    model: info.name,
    version: info.version,
    model_id: info.model_id,
    source: "Saved fitted model",
    horizon_seconds: 2,
  };
}
