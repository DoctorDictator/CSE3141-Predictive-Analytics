// Apply exactly the exported fitted StandardScaler + binary linear classifier.
// Parameters originate in the executed notebook; no distances-to-risk formula is used.
export function inferLaneModel(info, inputs) {
  const model = info?.browser_model;
  if (model?.kind !== "scaled-logit")
    return { eligible: false, reason: "Portable fitted model unavailable" };
  const n = info.features.length;
  if (
    ![model.mean, model.scale, model.coefficients].every(
      (a) => Array.isArray(a) && a.length === n && a.every(Number.isFinite),
    ) ||
    !Number.isFinite(model.intercept)
  )
    return { eligible: false, reason: "Invalid fitted parameters" };
  let z = model.intercept;
  for (let i = 0; i < n; i++) {
    const key = info.features[i],
      value = inputs[key],
      range = info.ranges[key];
    if (
      !Number.isFinite(value) ||
      !range ||
      value < range[0] ||
      value > range[1] ||
      model.scale[i] <= 0
    )
      return { eligible: false, reason: `Outside training support: ${key}` };
    z += ((value - model.mean[i]) / model.scale[i]) * model.coefficients[i];
  }
  const probability =
    z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));
  return {
    eligible: true,
    probability,
    model: info.name,
    version: info.version,
    source: "Saved fitted model",
  };
}
