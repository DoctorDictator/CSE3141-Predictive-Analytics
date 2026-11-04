const finite = Number.isFinite;
const arrayOf = (value, check) => Array.isArray(value) && value.length > 0 && value.every(check);
const score = value => finite(value) && value >= 0 && value <= 1;
const algorithms = ['Logistic Regression', 'LDA', 'QDA'];
const metrics = value => value && ['accuracy','precision','recall','f1','roc_auc'].every(key => score(value[key])) && Array.isArray(value.confusion_matrix) && value.confusion_matrix.length === 2 && value.confusion_matrix.every(row => Array.isArray(row) && row.length === 2 && row.every(n => Number.isInteger(n) && n >= 0));

// A reachable API may still be serving the previous project's response schema.
// Reject it before it reaches components that render NearMiss-specific fields.
export function validateResearch(name, data) {
  let valid = false;
  if (name === 'analytics') {
    const histograms = ['v_vel','v_acc','gap','headway','lane_id','v_class'];
    const correlations = ['v_vel','v_acc','leader_speed','leader_accel','gap','headway'];
    valid = data && ['rows','vehicles','recording_count','duration_seconds','eligible_rows'].every(key => finite(data[key]) && data[key] >= 0)
      && typeof data.sha256 === 'string' && score(data.positive_rate)
      && histograms.every(key => arrayOf(data.histograms?.[key], item => finite(item.value) && finite(item.count)))
      && correlations.every(key => correlations.every(other => finite(data.correlation?.[key]?.[other])))
      && arrayOf(data.speed_gap, item => finite(item.speed) && finite(item.gap))
      && data.threshold_sensitivity && Object.keys(data.threshold_sensitivity).length > 0 && Object.values(data.threshold_sensitivity).every(score);
  } else if (name === 'metrics') {
    valid = data && algorithms.includes(data.selected_model) && metrics(data.test) && score(data.persistence_baseline_f1)
      && typeof data.protocol === 'string' && ['train','validation','test'].every(key => finite(data.split_sizes?.[key]))
      && Array.isArray(data.comparison) && data.comparison.length === 3
      && algorithms.every(name => data.comparison.some(row => row.name === name && metrics(row.training) && metrics(row.validation) && score(row.cv_f1)));
  } else if (name === 'info') {
    valid = data && typeof data.version === 'string' && data.version.startsWith('nearmiss-') && algorithms.includes(data.name)
      && arrayOf(data.features, key => typeof key === 'string' && typeof data.units?.[key] === 'string' && Array.isArray(data.ranges?.[key]) && data.ranges[key].length === 2 && data.ranges[key].every(finite));
  }
  if (!valid) throw new Error('Incompatible NearMiss research response.');
  return data;
}

export function validatePredictions(data) {
  if (!Array.isArray(data?.predictions) || !data.predictions.every(item => typeof item.id === 'string' && typeof item.eligible === 'boolean' && (!item.eligible || score(item.probability)))) {
    throw new Error('Incompatible NearMiss prediction response.');
  }
  return data;
}
