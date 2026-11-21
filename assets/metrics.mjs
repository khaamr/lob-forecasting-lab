export function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  const columns = lines.shift().split(',');
  return lines.map(line => Object.fromEntries(line.split(',').map((value, index) => [
    columns[index], value === '' ? null : Number.isFinite(Number(value)) ? Number(value) : value
  ])));
}

export function average(values) {
  const valid = values.filter(value => typeof value === 'number' && Number.isFinite(value));
  return valid.length ? valid.reduce((total, value) => total + value, 0) / valid.length : null;
}

export function summarize(rows, { asset = 'all', horizon = 'all', threshold = 0.3, metric = 'mcc' } = {}) {
  const selected = rows.filter(row =>
    (asset === 'all' || row.asset === asset) &&
    (horizon === 'all' || row.horizon === Number(horizon)) &&
    row.threshold === Number(threshold)
  );
  const models = [...new Set(selected.map(row => row.model))];
  return models.map(model => {
    const runs = selected.filter(row => row.model === model);
    return {
      model,
      count: runs.length,
      definedMcc: runs.filter(row => Number.isFinite(row.mcc)).length,
      definedMetric: runs.filter(row => Number.isFinite(row[metric])).length,
      ...Object.fromEntries(['mcc', 'macro_f1', 'p_t', 'coverage'].map(key => [key, average(runs.map(row => row[key]))]))
    };
  }).sort((a, b) => (b[metric] ?? -Infinity) - (a[metric] ?? -Infinity));
}
