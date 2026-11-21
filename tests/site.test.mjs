import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { parseCsv, average, summarize } from '../assets/metrics.mjs';

const rows = parseCsv(readFileSync(new URL('../results/metrics.csv', import.meta.url), 'utf8'));

test('saved data keeps all 1029 combinations and missing metrics', () => {
  assert.equal(rows.length, 1029);
  assert.ok(rows.some(row => row.mcc === null));
  assert.equal(average([null, 0, 0.2]), 0.1);
  assert.equal(average([null]), null);
});

test('default explorer reproduces the published findings', () => {
  const results = summarize(rows);
  assert.equal(results.length, 7);
  assert.equal(results[0].model, 'logistic');
  assert.equal(results[0].mcc.toFixed(4), '0.1191');
  assert.ok(results.every(result => result.count === 21));
  const overlap = summarize(rows, { metric: 'p_t' });
  assert.equal(overlap[0].model, 'persistence');
  assert.equal(overlap[0].p_t.toFixed(4), '0.0675');
});

test('combined filters select only the exact saved experiments', () => {
  for (const asset of new Set(rows.map(row => row.asset))) {
    for (const horizon of [10, 50, 100]) {
      for (const threshold of [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]) {
        const filtered = summarize(rows, { asset, horizon: String(horizon), threshold: String(threshold) });
        assert.equal(filtered.length, 7);
        for (const model of filtered) {
          const source = rows.find(row => row.asset === asset && row.horizon === horizon && row.threshold === threshold && row.model === model.model);
          assert.equal(model.count, 1);
          assert.equal(model.mcc, source.mcc);
          assert.equal(model.coverage, source.coverage);
          assert.equal(model.p_t, source.p_t);
        }
      }
    }
  }
});

test('negative and undefined skill retain their scientific meaning', () => {
  const example = [
    { asset: 'aapl', horizon: 10, threshold: 0.9, model: 'abstain', mcc: null, coverage: 0 },
    { asset: 'aapl', horizon: 10, threshold: 0.9, model: 'negative', mcc: -0.2, coverage: 1 }
  ];
  const result = summarize(example, { threshold: 0.9 });
  assert.equal(result[0].model, 'negative');
  assert.equal(result[0].mcc, -0.2);
  assert.equal(result[1].mcc, null);
  assert.equal(result[1].definedMcc, 0);
  assert.equal(result[1].coverage, 0);
});
