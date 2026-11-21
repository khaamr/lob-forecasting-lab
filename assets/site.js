import { parseCsv, summarize } from './metrics.mjs';

const names = { logistic: 'Logistic regression', persistence: 'Persistence', transformer: 'Transformer', deeplob: 'DeepLOB', graph: 'Graph + GRU', diffusion: 'Cond. diffusion', prior: 'Class prior' };
const assetNames = { aapl: 'AAPL', amzn: 'AMZN', goog: 'GOOG', intc: 'INTC', msft: 'MSFT', binance_btcusdt: 'Binance · BTC/USDT', 'coinbase_btc-usd': 'Coinbase · BTC/USD' };
const metricNames = { mcc: 'Matthews correlation', macro_f1: 'Macro F1', p_t: 'Transaction overlap', coverage: 'Forecast coverage' };
const state = { asset: 'all', horizon: 'all', threshold: 0.3, metric: 'mcc' };
const element = id => document.getElementById(id);
const format = (value, percent = false) => value === null ? 'undefined' : percent ? `${(value * 100).toFixed(1)}%` : value.toFixed(4);

async function loadCsv(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`Could not load ${path} (${response.status})`);
  return parseCsv(await response.text());
}

function renderResults(rows) {
  const models = summarize(rows, state);
  const values = models.map(model => model[state.metric]).filter(Number.isFinite);
  const minimum = Math.min(0, ...values);
  const maximum = Math.max(0.01, ...values);
  const span = maximum - minimum;
  const zero = -minimum / span * 100;
  element('chart-title').textContent = metricNames[state.metric].toUpperCase();
  element('chart').replaceChildren(...models.map(model => {
    const row = document.createElement('div');
    row.className = 'bar-row';
    const label = document.createElement('span');
    label.className = 'bar-name';
    label.textContent = names[model.model] || model.model;
    const track = document.createElement('div');
    track.className = 'bar-track';
    const value = model[state.metric];
    if (value !== null) {
      const fill = document.createElement('span');
      fill.className = `bar-fill${value < 0 ? ' negative' : ''}`;
      fill.style.left = `${Math.min(zero, (value - minimum) / span * 100)}%`;
      fill.style.width = `${Math.abs(value) / span * 100}%`;
      fill.title = `${names[model.model]}: ${format(value)} (${model.definedMetric}/${model.count} defined runs)`;
      track.append(fill);
    }
    const line = document.createElement('span');
    line.className = 'bar-zero';
    line.style.left = `${zero}%`;
    track.append(line);
    const score = document.createElement('span');
    score.className = 'bar-value';
    score.textContent = value === null ? 'n/a' : format(value, state.metric === 'coverage');
    row.append(label, track, score);
    return row;
  }));
  element('scale-min').textContent = minimum.toFixed(3);
  element('scale-max').textContent = maximum.toFixed(3);
  const leader = models.find(model => model[state.metric] !== null);
  element('leader-name').textContent = leader ? names[leader.model] : 'No defined results';
  element('leader-score').textContent = leader ? format(leader[state.metric], state.metric === 'coverage') : 'n/a';
  element('leader-description').textContent = leader
    ? `Mean ${metricNames[state.metric].toLowerCase()} across ${leader.definedMetric} of ${leader.count} selected combinations.${state.metric === 'p_t' ? ' Overlap is not profitability.' : ''}`
    : 'No saved run has a defined value for this selection. Try a lower confidence threshold.';
  element('coverage').textContent = leader ? format(leader.coverage, true) : 'n/a';
  element('combinations').textContent = models[0]?.count ?? 0;
  element('filter-summary').textContent = `${state.asset === 'all' ? 'All 7 books' : assetNames[state.asset]} · ${state.horizon === 'all' ? 'all horizons' : `${state.horizon} updates`} · threshold ${Number(state.threshold).toFixed(2)}`;
  element('metric-table').replaceChildren(...models.map(model => {
    const row = document.createElement('tr');
    for (const value of [names[model.model], format(model.mcc), format(model.macro_f1), format(model.p_t), format(model.coverage, true), `${model.definedMcc}/${model.count}`]) {
      const cell = document.createElement('td');
      cell.textContent = value;
      row.append(cell);
    }
    return row;
  }));
}

function renderMarket(book, interval) {
  const crypto = book.asset.includes('btc');
  const venue = crypto ? (book.asset.startsWith('binance') ? 'BINANCE SNAPSHOTS' : 'COINBASE SNAPSHOTS') : 'LOBSTER PUBLIC SAMPLE';
  const date = crypto ? '2024-01-01' : '2012-06-21';
  const fractions = ['down', 'stable', 'up'].map(key => ({ key, count: interval[key], percent: interval[key] / interval.test_windows * 100 }));
  element('market-detail').innerHTML = `
    <div class="market-title"><div><p class="eyebrow">${venue}</p><h3>${assetNames[book.asset]}</h3><p>10 levels per side · ${Number(book.updates).toLocaleString()} updates</p></div><span class="market-date">${date}</span></div>
    <div class="market-metrics"><div><strong>${book.mean_spread_ticks.toFixed(2)}</strong><span>Mean spread · ticks</span></div><div><strong>${(book.one_tick_fraction * 100).toFixed(1)}%</strong><span>One-tick spread</span></div><div><strong>${book.information_richness.toFixed(2)}</strong><span>Information richness</span></div></div>
    <h4>EVALUATION LABEL DISTRIBUTION</h4><div class="class-bar" role="img" aria-label="${fractions.map(part => `${part.key} ${part.percent.toFixed(1)}%`).join(', ')}">${fractions.map(part => `<span class="class-${part.key}" style="width:${part.percent}%"></span>`).join('')}</div>
    <div class="class-legend">${fractions.map(part => `<span><i class="class-${part.key}"></i>${part.key.toUpperCase()} ${part.percent.toFixed(1)}%</span>`).join('')}</div>
    <p class="market-context">${Number(interval.test_windows).toLocaleString()} test windows over <strong>${interval.test_span_seconds.toFixed(1)} seconds</strong>. ${book.asset === 'msft' ? 'This interval contains only stable labels; apparent accuracy needs particular care.' : book.asset === 'intc' ? 'Stable labels dominate this interval. MCC and macro F1 help contextualize accuracy.' : 'The same event horizon can cover a different physical duration in another market.'}</p>`;
}

async function initializeMarkets() {
  try {
    const [books, intervals] = await Promise.all([loadCsv('results/microstructure.csv'), loadCsv('results/evaluation_windows.csv')]);
    element('market-list').replaceChildren(...books.map((book, index) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'market-button';
      button.setAttribute('aria-pressed', String(index === 0));
      button.innerHTML = `<span class="market-symbol">${book.asset.includes('btc') ? '₿' : book.asset.slice(0, 1).toUpperCase()}</span><span><strong>${assetNames[book.asset]}</strong><small>${book.asset.includes('btc') ? 'Crypto snapshot sample' : 'Equity order book'}</small></span><span class="market-regime">${book.tick_group.toUpperCase()} TICK</span>`;
      button.addEventListener('click', () => {
        document.querySelectorAll('.market-button').forEach(item => item.setAttribute('aria-pressed', String(item === button)));
        renderMarket(book, intervals.find(interval => interval.asset === book.asset));
      });
      return button;
    }));
    renderMarket(books[0], intervals.find(interval => interval.asset === books[0].asset));
  } catch (error) {
    element('market-detail').textContent = `${error.message}. Inspect the source CSVs using the links below.`;
  }
}

const architectures = {
  deeplob: { title: 'Learn the local structure.', flow: ['Convolution blocks', 'Inception branches', 'LSTM', '3 probabilities'], description: 'Three convolution blocks extract spatial and temporal patterns. Inception branches combine scales, and an LSTM maps the sequence to down, stable and up probabilities.', question: 'Does the convolutional structure beat simple baselines under this public-data training budget?' },
  transformer: { title: 'Attend across the sequence.', flow: ['Feature projection', 'Positional encoding', 'Attention encoder', '3 probabilities'], description: 'An independently implemented transformer models relationships across the historical order book sequence. Compare its held-out forecasts with the same baselines and event horizons.', question: 'Does attention improve classification without sacrificing the timing of complete transactions?' },
  graph: { title: 'Let book structure speak.', flow: ['Price-level nodes', 'Graph aggregation', 'GRU', '3 probabilities'], description: 'A graph model encodes relationships across book levels, followed by a GRU for temporal structure. Extensions test relative prices, log sizes and the contribution of graph edges.', question: 'Do graph connections add information beyond the same input features?' },
  diffusion: { title: 'Condition on the book.', flow: ['Book context', 'Conditional denoising', 'Class decoding', '3 probabilities'], description: 'A conditional diffusion alternative is implemented alongside the other neural models. Its saved forecasts use the same classification and transaction-overlap evaluation.', question: 'Does a generative alternative earn its added complexity in a limited-budget reproduction?' }
};

function selectArchitecture(name) {
  const selected = architectures[name];
  document.querySelectorAll('[role="tab"]').forEach(tab => {
    const active = tab.dataset.model === name;
    tab.setAttribute('aria-selected', String(active));
    tab.tabIndex = active ? 0 : -1;
  });
  element('model-panel').setAttribute('aria-labelledby', `tab-${name}`);
  element('model-panel').innerHTML = `<div class="model-flow">${selected.flow.map(step => `<span>${step}</span>`).join('')}</div><h3>${selected.title}</h3><p>${selected.description}</p><p class="model-baseline">RESEARCH QUESTION / ${selected.question}</p>`;
}

function initializeArchitectures() {
  const tabs = [...document.querySelectorAll('[role="tab"]')];
  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => selectArchitecture(tab.dataset.model));
    tab.addEventListener('keydown', event => {
      let target;
      if (event.key === 'ArrowRight') target = tabs[(index + 1) % tabs.length];
      if (event.key === 'ArrowLeft') target = tabs[(index - 1 + tabs.length) % tabs.length];
      if (event.key === 'Home') target = tabs[0];
      if (event.key === 'End') target = tabs.at(-1);
      if (target) {
        event.preventDefault();
        target.focus();
        selectArchitecture(target.dataset.model);
      }
    });
  });
  selectArchitecture('deeplob');
}

function initializeFilm() {
  const film = element('research-film');
  const toggle = element('film-toggle');
  function update() {
    toggle.textContent = film.paused ? '▷' : 'Ⅱ';
    toggle.setAttribute('aria-label', film.paused ? 'Play research film' : 'Pause research film');
  }
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    film.autoplay = false;
    film.pause();
  }
  film.addEventListener('play', update);
  film.addEventListener('pause', update);
  toggle.addEventListener('click', async () => {
    if (film.paused) {
      try { await film.play(); } catch { toggle.setAttribute('aria-label', 'Video playback unavailable'); }
    } else film.pause();
    update();
  });
  update();
}

async function initializeResults() {
  try {
    const rows = await loadCsv('results/metrics.csv');
    for (const name of ['asset', 'threshold', 'metric']) {
      element(name).addEventListener('change', event => {
        state[name] = event.target.value;
        renderResults(rows);
      });
    }
    element('filters').addEventListener('submit', event => event.preventDefault());
    element('horizons').addEventListener('click', event => {
      const button = event.target.closest('button');
      if (!button) return;
      state.horizon = button.dataset.value;
      element('horizons').querySelectorAll('button').forEach(item => item.setAttribute('aria-pressed', String(item === button)));
      renderResults(rows);
    });
    renderResults(rows);
  } catch (error) {
    element('chart').textContent = `${error.message}. Serve this folder over HTTP, or download results/metrics.csv directly.`;
    element('chart').classList.add('error');
    element('filter-summary').textContent = 'Results could not be loaded. Source CSV remains available.';
    element('leader-name').textContent = 'Results unavailable';
    element('leader-score').textContent = 'n/a';
    element('leader-description').textContent = 'Use the source CSV link to inspect the saved evidence.';
    for (const name of ['coverage', 'combinations']) element(name).textContent = 'n/a';
  }
}

initializeFilm();
initializeArchitectures();
initializeMarkets();
initializeResults();
