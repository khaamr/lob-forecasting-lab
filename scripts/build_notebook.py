#build the reproducible notebook with visible implementations
import ast
import json
from pathlib import Path as path
import nbformat as nbf
root=path(__file__).resolve().parents[1]
cells=[]

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))

def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))

def source_group(filename,names):
    source=(root/'scripts'/filename).read_text()
    tree=ast.parse(source)
    lines=source.splitlines()
    snippets=[]
    for node in tree.body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names:
            snippets.append('\n'.join(lines[node.lineno-1:node.end_lineno]))
    code('\n\n'.join(snippets))

md('''# deep limit order book forecasting: reproduction and implemented future directions

based on **briola, bartolucci and aste (2025), deep limit order book forecasting: a microstructural guide**, quantitative finance, 25(7), 1101–1131. [article](https://doi.org/10.1080/14697688.2025.2522911) · [authors’ code](https://github.com/FinancialComputingUCL/LOBFrame).

this notebook delivers an executable public-data study: data acquisition, cleaning, microstructure, exact tick labels, purged windows, four neural architectures, baselines, forecasting metrics, transaction overlap, and implemented extensions with interactive plotly graphs. definitions and model implementations are visible below; supporting scripts allow command-line reruns.

**scope:** the licensed 15-stock, 2017–2019 dataset is not supplied. the public stock experiments use the full available sample day, 2012-06-21, with bounded training and contiguous evaluation subsets. the crypto extension uses two real snapshot prefixes from 2024-01-01. newer stock sectors use daily bars from 2022–2025 in a separate pipeline. none of these are represented as the paper’s numerical reproduction.

all authored prose, headings, variables and comments use lowercase. python keywords and case-sensitive third-party api names retain their required spelling. comments follow the requested `#standard scaler` style.''')
md('''## 1. environment and reproducibility

run this notebook from the project folder with the `lob research` kernel. the supplied environment and downloaded data are already prepared. on another computer, create an environment with `requirements.txt` first. restart the kernel and run all cells in order. cached trained forecasts make review faster; set `retrain = True` below to repeat training.

raw file hashes, the experiment configuration, model checkpoints, forecast arrays, training histories and package versions are saved. all neural computation uses the cpu and fixed seeds. the smaller verification experiment trains twice independently to check determinism.''')
code('''#project paths
import os
import sys
import warnings
import platform
import json
import hashlib
import subprocess
import importlib.metadata as metadata
from pathlib import Path as path
root = path.cwd().resolve()
if not (root / 'scripts' / 'pipeline.py').exists() and (root.parent / 'scripts' / 'pipeline.py').exists():
    root = root.parent
assert (root / 'scripts' / 'pipeline.py').exists(), 'run from the supplied project folder'
sys.path.insert(0, str(root))
retrain = False

#installed versions
import numpy as np
import pandas as pd
import torch
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from IPython.display import display, Markdown as markdown
pio.renderers.default = 'plotly_mimetype+notebook'
#verify matrix arithmetic before filtering a documented apple accelerate warning
if platform.system() == 'Darwin' and platform.machine() == 'arm64' and np.__version__ == '2.2.6':
    rng = np.random.default_rng(17)
    matrix = rng.normal(size=(128,40))
    with warnings.catch_warnings(record=True):
        product = matrix.T @ matrix
    assert np.isfinite(product).all() and np.allclose(product,np.einsum('ij,ik->jk',matrix,matrix,optimize=False),atol=1e-10)
    warnings.filterwarnings('ignore', message='(divide by zero|overflow|invalid value) encountered in matmul', category=RuntimeWarning)
versions = pd.DataFrame([{'package': p, 'version': metadata.version(p)} for p in ['numpy','pandas','torch','scikit-learn','plotly','nbformat','nbclient','yfinance']])
display(versions)
''')
md('''## 2. what is reproduced, adapted, and extended

| component | article | implemented experiment | interpretation |
|---|---|---|---|
| stocks | 15 stocks, 2017–2019 | five stocks, 2012-06-21 | public-data method reproduction |
| cleaning | crossed quotes, last timestamp state, 09:40–15:50 | implemented plus depth validity checks | authentic ten-level books |
| labels | raw mid-price difference, inclusive one tick | integer half-tick arithmetic, h10/h50/h100 | no smoothed future labels |
| input | 100 × 40 | 100 × 40 | ten price/size levels per side |
| normalization | previous five complete days | training-only scaler for the single-day sample | rolling daily implementation supplied separately |
| split | 45 train, 5 validation, 10 test days per year | 60/20/20 chronology, purged boundaries | input and future labels stay inside their splits |
| sampling | at most 5000 per class per day | smaller configured balanced sample per epoch | public sample budget, not the original compute |
| deeplob | convolutions, inception, lstm; adamw | matching 143907-parameter architecture and optimizer settings | smaller epoch budget |
| evaluation | full sequential validation and test days | first configured contiguous validation/test windows | no random thinning before transaction scoring |
| confidence | 0.3–0.9 | same thresholds, explicit coverage | no-forecast metrics are undefined |
| transaction metric | correct transaction intersection over union | direction-regime matching checked against figure 12 | not a profitability estimate |
| future work | transformers, diffusion, graphs, other exchanges | all implemented below | research prototypes, not claims of superiority |

**paper ambiguity:** table 2 prints 2019-10-19 as the test start, a saturday. the licensed-data runner requires an explicit calendar manifest and rejects weekends instead of silently inventing a replacement. the current authors’ repository also contains implementation choices that differ from the paper’s prose; this study follows the paper’s stated 09:40–15:50 session filter.''')
md('''## 3. download and inspect real data

[official public stock samples](https://php.lobsterdata.com/info/DataSamples.php) supply `aapl`, `amzn`, `goog`, `intc`, and `msft`. `aapl` and `goog` overlap the paper; the other three extend its ticker universe. the legacy host now serves a webpage at old archive paths, so the downloader uses the working official `php` host and validates zip contents.

[tardis sample documentation](https://docs.tardis.dev/downloadable-csv-files/overview) permits unauthenticated downloads for the first day of a month. the crypto downloader preserves the first 60000 snapshots from each venue and keeps the first ten of the 25 levels. these are captured exchange snapshots, not nasdaq event messages. crypto labels use a one-cent price grid, an explicit assumption verified against all downloaded quotes rather than historical venue metadata. `btc/usdt` and `btc/usd` also differ in quote currency.

[yahoo finance historical data](https://finance.yahoo.com/quote/QQQ/history/) supplies adjusted daily prices. `enph`, `crsp`, `rklb`, and `ionq` are selected as solar energy, gene editing, space systems, and quantum computing examples. these labels are research groupings, not a claim that the stocks isolate one economic exposure. no daily bar is substituted for a limit order book.''')
code('''#download only if the archived inputs are absent
if not (root / 'data/raw/manifest.json').exists():
    subprocess.run([sys.executable, str(root / 'scripts/download_data.py')], check=True)
if not (root / 'data/raw/daily_manifest.json').exists():
    subprocess.run([sys.executable, str(root / 'scripts/download_prices.py')], check=True)
manifest = json.loads((root / 'data/raw/manifest.json').read_text())
daily_manifest = json.loads((root / 'data/raw/daily_manifest.json').read_text())
for record in manifest + daily_manifest:
    assert hashlib.sha256((root / record['path']).read_bytes()).hexdigest() == record['sha256']
display(pd.DataFrame(manifest)[['asset','kind','date','selection']])
''')
md('''## 4. preprocessing, labels and leakage controls

timestamps from stock messages are converted to integer nanoseconds before collapsing duplicate states. crossed or invalid books are removed; equity events are restricted to 09:40–15:50 eastern time. a target at event `t` uses `mid[t+h] - mid[t]`: down at or below minus one tick, up at or above one tick, stable otherwise. comparisons use integer half ticks to avoid floating-point errors at the boundary.

the public experiment fits its scaler only on the first 60 percent of rows. each validation and test window starts 99 events inside its own segment, and the last `h` labels are purged. the test cap selects a contiguous interval, preserving the chronology needed by the transaction metric. the full downloaded day is used for descriptive microstructure, not for tuning against test labels.''')
source=(root/'scripts/pipeline.py').read_text()
imports=source[:source.index('def seed_all')].replace("root = path(__file__).resolve().parents[1]", "#root is defined in the setup cell")
code(imports)
source_group('pipeline.py',['seed_all','clean_book','load_books','labels','stationary_features','split_indices','prepare','balanced_indices','windows'])
code('''#validated public books
seed_all(17)
books, data_audit = load_books()
display(data_audit)
assert len(books) == 7
''')
md('''## 5. model implementations

**deeplob:** three spatial/temporal convolution blocks, three inception branches and a 64-unit lstm. the input layout follows the paper’s alternating ask-price, ask-size, bid-price, bid-size order. adamw uses a learning rate of 0.00006, betas 0.9 and 0.95, and weight decay 0.01. the reference code informed an independent implementation; it is not imported with its external logging services.

**transformer:** two historical self-attention layers, four heads and learned temporal positions. attention may use any point inside the already observed 100-event window. no token comes from the prediction horizon. this tests whether flexible temporal relationships help with sparse books.

**graph:** twenty nodes represent ask and bid levels. edges connect adjacent levels on each side and corresponding levels across sides. normalized message passing precedes a temporal gru. an identity-adjacency ablation tests whether the graph’s connections add value.

**diffusion:** a conditional denoising diffusion model predicts three-dimensional class embeddings, conditioned on a historical gru. it trains a noise predictor under a cosine noise schedule and performs 16 reverse steps from 24 fixed monte carlo noise draws. smoothed class votes form forecast probabilities. this is a small diffusion classifier, not a full order-book generator.

new architectures use a learning rate of 0.001 with the same optimizer family. their capacities differ, so this is a structured prototype benchmark, not a parameter-matched architecture ranking. validation loss selects checkpoints. model selection and confidence calibration never use test labels.''')
source_group('pipeline.py',['deeplob','transformer','graph_lob','diffusion_lob'])
md('''## 6. training and prediction pipeline

balanced samples are drawn without replacement within each epoch and resampled deterministically between epochs. three baselines are included: the training class prior, trailing-horizon persistence, and a scaled snapshot logistic classifier. early stopping restores the best validation checkpoint. all forecasts and their original event indices are stored.''')
source_group('pipeline.py',['predict','train_model','baseline_probabilities'])
md('''## 7. transaction metric and evaluation

`p_t = correct / (potential + executed - correct)` is the overlap of complete target and predicted transactions. a stable signal holds the current direction. a directional reversal closes one position and opens the opposite one; an unclosed final position is not a completed transaction.

correctness is measured inside the corresponding directional target regimes, matching figure 12. it does **not** require identical event indices at entry and exit. confidence rejection becomes a hold signal on the original timeline; it does not remove timestamps. empty transaction unions and empty selected forecast sets produce undefined values. `p_t` is a signal-timing overlap score, not an empirical order fill probability or a guarantee of profit.''')
source_group('pipeline.py',['transaction_set','transaction_metrics','score','fit_temperature','microstructure','run_benchmark','block_bootstrap','execution_stress','rolling_daily_scale'])
code('''#the two examples from figure twelve
true = np.array([-1,-1,1,1,1,1,-1,-1,0,-1,1,1,1]) + 1
example_a = np.array([-1,-1,-1,-1,0,0,-1,-1,0,-1,1,1,1]) + 1
example_b = np.array([-1,0,0,1,0,0,0,-1,-1,0,0,-1,1]) + 1
examples = pd.DataFrame([dict(example='a', **transaction_metrics(true, example_a)), dict(example='b', **transaction_metrics(true, example_b))])
assert examples.p_t.tolist() == [0., 1.]
display(examples)
''')
code('''#experiment configuration and trained result cache
from scripts.run_experiments import config
signature = hashlib.sha256((json.dumps(config, sort_keys=True)+(root/'scripts/pipeline.py').read_text()+(root/'data/raw/manifest.json').read_text()).encode()).hexdigest()[:16]
directory = root / 'results' / signature
display(pd.DataFrame([config]).drop(columns='models'))
execution_config = dict(config, retrain=retrain)
metrics, training_history, runtime = run_benchmark(books, execution_config, directory)
metrics.to_csv(root/'results/metrics.csv', index=False)
training_history.to_csv(root/'results/training_history.csv', index=False)
runtime.to_csv(root/'results/runtime.csv', index=False)
data_audit.to_csv(root/'results/data_audit.csv', index=False)
#document the exact evaluation intervals rather than treating them as full days
window_rows = []
for asset, book in books.items():
    prepared = prepare(book, 50, validation_cap=config['validation_cap'], test_cap=config['test_cap'])
    indices = prepared['indices'][2]
    target = prepared['y'][indices]
    window_rows.append(dict(asset=asset, first_event=int(indices[0]), last_event=int(indices[-1]), test_windows=len(indices), test_span_seconds=(book['frame'].timestamp.iloc[indices[-1]]-book['frame'].timestamp.iloc[indices[0]])/1e9, down=int((target==0).sum()), stable=int((target==1).sum()), up=int((target==2).sum())))
evaluation_windows = pd.DataFrame(window_rows)
evaluation_windows.to_csv(root/'results/evaluation_windows.csv', index=False)
display(evaluation_windows)
microstructure(books).to_csv(root/'results/microstructure.csv', index=False)
(root/'results/active_run.json').write_text(json.dumps(dict(signature=signature,config=config),indent=2))
''')
md('''## 8. implemented future directions

| direction | concrete implementation | why it matters | evaluation |
|---|---|---|---|
| alternative architectures | transformer, graph and diffusion alongside deeplob | one architecture may not fit every book structure | all tickers and all three horizons |
| graph structure | connected graph versus identity adjacency | separates useful relationships from network capacity | fixed aapl/intc h50 ablation |
| stationary representation | prices relative to current mid, sizes transformed with log1p | absolute price shifts can impair transfer | graph raw versus stationary features |
| cross-exchange validation | train on one venue, test on the other with a shared physical cutoff | checks whether a signal survives a venue change | both transfer directions, same-venue controls |
| calibration | validation-only temperature scaling | a confidence score is useful only if its reliability is checked | test log loss, brier score, calibration error |
| timing and friction | delayed top-quote execution and per-side fees | accurate labels may disappear before execution or fail to cover the spread | delay and fee sensitivity curves |
| relaxed transaction timing | one-to-one matching within a stated event tolerance | tests sensitivity to a strict timing convention | tolerance 0, 1, 5 and 10 |
| training budget | expanded deeplob and graph fits on aapl/intc | short runs may undertrain an architecture | 30 epochs and 1024 samples per class versus the main budget |
| uncertainty | moving-block resampling and three training seeds | adjacent windows are dependent and training is stochastic | block sizes 128/512 and seeds 17/29/43 |
| new sectors | daily models for solar, gene editing, space and quantum stocks | tests a broader ticker universe where public daily data exists | separate walk-forward tests in 2024 and 2025 |

cross-exchange transfer deliberately uses a logistic model and stationary features, with the same source-fitted scaler applied to each target. the neural model families are also benchmarked within each exchange. the transfer test is a proof of pipeline, limited by a short overlapping sample and the usd/usdt denomination difference.''')
code('''#extension imports
from scipy.special import softmax
from sklearn.preprocessing import StandardScaler as standard_scaler
from sklearn.linear_model import LogisticRegression as logistic_regression
from sklearn.ensemble import HistGradientBoostingClassifier as gradient_boosting
from sklearn.metrics import log_loss, matthews_corrcoef, f1_score, accuracy_score
''')
source_group('extensions.py',['run_daily','load_prediction','run_sensitivities','run_ablations','run_cross_exchange','run_seed_robustness','run_budget_sensitivity'])
code('''#execute extensions or verify their current artifact manifest
extension_key = hashlib.sha256((signature+(root/'scripts/extensions.py').read_text()+(root/'data/raw/daily_manifest.json').read_text()).encode()).hexdigest()
extension_manifest_path = root/'results/extension_manifest.json'
extension_valid = False
if extension_manifest_path.exists():
    extension_manifest = json.loads(extension_manifest_path.read_text())
    extension_valid = extension_manifest['signature'] == extension_key and all((root/p).exists() and hashlib.sha256((root/p).read_bytes()).hexdigest() == digest for p,digest in extension_manifest['files'].items())
if retrain or not extension_valid:
    run_daily()
    run_cross_exchange(books)
    run_sensitivities(books, directory)
    run_ablations(books, config, directory)
    run_seed_robustness(books, config, directory)
    run_budget_sensitivity(books, directory)
    names = ['daily_summary','daily_metrics','daily_predictions','cross_exchange','calibration','execution_stress','tolerance','bootstrap','regime_scores','ablations','seed_robustness','budget_sensitivity']
    files = {f'results/{n}.csv':hashlib.sha256((root/f'results/{n}.csv').read_bytes()).hexdigest() for n in names}
    extension_manifest_path.write_text(json.dumps(dict(signature=extension_key, files=files),indent=2))
print('all future direction experiments are available and verified')
''')
md('''## 9. interactive results

the following figures use plotly. hover for exact values, click legends to isolate series, and use the toolbar to zoom or export an image. the offline dashboard embeds the complete plotly library and chart data. no connection is needed to view the supplied results.''')
code('''#build the chart collection and offline dashboard
from scripts.visuals import create_figures, export_gallery
figures = create_figures(books, directory)
dashboard_path = export_gallery(figures)
print(f'{len(figures)} interactive figures created')
''')
sections=[
('microstructure and class balance', '''the sample separates liquid narrow-spread stocks from wider and sparser books. the classification is measured from these sample data, not inherited from 2017–2019. in particular, aapl’s sample tick group can differ from its group in the article. crypto quantities use native units, so their volume levels are not directly comparable with share counts.''', ['01_microstructure_table','02_spread_regimes','03_spread_distribution','04_depth_liquidity','05_book_sparsity','06_event_clock','07_class_balance']),
('forecasting and transaction quality', '''compare every neural model with the prior, persistence and logistic baselines. a model that predicts mostly stable can obtain high accuracy while producing little useful transaction overlap. mcc and macro f1 expose different aspects of this behavior; `p_t` adds chronology. these charts include the actual observed results, including weak models.''',['08_architecture_comparison','09_transaction_comparison','10_confidence_coverage','11_confidence_transactions','12_learning_curves','13_confusion_matrices','26_metric_table']),
('architecture, calibration and exchange extensions', '''temperature scaling minimizes validation log loss, but its test effect may be positive or negative. graph ablations use the same predefined tickers and training budget. transfer values are from a common 270-second-scale interval, with target observations strictly later than training labels. diagonal and off-diagonal cells have different source training sets; the very short binance test has few possible transactions.''',['14_calibration','15_graph_ablation','16_cross_exchange','27_training_budget']),
('execution sensitivity and uncertainty', '''the execution calculation assumes one-unit marketable orders, fills at observed top quotes, no market impact and no queue or network model. fees are charged on both sides. delays are in filtered updates, not milliseconds. results are illustrative net ticks per completed trade and must not be interpreted as an executable backtest.

positive timing tolerance is an extension to the article’s metric and may use nearby labels retrospectively for scoring; it never changes the forecasts. moving-block intervals describe sampling variation inside this captured interval, not uncertainty across market days. the intervals for transaction overlap reset state at block boundaries, so they target block-local overlap rather than the full-sequence statistic. the seed study retrains the graph model at h50 on the two predefined stock examples.''',['17_latency_costs','18_timing_tolerance','19_seed_robustness','20_block_uncertainty']),
('new sectors: daily prices and separate forecasting', '''daily models use lagged returns, trailing volatility, high-low range, relative volume and the same-day qqq return, all known at the forecasting close. targets describe the adjusted close change after 1, 5 or 10 trading days, with a separate ±0.5 percent threshold. training target tails are purged before each test year. 2024 uses earlier years; 2025 expands training through 2024. no model is selected using 2025 outcomes.

the four newer fields broaden the stock comparison, but do not validate order-book forecasting in those sectors. selected survivors and a fixed small universe introduce selection bias. qqq is an etf tracking the nasdaq-100, not the nasdaq composite or an order-book dataset.''',['21_new_sector_prices','22_new_sector_risk','23_daily_forecasting','24_daily_correlation']),
('published references and interpretation', '''the published mcc values at threshold 0.3 are transcribed from appendix tables a2–a4, covering 135 ticker/year/horizon entries. the side-by-side chart compares aapl and goog with the reduced sample experiment. different dates, normalization and compute mean that a discrepancy does not establish a failed replication or refute the paper. the reference values are never used to tune a model.''',['25_paper_reference'])]
for title,text,names in sections:
    md('### '+title+'\n\n'+text)
    for i in range(0,len(names),2):
        code('\n'.join(f"figures['{n}'].show()" for n in names[i:i+2]))
md('''## 10. results-driven conclusions

at h50, the intc test window contains 50 down, 4046 stable and no up labels; the msft window contains 4096 stable labels. neither contains a complete potential transaction. these short intervals cannot establish the article’s large-tick transaction result. an expanded training budget does not lengthen the evaluation window. single-class mcc values and undefined transaction overlap must be read with these counts.

the following summary is calculated from the saved experiment tables. averages give each asset one vote and are descriptive; they are not significance tests. individual ticker/horizon tables remain the appropriate place to inspect weak or unstable results.''')
code('''#summarize observed results without presuming a winning architecture
base = metrics.query('threshold == .3')
rank = base.groupby('model')[['mcc','macro_f1','p_t']].mean().sort_values('mcc',ascending=False)
display(rank)
best_mcc = rank.index[0]
best_pt = rank.p_t.idxmax()
calibration = pd.read_csv(root/'results/calibration.csv').groupby('variant').log_loss.mean()
ablation = pd.read_csv(root/'results/ablations.csv')
daily_scores = pd.read_csv(root/'results/daily_metrics.csv')
beats_majority = (daily_scores.accuracy > daily_scores.majority_accuracy).mean()
display(markdown(f"the highest mean mcc in this public run belongs to **{best_mcc}** ({rank.loc[best_mcc,'mcc']:.3f}); the highest mean transaction overlap belongs to **{best_pt}** ({rank.loc[best_pt,'p_t']:.3f}). this distinction matters because label accuracy and complete signal sequences measure different things.\\n\\nmean held-out log loss is {calibration['raw']:.3f} before calibration and {calibration['temperature']:.3f} after it. calibration must be assessed on held-out observations instead of assumed beneficial.\\n\\n{beats_majority:.1%} of the separate daily ticker/model/horizon/year experiments exceed their training-majority accuracy baseline. this fraction is descriptive and includes dependent comparisons; it does not demonstrate a deployable strategy."))
display(ablation[['asset','representation','model','mcc','macro_f1','p_t']])
''')
md('''## 11. licensed-data pipeline

the supplied `scripts/paper_archive.py` implements the full day-based adapter: 5 warmup days, 45 training days, 5 validation days, 10 test days, previous-five-day scaling, daily balanced sampling up to 5000 examples per class, 100 maximum epochs and patience 15. it evaluates each test day independently. repeat its runner for each original ticker and year to obtain the 135 original design experiments.

this runner is implemented but **not empirically validated on the unavailable licensed archive**. it requires an explicit manifest of actual message/orderbook pairs; the notebook does not pretend to reconstruct that archive from daily data. large runs require substantial memory and compute. the article reports roughly 959 cumulative gpu hours for its study.''')
code('''#full archive entry point; the public run does not require a license or credentials
from scripts.paper_archive import prepare_archive, run_archive
licensed_manifest = None
if licensed_manifest is not None:
    licensed_metrics = run_archive(licensed_manifest, root/'results/licensed')
    display(licensed_metrics)
else:
    print('licensed experiment not run: the original archive is not present')
''')
md('''## 12. verification

checks cover exact tick boundaries, purged windows, balanced sampling, both paper transaction examples, no duplicate transaction credit, stable-only sequences, network shapes and gradients, deterministic inference, past-only rolling normalization, and fee monotonicity. the result audit then recomputes metrics from every saved forecast array and checks data shapes, finite probabilities, original sequence indices, confidence coverage and physical-time separation.

repeated notebook execution is recorded separately from independent model retraining. on this apple silicon runtime, a bounded matrix calculation is compared with a non-blas calculation before filtering only the [documented numpy 2.2.6 matrix warning](https://github.com/numpy/numpy/issues/28687). forecast arrays are separately checked for finiteness. other warnings remain visible.

cache reuse verifies replay; it is not evidence that a second full training run was performed.''')
code('''#run the scientific checks
checked = subprocess.run([sys.executable,'-m','pytest','-q','tests'],cwd=root,capture_output=True,text=True)
print(checked.stdout)
assert checked.returncode == 0, checked.stderr
from scripts.verify_results import verify
verification = verify(retrain=True)
display(pd.DataFrame([verification]))
''')
md('''## 13. saved outputs and sources

- `results/metrics.csv`: complete ticker/model/horizon/confidence table.
- `results/daily_metrics.csv`: the separate daily walk-forward comparisons.
- `results/cross_exchange.csv`: common-clock transfer and same-venue controls.
- `results/ablations.csv`, `results/calibration.csv`, `results/execution_stress.csv`, `results/tolerance.csv`, `results/bootstrap.csv`, and `results/seed_robustness.csv`: implemented future-work experiments.
- `results/data_audit.csv`, `data/raw/manifest.json`, and `data/raw/daily_manifest.json`: data cleaning and source lineage.
- `results/verification.json`: checks of computed outputs and independent retraining.
- `research_dashboard.html`: offline interactive plotly gallery.

sources: [published article](https://doi.org/10.1080/14697688.2025.2522911), [lobframe](https://github.com/FinancialComputingUCL/LOBFrame), [official public stock samples](https://php.lobsterdata.com/info/DataSamples.php), [tardis sample availability](https://docs.tardis.dev/downloadable-csv-files/overview), [tardis snapshot schema](https://docs.tardis.dev/downloadable-csv-files/data-types), and [yahoo finance historical data](https://finance.yahoo.com/quote/QQQ/history/). raw sample files are retained locally for replay; no license to redistribute vendor data is asserted.

**what remains unavailable:** exact 2017–2019 numerical replication, recent stock-sector order books, and evidence across many days/exchanges. those require additional data. all reported public experiments and graphs above are computed, not placeholders.''')
notebook=nbf.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(display_name='lob research',language='python',name='lob-research'),language_info=dict(name='python',version='3.10')))
nbf.write(notebook,root/'lob_reproduction_future_work.ipynb')
print(f'created {len(cells)} cells')
