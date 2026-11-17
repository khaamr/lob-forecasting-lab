# reproduction guide

## inspect the saved study

open [the executed notebook](../lob_reproduction_future_work.ipynb), [the presentation](presentation.pdf), or download [the standalone dashboard](../research_dashboard.html). review does not require raw data or training. github's notebook viewer is static; use a local notebook or dashboard for plotly interactivity.

## environment

use python 3.10. requirements.txt pins the direct dependencies of the publication validation runtime; requirements.lock records its resolved dependencies. environment/archived-requirements.txt preserves the supplied run environment separately. a different pytorch, numpy or platform build can produce different floating-point results. cached forecast metrics can be checked separately from checkpoint replay and independent retraining.

```sh
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name lob-research --display-name 'lob research'
python -m pytest -q tests
```

## acquire inputs

```sh
python scripts/download_data.py
python scripts/download_prices.py
```

the first command acquires five official lobster public stock samples and real binance/coinbase snapshot prefixes. the second downloads adjusted daily prices and creates the processed daily table. source urls, selection rules and file checksums are recorded in data/raw/manifest.json and data/raw/daily_manifest.json. the archived manifests under data/manifests preserve the lineage of the saved results.

raw downloads and processed vendor price tables are ignored by git. the original paper's licensed archive is not included. public source availability and adjusted historical prices can change; if hashes or code/configuration signatures differ, regenerate the affected experiment results rather than asserting identical replay.

## run the notebook

start jupyter or open the notebook in an editor, choose the lob research kernel, and run all cells from the repository root. the notebook downloads missing input manifests and validates hashes. it includes all major model and metric implementations directly in visible cells.

retrain is false by default. cached predictions and checkpoints are reused when the code/configuration/data signature matches. set retrain = True for the main experiment's independent training. extension artifacts have a separate signature and file hashes. the final verification cell also replays selected checkpoints and performs a small independent deterministic training comparison; exact equality is environment-sensitive.

```sh
python scripts/execute_notebook.py 1
```

this command uses a local jupyter kernel and writes the execution report under results. it executes the whole notebook and checks for errors and at least 27 plotly outputs. scripts/build_notebook.py rebuilds notebook source and clears its outputs; execute the notebook after rebuilding.

## command-line experiments

```sh
python scripts/run_experiments.py
python scripts/run_extensions.py
python scripts/run_remaining.py
python scripts/build_notebook.py
python scripts/execute_notebook.py 1
python scripts/verify_results.py
python -m pytest -q tests
```

run_remaining.py expects the main benchmark metrics to exist and waits for them when they are missing. run the main experiment first. model checkpoints and predictions are included under the active signature in results/active_run.json. regenerating figures and the dashboard is also performed by the notebook.

## scientific scope

| component | original study | public-data reproduction |
| --- | --- | --- |
| stock sample | 15 stocks, 2017–2019 | five stocks, 2012-06-21 |
| representation | ten levels per side | 100 historical updates × 40 features |
| prediction horizons | 10, 50, 100 updates | same event horizons |
| normalization | previous five complete days | training-only scaler for the one-day sample |
| split | multiple train, validation and test days | chronological 60/20/20 with purged boundaries |
| base training | larger balanced daily samples and longer training | 128 examples/class/epoch, eight epochs, patience three |
| evaluation | full sequential test days | first 1,024 validation and 4,096 test windows |

expanded-budget experiments use 1,024 examples per class, 30 epochs and patience eight for deeplob and graph on aapl/intc at h50. seed robustness, moving-block intervals and graph-edge/feature ablations are reported separately. architectures differ in capacity and are not parameter matched.

the seven-book benchmark combines five stock event streams and two crypto snapshot streams. those measurement processes differ, and crypto samples are prefixes rather than full days. the one-cent crypto label grid is an explicit modeling assumption. averaged metrics are descriptive and reflect very short evaluation spans.

transaction overlap is correct complete directional transactions divided by the union of target and predicted transactions. the implementation is checked against the paper's figure 12 examples. it is not an observed fill probability. the execution sensitivity calculation assumes top-quote fills, no impact and one-unit orders; it is not a deployable backtest. daily forecasting uses a separate daily-bar pipeline.

## licensed archive adapter

scripts/paper_archive.py supplies prepare_archive and run_archive for a locally licensed original ticker/year archive. its manifest requires 65 sessions: five warmup, 45 train, five validation and ten test days, with ten-level order books and matching messages.

```json
{
  "asset": "aapl",
  "sessions": [
    {
      "date": "2017-03-06",
      "split": "warmup",
      "orderbook": "aapl_2017-03-06_orderbook_10.csv",
      "message": "aapl_2017-03-06_message_10.csv"
    }
  ]
}
```

the snippet illustrates the schema and intentionally fails the required session count. supply a complete licensed calendar. the adapter uses previous-five-day normalization, daily balanced sampling, 100 maximum epochs, patience 15 and separate daily transaction evaluation. it has not been run against the unavailable licensed archive and assembles each ticker/year in memory.

the paper's table 2 lists a saturday test-start date in 2019. the adapter rejects weekends and requires an explicit valid calendar rather than inventing a replacement.

## provenance

[article](https://doi.org/10.1080/14697688.2025.2522911) · [authors' lobframe repository](https://github.com/FinancialComputingUCL/LOBFrame) · [upstream revision](../results/reference_revision.json) · [archived manifests](../data/manifests) · [archived verification record](../results/verification.json) · [publication validation](../results/publication-validation.json)
