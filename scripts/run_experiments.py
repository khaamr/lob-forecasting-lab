#run reproducible public sample experiments
import hashlib
import json
from pathlib import Path as path
import sys
sys.path.insert(0, str(path(__file__).resolve().parents[1]))
from scripts.pipeline import load_books, microstructure, run_benchmark

root = path(__file__).resolve().parents[1]
config = dict(epochs=8, patience=3, per_class=128, validation_cap=1024, test_cap=4096,
              models=['deeplob', 'transformer', 'graph', 'diffusion'], seed=17)

def main():
    signature = hashlib.sha256((json.dumps(config, sort_keys=True)+(root/'scripts/pipeline.py').read_text()+(root/'data/raw/manifest.json').read_text()).encode()).hexdigest()[:16]
    directory = root / 'results' / signature
    directory.mkdir(parents=True, exist_ok=True)
    (root/'results/active_run.json').write_text(json.dumps(dict(signature=signature, config=config), indent=2))
    books, audit = load_books()
    audit.to_csv(root/'results/data_audit.csv', index=False)
    microstructure(books).to_csv(root/'results/microstructure.csv', index=False)
    scores, history, runtime = run_benchmark(books, config, directory)
    scores.to_csv(root/'results/metrics.csv', index=False)
    history.to_csv(root/'results/training_history.csv', index=False)
    runtime.to_csv(root/'results/runtime.csv', index=False)
    print(scores.query('threshold == .3').groupby('model')[['mcc','macro_f1','p_t']].mean().to_string(), flush=True)

if __name__ == '__main__':
    main()
