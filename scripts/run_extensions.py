#execute future work experiments
import json
import sys
from pathlib import Path as path
sys.path.insert(0,str(path(__file__).resolve().parents[1]))
from scripts.pipeline import load_books
from scripts.extensions import run_daily,run_cross_exchange,run_sensitivities,run_ablations,run_seed_robustness
root=path(__file__).resolve().parents[1]
if __name__=='__main__':
    print('running daily walk forward comparisons',flush=True)
    run_daily()
    books,_=load_books()
    print(run_cross_exchange(books).to_string(index=False),flush=True)
    active=json.loads((root/'results/active_run.json').read_text())
    directory=root/'results'/active['signature']
    if (root/'results/metrics.csv').exists():
        run_sensitivities(books,directory)
        run_ablations(books,active['config'],directory)
        run_seed_robustness(books,active['config'],directory)
    print('completed available extensions',flush=True)
