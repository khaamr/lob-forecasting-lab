#complete the public experiment extensions
import sys,json,time
from pathlib import Path as path
sys.path.insert(0,str(path(__file__).resolve().parents[1]))
from scripts.pipeline import load_books
from scripts.extensions import run_sensitivities,run_ablations,run_seed_robustness,run_budget_sensitivity
root=path(__file__).resolve().parents[1]
if __name__=='__main__':
    while not (root/'results/metrics.csv').exists():
        time.sleep(5)
    active=json.loads((root/'results/active_run.json').read_text())
    directory=root/'results'/active['signature']
    books,_=load_books()
    run_sensitivities(books,directory)
    print('completed calibration, execution and uncertainty',flush=True)
    run_ablations(books,active['config'],directory)
    print('completed graph ablations',flush=True)
    run_seed_robustness(books,active['config'],directory)
    print('completed seed robustness',flush=True)
    run_budget_sensitivity(books,directory)
    print('completed all extensions',flush=True)
