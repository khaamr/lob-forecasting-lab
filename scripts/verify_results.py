#verify artifacts, predictions and deterministic retraining
import json
import sys
from pathlib import Path as path
import numpy as np
import torch
import pandas as pd
sys.path.insert(0,str(path(__file__).resolve().parents[1]))
from scripts.pipeline import load_books,prepare,train_model,predict,score,seed_all,deeplob,transformer,graph_lob,diffusion_lob
root=path(__file__).resolve().parents[1]

def verify(retrain=True):
    seed_all(17)
    active=json.loads((root/'results/active_run.json').read_text())
    directory=root/'results'/active['signature']
    metrics=pd.read_csv(root/'results/metrics.csv')
    checks=[]
    for p in directory.glob('*_predictions.npz'):
        d=np.load(p)
        prob,y,indices=d['test'],d['targets'],d['indices']
        assert prob.shape==(len(y),3)
        assert np.isfinite(prob).all() and (prob>=0).all() and np.allclose(prob.sum(1),1,atol=1e-6)
        assert (np.diff(indices)==1).all(), 'transaction evaluation needs a contiguous sequence'
        asset,horizon,name=p.stem.removesuffix('_predictions').rsplit('_',2)
        h=int(horizon[1:])
        actual=score(y,prob)
        saved=metrics.query('asset == @asset and horizon == @h and model == @name and threshold == .3').iloc[0]
        for key in ['mcc','macro_f1','accuracy','p_t','potential','executed','correct']:
            assert np.isclose(actual[key],saved[key],equal_nan=True,atol=1e-12), (p,key)
        checks.append(str(p.name))
    assert len(checks)==147, len(checks)
    for _,g in metrics.groupby(['asset','horizon','model']):
        assert (np.diff(g.sort_values('threshold').coverage)<=0).all()
        assert (g.correct<=g[['potential','executed']].min(axis=1)).all()
        assert g.p_t.dropna().between(0,1).all()
    for _,g in pd.read_csv(root/'results/physical_horizons.csv').groupby(['asset','horizon']):
        assert np.isclose(g.probability.sum(),1)
    daily=pd.read_csv(root/'results/daily_metrics.csv')
    assert len(daily)==132 and daily.mcc.between(-1,1).all()
    cross=pd.read_csv(root/'results/cross_exchange.csv')
    assert (cross.train_last_timestamp<cross.test_first_timestamp).all()
    report=dict(prediction_files_checked=len(checks),metric_rows=len(metrics),daily_experiments=len(daily),
                model_experiments=84,finite_probabilities=True,metrics_recomputed=True,
                monotone_coverage=True,cross_exchange_time_separation=True)
    if retrain:
        books,_=load_books()
        constructors=dict(deeplob=deeplob,transformer=transformer,graph=graph_lob,diffusion=diffusion_lob)
        checkpoint_count=0
        for asset in ['aapl','intc']:
            prepared=prepare(books[asset],50,validation_cap=1024,test_cap=4096)
            for name,constructor in constructors.items():
                model=constructor()
                model.load_state_dict(torch.load(directory/f'{asset}_h50_{name}.pt',weights_only=True))
                observed=predict(model,prepared,prepared['indices'][2][:256])
                expected=np.load(directory/f'{asset}_h50_{name}_predictions.npz')['test'][:256]
                assert np.array_equal(observed,expected), (asset,name,float(np.max(np.abs(observed-expected))))
                checkpoint_count+=1
        report['checkpoints_reloaded_and_replayed']=checkpoint_count
        p=prepare(books['aapl'],50,validation_cap=128,test_cap=256)
        config=dict(epochs=2,patience=2,per_class=32)
        first,_,_=train_model('graph',p,config,17)
        second,_,_=train_model('graph',p,config,17)
        pa,pb=predict(first,p,p['indices'][2]),predict(second,p,p['indices'][2])
        assert np.array_equal(pa,pb)
        report['independent_retraining_max_probability_difference']=float(np.max(np.abs(pa-pb)))
    (root/'results/verification.json').write_text(json.dumps(report,indent=2))
    return report

if __name__=='__main__':
    print(verify())
