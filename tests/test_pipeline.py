#test the scientific contracts
import numpy as np
import pandas as pd
import torch
from scripts.pipeline import labels, split_indices, transaction_metrics, transaction_set, balanced_indices, seed_all, deeplob, transformer, graph_lob, diffusion_lob, features, rolling_daily_scale, execution_stress


def test_labels_at_tick_boundaries():
    assert labels([1, 1.005, 1.015, 1.005, 1.005], 1, .01).tolist() == [1, 2, 0, 1]
    assert len(labels(np.arange(200)*.01, 100, .01)) == 100


def test_purged_windows():
    for h in [10, 50, 100]:
        tr, va, te = split_indices(10000, h)
        assert tr[-1]+h < va[0]-99
        assert va[-1]+h < te[0]-99
        assert te[-1]+h < 10000


def test_balancing():
    y = np.array([0]*200+[1]*100+[2]*50)
    ii = balanced_indices(y, np.arange(len(y)), 128, 17)
    assert np.bincount(y[ii]).tolist() == [50, 50, 50]
    assert len(set(ii)) == len(ii)


def test_paper_figure_twelve():
    true = np.array([-1,-1,1,1,1,1,-1,-1,0,-1,1,1,1])+1
    pred_a = np.array([-1,-1,-1,-1,0,0,-1,-1,0,-1,1,1,1])+1
    pred_b = np.array([-1,0,0,1,0,0,0,-1,-1,0,0,-1,1])+1
    a, b = transaction_metrics(true, pred_a), transaction_metrics(true, pred_b)
    assert a == dict(potential=3, executed=1, correct=0, p_t=0.)
    assert b == dict(potential=3, executed=3, correct=3, p_t=1.)


def test_empty_and_perfect_transactions():
    assert np.isnan(transaction_metrics([1]*10, [1]*10)['p_t'])
    y = [0,0,1,2,2,1,0]
    assert transaction_metrics(y, y)['p_t'] == 1
    assert transaction_metrics(y, y, np.ones(7)*.4, .9)['executed'] == 0
    assert transaction_set([1,1,0,1]) == []


def test_no_duplicate_transaction_credit():
    y = [0,0,0,2,2,2,0,0,0]
    p = [0,2,0,2,0,2,0,2,0]
    v = transaction_metrics(y, p, tolerance=1)
    assert v['correct'] <= min(v['potential'], v['executed'])
    assert 0 <= v['p_t'] <= 1


def test_models_and_diffusion_gradients():
    seed_all()
    x = torch.randn(2,100,40)
    for constructor in [deeplob, transformer, graph_lob, diffusion_lob]:
        model = constructor()
        out = model(x)
        assert out.shape == (2,3) and torch.isfinite(out).all()
        model.eval()
        assert torch.equal(model(x), model(x))
    model = diffusion_lob()
    loss = model.training_loss(x, torch.tensor([0,2]))
    loss.backward()
    assert model.denoise[0].weight.grad.abs().sum() > 0


def test_rolling_scale_excludes_today():
    days = {f'2024-01-{i:02}': pd.DataFrame(np.ones((2,40))*i, columns=features) for i in range(1,8)}
    a, lineage = rolling_daily_scale(days)
    days['2024-01-07'][:] = 9999
    b, _ = rolling_daily_scale(days)
    assert np.array_equal(a['2024-01-06'], b['2024-01-06'])
    assert lineage[0]['fitted_days'] == list(days)[:5]


def test_costs_reduce_net_result():
    f = pd.DataFrame({'ask_price_1':[10.01,10.03,10.05,10.07], 'bid_price_1':[10,10.02,10.04,10.06]})
    p = np.eye(3)[[2,2,0,0]]
    a = execution_stress(f,np.arange(4),p,.01,fee_bps=0)
    b = execution_stress(f,np.arange(4),p,.01,fee_bps=1)
    assert a['mean_net_ticks'] > b['mean_net_ticks']


def test_cleaning_session_and_duplicate_order():
    from scripts.pipeline import clean_book
    rows=[]
    for t in [34799,34800,34800,35000,57001]:
        row={}
        for level in range(1,11):
            row[f'ask_price_{level}']=10+.01*level
            row[f'bid_price_{level}']=10-.01*level
            row[f'ask_size_{level}']=100
            row[f'bid_size_{level}']=100
        row.update(seconds=t,timestamp=t*10**9)
        rows.append(row)
    rows[2]['ask_size_1']=222
    rows[3]['bid_price_1']=11
    result,audit=clean_book(pd.DataFrame(rows))
    assert len(result)==1 and result.ask_size_1.iloc[0]==222
    assert audit['outside_session']==2 and audit['invalid_rows']==1
    assert audit['duplicate_timestamps']==1


def test_scaler_cannot_see_test_rows():
    from scripts.pipeline import prepare
    rows=[]
    for t in range(1000):
        row={f:100+t*.001+j for j,f in enumerate(features)}
        row['mid']=100+t*.001
        rows.append(row)
    frame=pd.DataFrame(rows)
    first=prepare(dict(frame=frame,tick=.01),10)
    frame.loc[800:,features]=1e6
    second=prepare(dict(frame=frame,tick=.01),10)
    assert np.array_equal(first['scaler'].mean_,second['scaler'].mean_)
    assert np.array_equal(first['x'][:600],second['x'][:600])
