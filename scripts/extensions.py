#future directions and additional stock sectors
import json
from pathlib import Path as path
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.preprocessing import StandardScaler as standard_scaler
from sklearn.linear_model import LogisticRegression as logistic_regression
from sklearn.ensemble import HistGradientBoostingClassifier as gradient_boosting
from sklearn.metrics import log_loss, matthews_corrcoef, f1_score, accuracy_score
from scripts.pipeline import (prepare, train_model, predict, score, transaction_metrics, fit_temperature,
                              execution_stress, stationary_features, labels, block_bootstrap, seed_all)
root = path(__file__).resolve().parents[1]


def run_daily():
    frame = pd.read_csv(root/'data/processed/daily_prices.csv', parse_dates=['date'])
    prices = frame.pivot(index='date', columns='asset', values='close').sort_index()
    returns = prices.pct_change(fill_method=None)
    summary = []
    for asset in prices:
        close = prices[asset].dropna()
        r = returns[asset].dropna()
        paired = returns[[asset, 'qqq']].dropna() if asset != 'qqq' else None
        beta = paired.iloc[:,0].cov(paired.iloc[:,1])/paired.iloc[:,1].var() if paired is not None else 1.
        summary.append(dict(asset=asset, field=frame.loc[frame.asset==asset,'field'].iloc[0],
                            observations=len(close), total_return=close.iloc[-1]/close.iloc[0]-1,
                            annualized_volatility=r.std()*np.sqrt(252),
                            maximum_drawdown=(close/close.cummax()-1).min(), beta_to_qqq=beta))
    scores, predictions = [], []
    for asset, group in frame.groupby('asset'):
        group = group.sort_values('date').set_index('date')
        close, volume = group.close, group.volume
        x = pd.DataFrame(index=group.index)
        for lag in [1,2,5,10,20]:
            x[f'return_{lag}'] = close.pct_change(lag, fill_method=None)
        x['volatility_20'] = close.pct_change(fill_method=None).rolling(20).std()
        x['range'] = (group.high-group.low)/close
        x['volume_ratio'] = volume/volume.rolling(20).mean()-1
        x['benchmark_return'] = prices.qqq.pct_change(fill_method=None)
        for horizon in [1,5,10]:
            forward = close.shift(-horizon)/close-1
            #a separate daily label rule; never described as a lob tick label
            y = pd.Series(np.where(forward < -.005, 0, np.where(forward > .005, 2, 1)), index=x.index)
            y[forward.isna()] = np.nan
            valid = x.notna().all(axis=1)&y.notna()
            for year in [2024,2025]:
                train = valid&(x.index.year<year)
                test = valid&(x.index.year==year)
                #purge the training tail whose forward return enters the test year
                positions = np.flatnonzero(train)
                positions = positions[positions+horizon < np.flatnonzero(x.index.year==year)[0]]
                te = np.flatnonzero(test)
                scaler = standard_scaler().fit(x.iloc[positions])
                xt, xv = scaler.transform(x.iloc[positions]), scaler.transform(x.iloc[te])
                yt, yv = y.iloc[positions].astype(int), y.iloc[te].astype(int)
                for name, model in [('logistic', logistic_regression(max_iter=500, random_state=17)),
                                    ('boosted_trees', gradient_boosting(max_iter=80,max_leaf_nodes=7,l2_regularization=5,random_state=17))]:
                    model.fit(xt,yt)
                    prob = model.predict_proba(xv)
                    pred = prob.argmax(1)
                    scores.append(dict(asset=asset,horizon_days=horizon,test_year=year,model=name,
                                       train_count=len(yt),test_count=len(yv),mcc=matthews_corrcoef(yv,pred),
                                       macro_f1=f1_score(yv,pred,labels=[0,1,2],average='macro',zero_division=0),
                                       accuracy=accuracy_score(yv,pred),majority_accuracy=np.mean(yv==yt.mode().iloc[0])))
                    predictions.extend(dict(asset=asset,horizon_days=horizon,test_year=year,model=name,date=str(d.date()),target=int(a),prediction=int(b)) for d,a,b in zip(x.index[te],yv,pred))
    summary = pd.DataFrame(summary)
    daily_scores = pd.DataFrame(scores)
    summary.to_csv(root/'results/daily_summary.csv',index=False)
    daily_scores.to_csv(root/'results/daily_metrics.csv',index=False)
    pd.DataFrame(predictions).to_csv(root/'results/daily_predictions.csv',index=False)
    return summary,daily_scores


def load_prediction(directory, asset, h, name):
    return np.load(directory/f'{asset}_h{h}_{name}_predictions.npz')


def run_sensitivities(books, directory):
    calibration, stress, tolerance, intervals, regime_scores = [], [], [], [], []
    for asset, book in books.items():
        for horizon in [10,50,100]:
            prepared = prepare(book,horizon,validation_cap=1024,test_cap=4096)
            _, va, te = prepared['indices']
            for name in ['deeplob','transformer','graph','diffusion','logistic']:
                result = load_prediction(directory,asset,horizon,name)
                pv,pt,y = result['validation'],result['test'],result['targets']
                temperature = fit_temperature(prepared['y'][va],pv)
                calibrated = softmax(np.log(pt.clip(1e-9))/temperature,axis=1)
                for variant,prob in [('raw',pt),('temperature',calibrated)]:
                    confidence=prob.max(1)
                    correct=prob.argmax(1)==y
                    ece=0.
                    for a,b in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
                        keep=(confidence>=a)&(confidence<b if b<1 else confidence<=b)
                        if keep.any():
                            ece+=keep.mean()*abs(confidence[keep].mean()-correct[keep].mean())
                    calibration.append(dict(asset=asset,horizon=horizon,model=name,variant=variant,temperature=temperature,
                                             log_loss=log_loss(y,prob,labels=[0,1,2]),ece=ece,
                                             brier=np.mean(np.sum((prob-np.eye(3)[y])**2,axis=1))))
                for delay in [0,1,5,10,50,100]:
                    for fee in [0,.1,1]:
                        stress.append(dict(asset=asset,horizon=horizon,model=name,
                                           **execution_stress(book['frame'],te,pt,book['tick'],delay,fee)))
                for tol in [0,1,5,10]:
                    tolerance.append(dict(asset=asset,horizon=horizon,model=name,tolerance_updates=tol,
                                          **transaction_metrics(y,pt.argmax(1),tolerance=tol)))
                #classification by causal trailing volatility, with cutoff fitted on training only
                volatility=book['frame'].mid.diff().rolling(100).std().to_numpy()
                cutoff=np.nanmedian(volatility[:int(len(volatility)*.6)])
                for regime,keep in [('quiet',volatility[te]<=cutoff),('active',volatility[te]>cutoff)]:
                    if keep.any():
                        regime_scores.append(dict(asset=asset,horizon=horizon,model=name,regime=regime,
                                                   count=keep.sum(),mcc=matthews_corrcoef(y[keep],pt[keep].argmax(1))))
                if horizon==50:
                    for block in [128,512]:
                        bounds=block_bootstrap(y,pt,block=block,repeats=100)
                        intervals.append(dict(asset=asset,model=name,block_updates=block,mcc_low=bounds[0,0],mcc_high=bounds[1,0],p_t_low=bounds[0,1],p_t_high=bounds[1,1]))
    for name,rows in [('calibration',calibration),('execution_stress',stress),('tolerance',tolerance),('bootstrap',intervals),('regime_scores',regime_scores)]:
        pd.DataFrame(rows).to_csv(root/f'results/{name}.csv',index=False)


def run_ablations(books, config, directory):
    rows=[]
    #predeclared small and large tick representatives
    for asset in ['aapl','intc']:
        for representation,name in [('raw','graph_no_edges'),('stationary','graph')]:
            prepared=prepare(books[asset],50,representation,validation_cap=1024,test_cap=4096)
            cache=directory/f'ablation_{asset}_{representation}_{name}.npz'
            if cache.exists():
                result=np.load(cache);prob=result['probability']
            else:
                model,history,seconds=train_model(name,prepared,config)
                prob=predict(model,prepared,prepared['indices'][2])
                np.savez_compressed(cache,probability=prob)
            rows.append(dict(asset=asset,representation=representation,model=name,**score(prepared['y'][prepared['indices'][2]],prob)))
        base=load_prediction(directory,asset,50,'graph')
        rows.append(dict(asset=asset,representation='raw',model='graph',**score(base['targets'],base['test'])))
    pd.DataFrame(rows).to_csv(root/'results/ablations.csv',index=False)


def run_cross_exchange(books):
    assets=['binance_btcusdt','coinbase_btc-usd']
    #a common physical time prevents training on one venue after the other venue test
    start=max(books[a]['frame'].timestamp.min() for a in assets)
    end=min(books[a]['frame'].timestamp.max() for a in assets)
    cut=start+.6*(end-start)
    test_start=start+.8*(end-start)
    rows=[]
    for source in assets:
        source_book=books[source]
        frame=source_book['frame']
        x=stationary_features(frame,source_book['tick'])
        y=labels(frame.mid,50,source_book['tick'])
        indices=np.arange(99,len(y))
        tr=indices[(frame.timestamp.to_numpy()[indices]>=start)&(frame.timestamp.to_numpy()[indices+50]<cut)]
        assert len(tr)>50
        scaler=standard_scaler().fit(x[tr])
        model=logistic_regression(max_iter=1000,random_state=17).fit(scaler.transform(x[tr]),y[tr])
        assert len(model.classes_)==3
        for target in assets:
            target_book=books[target]
            tf=target_book['frame']
            tx=stationary_features(tf,target_book['tick'])
            ty=labels(tf.mid,50,target_book['tick'])
            jj=np.arange(99,len(ty))
            te=jj[(tf.timestamp.to_numpy()[jj-99]>=test_start)&(tf.timestamp.to_numpy()[jj+50]<=end)]
            assert len(te)>0 and frame.timestamp.iloc[tr[-1]+50]<tf.timestamp.iloc[te[0]-99]
            prob=model.predict_proba(scaler.transform(tx[te]))
            rows.append(dict(source=source,target=target,train_count=len(tr),common_span_seconds=(end-start)/1e9,
                             train_last_timestamp=int(frame.timestamp.iloc[tr[-1]+50]),
                             test_first_timestamp=int(tf.timestamp.iloc[te[0]-99]),**score(ty[te],prob)))
    result=pd.DataFrame(rows)
    result.to_csv(root/'results/cross_exchange.csv',index=False)
    return result


def run_seed_robustness(books,config,directory):
    rows=[]
    for asset in ['aapl','intc']:
        p=prepare(books[asset],50,validation_cap=1024,test_cap=4096)
        for seed in [17,29,43]:
            cache=directory/f'seed_{asset}_{seed}_graph.npz'
            if seed==17:
                prob=load_prediction(directory,asset,50,'graph')['test']
            elif cache.exists():
                prob=np.load(cache)['probability']
            else:
                model,_,_=train_model('graph',p,config,seed)
                prob=predict(model,p,p['indices'][2])
                np.savez_compressed(cache,probability=prob)
            rows.append(dict(asset=asset,seed=seed,**score(p['y'][p['indices'][2]],prob)))
    pd.DataFrame(rows).to_csv(root/'results/seed_robustness.csv',index=False)


def run_budget_sensitivity(books,directory):
    rows=[]
    expanded=dict(epochs=30,patience=8,per_class=1024)
    main_config=json.loads((root/'results/active_run.json').read_text())['config']
    for asset in ['aapl','intc']:
        p=prepare(books[asset],50,validation_cap=1024,test_cap=4096)
        for name in ['deeplob','graph']:
            base=load_prediction(directory,asset,50,name)
            rows.append(dict(asset=asset,model=name,budget='main sample budget',epochs_cap=main_config['epochs'],per_class=main_config['per_class'],**score(base['targets'],base['test'])))
            cache=directory/f'budget_{asset}_{name}.npz'
            if cache.exists():
                probability=np.load(cache)['probability']
            else:
                model,history,seconds=train_model(name,p,expanded)
                probability=predict(model,p,p['indices'][2])
                np.savez_compressed(cache,probability=probability)
                (directory/f'budget_{asset}_{name}.json').write_text(json.dumps(dict(history=history,seconds=seconds),indent=2))
            rows.append(dict(asset=asset,model=name,budget='expanded sample budget',epochs_cap=30,per_class=1024,**score(p['y'][p['indices'][2]],probability)))
            print(f'completed budget sensitivity: {asset} {name}',flush=True)
    pd.DataFrame(rows).to_csv(root/'results/budget_sensitivity.csv',index=False)
