#licensed archive pipeline with explicit date manifests
import json
from pathlib import Path as path
import zipfile
import numpy as np
import pandas as pd
from scripts.pipeline import features,clean_book,rolling_daily_scale,labels,train_model,predict,score,balanced_indices

paper_assets=['chtr','goog','gs','ibm','mcd','nvda','aapl','abbv','pm','bac','csco','ko','orcl','pfe','vz']

def read_pair(book_path,message_path):
    frame=pd.read_csv(book_path,header=None,names=features)
    messages=pd.read_csv(message_path,header=None,dtype={0:str})
    assert len(frame)==len(messages)
    parts=messages[0].str.split('.',expand=True)
    frame['timestamp']=parts[0].astype('int64')*10**9+parts[1].fillna('0').str.pad(9,side='right',fillchar='0').str[:9].astype('int64')
    frame['seconds']=frame.timestamp/1e9
    frame[features[0::2]]=frame[features[0::2]]/10000
    return clean_book(frame,True)[0]


def prepare_archive(manifest_path,horizon):
    #one ticker and one year per manifest, including five warmup days
    manifest_path=path(manifest_path)
    manifest=json.loads(manifest_path.read_text())
    asset=manifest['asset']
    if asset not in paper_assets:
        raise ValueError('the original paper asset must be listed explicitly')
    records=manifest['sessions']
    dates=[r['date'] for r in records]
    assert len(dates)==len(set(dates)), 'duplicate sessions'
    assert all(pd.Timestamp(d).dayofweek<5 for d in dates), 'a weekend appears in the session manifest; the paper table has a 2019 date ambiguity'
    counts=pd.Series([r['split'] for r in records]).value_counts().to_dict()
    assert counts=={'train':45,'test':10,'validation':5,'warmup':5}, f'unexpected split counts: {counts}'
    days={r['date']:read_pair(manifest_path.parent/r['orderbook'],manifest_path.parent/r['message']) for r in records}
    scaled,lineage=rolling_daily_scale(days)
    arrays,ys,groups,lookup=[],[],[],{'train':[],'validation':[],'test':[]}
    offset=0
    frames=[]
    for record in sorted(records,key=lambda r:r['date']):
        date=record['date']
        if record['split']=='warmup':
            continue
        assert date in scaled, 'five preceding warmup days are required'
        frame=days[date]
        n=len(frame)
        label=labels(frame.mid,horizon,.01)
        padded=np.concatenate([label,np.full(horizon,-1)])
        ii=np.arange(99,n-horizon)+offset
        lookup[record['split']].append(ii)
        if record['split']=='train':
            groups.append(ii)
        arrays.append(scaled[date]);ys.append(padded)
        frames.append(dict(date=date,split=record['split'],start=offset,stop=offset+n))
        offset+=n
    prepared=dict(x=np.concatenate(arrays),y=np.concatenate(ys),
                  indices=tuple(np.concatenate(lookup[s]) for s in ['train','validation','test']),
                  training_groups=groups,horizon=horizon,scaler=None,representation='rolling five day')
    return prepared,frames,lineage


def run_archive(manifest_path,output_dir):
    #the public notebook never invokes this without a real licensed manifest
    from scripts import pipeline
    output_dir=path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    rows=[]
    for h in [10,50,100]:
        prepared,frames,lineage=prepare_archive(manifest_path,h)
        #balanced sampling must be daily rather than pooled across forty five days
        original=pipeline.balanced_indices
        def daily_balanced(y,indices,cap,seed):
            return np.concatenate([original(y,g,cap,seed+i) for i,g in enumerate(prepared['training_groups'])])
        pipeline.balanced_indices=daily_balanced
        try:
            model,history,seconds=train_model('deeplob',prepared,dict(epochs=100,patience=15,per_class=5000))
        finally:
            pipeline.balanced_indices=original
        #score each day independently so positions never cross a session boundary
        for session in frames:
            if session['split']!='test':
                continue
            ii=np.arange(session['start']+99,session['stop']-h)
            probability=predict(model,prepared,ii)
            np.savez_compressed(output_dir/f"h{h}_{session['date']}.npz",probability=probability,targets=prepared['y'][ii],indices=ii)
            for threshold in [.3,.4,.5,.6,.7,.8,.9]:
                rows.append(dict(date=session['date'],horizon=h,**score(prepared['y'][ii],probability,threshold)))
        (output_dir/f'h{h}_lineage.json').write_text(json.dumps(lineage,indent=2))
        (output_dir/f'h{h}_training.json').write_text(json.dumps(history,indent=2))
    result=pd.DataFrame(rows)
    result.to_csv(output_dir/'metrics.csv',index=False)
    return result
