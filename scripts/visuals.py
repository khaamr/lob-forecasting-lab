#interactive plotly figures
import json
from pathlib import Path as path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix
from scripts.pipeline import labels,classes,transaction_metrics
root=path(__file__).resolve().parents[1]
colors=['#2563eb','#e76f51','#2a9d8f','#8b5cf6','#d99c20','#e05294','#64748b','#0891b2','#be123c','#6d28d9','#4d7c0f','#a16207']
px.defaults.template='plotly_white'
px.defaults.color_discrete_sequence=colors


def polish(fig,title=None):
    if title:
        fig.update_layout(title=title)
    fig.update_layout(font=dict(family='Arial, sans-serif',size=13,color='#17283b'),
                      title_font_size=21,paper_bgcolor='#f8fafc',plot_bgcolor='white',
                      margin=dict(l=65,r=35,t=85,b=70),height=480,
                      legend_title_text='',hovermode='closest')
    for annotation in fig.layout.annotations or []:
        annotation.text=annotation.text.replace('_',' ')
    return fig


def plotly_table(frame,title):
    frame=frame.copy()
    for c in frame.select_dtypes('number'):
        frame[c]=frame[c].map(lambda v:f'{v:.4f}' if pd.notna(v) else 'undefined')
    fig=go.Figure(go.Table(header=dict(values=[c.replace('_',' ') for c in frame],fill_color='#17283b',font_color='white',align='left',height=32),
                          cells=dict(values=[frame[c].astype(str).tolist() for c in frame],fill_color='#edf2f7',align='left',height=29)))
    polish(fig,title)
    fig.update_layout(height=max(300,135+len(frame)*30))
    return fig


def ccdf(series,points=180):
    values=np.sort(np.asarray(series))
    index=np.unique(np.linspace(0,len(values)-1,min(points,len(values))).astype(int))
    return values[index],1-index/len(values)


def create_figures(books,directory):
    figures={}
    micro=pd.read_csv(root/'results/microstructure.csv')
    equity=micro.loc[micro.asset.isin([k for k,v in books.items() if v['equity']])]
    figures['01_microstructure_table']=plotly_table(equity[['asset','tick_group','mean_spread_ticks','mean_top_depth','mean_book_width_ticks','information_richness']], 'public stock sample: measured microstructure')
    figures['02_spread_regimes']=polish(px.bar(equity,x='asset',y='mean_spread_ticks',color='tick_group',log_y=True), 'spread in ticks separates the stock regimes')
    figures['02_spread_regimes'].add_hline(y=1.5,line_dash='dot',annotation_text='large tick boundary: 1.5')
    figures['02_spread_regimes'].add_hline(y=3,line_dash='dot',annotation_text='small tick boundary: 3')
    distributions=[]
    for asset,book in books.items():
        if not book['equity']:
            continue
        frame=book['frame']
        spread=np.rint((frame.ask_price_1-frame.bid_price_1)/book['tick']).astype(int)
        counts=spread.value_counts(normalize=True).sort_index()
        distributions.extend(dict(asset=asset,spread_ticks=s,probability=p) for s,p in counts.items())
    figures['03_spread_distribution']=polish(px.line(pd.DataFrame(distributions),x='spread_ticks',y='probability',color='asset',log_y=True), 'spread probability distribution: the public sample analogue of figure 4')
    figures['03_spread_distribution'].update_xaxes(range=[0,80])
    liquidity=[]
    width=[]
    clock=[]
    class_rows=[]
    for asset,book in books.items():
        f=book['frame']
        for side in ['ask','bid']:
            values,survival=ccdf(f[f'{side}_size_1'])
            liquidity.extend(dict(asset=asset,side=side,volume=v,survival=p) for v,p in zip(values,survival))
            d=(f.ask_price_10-f.ask_price_1)/book['tick'] if side=='ask' else (f.bid_price_1-f.bid_price_10)/book['tick']
            values,survival=ccdf(d)
            width.extend(dict(asset=asset,side=side,depth_ticks=v,survival=p) for v,p in zip(values,survival))
        for h in [10,50,100]:
            duration=np.diff(f.timestamp.to_numpy()[::h])/1e9
            exact=(f.timestamp.to_numpy()[h:]-f.timestamp.to_numpy()[:-h])/1e9
            for bucket,mask in [('less than 1 second',exact<1),('1 to 10 seconds',(exact>=1)&(exact<10)),('at least 10 seconds',exact>=10)]:
                clock.append(dict(asset=asset,horizon=h,bucket=bucket,probability=mask.mean()))
            y=labels(f.mid,h,book['tick'])
            for c,p in enumerate(np.bincount(y,minlength=3)/len(y)):
                class_rows.append(dict(asset=asset,horizon=h,label=classes[c],fraction=p))
    liquidity=pd.DataFrame(liquidity)
    width=pd.DataFrame(width)
    figures['04_depth_liquidity']=polish(px.line(liquidity[liquidity.asset.isin(equity.asset)],x='volume',y='survival',color='asset',line_dash='side',log_x=True,log_y=True), 'best quote liquidity: ask and bid volume survival')
    figures['05_book_sparsity']=polish(px.line(width[width.asset.isin(equity.asset)],x='depth_ticks',y='survival',color='asset',line_dash='side',log_x=True), 'actual book depth: wider spacing means a sparser representation')
    pd.DataFrame(clock).to_csv(root/'results/physical_horizons.csv',index=False)
    pd.DataFrame(class_rows).to_csv(root/'results/class_balance.csv',index=False)
    figures['06_event_clock']=polish(px.bar(pd.DataFrame(clock),x='asset',y='probability',color='bucket',facet_col='horizon'), 'the same event horizon has different physical durations')
    figures['06_event_clock'].update_layout(height=520)
    figures['07_class_balance']=polish(px.bar(pd.DataFrame(class_rows),x='asset',y='fraction',color='label',facet_col='horizon'), 'class imbalance changes with the forecasting horizon')
    metrics=pd.read_csv(root/'results/metrics.csv')
    base=metrics.query('threshold == .3')
    figures['08_architecture_comparison']=polish(px.bar(base,x='asset',y='mcc',color='model',facet_col='horizon',barmode='group'), 'all model families and baselines: held out mcc')
    figures['08_architecture_comparison'].update_layout(height=570,legend=dict(orientation='h',y=-.28))
    figures['09_transaction_comparison']=polish(px.bar(base,x='asset',y='p_t',color='model',facet_col='horizon',barmode='group'), 'correct transaction overlap can remain low despite classification skill')
    figures['09_transaction_comparison'].update_layout(height=570,legend=dict(orientation='h',y=-.28))
    curves=metrics.query('model == "deeplob"').merge(micro[['asset','tick_group']],on='asset')
    figures['10_confidence_coverage']=polish(px.scatter(curves,x='coverage',y='mcc',color='asset',facet_col='horizon',size='count',hover_data=['threshold','p_t']), 'confidence filtering: report coverage alongside apparent skill')
    figures['11_confidence_transactions']=polish(px.line(curves,x='threshold',y='p_t',color='asset',facet_col='horizon',markers=True), 'confidence thresholds and transaction overlap')
    history=pd.read_csv(root/'results/training_history.csv')
    history=history.query('asset == "aapl" and horizon == 50')
    figures['12_learning_curves']=polish(px.line(history,x='epoch',y='validation_loss',color='model',markers=True), 'validation loss determines the restored checkpoint: aapl, h50')
    matrix=make_subplots(rows=1,cols=4,subplot_titles=['deeplob','transformer','graph','diffusion'])
    for j,name in enumerate(['deeplob','transformer','graph','diffusion']):
        result=np.load(directory/f'intc_h50_{name}_predictions.npz')
        cm=confusion_matrix(result['targets'],result['test'].argmax(1),labels=[0,1,2],normalize='true')
        matrix.add_trace(go.Heatmap(z=cm,x=classes,y=classes,zmin=0,zmax=1,colorscale='Blues',showscale=j==3,text=np.round(cm,2),texttemplate='%{text}'),row=1,col=j+1)
    figures['13_confusion_matrices']=polish(matrix,'intc, h50: row normalized confusion matrices')
    calibration=pd.read_csv(root/'results/calibration.csv')
    figures['14_calibration']=polish(px.box(calibration,x='model',y='log_loss',color='variant',points='all'), 'validation fitted temperature scaling: held out log loss')
    ablations=pd.read_csv(root/'results/ablations.csv')
    ablations['variant']=ablations.model+' / '+ablations.representation
    figures['15_graph_ablation']=polish(px.bar(ablations,x='asset',y='mcc',color='variant',barmode='group'), 'why graph structure and price normalization matter: controlled ablations')
    cross=pd.read_csv(root/'results/cross_exchange.csv')
    pivot=cross.pivot(index='source',columns='target',values='mcc')
    figures['16_cross_exchange']=polish(px.imshow(pivot,text_auto='.3f',color_continuous_scale='RdBu',range_color=[-.5,.5],aspect='auto'), 'common clock transfer: logistic model, h50')
    stress=pd.read_csv(root/'results/execution_stress.csv')
    view=stress.query('horizon == 50 and model == "graph"').copy()
    view['fee_scenario']=view.fee_bps.map(lambda v:f'{v:g} bps each side')
    figures['17_latency_costs']=polish(px.line(view,x='delay_updates',y='mean_net_ticks',color='fee_scenario',facet_col='asset',facet_col_wrap=3,markers=True), 'execution sensitivity: top quote fills with spread and stated fees')
    figures['17_latency_costs'].update_layout(height=850)
    figures['17_latency_costs'].update_yaxes(matches=None)
    tolerance=pd.read_csv(root/'results/tolerance.csv').query('horizon == 50 and model == "graph"')
    figures['18_timing_tolerance']=polish(px.line(tolerance,x='tolerance_updates',y='p_t',color='asset',markers=True), 'relaxing signal timing: an explicitly different transaction metric')
    seeds=pd.read_csv(root/'results/seed_robustness.csv')
    figures['19_seed_robustness']=polish(px.scatter(seeds,x='seed',y='mcc',color='asset'), 'graph model, h50: three independently trained seeds')
    bootstrap=pd.read_csv(root/'results/bootstrap.csv')
    boot=bootstrap.query('block_updates == 512').merge(base.query('horizon == 50')[['asset','model','mcc']],on=['asset','model'])
    boot['upper']=(boot.mcc_high-boot.mcc).clip(lower=0)
    boot['lower']=(boot.mcc-boot.mcc_low).clip(lower=0)
    figures['20_block_uncertainty']=polish(px.scatter(boot,x='asset',y='mcc',color='model',error_y='upper',error_y_minus='lower',hover_data=['mcc_low','mcc_high']), 'within sample uncertainty: 95 percent moving block intervals')
    figures['20_block_uncertainty'].update_layout(height=540)
    daily=pd.read_csv(root/'data/processed/daily_prices.csv',parse_dates=['date'])
    daily['rebased']=daily.close/daily.groupby('asset').close.transform('first')*100
    figures['21_new_sector_prices']=polish(px.line(daily,x='date',y='rebased',color='asset',log_y=True), 'additional sectors and qqq: adjusted prices rebased to 100')
    figures['21_new_sector_prices'].update_yaxes(tickvals=[10,20,50,100,200,500,1000],ticktext=['10','20','50','100','200','500','1000'])
    summary=pd.read_csv(root/'results/daily_summary.csv')
    figures['22_new_sector_risk']=polish(px.scatter(summary,x='annualized_volatility',y='total_return',color='field',text='asset',size='observations',hover_data=['maximum_drawdown','beta_to_qqq']), 'new sector comparison: realized return, volatility and drawdown')
    figures['22_new_sector_risk'].update_traces(textposition='top center')
    daily_scores=pd.read_csv(root/'results/daily_metrics.csv')
    figures['23_daily_forecasting']=polish(px.bar(daily_scores.query('horizon_days == 1'),x='asset',y='mcc',color='model',facet_col='test_year',barmode='group'), 'daily direction forecasting: separate 2024 and 2025 walk forward tests')
    correlation=daily.pivot(index='date',columns='asset',values='close').pct_change(fill_method=None).corr()
    figures['24_daily_correlation']=polish(px.imshow(correlation,zmin=-1,zmax=1,text_auto='.2f',color_continuous_scale='RdBu',aspect='auto'), 'daily return correlations: sector labels do not guarantee diversification')
    figures['24_daily_correlation'].update_layout(height=680)
    published=pd.read_csv(root/'data/processed/paper_reference_mcc.csv')
    comparison=published.query('asset in ["aapl","goog"]').groupby(['asset','horizon'],as_index=False).published_mcc.mean().rename(columns={'published_mcc':'mcc'})
    comparison['experiment']='published 2017 to 2019'
    sample=base.query('asset in ["aapl","goog"] and model == "deeplob"')[['asset','horizon','mcc']].copy()
    sample['experiment']='public sample 2012, reduced training'
    figures['25_paper_reference']=polish(px.bar(pd.concat([comparison,sample]),x='asset',y='mcc',color='experiment',facet_col='horizon',barmode='group'), 'published reference versus sample experiment: different dates and budgets')
    table=base.query('horizon == 50').groupby('model',as_index=False)[['mcc','macro_f1','accuracy','p_t']].mean()
    figures['26_metric_table']=plotly_table(table,'h50 model comparison: equal weight per asset, descriptive only')
    budget=pd.read_csv(root/'results/budget_sensitivity.csv')
    figures['27_training_budget']=polish(px.bar(budget,x='asset',y='mcc',color='budget',facet_col='model',barmode='group'), 'training budget sensitivity: early stopping and larger balanced samples')
    pd.DataFrame([dict(name=name,traces=len(fig.data)) for name,fig in figures.items()]).to_csv(root/'results/figure_inventory.csv',index=False)
    return figures


def export_gallery(figures):
    folder=root/'figures'
    folder.mkdir(exist_ok=True)
    fragments=[]
    for i,(name,fig) in enumerate(figures.items()):
        fig.write_json(folder/f'{name}.json')
        fragment=fig.to_html(full_html=False,include_plotlyjs=True if i==0 else False,div_id=name)
        fragments.append(f'<section id="section-{name}">{fragment}</section>')
    nav=''.join(f'<a href="#section-{name}">{name.replace("_"," ")}</a>' for name in figures)
    html='''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>lob forecasting research notebook</title>
<style>body{margin:0;background:#f8fafc;color:#17283b;font:16px Arial,sans-serif}header{padding:45px 5%;background:#17283b;color:white}h1{font-size:34px;font-weight:500}p{max-width:1000px;line-height:1.6}nav{padding:25px 5%;display:flex;gap:10px;flex-wrap:wrap}a{color:#2563eb;text-decoration:none;background:#e7eef7;padding:8px;border-radius:6px;font-size:12px}section{margin:22px 3%;padding:12px;background:white;border-radius:12px;border:1px solid #dde5ee}footer{padding:30px 5%}</style></head><body>
<header><h1>deep limit order book forecasting</h1><p>public data reproduction and implemented future directions. all charts are interactive and embedded for offline use. the 2012 stock sample and 2024 crypto prefixes are reduced experiments; they do not reproduce the licensed 2017–2019 results. daily stock bars are a separate analysis.</p></header>'''+f'<nav>{nav}</nav>'+''.join(fragments)+'''<footer>source: briola, bartolucci and aste (2025), doi: 10.1080/14697688.2025.2522911. see the notebook for methods, assumptions, source links and verification.</footer></body></html>'''
    (root/'research_dashboard.html').write_text(html)
    return root/'research_dashboard.html'
