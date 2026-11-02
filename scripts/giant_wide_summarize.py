#!/usr/bin/env python3
"""Summarize frozen-model comparison without optimizing on evaluation scores."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'.deps')]
import numpy as np
import fitsio
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from photdis.giant_information import wilson
from photdis.stage1 import require_compute
OUT=Path('data/giant_wide_v1');FIG=OUT/'figures'
NAMES=['B-K','A-wide','B-wide'];COLORS=['#999999','#e69f00','#0072b2']


def identity_text(value):
    return (value.decode() if isinstance(value, bytes) else str(value)).strip()


def sum_counts(parts):
    return wilson(sum(p['success'] for p in parts),sum(p['n'] for p in parts))


def paired_bootstrap(folds,domain,key,category,stratum,first,second):
    values=[]
    for r in folds:
        a=r['models'][first]['domains'][domain]['test'][key]['strata'][stratum].get(category,{'n':0,'success':0})
        b=r['models'][second]['domains'][domain]['test'][key]['strata'][stratum].get(category,{'n':0,'success':0})
        if a['n'] and b['n']:values.append([a['success'],a['n'],b['success'],b['n']])
    if not values:return None
    a=np.asarray(values);rng=np.random.default_rng(20261102);draw=a[rng.integers(len(a),size=(2000,len(a)))].sum(axis=1)
    delta=draw[:,2]/draw[:,3]-draw[:,0]/draw[:,1]
    return {'first':first,'second':second,'n_fields':len(a),
            'pooled_delta':float(a[:,2].sum()/a[:,3].sum()-a[:,0].sum()/a[:,1].sum()),
            'field_bootstrap_interval95':np.quantile(delta,[.025,.975]).tolist()}


def main():
    require_compute();FIG.mkdir(parents=True,exist_ok=True)
    source=json.loads((OUT/'results.json').read_text());folds=source['folds'];aggregate={}
    for domain in ('common_K','own'):
        aggregate[domain]={}
        for key in ('0.5','0.01','0.001'):
            aggregate[domain][key]={}
            for name in NAMES:
                inputs=[r['models'][name]['domains'][domain]['test'][key]['strata'] for r in folds]
                agg={}
                for stratum in inputs[0]:
                    categories=set().union(*(p[stratum] for p in inputs))
                    agg[stratum]={cat:sum_counts([p[stratum].get(cat,{'n':0,'success':0}) for p in inputs]) for cat in sorted(categories)}
                rates=agg['all'];tpr=rates['K_giant_total']['rate'];fpr=rates['all_dwarf_total']['rate']
                aggregate[domain][key][name]={'strata':agg,'efficiency':tpr/fpr if fpr else None,
                     'hypothetical_two_class_purity_at_one_percent':.01*tpr/(.01*tpr+.99*fpr) if tpr+fpr else None}
    pairs={}
    for key in ('0.5','0.01','0.001'):
        pairs[key]={}
        for cat in ('K_giant_total','all_dwarf_total','warm_dwarf'):
            pairs[key][cat]={f'{a}_to_{b}':paired_bootstrap(folds,'common_K',key,cat,'all',a,b) for a,b in [('B-K','B-wide'),('A-wide','B-wide')]}
    # Population evidence for actual held-out selected K giants, not all selected giants.
    with np.load('data/giant_information_v1/labels.npz') as f:t={k:f[k] for k in f.files}
    # Reuse identity-audited small VAC joins; never infer distance from class score.
    vac_paths=[Path('data/giant_vac_v3/same_star.fits'),Path('data/giant_ranked_vac_v1/same_star.fits')]
    vac={};ambiguous=set();vac_hashes={}
    for path in vac_paths:
        if not path.exists():continue
        vac_hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        for row in fitsio.read(str(path),columns=['source_id','desi_object_id','specdis_valid','spdist_valid','specdis_dist_kpc','spdist_dist_kpc']):
            key=(int(row['source_id']),identity_text(row['desi_object_id']))
            if key in vac:ambiguous.add(key)
            else:vac[key]=row
    candidates={}
    for key in ('0.5','0.01','0.001'):
        selected=[]
        for r in folds:
            with np.load(OUT/f"{r['test_field']}_scores.npz") as f:
                rows=f['rows'];cut=r['models']['B-wide']['domains']['common_K']['thresholds'][key]['threshold']
                use=(f['score_B-wide']>=cut)&~f['outside_B-wide']&(t['label'][rows]==1)&(t['desi_teff'][rows]>=4200)&(t['desi_teff'][rows]<5000)
                selected.extend(rows[use].tolist())
        ids=np.array(selected,int);params={}
        for col in ('desi_teff','desi_logg','feh_corrected','r'):
            good=np.isfinite(t[col][ids])
            if col=='feh_corrected':good &= t['abundance_calibration_valid'][ids]
            v=t[col][ids][good];params[col]={'n_valid':len(v),'q16_50_84':np.quantile(v,[.16,.5,.84]).tolist() if len(v) else None}
        valid=t['abundance_calibration_valid'][ids]&np.isfinite(t['feh_corrected'][ids])
        vac_pairs=[vac.get((int(t['source_id'][i]),identity_text(t['desi_object_id'][i]))) for i in ids
               if (int(t['source_id'][i]),identity_text(t['desi_object_id'][i])) not in ambiguous]
        vac_pairs=[row for row in vac_pairs if row is not None]
        teacher={kind:{'valid_n':int(sum(row[kind+'_valid'] for row in vac_pairs)),
                       'dist_kpc_q16_50_84':np.quantile([row[kind+'_dist_kpc'] for row in vac_pairs if row[kind+'_valid']],[.16,.5,.84]).tolist() if any(row[kind+'_valid'] for row in vac_pairs) else None}
                 for kind in ('spdist','specdis')}
        candidates[key]={'vac_unique_pair_matches':len(vac_pairs),'spectroscopic_teachers':teacher,
            'vac_source_hashes':vac_hashes,'n':len(ids),'fields':{str(f):int((t['field_index'][ids]==f).sum()) for f in np.unique(t['field_index'][ids])},'parameters':params,
            'feh_lt_minus1p5':int((valid&(t['feh_corrected'][ids]<-1.5)).sum()),
            'feh_ge_minus1p5':int((valid&(t['feh_corrected'][ids]>=-1.5)).sum()),
            'phase_confirmed_n':None,'phase_status':'Gravity evidence does not identify RGB/RC/HB; no independent phase label in this dataset.'}
    result={'aggregate':aggregate,'paired_field_bootstrap':pairs,'selected_K_population':candidates,'source_hashes':source['source_hashes'],
            'n_fields':len(folds),'scope':source['scope'],'uncertainty':'Wilson describes source counts; field bootstrap descriptive with overlapping training folds'}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    # Comparison at predeclared operating points; all full-reference denominators.
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
    for ax,key in zip(axes,('0.5','0.01','0.001')):
        for i,(name,color) in enumerate(zip(NAMES,COLORS)):
            stats=aggregate['common_K'][key][name]['strata']['all']
            for j,cat in enumerate(('K_giant_total','all_dwarf_total','warm_dwarf')):
                v=stats[cat];p=v['rate'];lo,hi=v['interval95']
                ax.errorbar(j+(i-1)*.2,p,yerr=[[max(0,p-lo)],[max(0,hi-p)]],fmt='o',color=color,label=name if j==0 else None,capsize=3)
                ax.text(j+(i-1)*.2,p+.025,f"{v['success']}/{v['n']}",rotation=90,fontsize=6,ha='center',va='bottom')
        ax.set(xticks=[0,1,2],xticklabels=['K giant recall','Dwarf FPR','Warm dwarf FPR'],ylim=(-.02,1.2),title='Score 0.5' if key=='0.5' else f'Calibration target FPR {float(key):.1%}')
        ax.tick_params(axis='x',labelrotation=15);ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('Fixed-model wide-negative comparison | common K-training color box\nCounts use full W1 reference denominator; Wilson intervals are descriptive')
    fig.savefig(FIG/'01_operating_points.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for col,key in enumerate(('0.01','0.001')):
        for row,cat in enumerate(('K_giant_total','all_dwarf_total')):
            ax=axes[row,col]
            for name,color in zip(NAMES,COLORS):
                vv=[r['models'][name]['domains']['common_K']['test'][key]['strata']['all'][cat] for r in folds]
                ax.plot([r['test_field'] for r in folds],[v['rate'] if v['rate'] is not None else np.nan for v in vv],'-o',ms=4,color=color,label=name)
            ax.set(title=f'{cat} | target FPR {float(key):.1%}',xlabel='Held-out field index',ylabel='Observed rate');ax.grid(alpha=.2)
    axes[0,0].legend();fig.suptitle('Whole-field variation | same frozen model used for calibration and test')
    fig.savefig(FIG/'02_field_variation.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,stratum in zip(axes,('r16_18','r18_20')):
        for name,color in zip(NAMES,COLORS):
            values=[aggregate['common_K'][key][name]['strata'][stratum] for key in ('0.5','0.01','0.001')]
            x=[v['all_dwarf_total']['rate'] for v in values];y=[v['K_giant_total']['rate'] for v in values]
            ax.plot(x,y,'-o',color=color,label=name)
        ax.set(xscale='symlog',xlim=(-.0001,1),ylim=(-.02,1.02),xlabel='Actual dwarf FPR (symlog)',ylabel='K giant recall',title=stratum);ax.grid(alpha=.2)
    axes[0].legend();fig.suptitle('Fixed operating points by observed r | not a test-optimized ROC')
    fig.savefig(FIG/'03_brightness_tradeoff.png',dpi=170);plt.close(fig)

if __name__=='__main__':main()
