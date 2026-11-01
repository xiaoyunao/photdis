#!/usr/bin/env python3
"""Aggregate fixed outer predictions; ROC thresholds here are descriptive only."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'.deps')]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from photdis.giant_information import wilson,field_bootstrap,purity
from photdis.giant_candidate import feasibility
from photdis.stage1 import require_compute,write_json
plt.rcParams.update({'figure.dpi':140,'font.size':10})


def pool_summaries(summaries):
    keys=sorted(set().union(*(s['strata'].keys() for s in summaries)))
    result={}
    for key in keys:
        rows=[s['strata'][key] for s in summaries if key in s['strata']]
        categories=sorted(set().union(*(r.keys() for r in rows)))
        result[key]={c:wilson(sum(r[c]['success'] for r in rows if c in r),sum(r[c]['n'] for r in rows if c in r)) for c in categories}
    return result


def aggregate(result):
    cfg=result['parent_config'];output={'panels':{},'mechanism':{},'validation':{}}
    for panel in ('optical','common5','w1'):
        cells=[r for r in result['application'] if r['panel']==panel]
        models=sorted(set().union(*(r['models'].keys() for r in cells)))
        p={}
        for name in models:
            usable=[r['models'][name] for r in cells if name in r['models']]
            modes={'fixed_half':[m['fixed_half'] for m in usable], 'inside_half':[m['inside_half'] for m in usable]}
            for alpha in result['config']['target_fpr']:
                modes['nested_'+str(alpha)]=[m['nested_results'][str(alpha)] for m in usable if str(alpha) in m.get('nested_results',{})]
            p[name]={mode:pool_summaries(values) for mode,values in modes.items() if values}
            for mode in p[name].values():
                for row in mode.values():
                    row['two_class_K_purity_if_assumed_base_rate']=purity(row['K_giant_total']['rate'],row['all_dwarf_total']['rate'],cfg['purity_giant_fractions'])
        paired={}
        for name,base in [('B','A'),('C','B')]:
            modes=['fixed_half','inside_half']+['nested_'+str(a) for a in result['config']['target_fpr']]
            for mode in modes:
                def get(m):
                    return m.get('nested_results',{}).get(mode[7:]) if mode.startswith('nested_') else m.get(mode)
                for cls in ('K_giant_total','all_dwarf_total'):
                    pairs=[]
                    for r in cells:
                        if name not in r['models'] or base not in r['models']:continue
                        a,b=get(r['models'][name]),get(r['models'][base])
                        if a and b and a['strata']['all'][cls]['n']>=10:
                            pairs.append(a['strata']['all'][cls]['rate']-b['strata']['all'][cls]['rate'])
                    paired[f'{name}-{base}_{mode}_{cls}']=field_bootstrap(pairs,cfg['bootstrap_draws'],cfg['seed'])
        output['panels'][panel]={'models':p,'paired_field_changes':paired}
    for panel in result['config']['mechanism_panels']:
        cells=[r for r in result['mechanism'] if r['panel']==panel]
        output['mechanism'][panel]={}
        for name in ('spec','A','B','C'):
            usable=[r for r in cells if name in r['models']]
            if not usable:continue
            n=sum(r['n_test'] for r in usable)
            item={'n':n,'mean_log_density':sum(r['n_test']*r['models'][name]['mean_log_density'] for r in usable)/n}
            for key in ('coverage68','coverage95'):
                item[key]=wilson(sum(r['models'][name][key]['success'] for r in usable),n)
            if name!='spec':
                vals=[r['models'][name]['gain_vs_spec'] for r in usable if r['n_test']>=10]
                gain=field_bootstrap(vals,cfg['bootstrap_draws'],cfg['seed'])
                gain['stable_gain']=bool(gain['n_fields']>=3 and gain['interval95'][0]>0 and gain['positive_fraction']>=.75)
                item['equal_field_gain_vs_spec']=gain
            output['mechanism'][panel][name]=item
    checks=output['validation']
    checks['outer_fields_excluded']=all(r['field_index'] not in r['training_fields'] for r in result['application']+result['mechanism'])
    checks['inner_fields_isolated']=all(r['field_index'] not in x['training_fields']+x['held_fields'] and not set(x['training_fields']) & set(x['held_fields']) for r in result['application'] for x in r.get('inner_splits',[]))
    checks['fixed_W1_pair_counts']=all(r['models']['A']['fixed_half']['strata']['all']['reference_total']['n']==r['models']['B']['fixed_half']['strata']['all']['reference_total']['n'] for r in result['application'] if r['panel']=='w1' and r['models'])
    checks['calibration_empirical_FPR_budget']=all(v['calibration_false_positives']<=v['allowed_false_positives'] for r in result['application'] for m in r['models'].values() for v in m.get('nested_thresholds',{}).values() if v['threshold'] is not None)
    checks['population_partition']=all(sum(v['n'] for k,v in m['fixed_half']['strata']['all'].items() if k not in ('K_giant_total','all_dwarf_total','reference_total'))==r['n_reference'] for r in result['application'] for m in r['models'].values())
    assert all(checks.values()),checks
    return output


def curves(result,t,folder):
    output={}
    for panel,names in [('optical',['A']),('common5',['A','B','C']),('w1',['A','B'])]:
        output[panel]={}
        for name in names:
            rows,scores,outs=[],[],[]
            for r in result['application']:
                if r['panel']!=panel or name not in r['models']:continue
                with np.load(folder/f"{r['field_index']}_{panel}_scores.npz") as f:
                    rows.append(f['rows']);scores.append(f['score_'+name]);outs.append(f['outside_'+name])
            if not rows:continue
            rows=np.concatenate(rows);score=np.concatenate(scores);outside=np.concatenate(outs)
            label=t['label'][rows];k=(t['desi_teff'][rows]>=4200)&(t['desi_teff'][rows]<5000)
            use=(label==0)|((label==1)&k)
            fpr,tpr,threshold=roc_curve(label[use],np.where(outside[use],-1.,score[use]))
            physical=threshold>=0
            fpr,tpr=fpr[physical],tpr[physical]
            choose=np.unique(np.r_[np.arange(min(200,len(fpr))),np.linspace(0,len(fpr)-1,min(400,len(fpr))).astype(int)])
            output[panel][name]={'n_dwarfs':int((label==0).sum()),'n_K_giants':int(((label==1)&k).sum()),
                'fpr':fpr[choose].tolist(),'tpr':tpr[choose].tolist(),
                'pooled_descriptive_points':feasibility(t,rows,score,outside,result['config']['target_fpr'])}
    return output


def figures(result,summary,folder):
    folder.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,panel in zip(axes,('common5','w1')):
        for name,c in summary['roc'][panel].items():ax.plot(np.maximum(c['fpr'],1e-5),c['tpr'],label=name)
        ax.set(xscale='log',xlim=(1e-4,1),ylim=(0,1.02),xlabel='All qualified dwarf FPR',ylabel='K giant TPR',title=panel+'; in training color box')
        for a in (.001,.01):ax.axvline(a,color='gray',ls=':',lw=.8)
        ax.legend()
    fig.suptitle('Descriptive outer-score ROC; thresholds from this plot are not validated');fig.tight_layout();fig.savefig(folder/'01_low_fpr_roc.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    keys=['cool_giant','K_giant','warm_giant','cool_dwarf','K_dwarf','warm_dwarf','intermediate_or_transition','hot_qualified_unclassified','raw_nonstellar_unassessed','unknown']
    for ax,mode in zip(axes,['inside_half','nested_0.01']):
        data=summary['panels']['w1']['models']['B'][mode]['all']
        count=[data.get(k,{}).get('success',0) for k in keys]
        ax.barh(keys,count);ax.invert_yaxis();ax.set(xscale='symlog',xlabel='Selected reference objects',title='W1 B: '+mode)
        for i,n in enumerate(count):ax.text(max(n,1),i,str(n),va='center',fontsize=8)
    fig.suptitle('Selected spectral-reference composition; unknown is not a physical class');fig.tight_layout();fig.savefig(folder/'02_selected_composition.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for name,offset in [('A',-.08),('B',.08)]:
        cells=[r for r in result['application'] if r['panel']=='w1' and name in r['models']]
        for ax,cls in zip(axes,('K_giant_total','all_dwarf_total')):
            for r in cells:
                s=r['models'][name].get('nested_results',{}).get('0.01')
                if not s:continue
                v=s['strata']['all'][cls]
                if v['rate'] is None:continue
                ax.errorbar(r['field_index']+offset,v['rate'],yerr=[[v['rate']-v['interval95'][0]],[v['interval95'][1]-v['rate']]],fmt='o',color='C0' if name=='A' else 'C1',ms=4)
            ax.plot([],[],'o',color='C0' if name=='A' else 'C1',label=name)
            ax.set(xlabel='Outer held-out field',ylabel=cls+' selected fraction');ax.legend()
    axes[1].axhline(.01,color='gray',ls=':');fig.suptitle('Train-only 1% FPR thresholds applied outside training fields');fig.tight_layout();fig.savefig(folder/'03_nested_transfer.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5))
    for name,off in [('A',-.1),('B',0),('C',.1)]:
        for r in result['mechanism']:
            if r['panel']=='common5' and name in r['models']:
                ax.plot(r['field_index']+off,r['models'][name]['gain_vs_spec'],'o',color={'A':'C0','B':'C1','C':'C2'}[name],mfc={'A':'C0','B':'C1','C':'C2'}[name] if r['n_test']>=10 else 'none')
        ax.plot([],[],'o',label=name)
    ax.axhline(0,color='gray');ax.set(xlabel='Held-out field',ylabel='Mean log-score gain vs observed T/logg',title='FeH mechanism contrast; hollow = fewer than 10 test giants');ax.legend();fig.tight_layout();fig.savefig(folder/'04_feh_mechanism.png');plt.close(fig)
    return [str(p) for p in sorted(folder.glob('*.png'))]


def main():
    require_compute()
    config=json.loads(Path('configs/giant_candidate_v1.json').read_text());folder=Path(config['output_directory'])
    result=json.loads((folder/'results.json').read_text());summary=aggregate(result)
    with np.load(Path(config['parent_directory'])/'labels.npz',allow_pickle=False) as f:
        t={k:f[k] for k in ('label','desi_teff')}
    summary['roc']=curves(result,t,folder)
    summary['source_results_sha256']=hashlib.sha256((folder/'results.json').read_bytes()).hexdigest()
    summary['summary_code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary['figures']=figures(result,summary,Path('figures/giant_candidate_v1'))
    write_json(Path('manifests/giant_candidate_v1.json'),result)
    write_json(Path('manifests/giant_candidate_summary_v1.json'),summary)
    print(json.dumps(summary['validation']),flush=True)


if __name__=='__main__':main()
