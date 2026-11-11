#!/usr/bin/env python3
"""Matched information, score-tail and near-selection audits without refitting."""
import json
import os
import numpy as np
from v2_1_near import ROOT,load,sha,save_json,csv_rows
from photdis.stage1 import require_compute
from photdis.v3.controls import probability_metrics,reliability_rows
from photdis.v3.metrics import distance_summary
D=ROOT/'data/v3_0'


def read_checked(folder):
    m=json.loads((folder/'complete.json').read_text());p=folder/'per_star.npz';assert sha(p)==m['outputs'][p.name]
    return load(p)


def paired_interval(values,cells):
    groups=np.unique(cells)
    if len(groups)<2:return dict(cells=len(groups),lo95=None,hi95=None)
    sums=np.array([values[cells==c].sum() for c in groups]);sizes=np.array([(cells==c).sum() for c in groups])
    index=np.random.default_rng(309003).integers(0,len(groups),(512,len(groups)))
    means=sums[index].sum(axis=1)/sizes[index].sum(axis=1)
    return dict(cells=len(groups),lo95=float(np.quantile(means,.025)),hi95=float(np.quantile(means,.975)))


def main():
    require_compute();out=D/'diagnostics';out.mkdir(exist_ok=True)
    views={v:read_checked(D/'evaluation_results'/f'development_{v}_fit_0') for v in ('grz','griz','multiband')}
    qrf=read_checked(D/'evaluation_results/development_qrf_0');base=views['grz'];ids=base['identity']
    for t in [qrf,*views.values()]:
        np.testing.assert_array_equal(t['identity'],ids);assert (t['TYPE']=='PSF').all()
    p=load(D/'controls/probe_development.npz');known=base['class_label']>=0
    np.testing.assert_array_equal(p['identity'],ids[known]);assert (p['TYPE']=='PSF').all()
    groups=dict(all=np.ones(len(ids),bool),dwarf=base['class_label']==0,evolved=base['class_label']==1,
        other=base['class_label']==2,unknown=~known,red=base['gr']>1.2,
        poor_dwarf=(base['class_label']==0)&base['corrected_feh_valid']&(base['corrected_feh']< -1.5))
    for lo,hi in [(0,18),(18,20),(20,22),(22,99)]:groups[f'r_{lo}_{hi}']=(base['r']>=lo)&(base['r']<hi)
    t=load(D/'cohorts/development.npz');np.testing.assert_array_equal(t['identity'],ids)
    groups.update(weak_W1=t['band_state'][:,4]==3,no_W1=~t['band_mask'][:,4],all_optional=t['band_mask'][:,[2,4,5]].all(axis=1))
    class_rows=[];paired=[];reliability=[]
    probabilities={f'probe_{v}':p[v] for v in ('no_color','grz','griz','multiband')}
    probabilities.update({f'v3_{v}':t['class_probability'][known] for v,t in views.items()})
    labels=base['class_label'][known]
    baseline=probabilities['probe_no_color'];base_score=np.log(np.maximum(baseline[np.arange(len(labels)),labels],1e-300))
    for method,prob in probabilities.items():
        score=np.log(np.maximum(prob[np.arange(len(labels)),labels],1e-300))
        for group,use in groups.items():
            mask=use[known]
            class_rows.append(dict(method=method,group=group,**probability_metrics(prob[mask],labels[mask])))
            if mask.any():paired.append(dict(method=method,baseline='probe_no_color',group=group,n=int(mask.sum()),
                logscore_gain=float((score-base_score)[mask].mean()),**paired_interval((score-base_score)[mask],base['cell'][known][mask])))
        reliability.extend(dict(method=method,**r) for r in reliability_rows(prob,labels))
    tails=[];near=[];band=[]
    models={'qrf':qrf,**views}
    for model,x in models.items():
        for source in ('primary','specdis','spdist','gaia_precise'):
            valid=x[source+'_valid'];reference=x[source+'_mu'];distance=10**((reference-10)/5)
            for group,use in groups.items():
                mask=use&valid
                if mask.any():
                    value=x[source+'_logscore'][mask];old=qrf[source+'_logscore'][mask];delta=value-old
                    order=np.argsort(-abs(delta));cut=max(1,int(np.ceil(.01*len(delta))))
                    tails.append(dict(model=model,reference=source,group=group,n=len(value),mean=float(value.mean()),
                        median=float(np.median(value)),q01=float(np.quantile(value,.01)),q99=float(np.quantile(value,.99)),
                        below_minus20=int((value< -20).sum()),gain_mean=float(delta.mean()),gain_median=float(np.median(delta)),
                        top1percent_absolute_gain_contribution=float(delta[order[:cut]].sum()/len(delta)),
                        remainder_gain_mean=float(delta[order[cut:]].mean()) if len(delta)>cut else None))
                for threshold in (.5,.8,.95):
                    selected=mask&(x['qnear']>threshold);n=int(selected.sum());refnear=mask&(distance<10)
                    near.append(dict(model=model,reference=source,group=group,threshold=threshold,n=n,
                        reference_near_eligible=int(refnear.sum()),near_retained=int((selected&refnear).sum()),
                        far_contaminants=int((selected&(distance>=10)).sum()),
                        far_fraction=float((distance[selected]>=10).mean()) if n else None,
                        **{k:v for k,v in distance_summary(x['mu'][selected],reference[selected]).items() if k!='n'}))
    for old,new in [('grz','griz'),('grz','multiband')]:
        for group,use in groups.items():
            for reference in ('primary','specdis','spdist','gaia_precise'):
                mask=use&base[reference+'_valid']
                if not mask.any():continue
                delta=views[new][reference+'_logscore'][mask]-views[old][reference+'_logscore'][mask]
                band.append(dict(old=old,new=new,group=group,reference=reference,n=int(mask.sum()),mean_logscore_gain=float(delta.mean()),
                    median_logscore_gain=float(np.median(delta)),**paired_interval(delta,base['cell'][mask])))
    for name,rows in [('information',class_rows),('information_gain',paired),('reliability',reliability),('score_tails',tails),('near_selection',near),('band_gains',band)]:csv_rows(out/(name+'.csv'),rows)
    paths=[ROOT/'scripts/v3_0_diagnostics.py',D/'controls/probe_development.npz']+[D/'evaluation_results'/f'development_{v}_fit_0/complete.json' for v in views]+[D/'evaluation_results/development_qrf_0/complete.json']
    save_json(out/'complete.json',dict(inputs={str(p.relative_to(ROOT)):sha(p) for p in paths},n=len(ids),known_labels=int(known.sum()),
        outputs={p.name:sha(p) for p in out.glob('*.csv')},sky_bootstrap_replicates=512,
        scope='Identical seen development sources and empirical class composition; paired sky bootstrap, not independent absolute truth or new blind validation.',job=os.getenv('SLURM_JOB_ID')))
    print('MATCHED DIAGNOSTICS COMPLETE',len(ids),flush=True)


if __name__=='__main__':main()
