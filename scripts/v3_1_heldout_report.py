#!/usr/bin/env python3
"""Complete-state heldout-sky transfer and shared predictive support report."""
import json
import os
import numpy as np
from scipy.special import logsumexp
from v2_1_near import ROOT,load,sha,save_json,csv_rows,atomic_npz
from photdis.stage1 import require_compute
from photdis.v3_1.count_mixture import mix_counts,realized_count_intervals
from photdis.v3_1.predictive import mix_predictive
from photdis.v3_1.comparison import paired_mean


def main():
    require_compute();root=ROOT/'data/v3_1';out=root/'heldout_transfer/report';out.mkdir(parents=True,exist_ok=True)
    paths=[ROOT/'scripts/v3_1_heldout_report.py',ROOT/'src/photdis/v3_1/count_mixture.py',ROOT/'src/photdis/v3_1/predictive.py',
        ROOT/'src/photdis/v3_1/comparison.py',ROOT/'docs/v3_1_heldout_transfer_protocol.md']
    inputs={str(p.relative_to(ROOT)):sha(p) for p in paths};missing=[];inventory=[];source={};ensembles={}
    for case in ['broad','candidate','holdout']:
        p=root/f'target_ensemble/{case}/complete.json'
        if not p.exists():missing.append(f'ensemble/{case}')
        else:inputs[str(p.relative_to(ROOT))]=sha(p);ensembles[case]=json.loads(p.read_text())
        for index in range(34):
            p=root/f'heldout_transfer/{case}/correction_{index:02}/complete.json'
            if not p.exists():missing.append(f'{case}/{index}');continue
            r=json.loads(p.read_text());assert r['case']==case and r['correction']==index;source[case,index]=r;inputs[str(p.relative_to(ROOT))]=sha(p)
            if r['available']:
                for name,digest in r['outputs'].items():assert sha(p.parent/name)==digest
                assert r['all_PSF'] and r['population_fit_excludes_heldout']==(case=='holdout')
            inventory.append(dict(case=case,correction=index,available=r['available'],passed=r['passed']))
    csv_rows(out/'inventory.csv',inventory)
    if missing:
        save_json(out/'progress.json',dict(inputs=inputs,missing=missing,complete=False));raise RuntimeError('Complete102state outcomes required; no missing-state normalization')
    summaries=[];count_rows=[];field_rows=[];checks=[];view_status=[];base_objects={}
    for case,ensemble in ensembles.items():
        for view in ensemble['views']:
            label=view['view'];indices=view['states'];available=view['inputs_passed'] and all(source[case,i]['available'] for i in indices)
            if not available:
                view_status.append(dict(case=case,view=label,available=False,accepted=False));continue
            inherited=view['accepted'] and all(source[case,i]['passed'] for i in indices)
            vm=root/f'target_ensemble/{case}/{label}/complete.json';assert sha(vm)==ensemble['view_markers'][f'{label}/complete.json'];vd=json.loads(vm.read_text())
            for envelope in ['broad','candidate']:
                folder=out/f'{case}_{label}_{envelope}';folder.mkdir(exist_ok=True);means={};results={};scores={};good=inherited
                for variant in ['base','entry_seed','entry_level','native_fine']:
                    vp=vm.parent/f'{variant}_probabilities.npz';assert sha(vp)==vd['outputs'][vp.name];wdata=load(vp)
                    np.testing.assert_array_equal(wdata['state_ids'],indices);weight=wdata['state_weights'];logweight=wdata['state_log_weights']
                    objects=[];counts=[];pred=[]
                    for index in indices:
                        p=root/f'heldout_transfer/{case}/correction_{index:02}';objects.append(load(p/f'{envelope}_{variant}_objects.npz'))
                        counts.append(load(p/f'{envelope}_{variant}_counts.npz'));pred.append(load(p/f'{envelope}_{variant}_predictive.npz'))
                    first=objects[0]
                    for value in objects:
                        for key in ['identity','TYPE','rows','field','fine_audited','design_weight']:np.testing.assert_array_equal(first[key],value[key])
                        assert (value['TYPE']=='PSF').all()
                    mean=np.einsum('s,sncd->ncd',weight,np.array([r['class_distance'] for r in objects]));means[variant]=mean
                    point=logsumexp(logweight[:,None]+np.array([r['pointwise_log_predictive'] for r in objects]),axis=0);scores[variant]=point
                    result=mix_counts(counts,weight,indices);results[variant]=result;predictive=mix_predictive(pred,weight)
                    np.testing.assert_allclose(mean.sum(axis=0).reshape(-1),result['mean'],atol=1e-7,rtol=1e-12)
                    interval,half=realized_count_intervals(result)
                    atomic_npz(folder/f'{variant}_counts.npz',**result,intervals95=interval,count_CDF_half_delta=np.array(half))
                    merged={k:first[k] for k in ['identity','TYPE','rows','field','fine_audited','design_weight']};merged.update(class_distance=mean,pointwise_log_predictive=point)
                    atomic_npz(folder/f'{variant}_objects.npz',**merged)
                    if variant=='base':base_objects[case,label,envelope]=merged
                    pd=float(abs(mean-means['base']).max()) if len(mean) else 0.;sd=float(abs(point-scores['base']).max()) if len(point) else 0.;nd=float(abs(result['mean']-results['base']['mean']).max())
                    passed=pd<.003 and sd<.02 and nd<.5;good &= passed
                    checks.append(dict(case=case,view=label,envelope=envelope,variant=variant,maximum_probability_delta=pd,maximum_pointwise_logscore_delta=sd,maximum_count_mean_delta=nd,passed=bool(passed)))
                    summaries.append(dict(case=case,view=label,envelope=envelope,variant=variant,population_fit_excludes_heldout=case=='holdout',
                        pointwise_logscore_sum=float(point.sum()),design_mean_pointwise_logscore=float(first['design_weight']@point/len(point)) if len(point) else None,
                        fine_audited=int(first['fine_audited'].sum()),**predictive))
                    for k,(lo,median,hi) in enumerate(interval):
                        count_rows.append(dict(case=case,view=label,envelope=envelope,variant=variant,class_index=k//6,distance_bin=k%6,n=int(result['n']),mean=float(result['mean'][k]),
                            lower95=int(lo),median=int(median),upper95=int(hi),variance=float(result['covariance'][k,k]),conditional_variance=float(result['conditional_covariance'][k,k]),shared_variance=float(result['shared_covariance'][k,k]),count_CDF_half_delta=half))
                    if variant=='base':
                        for f in np.unique(first['field']):
                            mask=first['field']==f;w=first['design_weight'][mask];w/=w.sum();prob=np.einsum('n,ncd->cd',w,mean[mask])
                            field_rows.append(dict(case=case,view=label,envelope=envelope,field=int(f),n=int(mask.sum()),
                                mean_logscore=float(w@point[mask]),qnear=float(prob[:,:3].sum()),**{f'class{c}_distance{d}':float(prob[c,d]) for c in range(3) for d in range(6)}))
                status=dict(case=case,view=label,envelope=envelope,available=True,accepted=bool(good),inherited_inputs_passed=bool(inherited))
                view_status.append(status);save_json(folder/'complete.json',dict(inputs=inputs,status=status,outputs={p.name:sha(p) for p in folder.iterdir() if p.suffix=='.npz'}))
    paired=[]
    for label in ['zero','point','bootstrap_evidence','bootstrap_cut']:
        for envelope in ['broad','candidate']:
            reference=base_objects.get(('holdout',label,envelope))
            if reference is None:continue
            for case in ['broad','candidate']:
                other=base_objects.get((case,label,envelope))
                if other is None:continue
                np.testing.assert_array_equal(other['identity'],reference['identity']);weight=reference['design_weight'];cells=reference['field']
                metrics={'pointwise_logscore_gain':other['pointwise_log_predictive']-reference['pointwise_log_predictive'],
                    'qnear_shift':other['class_distance'][:,:,:3].sum(axis=(1,2))-reference['class_distance'][:,:,:3].sum(axis=(1,2))}
                metrics.update({f'class{c}_distance{d}_shift':other['class_distance'][:,c,d]-reference['class_distance'][:,c,d] for c in range(3) for d in range(6)})
                for metric,value in metrics.items():
                    if len(value):paired.append(dict(case=case,baseline='holdout',view=label,envelope=envelope,metric=metric,**paired_mean(value,weight,cells)))
    csv_rows(out/'predictive_summary.csv',summaries);csv_rows(out/'counts.csv',count_rows);csv_rows(out/'field_summary.csv',field_rows);csv_rows(out/'checks.csv',checks);csv_rows(out/'paired.csv',paired)
    save_json(out/'views.json',view_status)
    save_json(out/'complete.json',dict(inputs=inputs,states=102,all_PSF=True,views=view_status,passed=all(v['accepted'] for v in view_status),
        outputs={p.name:sha(p) for p in out.iterdir() if p.suffix=='.csv' or p.name=='views.json'},references_read=False,job=os.getenv('SLURM_JOB_ID'),
        scope='Three fixed skyfields; broad/candidate fits include their data, only heldout fit excludes them. Shared unit-catalog and generalized-design photometric scores separate; support failures retained. Cut class/distance counts are not true-distance validation or independent contamination.'))


if __name__=='__main__':main()
