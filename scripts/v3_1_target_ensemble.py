#!/usr/bin/env python3
"""All-state conditional population ensemble, count moments and numerical transfers."""
import argparse
import json
import os
import numpy as np
from scipy.special import logsumexp,softmax
from v2_1_near import ROOT,load,sha,save_json,csv_rows,atomic_npz
from photdis.stage1 import require_compute
from photdis.v3_1.count_mixture import mix_counts,realized_count_intervals


def main():
    require_compute();p=argparse.ArgumentParser();p.add_argument('--case',choices=['broad','candidate','holdout'],required=True);a=p.parse_args()
    root=ROOT/'data/v3_1';out=root/'target_ensemble'/a.case;out.mkdir(parents=True,exist_ok=True)
    paths=[ROOT/'scripts/v3_1_target_ensemble.py',ROOT/'src/photdis/v3_1/count_mixture.py',ROOT/'docs/v3_1_target_ensemble_protocol.md',
        root/'luminosity_fit_refined/complete.json',root/'luminosity_fit_refined/parameters.npz']
    inputs={str(path.relative_to(ROOT)):sha(path) for path in paths};correction_done=json.loads(paths[-2].read_text())
    assert correction_done['outputs']['parameters.npz']==sha(paths[-1]);correction=load(paths[-1])
    parameters=np.concatenate([np.zeros((1,3,2)),correction['point'][None],correction['bootstrap']]);assert parameters.shape==(34,3,2)
    selected=[];missing=[];inventory=[]
    for index in range(34):
        marker=root/f'target_refinement/{a.case}/correction_{index:02}/complete.json'
        if not marker.exists():missing.append(index);continue
        r=json.loads(marker.read_text());assert r['case']==a.case and r['correction']==index
        inputs[str(marker.relative_to(ROOT))]=sha(marker);last=r['attempts'][-1];fit=ROOT/last['fit'];assert sha(fit)==last['fit_sha256']
        fd=json.loads(fit.read_text());assert fd['case']==a.case and fd['correction']==index and fd['draws']==r['selected_draws']
        row=dict(correction=index,draws=fd['draws'],passed=r['passed'],fit=str(fit.relative_to(ROOT)),fit_sha256=sha(fit),
            population=str((fit.parent/'population.npz').relative_to(ROOT)),population_sha256=fd['outputs']['population.npz'],
            parameters=parameters[index].tolist(),sampler_passed=fd['sampler_passed'],bridge_passed=fd['bridge_passed'])
        if 'counts' in last:
            cp=ROOT/last['counts'];assert sha(cp)==last['counts_sha256'];cd=json.loads(cp.read_text())
            assert cd['case']==a.case and cd['correction']==index and cd['inputs'][str(fit.relative_to(ROOT))]==sha(fit)
            for name,digest in cd['outputs'].items():assert sha(cp.parent/name)==digest
            row.update(counts=str(cp.relative_to(ROOT)),counts_sha256=sha(cp),counts_passed=cd['passed'])
        if r['passed']:
            assert fd['sampler_passed'] and fd['bridge_passed'] and row['counts_passed']
            row['log_evidence']=fd['bridge']['log_evidence']
        inventory.append({k:v for k,v in row.items() if k not in ('parameters',)})
        selected.append(row)
    csv_rows(out/'inventory.csv',inventory)
    if missing:
        save_json(out/'progress.json',dict(inputs=inputs,missing=missing,complete=False,job=os.getenv('SLURM_JOB_ID')))
        raise RuntimeError('All34registered outcomes required; no incomplete ensemble normalization')
    assert [r['correction'] for r in selected]==list(range(34))
    freeze=out/'inputs.json'
    if freeze.exists():assert json.loads(freeze.read_text())==inputs
    else:save_json(freeze,inputs)
    save_json(out/'states.json',dict(case=a.case,states=selected,response_draw=12,calibration_draw=12,within_class_tilts=False,
        main_final_candidate_case='candidate',main_view='bootstrap_evidence',references_read=False))
    variants=['base','entry_seed','entry_level','native_fine'];probability={};conditional_counts={};logevidence={};identity=None
    for row in selected:
        if not row['passed']:continue
        index=row['correction'];folder=(ROOT/row['counts']).parent;stars=load(folder/'per_star.npz');num=load(folder/'numerical_weights.npz')
        assert (stars['TYPE']=='PSF').all()
        if identity is None:identity=stars['identity'];rows=stars['rows'];types=stars['TYPE'];fields=stars['field'];audit=stars['fine_audited']
        else:
            for value,key in [(identity,'identity'),(rows,'rows'),(types,'TYPE'),(fields,'field'),(audit,'fine_audited')]:np.testing.assert_array_equal(value,stars[key])
        for variant in variants:
            key=(index,variant);probability[key]=stars[variant+'_probability'];conditional_counts[key]=load(folder/f'{variant}_counts.npz')
            lr=np.zeros(len(num['draw_ids'])) if variant=='base' else num[variant+'_logratio']
            logevidence[key]=row['log_evidence']+float(logsumexp(lr)-np.log(len(lr)))
            np.testing.assert_array_equal(conditional_counts[key]['draw_ids'],num['draw_ids'])
    views=[];checks=[];weight_rows=[];count_rows=[]
    for label,indices,evidence_weighted in [('zero',[0],True),('point',[1],True),('bootstrap_evidence',list(range(2,34)),True),('bootstrap_cut',list(range(2,34)),False)]:
        allpass=all(selected[i]['passed'] for i in indices);view=dict(view=label,states=indices,inputs_passed=allpass,accepted=False)
        if not allpass:
            view['failed_states']=[i for i in indices if not selected[i]['passed']];views.append(view);continue
        folder=out/label;folder.mkdir(exist_ok=True);means={};results={};ws={};zs={};good=True
        for variant in variants:
            le=np.array([logevidence[i,variant] for i in indices]);lw=le-logsumexp(le) if evidence_weighted else np.full(len(indices),-np.log(len(indices)))
            w=np.exp(lw);ws[variant]=w;zs[variant]=float(logsumexp(le)-np.log(len(le)))
            mean=np.einsum('s,sncd->ncd',w,np.array([probability[i,variant] for i in indices]));means[variant]=mean
            result=mix_counts([conditional_counts[i,variant] for i in indices],w,indices);results[variant]=result
            np.testing.assert_allclose(mean.sum(axis=0).reshape(-1),result['mean'],atol=1e-7,rtol=1e-12)
            assert abs(result['covariance'].sum(axis=1)).max()<1e-6 and np.linalg.eigvalsh(result['covariance']).min()>-1e-6
            intervals,half=realized_count_intervals(result)
            atomic_npz(folder/f'{variant}_counts.npz',**result,intervals95=intervals,count_CDF_half_delta=np.array(half))
            atomic_npz(folder/f'{variant}_probabilities.npz',identity=identity,TYPE=types,rows=rows,field=fields,fine_audited=audit,
                class_distance=mean,qnear=mean[:,:,:3].sum(axis=(1,2)),state_ids=np.array(indices),state_weights=w,state_log_weights=lw)
            weight_rows.extend(dict(view=label,variant=variant,correction=int(i),weight=float(v),log_weight=float(l),log_evidence=float(e)) for i,v,l,e in zip(indices,w,lw,le))
            pd=float(abs(mean-means['base']).max());cd=float(abs(result['mean']-results['base']['mean']).max())
            wd=float(abs(w-ws['base']).max());zd=zs[variant]-zs['base']
            # Evidence changes do not change an explicitly cut state distribution.
            passed=pd<.003 and cd<.5 and wd<.01 and (abs(zd)<.05 or not evidence_weighted)
            good &= passed
            q=mean[:,:,:3].sum(axis=(1,2));qbase=means['base'][:,:,:3].sum(axis=(1,2))
            checks.append(dict(view=label,variant=variant,maximum_probability_delta=pd,maximum_count_mean_delta=cd,
                maximum_state_weight_delta=wd,log_evidence_delta=zd,evidence_weighted=evidence_weighted,
                q95_crossings=int(((q>.95)!=(qbase>.95)).sum()),count_CDF_half_delta=half,passed=bool(passed)))
            for k,(lo,median,hi) in enumerate(intervals):
                count_rows.append(dict(view=label,variant=variant,class_index=k//6,distance_bin=k%6,n=int(result['n']),mean=float(result['mean'][k]),
                    lower95=int(lo),median=int(median),upper95=int(hi),variance=float(result['covariance'][k,k]),
                    conditional_variance=float(result['conditional_covariance'][k,k]),within_state_shared_variance=float(result['within_state_shared_covariance'][k,k]),
                    between_state_variance=float(result['between_state_covariance'][k,k]),count_CDF_half_delta=half))
        w=ws['base'];view.update(accepted=bool(good),n=len(identity),fine_audited=int(audit.sum()),state_ESS=float(1/(w@w)),
            maximum_state_weight=float(w.max()),top_state=int(indices[w.argmax()]),q95_selected=int((means['base'][:,:,:3].sum(axis=(1,2))>.95).sum()))
        save_json(folder/'complete.json',dict(inputs=inputs,view=view,outputs={p.name:sha(p) for p in folder.iterdir() if p.suffix=='.npz'},references_read=False))
        views.append(view)
    csv_rows(out/'weights.csv',weight_rows);csv_rows(out/'checks.csv',checks);csv_rows(out/'counts.csv',count_rows);save_json(out/'views.json',views)
    save_json(out/'complete.json',dict(inputs=inputs,case=a.case,states=34,all_PSF=True,views=views,
        passed=all(v['accepted'] for v in views),main_candidate_accepted=next(v['accepted'] for v in views if v['view']=='bootstrap_evidence'),
        outputs={p.name:sha(p) for p in out.iterdir() if p.name in ('inventory.csv','states.json','weights.csv','checks.csv','counts.csv','views.json')},
        view_markers={f'{v["view"]}/complete.json':sha(out/v['view']/'complete.json') for v in views if v['inputs_passed']},
        references_read=False,job=os.getenv('SLURM_JOB_ID'),scope='Observed cohort only; full conditional/shared/calibration count covariance. Finite bootstrap evidence and cut sensitivity distinct. High-q subset covariance, heldout predictions and new validation remain separate required products.'))


if __name__=='__main__':main()
