#!/usr/bin/env python3
"""One frozen population state transferred to the three heldout sky fields."""
import argparse
import json
import os
import numpy as np
from scipy.special import softmax
from v2_1_near import ROOT,load,sha,save_json,csv_rows,atomic_npz
from photdis.stage1 import require_compute
from photdis.v3.population import SharedCounts
from photdis.v3_1.predictive import SharedPredictive

VARIANTS=['base','entry_seed','entry_level','native_fine']


def main():
    require_compute();p=argparse.ArgumentParser();p.add_argument('--case',choices=['broad','candidate','holdout'],required=True);p.add_argument('--correction',type=int,required=True);a=p.parse_args();assert 0<=a.correction<34
    root=ROOT/'data/v3_1';matrix=root/'target_matrices';jointroot=root/'target_joint_matrices';refinement=root/f'target_refinement/{a.case}/correction_{a.correction:02}/complete.json'
    out=root/f'heldout_transfer/{a.case}/correction_{a.correction:02}';out.mkdir(parents=True,exist_ok=True)
    rd=json.loads(refinement.read_text());assert rd['case']==a.case and rd['correction']==a.correction
    if not rd['passed']:
        save_json(out/'complete.json',dict(case=a.case,correction=a.correction,available=False,passed=False,source_refinement=str(refinement.relative_to(ROOT)),source_sha256=sha(refinement),reason='Conditional population/count gates failed; no state deletion or survivor normalization'))
        return
    last=rd['attempts'][-1];fit=ROOT/last['fit'];counts=ROOT/last['counts'];assert sha(fit)==last['fit_sha256'] and sha(counts)==last['counts_sha256']
    paths=[refinement,fit,fit.parent/'population.npz',counts,counts.parent/'numerical_weights.npz',matrix/'complete.json',matrix/'matrices.npz',
        jointroot/'complete.json',jointroot/'metadata.npz',jointroot/f'correction_{a.correction:02}.npz',
        ROOT/'scripts/v3_1_heldout_transfer.py',ROOT/'docs/v3_1_heldout_transfer_protocol.md',ROOT/'src/photdis/v3_1/predictive.py',ROOT/'src/photdis/v3/population.py']
    inputs={str(path.relative_to(ROOT)):sha(path) for path in paths};fd=json.loads(fit.read_text());cd=json.loads(counts.read_text());md=json.loads(paths[5].read_text());jd=json.loads(paths[7].read_text())
    assert fd['sampler_passed'] and fd['bridge_passed'] and cd['passed'] and md['passed'] and jd['passed']
    assert sha(paths[2])==fd['outputs']['population.npz'] and sha(paths[4])==cd['outputs']['numerical_weights.npz'] and sha(paths[6])==md['outputs']['matrices.npz']
    for path in paths[8:10]:assert sha(path)==jd['outputs'][path.name]
    marker=out/'complete.json'
    if marker.exists():
        r=json.loads(marker.read_text());assert r['inputs']==inputs
        for name,digest in r['outputs'].items():assert sha(out/name)==digest
        return
    freeze=out/'inputs.json'
    if freeze.exists():assert json.loads(freeze.read_text())==inputs
    else:save_json(freeze,inputs)
    population=load(paths[2]);weights=load(paths[4]);data=load(paths[6]);meta=load(paths[8]);joint=load(paths[9]);draw_ids=population['draw_ids'];s=len(draw_ids)
    np.testing.assert_array_equal(draw_ids,weights['draw_ids']);np.testing.assert_array_equal(meta['identity'],data['identity']);np.testing.assert_array_equal(meta['logscale'],data['logscale'])
    rows=np.flatnonzero(data['heldout']);identity=data['identity'][rows];types=data['TYPE'][rows];assert (types=='PSF').all() and (population['TYPE']=='PSF').all()
    if a.case=='holdout':assert not np.intersect1d(identity,population['identity']).size and not np.intersect1d(data['field'][rows],population['fields']).size
    candidate=data['candidate'][rows];field=data['field'][rows];fine_mask=data['fine_audited'][rows];fine_lookup={int(r):j for j,r in enumerate(meta['fine_rows'])}
    basejoint=joint['coarse_joint_response'][rows];finejoint=basejoint.copy()
    for k,row in enumerate(rows):
        if fine_mask[k]:finejoint[k]=joint['fine_joint_response'][fine_lookup[int(row)]]
    np.testing.assert_allclose(basejoint.sum(axis=-1),data['response'][rows,a.correction],atol=1e-12,rtol=1e-8)
    theta={};existing={}
    for f in np.unique(field):
        present=f in population['fields'];existing[int(f)]=present
        if present:value=population['theta'][:,int(np.flatnonzero(population['fields']==f)[0])]
        else:
            rng=np.random.default_rng(np.random.SeedSequence([320401,a.correction,int(f)]));local=population['global_logits']+.35*rng.normal(size=population['global_logits'].shape)
            value=softmax(np.c_[local,np.zeros(len(local))],axis=1).reshape(-1,3,5)
        theta[int(f)]=value
    masks={'broad':np.ones(len(rows),bool),'candidate':candidate};acc={};pred={};probability={};score={};design={}
    for label,mask in masks.items():
        design[label]=data['design_weight'][rows].copy()
        if mask.any():design[label]/=design[label][mask].mean()
        for variant in VARIANTS:
            key=label,variant;w=weights[variant+'_weight'];acc[key]=SharedCounts(draw_ids,18,w,seed=320431);pred[key]=SharedPredictive(w)
            probability[key]=np.full((len(rows),3,6),np.nan);score[key]=np.full(len(rows),np.nan)
    for f in np.unique(field):
        all_use=np.flatnonzero(field==f);t=theta[int(f)];flat=t.reshape(s,-1)
        for start in range(0,len(all_use),32):
            use=all_use[start:start+32];probs={};totals={}
            for name,response in [('base',basejoint),('native_fine',finejoint)]:
                value=np.einsum('scd,ncdb->sncb',t,response[use],optimize=True);total=value.sum(axis=(2,3));assert (total>0).all()
                probs[name]=value/total[:,:,None,None];totals[name]=total
            for label,mask in masks.items():
                pick=np.flatnonzero(mask[use]);selected=use[pick]
                if not len(pick):continue
                global_rows=rows[selected]
                for variant in VARIANTS:
                    key=label,variant;native='native_fine' if variant=='native_fine' else 'base'
                    suffix={'entry_seed':'selection_second_','entry_level':'selection_previous_'}.get(variant,'selection_')+label
                    alpha=data[suffix][global_rows,a.correction];entry=flat@alpha.reshape(len(selected),-1).T;assert (entry>0).all()
                    logdensity=np.log(totals[native][:,pick])-np.log(entry)+data['logscale'][global_rows][None]
                    score[key][selected]=pred[key].add(logdensity,design[label][selected])
                    probability[key][selected]=acc[key].add(probs[native][:,pick].reshape(s,len(pick),18),draw_ids).reshape(len(pick),3,6)
    checks=[];summaries=[]
    for label,mask in masks.items():
        base=acc[label,'base'].result()
        for variant in VARIANTS:
            key=label,variant;result=acc[key].result();predictive=pred[key].result();assert result['n']==int(mask.sum())==predictive['n']
            atomic_npz(out/f'{label}_{variant}_counts.npz',**result)
            atomic_npz(out/f'{label}_{variant}_predictive.npz',**predictive,draw_ids=draw_ids,draw_weights=weights[variant+'_weight'])
            atomic_npz(out/f'{label}_{variant}_objects.npz',identity=identity[mask],TYPE=types[mask],rows=rows[mask],field=field[mask],fine_audited=fine_mask[mask],
                design_weight=design[label][mask],class_distance=probability[key][mask],pointwise_log_predictive=score[key][mask])
            pd=float(abs(probability[key][mask]-probability[label,'base'][mask]).max()) if mask.any() else 0.
            sd=float(abs(score[key][mask]-score[label,'base'][mask]).max()) if mask.any() else 0.;nd=float(abs(result['mean']-base['mean']).max())
            passed=pd<.003 and sd<.02 and nd<.5 and result['marginal_half_delta']<.005
            checks.append(dict(envelope=label,variant=variant,maximum_probability_delta=pd,maximum_pointwise_logscore_delta=sd,maximum_count_mean_delta=nd,marginal_half_delta=result['marginal_half_delta'],passed=bool(passed)))
            summaries.append(dict(envelope=label,variant=variant,n=int(mask.sum()),fine_audited=int(fine_mask[mask].sum()),pointwise_logscore_sum=float(score[key][mask].sum()),
                design_mean_pointwise_logscore=float(design[label][mask]@score[key][mask]/mask.sum()) if mask.any() else None,
                **{k:v for k,v in predictive.items() if np.ndim(v)==0 and k!='n'}))
    atomic_npz(out/'shared_parameters.npz',fields=np.array(sorted(theta)),theta=np.stack([theta[f] for f in sorted(theta)],axis=1),
        field_was_fitted=np.array([existing[f] for f in sorted(theta)]),draw_ids=draw_ids,identity=identity,TYPE=types,rows=rows)
    csv_rows(out/'checks.csv',checks);csv_rows(out/'predictive_summary.csv',summaries)
    save_json(marker,dict(inputs=inputs,case=a.case,correction=a.correction,available=True,passed=all(c['passed'] for c in checks),all_PSF=True,
        heldout_rows=len(rows),candidate_rows=int(candidate.sum()),heldout_fields=np.unique(field).tolist(),fitted_fields=population['fields'].tolist(),
        population_fit_excludes_heldout=a.case=='holdout',outputs={p.name:sha(p) for p in out.iterdir() if p.suffix in ('.npz','.csv')},
        references_read=False,job=os.getenv('SLURM_JOB_ID'),scope='Frozen cut-transfer class/distance counts and distinct selected photometric predictive densities. Unit catalog and generalized design scores separate; no distance truth or independent contamination measurement.'))


if __name__=='__main__':main()
