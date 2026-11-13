#!/usr/bin/env python3
"""Shared realized-count sidecars and post-fit seen-system membership comparison."""
import json
import os
import numpy as np
from scipy.special import logsumexp
from v2_1_near import ROOT,load,sha,save_json,atomic_npz,csv_rows
from photdis.stage1 import require_compute
from photdis.v3_1.structure_draws import realized_structure_counts,count_draw_diagnostics


def main():
    require_compute(); root=ROOT/'data/v3_1/structure'; posterior=root/'posterior'; out=root/'report'; out.mkdir(exist_ok=True)
    paths=[posterior/'complete.json',root/'matrices/complete.json',root/'matrices/matrices.npz',
        root/'complete.json',root/'photometry.npz',ROOT/'scripts/v3_1_structure_report.py',
        ROOT/'src/photdis/v3_1/structure_draws.py',ROOT/'docs/v3_1_structure_counts_protocol.md']
    pd=json.loads(paths[0].read_text()); md=json.loads(paths[1].read_text()); prep=json.loads(paths[3].read_text())
    assert pd['n']==md['n']==prep['n']==512 and pd['states']==md['states']==32
    assert pd['all_PSF'] and md['all_PSF'] and prep['all_PSF']
    assert sha(paths[2])==md['outputs']['matrices.npz'] and sha(paths[4])==prep['outputs']['photometry.npz']
    assert pd['inputs'][str(paths[1].relative_to(ROOT))]==sha(paths[1]) and pd['inputs'][str(paths[2].relative_to(ROOT))]==sha(paths[2])
    data=load(paths[2]); photo=load(paths[4]); np.testing.assert_array_equal(data['identity'],photo['identity'])
    assert (photo['TYPE']=='PSF').all() and (data['TYPE']=='PSF').all()
    results={}
    for branch in ['full','gaussian']:
        name=f'{branch}/base_m64_z256.json'; marker=posterior/name; assert sha(marker)==pd['results'][name]
        record=json.loads(marker.read_text()); assert record['inputs']==pd['inputs']
        payload=marker.with_suffix('.npz'); assert sha(payload)==record['sha256']; paths.extend([marker,payload])
        r=load(payload); np.testing.assert_array_equal(r['identity'],photo['identity']); assert (r['TYPE']=='PSF').all()
        np.testing.assert_array_equal(r['response_ids'],np.arange(32)); results[branch]=r
    inputs={str(p.relative_to(ROOT)):sha(p) for p in paths}; freeze=out/'inputs.json'
    if freeze.exists():assert json.loads(freeze.read_text())==inputs
    else:save_json(freeze,inputs)
    attempts=[]; summaries=[]; outputs=[]
    for bi,branch in enumerate(['full','gaussian']):
        result=results[branch]; indices=np.searchsorted(data['common_mu'],result['common_mu'])
        np.testing.assert_array_equal(data['common_mu'][indices],result['common_mu'])
        response=data['logresponse'][1,bi]; rs=np.take(response,5+indices,axis=-1); rb=logsumexp(response[:,:,:5],axis=-1)-np.log(5)
        for draws in [65536,262144]:
            name=f'{branch}_n{draws}.npz'; marker=out/f'{branch}_n{draws}.json'
            if marker.exists():
                check=json.loads(marker.read_text()); assert check['inputs']==inputs and sha(out/name)==check['sha256']
            else:
                samples=realized_structure_counts(rs,rb,data['radius_deg'],result['zeta'],result['global_weights'],draws=draws,seed=320601)
                check=count_draw_diagnostics(samples['realized_counts'],result['mean'],result['covariance'])
                atomic_npz(out/name,**samples,quantiles=np.array(check['quantiles'],dtype='i8'),seed=np.array(320601))
                check.update(inputs=inputs,sha256=sha(out/name),branch=branch,job=os.getenv('SLURM_JOB_ID')); save_json(marker,check)
            attempts.append(dict(branch=branch,draws=draws,passed=check['passed'],marker=marker.name,sha256=sha(marker)))
            outputs.extend([name,marker.name])
            if check['passed']:break
        numerical=next(s['numerically_passed'] for s in pd['summary'] if s['branch']==branch)
        for c,label in enumerate(['member','background']):
            q=check['quantiles'][c]
            summaries.append(dict(branch=branch,component=label,n=512,draws=draws,original_numerical_passed=numerical,
                sampling_passed=check['passed'],accepted=bool(numerical and check['passed']),mean=float(result['mean'][c]),
                variance=float(result['covariance'][c,c]),conditional_variance=float(result['conditional_covariance'][c,c]),
                shared_variance=float(result['shared_covariance'][c,c]),lower95=q[0],lower68=q[1],median=q[2],upper68=q[3],upper95=q[4],
                CDF_half_delta=check['CDF_half_delta'],mean_MC_standard_error=check['mean_MC_standard_error']))
    # Association is deliberately after authentication of both completed fits.
    old=ROOT/'data/v3_0/independent_scale'; association=json.loads((old/'association.json').read_text())
    assert sha(old/'photometry.npz')==association['input_sha256']
    assert sha(ROOT/'configs/v3_0_independent_scale.json')==association['config_sha256']
    member=load(old/'membership.npz'); original=load(old/'photometry.npz'); original_rows=photo['original_rows']
    np.testing.assert_array_equal(member['identity'],original['identity'])
    np.testing.assert_array_equal(member['identity'][original_rows],photo['identity'])
    assert (member['TYPE'][original_rows]=='PSF').all()
    probability=member['membership_probability'][original_rows]; available=member['membership_available'][original_rows]
    assert np.isfinite(probability[available]).all() and ((probability[available]>=0) & (probability[available]<=1)).all()
    selected=available & (probability>=.9); np.testing.assert_array_equal(selected,member['selected_member'][original_rows])
    evidence_inputs={str(p.relative_to(ROOT)):sha(p) for p in [old/'membership.npz',old/'association.json',old/'photometry.npz',ROOT/'configs/v3_0_independent_scale.json']}
    ep=out/'membership_inputs.json'
    if ep.exists():assert json.loads(ep.read_text())==evidence_inputs
    else:save_json(ep,evidence_inputs)
    groups={'all_photometric':np.ones(512,bool),'membership_available':available,'published_p_ge_0.9':selected,
        'available_p_lt_0.9':available & ~selected,'membership_missing':~available}; comparisons=[]; star_rows=[]
    for branch,result in results.items():
        p=result['per_object_member_probability']; assert np.isfinite(p).all() and ((p>=0) & (p<=1)).all()
        for label,mask in groups.items():
            valid=mask & available; count=int(valid.sum())
            comparisons.append(dict(branch=branch,group=label,n=int(mask.sum()),membership_available_n=count,
                expected_model_members=float(p[mask].sum()),published_probability_sum=float(probability[valid].sum()) if count else None,
                mean_model_minus_published=float((p[valid]-probability[valid]).mean()) if count else None,
                probability_MSE_descriptive=float(np.mean((p[valid]-probability[valid])**2)) if count else None,
                model_p_gt_0_5_n=int((mask & (p>.5)).sum()),model_p_gt_0_9_n=int((mask & (p>.9)).sum()),
                p_gt_0_5_agrees_with_published_p_ge_0_9_n=int(((p[valid]>.5)==selected[valid]).sum()) if count else None,
                p_gt_0_9_agrees_with_published_p_ge_0_9_n=int(((p[valid]>.9)==selected[valid]).sum()) if count else None,
                original_numerical_passed=next(s['numerically_passed'] for s in pd['summary'] if s['branch']==branch)))
        for i,identity in enumerate(photo['identity']):
            star_rows.append(dict(branch=branch,identity=int(identity),TYPE='PSF',radius_deg=float(data['radius_deg'][i]),
                model_member_probability=float(p[i]),membership_available=bool(available[i]),
                published_member_probability=float(probability[i]) if available[i] else None,
                published_p_ge_0_9=bool(selected[i]) if available[i] else None))
    csv_rows(out/'counts.csv',summaries); csv_rows(out/'membership_comparison.csv',comparisons); csv_rows(out/'membership.csv',star_rows)
    outputs+=['counts.csv','membership_comparison.csv','membership.csv','membership_inputs.json']
    save_json(out/'complete.json',dict(inputs=inputs,evidence_inputs=evidence_inputs,outputs={name:sha(out/name) for name in outputs},
        attempts=attempts,n=512,states=32,all_PSF=True,passed=all(r['accepted'] for r in summaries),summary=summaries,
        membership_available=int(available.sum()),published_p_ge_0_9=int(selected.sum()),job=os.getenv('SLURM_JOB_ID'),
        scope='Shared-state realized counts and descriptive already-seen Gaia-membership model comparison. Missing membership is not nonmembership. Supplied dynamical scale is a common prior, not a newly recovered independent scale. No density or finalcandidate validation claim.'))


if __name__=='__main__':main()
