#!/usr/bin/env python3
"""Shared structure posterior and numerical transfers for both fair branches."""
import json
import os
import numpy as np
from scipy.special import logsumexp
from v2_1_near import ROOT,load,sha,save_json,atomic_npz,csv_rows
from photdis.stage1 import require_compute
from photdis.v3_1.structure import structure_quadrature
from photdis.v3_1.structure_posterior import common_structure_posterior


def compare(value,reference):
    p=float(abs(value['per_object_member_probability']-reference['per_object_member_probability']).max())
    c=float(abs(value['mean']-reference['mean']).max());m=float(abs(value['common_mu_quantiles']-reference['common_mu_quantiles']).max())
    z=float(value['log_evidence']-reference['log_evidence'])
    return dict(maximum_member_probability_delta=p,maximum_mean_count_delta=c,maximum_common_mu_quantile_delta=m,
        log_evidence_delta=z,passed=bool(p<.003 and c<.5 and m<.03 and abs(z)<.05))


def main():
    require_compute();root=ROOT/'data/v3_1/structure';source=root/'matrices';out=root/'posterior';out.mkdir(exist_ok=True)
    paths=[source/'complete.json',source/'matrices.npz',ROOT/'scripts/v3_1_structure_infer.py',
        ROOT/'src/photdis/v3_1/structure.py',ROOT/'src/photdis/v3_1/structure_posterior.py',ROOT/'docs/v3_1_structure_protocol.md',
        ROOT/'docs/v3_1_structure_numerical_protocol.md']
    inputs={str(p.relative_to(ROOT)):sha(p) for p in paths};md=json.loads(paths[0].read_text())
    assert md['passed'] and md['all_PSF'] and md['states']==32 and md['n']==512
    assert sha(paths[1])==md['outputs']['matrices.npz'];data=load(paths[1])
    assert (data['TYPE']=='PSF').all();np.testing.assert_array_equal(data['response_ids'],np.arange(32))
    freeze=out/'inputs.json'
    if freeze.exists():assert json.loads(freeze.read_text())==inputs
    else:save_json(freeze,inputs)
    allchecks=[];summary=[];membership=[];final={};markers={}
    for branch_index,branch in enumerate(('full','gaussian')):
        folder=out/branch;folder.mkdir(exist_ok=True);results={}
        tasks=[('base',16,64),('base',32,128),('base',64,256),('native_coarse',64,256),('entry_second_seed',64,256),('entry_previous_level',64,256)]
        for variant,distance_order,odds_order in tasks:
            key=f'{variant}_m{distance_order}_z{odds_order}';path=folder/f'{key}.npz';marker=path.with_suffix('.json')
            if marker.exists():
                old=json.loads(marker.read_text());assert old['inputs']==inputs and sha(path)==old['sha256'];r=load(path)
            else:
                mu,lmu,z,lz=structure_quadrature(distance_order,odds_order);indices=np.searchsorted(data['common_mu'],mu)
                np.testing.assert_array_equal(data['common_mu'][indices],mu)
                grid=0 if variant=='native_coarse' else 1
                entry_index={'base':0,'native_coarse':0,'entry_second_seed':1,'entry_previous_level':2}[variant]
                response=data['logresponse'][grid,branch_index];entry=data['entry'][entry_index]
                rb=logsumexp(response[:,:,:5],axis=-1)-np.log(5);ab=entry[:,:,:5].mean(axis=-1)
                rs=np.take(response,5+indices,axis=-1);as_=np.take(entry,5+indices,axis=-1)
                r=common_structure_posterior(rs,rb,as_,ab,data['radius_deg'],mu,lmu,z,lz)
                atomic_npz(path,**r,identity=data['identity'],TYPE=data['TYPE'],response_ids=data['response_ids'])
                save_json(marker,dict(inputs=inputs,sha256=sha(path),branch=branch,variant=variant,distance_order=distance_order,odds_order=odds_order,job=os.getenv('SLURM_JOB_ID')))
            results[key]=r;markers[str(marker.relative_to(out))]=sha(marker)
            assert abs(r['mean'].sum()-512)<1e-8 and abs(r['covariance'].sum(axis=1)).max()<1e-8 and np.linalg.eigvalsh(r['covariance']).min()>-1e-8
            print('STRUCTURE POSTERIOR',branch,key,float(r['log_evidence']),r['mean'].tolist(),flush=True)
        reference=results['base_m64_z256'];checks=[]
        for current,prior,label in [('base_m32_z128','base_m16_z64','quadrature_first'),('base_m64_z256','base_m32_z128','quadrature_final'),
            ('native_coarse_m64_z256','base_m64_z256','native_coarse'),('entry_second_seed_m64_z256','base_m64_z256','entry_second_seed'),
            ('entry_previous_level_m64_z256','base_m64_z256','entry_previous_level')]:
            checks.append(dict(branch=branch,comparison=label,**compare(results[current],results[prior])))
        passed=all(r['passed'] for r in checks if r['comparison']!='quadrature_first');allchecks.extend(checks)
        final[branch]=reference
        weight=reference['response_weights'];mu=reference['common_mu_quantiles']
        summary.append(dict(branch=branch,n=512,numerically_passed=passed,member_mean=float(reference['mean'][0]),background_mean=float(reference['mean'][1]),
            member_sigma=float(np.sqrt(reference['covariance'][0,0])),shared_member_variance=float(reference['shared_covariance'][0,0]),
            conditional_member_variance=float(reference['conditional_covariance'][0,0]),response_ESS=float(1/(weight@weight)),top_response=int(weight.argmax()),
            common_distance_median_kpc=float(10**((mu[2]-10)/5)),common_distance_lower95_kpc=float(10**((mu[0]-10)/5)),common_distance_upper95_kpc=float(10**((mu[4]-10)/5)),
            log_evidence=float(reference['log_evidence'])))
        membership.extend(dict(branch=branch,identity=int(identity),TYPE='PSF',radius_deg=float(data['radius_deg'][row]),
            member_probability=float(reference['per_object_member_probability'][row]),numerically_passed=passed) for row,identity in enumerate(data['identity']))
    csv_rows(out/'numerical_checks.csv',allchecks);csv_rows(out/'summary.csv',summary);csv_rows(out/'membership.csv',membership)
    save_json(out/'comparison.json',dict(member_mean_full_minus_gaussian=float(final['full']['mean'][0]-final['gaussian']['mean'][0]),
        common_mu_quantile_full_minus_gaussian=(final['full']['common_mu_quantiles']-final['gaussian']['common_mu_quantiles']).tolist(),
        maximum_member_probability_difference=float(abs(final['full']['per_object_member_probability']-final['gaussian']['per_object_member_probability']).max()),
        scope='Fair likelihood-summary comparison, not a Bayes factor between normalized generative models or independent accuracy measurement. Common distance anchor was supplied once as a prior.'))
    save_json(out/'complete.json',dict(inputs=inputs,results=markers,outputs={n:sha(out/n) for n in ['numerical_checks.csv','summary.csv','membership.csv','comparison.json']},
        n=512,states=32,all_PSF=True,passed=all(r['numerically_passed'] for r in summary),summary=summary,job=os.getenv('SLURM_JOB_ID'),
        scope='SeenNGC288 shared response/common-distance/odds posterior, selected512object counts. No effective-area density, physical depth, final-candidate validation, or new independent distance accuracy claim. Numerical failures retained.'))


if __name__=='__main__':main()
