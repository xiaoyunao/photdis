#!/usr/bin/env python3
"""Coarse/fine native radial audit using identical shared parameter draws."""
import argparse
import json
import os
import time
import numpy as np
from scipy.special import logsumexp
from v2_1_near import ROOT,load,sha,save_json,atomic_npz
from photdis.stage1 import require_compute
from photdis.v3.population import SharedCounts
from photdis.v3.likelihood import trapezoid_weights
from photdis.v3_1.radial import RadialPopulation
from photdis.v3_1.radial_products import bin_integrals


def main():
    require_compute();p=argparse.ArgumentParser();p.add_argument('--catalog',type=int,required=True)
    p.add_argument('--arm',choices=['broad','local'],required=True)
    p.add_argument('--draws',type=int,default=512);p.add_argument('--warmup',type=int,default=512)
    a=p.parse_args();start=time.monotonic();root=ROOT/'data/v3_1'
    recovery=root/'radial_recovery'/f'catalog_{a.catalog:03}_{a.arm}_step0.02_s{a.draws}_w{a.warmup}'
    coarse=root/'radial_likelihood'/f'catalog_{a.catalog:03}_step0.02'
    fine=root/'radial_likelihood'/f'catalog_{a.catalog:03}_step0.01'
    paths=[recovery/'complete.json',recovery/'chains.npz',recovery/'counts.npz',recovery/'per_star.npz',
        coarse/'curves.npz',coarse/'complete.json',fine/'curves.npz',fine/'complete.json',
        ROOT/'scripts/v3_1_radial_quadrature.py',ROOT/'src/photdis/v3_1/radial.py',ROOT/'src/photdis/v3_1/radial_products.py']
    inputs={str(p.relative_to(ROOT)):sha(p) for p in paths};out=recovery/'fine_grid_audit';out.mkdir(exist_ok=True)
    marker=out/'complete.json'
    if marker.exists():
        record=json.loads(marker.read_text());assert record['inputs']==inputs
        for name,digest in record['outputs'].items():assert sha(out/name)==digest
        return
    previous=json.loads(paths[0].read_text())
    for p in paths[1:4]:assert sha(p)==previous['outputs'][p.name]
    records=[json.loads((x/'complete.json').read_text()) for x in (coarse,fine)]
    for folder,record in zip((coarse,fine),records):
        assert record['selection_passed'] and sha(folder/'curves.npz')==record['outputs']['curves.npz']
    c,f=load(paths[4]),load(paths[6]);np.testing.assert_array_equal(c['identity'],f['identity'])
    assert (c['TYPE']=='PSF').all() and (f['TYPE']=='PSF').all()
    np.testing.assert_array_equal(c['class_frequency'],f['class_frequency'])
    models=[];offsets=[]
    for data in (c,f):
        offset=data['log_likelihood'].max(axis=(1,2));offsets.append(offset.sum())
        models.append(RadialPopulation(data['mu'],np.exp(data['log_likelihood']-offset[:,None,None]),
            data['entry'],data['class_frequency'],local=a.arm=='local'))
    chains=load(paths[1])['samples'];z=chains.transpose(1,0,2).reshape(-1,chains.shape[-1]);s=len(z)
    delta=np.empty(s);probabilities=np.empty((s,len(f['identity']),18))
    for i,value in enumerate(z):
        # Constant log-likelihood scales differ between grids; include them in
        # the exact selected-likelihood correction before weight normalization.
        delta[i]=(models[1].objective(value)[0]-offsets[1])-(models[0].objective(value)[0]-offsets[0])
        probabilities[i]=bin_integrals(models[1].joint_density(value),f['mu']).reshape(-1,18)
    weights=np.exp(-delta-logsumexp(-delta));ess=float(1/(weights@weights))
    accumulator=SharedCounts(np.arange(s),18,weights,seed=315000+a.catalog+(10000 if a.arm=='local' else 0))
    mean=accumulator.add(probabilities,np.arange(s)).reshape(-1,3,6);counts=accumulator.result()
    oldprob=load(paths[3])['class_distance'];oldcounts=load(paths[2])
    raw_mean=probabilities.mean(axis=0).reshape(mean.shape)
    probability_delta=float(abs(mean-oldprob).max());count_delta=float(abs(counts['mean']-oldcounts['mean']).max())
    acceptance=bool(previous['passed'] and ess>=max(100,.5*s) and probability_delta<.003 and count_delta<.5
        and counts['marginal_half_delta']<.005)
    atomic_npz(out/'correction.npz',draw_ids=np.arange(s),log_objective_delta=delta,weights=weights,
        identity=f['identity'],TYPE=f['TYPE'],class_distance=mean)
    atomic_npz(out/'counts.npz',**counts)
    save_json(marker,dict(inputs=inputs,outputs={name:sha(out/name) for name in ['correction.npz','counts.npz']},
        job=os.getenv('SLURM_JOB_ID'),elapsed_seconds=time.monotonic()-start,catalog=a.catalog,arm=a.arm,
        coarse_sampler_passed=previous['sampler_passed'],coarse_all_checks_passed=previous['passed'],
        correction_ess=ess,correction_ess_fraction=ess/s,max_probability_delta=probability_delta,
        max_count_mean_delta=count_delta,max_same_draw_unweighted_probability_delta=float(abs(raw_mean-oldprob).max()),
        fine_marginal_half_delta=counts['marginal_half_delta'],passed=acceptance,
        scope='Same shared draws; exact selected-likelihood importance correction from .02 to .01 native grid. Original fit/products retained. Failed correction is not accepted fine-grid inference; numerical refinement required.'))
    print('RADIAL FINE GRID AUDIT',a.catalog,a.arm,acceptance,probability_delta,count_delta,ess,flush=True)


if __name__=='__main__':main()
