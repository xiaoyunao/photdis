#!/usr/bin/env python3
"""Bounded finite-response compression for large conditional catalog products.

This removes only negligible *global response weight*, never stellar modes or
SED kernels. Finite-prior support adequacy is explicitly a separate limitation.
"""
import argparse
import csv
import json
import os
import numpy as np
from scipy.special import logsumexp
from v2_1_near import ROOT,sha,save_json
from photdis.stage1 import require_compute
from v3_0_validation_inputs import check_freeze
D=ROOT/'data/v3_0'


def retain_responses(weights,tolerance):
    w=np.asarray(weights,float)
    if not np.isfinite(w).all() or (w<0).any() or not np.isclose(w.sum(),1,atol=1e-12):
        raise ValueError('Normalized nonnegative response weights required')
    if not 0<tolerance<1e-5:raise ValueError('Tight numerical compression bound required')
    order=np.argsort(-w,kind='stable');n=len(w)
    for i in range(1,len(w)+1):
        if w[order[i:]].sum()<=tolerance:n=i;break
    selected=order[:n];discarded=float(w[order[n:]].sum())
    return selected,w[selected]/w[selected].sum(),discarded


def main():
    require_compute();p=argparse.ArgumentParser();p.add_argument('--ensemble',default='all32_native_refined')
    p.add_argument('--tolerance',type=float,default=1e-10);a=p.parse_args();method=check_freeze()
    folder=D/'ensemble_products'/a.ensemble;record=json.loads((folder/'complete.json').read_text())
    assert record['passed'],'Final per-star numerical gate must pass first'
    rows=list(csv.DictReader((folder/'response_weights.csv').open()))
    index,weights,discarded=retain_responses([float(r['posterior_response_weight']) for r in rows],a.tolerance)
    counts=[json.loads((D/'catalog'/f'field_{i:02}'/'prepared.json').read_text())['candidate'] for i in range(14)]
    selected=[dict(draw=int(rows[i]['global_response_draw']),draws=int(rows[i]['chain_draws']),
        original_weight=float(rows[i]['posterior_response_weight']),retained_weight=float(w)) for i,w in zip(index,weights)]
    logz=np.array([float(r['log_evidence']) for r in rows]);omitted=np.ones(len(rows),bool);omitted[index]=False
    log_discarded=float(logsumexp(logz[omitted])-logsumexp(logz)) if omitted.any() else None
    n=sum(counts);out=D/'catalog/transfer_plan.json'
    if out.exists():raise RuntimeError('Transfer plan immutable; choose explicit migration rather than overwrite')
    save_json(out,dict(method_freeze=method,ensemble=a.ensemble,ensemble_sha256=sha(folder/'complete.json'),
        response_weights_sha256=sha(folder/'response_weights.csv'),responses=selected,discarded_response_mass=discarded,
        discarded_response_log_mass=log_discarded,
        density_warning='Total variation bound controls probabilities/CDF/PIT/counts, not relative densities or log scores. Score a pruned mixture only with observation-likelihood bounds or retain every response in log space.',
        tolerance=a.tolerance,qualified_parent=n,field_counts=counts,
        probability_absolute_error_bound=discarded,count_mean_absolute_error_bound=n*discarded,
        count_covariance_element_absolute_error_bound=3*n*n*discarded,
        scope='Conditional transfer of finite target-sample response posterior; not full-catalog population refit. All within-response stellar modes/kernels retained.',
        reference_response_ESS=record['response_effective_sample_size'],
        scientific_limit='Low finite response ESS means unresolved reference/calibration prior integration. Tiny discarded discrete weight does not establish continuous-response uncertainty convergence.',job=os.getenv('SLURM_JOB_ID')))
    print('TRANSFER PLAN',selected,'discarded mass',discarded,flush=True)


if __name__=='__main__':main()
