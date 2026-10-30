#!/usr/bin/env python3
"""Summarize correlated contrasts from saved synthetic clusters; no rescoring."""
import hashlib
import json
from pathlib import Path
import numpy as np
from photdis.cluster_rgb_external import paired_bin_contrast
from photdis.stage1 import write_json


def main():
    result_path=Path('manifests/cluster_rgb_ngc4147_result_v1.json')
    simulation_path=Path('manifests/cluster_rgb_ngc4147_recovery_v1.json')
    out=Path('manifests/cluster_rgb_ngc4147_summary_v1.json')
    r=json.loads(result_path.read_text())
    sims=json.loads(simulation_path.read_text())['results']
    actual=r['results'][r['primary']]['reference']
    observed=actual['bins']['2.5_3.0']['median']-actual['bins']['1.0_1.5']['median']
    comparisons={}
    for name,sim in sims.items():
        for key in ('per_cluster','per_cluster_proxy_bins'):
            c=paired_bin_contrast(sim[key])
            lo,mid,hi=c['interval95']
            c.update(observed=observed, observed_minus_predictive_median=observed-mid,
                     observed_inside95=bool(lo<=observed<=hi))
            comparisons[name+'__'+key]=c
    write_json(out,dict(
        role='derived_summary_of_frozen_test_not_model_or_prior_selection',
        result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
        recovery_sha256=hashlib.sha256(simulation_path.read_bytes()).hexdigest(),
        bright_bin='1.0_1.5',faint_bin='2.5_3.0',paired_contrasts=comparisons,
        primary_distance_ratio=10**(actual['overall']['median']/5),
        primary_interval_endpoint_ratio68=10**(actual['overall']['width68']/5),
        simulated_feh_outside_development={name:sum(
            d['feh'] < -2 or d['feh'] > -1.46 for d in sim['nuisances'])
            for name,sim in sims.items()}))
    print(json.dumps({k:{x:v[x] for x in ('interval95','observed','observed_inside95')}
                      for k,v in comparisons.items()},indent=2))


if __name__=='__main__':main()
