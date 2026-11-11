#!/usr/bin/env python3
"""Explicit sampling-design diagnostic, separate from full-catalog posterior counts."""
import argparse
import json
import os
import numpy as np
from v2_1_near import ROOT,load,sha,save_json,atomic_npz,csv_rows
from photdis.stage1 import require_compute
from photdis.v3.metrics import design_count_covariance
D=ROOT/'data/v3_0'


def main():
    require_compute();p=argparse.ArgumentParser();p.add_argument('--ensemble',default='all32_native_refined');a=p.parse_args()
    folder=D/'ensemble_products'/a.ensemble;marker=folder/'complete.json';record=json.loads(marker.read_text());assert record['passed']
    source=load(D/'cohorts/application.npz');pieces=[];ids=[]
    for path in sorted((folder/'chunks').glob('chunk_*.npz')):
        m=json.loads(path.with_suffix('.json').read_text());assert sha(path)==m['sha256'];t=load(path)
        assert (t['TYPE']=='PSF').all();pieces.append(t['probability'].reshape(-1,18));ids.append(t['identity'])
    ids=np.concatenate(ids);order=np.argsort(source['identity']);rows=order[np.searchsorted(source['identity'][order],ids)]
    np.testing.assert_array_equal(source['identity'][rows],ids);prob=np.concatenate(pieces);field=source['field'][rows]
    populations={}
    for f in np.unique(field):
        w=source['design_weight'][rows][field==f];assert np.allclose(w,w[0]);populations[int(f)]=int(round(w.sum()))
    total,cov=design_count_covariance(prob,field,populations)
    assert abs(total.sum()-sum(populations.values()))<1e-6
    out=folder/'design_diagnostic';out.mkdir(exist_ok=True)
    atomic_npz(out/'sampling_design.npz',total=total,conditional_design_covariance=cov)
    csv_rows(out/'sampling_design.csv',[dict(class_index=i//6,distance_bin=i%6,estimated_total=float(total[i]),conditional_design_sd=float(np.sqrt(max(cov[i,i],0)))) for i in range(18)])
    save_json(out/'complete.json',dict(ensemble_sha256=sha(marker),source_sha256=sha(D/'cohorts/application.npz'),n=len(ids),represented_parent=sum(populations.values()),stratum_population=populations,
        interpretation='Horvitz-Thompson plug-in total with SRS finite-population design covariance conditional on fitted marginal probabilities. This is a sampling-design diagnostic, not an independent validation count or a full posterior interval. Dependence of fitted population parameters on survey resampling is not included. The actual qualified full-catalog product uses unit weights and separately retains shared posterior covariance.',
        outputs={x.name:sha(x) for x in out.iterdir() if x.suffix in ('.npz','.csv')},job=os.getenv('SLURM_JOB_ID')))


if __name__=='__main__':main()
