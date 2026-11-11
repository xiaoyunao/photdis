#!/usr/bin/env python3
"""Allocated exact-equivalence and timing audit on the real target likelihood."""
import json
import os
import time
import joblib
import numpy as np
from v2_1_near import ROOT, load, sha, save_json
from photdis.stage1 import require_compute
from photdis.v3.model import inference_model
from photdis.v3.hierarchy import FieldPopulation
from photdis.v3.fast_hierarchy import FastFieldPopulation
D = ROOT/'data/v3_0'


def main():
    require_compute(); target = D/'targets/application_fit_0'
    r = load(target/'component_response.npz'); source = load(D/'cohorts/application.npz')
    model = inference_model(joblib.load(D/'sed_fit.joblib')); fit = joblib.load(target/'map.joblib')
    args = (r['response'].sum(axis=3), r['selection'], model, source['field'][r['rows']], source['design_weight'][r['rows']])
    slow = FieldPopulation(*args); fast = FastFieldPopulation(*args)
    rng = np.random.default_rng(305101); records = []
    for offset in [0., .1, 1.]:
        x = fit['map']+rng.normal(size=slow.dimension)*offset
        t = time.monotonic(); sv, sg = slow.objective(x); st = time.monotonic()-t
        t = time.monotonic(); fv, fg = fast.objective(x); ft = time.monotonic()-t
        np.testing.assert_allclose(sv, fv, atol=1e-7, rtol=1e-12)
        np.testing.assert_allclose(sg, fg, atol=1e-7, rtol=1e-10)
        records.append(dict(offset=offset, slow_seconds=st, fast_seconds=ft,
            value_difference=abs(sv-fv), gradient_max_difference=float(abs(sg-fg).max())))
    save_json(target/'objective_benchmark.json', dict(records=records, n=len(r['identity']), job=os.getenv('SLURM_JOB_ID'),
        inputs={str(p.relative_to(ROOT)): sha(p) for p in [target/'component_response.npz', ROOT/'src/photdis/v3/fast_hierarchy.py']}))
    print(json.dumps(records), flush=True)


if __name__ == '__main__': main()
