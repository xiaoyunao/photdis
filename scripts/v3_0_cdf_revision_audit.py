#!/usr/bin/env python3
"""Check that source-knot CDF repair preserves native densities and joint counts."""
import argparse
import json
import os
import numpy as np
from v2_1_near import ROOT,load,sha,save_json
from photdis.stage1 import require_compute


def main():
    require_compute();p=argparse.ArgumentParser();p.add_argument('--previous',required=True);a=p.parse_args()
    old=ROOT/a.previous;new=ROOT/'data/v3_0/ensemble_products/all32_native_refined'
    before=json.loads((old/'complete.json').read_text());after=json.loads((new/'complete.json').read_text())
    assert not before['passed'] and after['passed']
    old_inputs=json.loads((old/'inputs.json').read_text());new_inputs=json.loads((new/'inputs.json').read_text())
    code='scripts/v3_0_ensemble.py'
    assert {k:v for k,v in old_inputs.items() if k!=code}=={k:v for k,v in new_inputs.items() if k!=code}
    c0=load(old/'counts.npz');c1=load(new/'counts.npz');assert c0.keys()==c1.keys()
    for k in c0:np.testing.assert_array_equal(c0[k],c1[k])
    names=sorted(x.name for x in (old/'chunks').glob('chunk_*.npz'))
    assert names==sorted(x.name for x in (new/'chunks').glob('chunk_*.npz'))
    total=0;max_mu=0.;checks=0
    fixed=['identity','TYPE','probability','qnear','binary_probability','discrepancy_probability',
           'full_mu','full_logpdf','full_grid_offset']
    for name in names:
        for base in [old,new]:
            meta=json.loads((base/'chunks'/name).with_suffix('.json').read_text())
            assert sha(base/'chunks'/name)==meta['sha256']
        left=load(old/'chunks'/name);right=load(new/'chunks'/name)
        for key in fixed:np.testing.assert_array_equal(left[key],right[key]);checks+=1
        max_mu=max(max_mu,float(np.max(abs(left['mu']-right['mu']))));total+=len(right['identity'])
    assert total==26000
    save_json(new/'cdf_revision_audit.json',dict(passed=True,objects=total,exact_array_checks=checks+len(c0),
        previous_complete_sha256=sha(old/'complete.json'),new_complete_sha256=sha(new/'complete.json'),
        previous_path=str(old.relative_to(ROOT)),old_errors=before['errors'],new_errors=after['errors'],
        maximum_quantile_modulus_change=max_mu,
        interpretation='Numerical CDF/quantile interpolation repair only. Source model hashes, response evidence, native log densities, per-star joint masses and all count arrays identical.',
        job=os.getenv('SLURM_JOB_ID')))
    print('CDF REVISION AUDIT',total,'objects; exact preserved arrays',checks+len(c0),'maximum quantile change',max_mu,flush=True)


if __name__=='__main__':main()
