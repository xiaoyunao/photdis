#!/usr/bin/env python3
"""Recover signed WISE measurement columns by frozen native object identity."""
import json
import os

import numpy as np

from baseline_v1_4_experiments import ROOT, OUT, load, save_json, sha, atomic_npz
from photdis.stage1 import legacy_ids, require_compute
from photdis.reference_v1_4 import unique_join

DEST = ROOT/'data/baseline_v1_6_wise'
FIELDS = [f'{k}_{b}' for b in ('W1','W2') for k in
          ('FLUX','FLUX_IVAR','MW_TRANSMISSION','NOBS','WISEMASK','FRACFLUX')]


def states(raw):
    output = []
    for band in ('W1','W2'):
        f, v, tr, n, mask, frac = (raw[f'{k}_{band}'] for k in
                                  ('FLUX','FLUX_IVAR','MW_TRANSMISSION','NOBS','WISEMASK','FRACFLUX'))
        observed = n > 0
        measured = observed & np.isfinite(f) & np.isfinite(v) & (v>0) & np.isfinite(tr) & (tr>0)
        clean = measured & (mask==0) & (frac<1)
        snr = f*np.sqrt(np.maximum(v,0))
        output.append(np.select([clean & (snr>=5),clean,measured,observed],[4,3,2,1],0))
    return np.column_stack(output)


def main():
    require_compute()
    DEST.mkdir(exist_ok=True)
    path = DEST/'raw.npz'
    manifest = DEST/'raw.json'
    if manifest.exists():
        assert sha(path) == json.loads(manifest.read_text())['sha256']
        return
    refs = OUT/'references.npz'
    ref_hash = sha(refs)
    assert ref_hash == json.loads((OUT/'assembly.json').read_text())['reference_sha256']
    ids = load(OUT/'experiments/expanded_A_identities.npz')
    want = np.concatenate([ids['distance_train_identity'],ids['distance_eval_identity']])
    assert len(np.unique(want)) == len(want)
    with np.load(refs) as z:
        rows = unique_join(want,z['identity'])
        assert (rows>=0).all()
        ls = z['ls_id'][rows]
        expected = np.column_stack([z[f'{b}_measurement_state'][rows] for b in ('w1','w2')])
    assert len(np.unique(ls)) == len(ls)
    order = np.argsort(ls)
    sorted_ls = ls[order]
    raw = {k:np.full(len(ls),np.nan) for k in FIELDS}
    seen = np.zeros(len(ls),bool)
    inputs = {}
    sweep_hash = json.loads((OUT/'native_photometry.json').read_text())['sweep_hashes']
    for j,(name,digest) in enumerate(sorted(sweep_hash.items()),1):
        sweep = OUT/'photometry'/f'{name}.npz'
        with np.load(sweep) as z:
            if 'RELEASE' in z.files:
                cat = legacy_ids({k:z[k] for k in ('RELEASE','BRICKID','OBJID')})
                ix = np.searchsorted(sorted_ls,cat)
                keep = (ix<len(ls)) & (sorted_ls[np.minimum(ix,len(ls)-1)]==cat)
                if keep.any():
                    assert sha(sweep)==digest
                    inputs[sweep.name] = digest
                    dest = order[ix[keep]]
                    assert len(np.unique(dest))==len(dest) and not seen[dest].any()
                    for k in FIELDS:
                        raw[k][dest] = z[k][keep]
                    seen[dest] = True
        if j%200==0:
            print(f'checked {j}/1600 sweeps; matched {seen.sum()}/{len(ls)}',flush=True)
    assert len(sweep_hash)==1600
    zero_path = OUT/'zero_candidates_raw.npz'
    assert sha(zero_path)==json.loads((OUT/'zero_candidates_raw.json').read_text())['sha256']
    with np.load(zero_path) as z:
        missing = np.flatnonzero(~seen)
        ix = unique_join(ls[missing],z['ls_id'])
        assert (ix>=0).all()
        for k in FIELDS:
            raw[k][missing] = z[k][ix]
    actual = states(raw)
    assert np.array_equal(actual,expected)
    atomic_npz(path,identity=want,rows=rows,ls_id=ls,state=actual,**raw)
    save_json(manifest,dict(sha256=sha(path),reference_sha256=ref_hash,sweep_hashes=inputs,
              zero_fallback=len(missing),rows=len(rows),state_match=True,job_id=os.getenv('SLURM_JOB_ID')))
    print('COMPLETE raw',len(rows),'zero fallback',len(missing),flush=True)


if __name__=='__main__':
    main()
