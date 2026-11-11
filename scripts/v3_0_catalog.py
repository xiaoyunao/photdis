#!/usr/bin/env python3
"""Photometry-only candidate scope inventory before any final validation outcomes.

A candidate envelope is not scientific acceptance. All original parent rows keep
an explicit status, including north/no model and photometric exclusions.
"""
import argparse
import json
import os
import joblib
import numpy as np
from v2_1_near import ROOT, load, sha, save_json, atomic_npz
from photdis.stage1 import require_compute
from photdis.baseline_v1 import support
from photdis.v3.controls import optical_magnitudes
from v3_0_cohorts import INPUT_KEYS
D = ROOT/'data/v3_0'


def prepare(field):
    source = D/'application_inputs'/f'field_{field:02}.npz'
    meta_path = source.with_suffix('.json'); meta = json.loads(meta_path.read_text())
    assert sha(source) == meta['sha256']
    model_path = D/'controls/qrf_grz.joblib'; model = joblib.load(model_path)
    out = D/'catalog'/f'field_{field:02}'; out.mkdir(parents=True, exist_ok=True)
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in [source, meta_path, model_path, ROOT/'scripts/v3_0_catalog.py']}
    if (out/'prepared.json').exists():
        old = json.loads((out/'prepared.json').read_text()); assert old['inputs'] == inputs
        for name, digest in old['outputs'].items(): assert sha(out/name) == digest
        return
    t = load(source); assert (t['TYPE'] == 'PSF').all()
    mag, error = optical_magnitudes(t)
    supported = support(model, mag)
    decam = t['photometric_system'] == 'DECam_DR11'
    # Fixed pre-existing near-application observable criteria, no truth or qnear
    # selection. Full distance tails of these stars remain in every product.
    observable = (t['r'] < 20) & (error.max(axis=1) < .1) & supported & decam
    rows = np.flatnonzero(observable)
    payload = {k: t[k][rows] for k in INPUT_KEYS}
    payload.update(field=t['field'][rows], parent_row=t['parent_row'][rows], design_weight=np.ones(len(rows)))
    atomic_npz(out/'inputs.npz', **payload)
    # Every original catalog identity is accounted for, including no inference.
    status = np.full(meta['original_parent'], 1, 'i1')
    status[t['parent_row']] = 2
    status[t['parent_row'][~decam]] = 3
    status[t['parent_row'][observable]] = 0
    atomic_npz(out/'parent_status.npz', parent_row=np.arange(meta['original_parent'], dtype='i8'), status=status)
    save_json(out/'prepared.json', dict(inputs=inputs, field=field, original_parent=meta['original_parent'],
        native_PSF_entry=len(t['identity']), candidate=len(rows), nonPSF=meta['nonPSF'], missing_TYPE=meta['missing_TYPE'],
        outside_r20=int((t['r'] >= 20).sum()), optical_error_ge_point1=int((error.max(axis=1) >= .1).sum()),
        outside_reference_support=int((~supported).sum()), system_unmodeled=int((~decam).sum()),
        status_definitions={'0':'candidate input; prediction not yet done', '1':'failed native morphology/photometry admission',
                            '2':'outside predeclared observable candidate envelope', '3':'unmodeled northern system'},
        qualification='VerifiedPSF, nativeclean grzSNR>=5,absb>30,0<=EBV<.03,DECam,r<20,opticalerr<.1,referencecolorsupport; no qnear/Gaia/spectral admission',
        acceptance='Candidate research envelope only; conditional calibration/independent evaluation still required',
        reference_outcomes_read=False, outputs={p.name: sha(p) for p in (out/'inputs.npz', out/'parent_status.npz')}, job=os.getenv('SLURM_JOB_ID')))
    print('CANDIDATE INVENTORY', field, len(rows), 'of', meta['original_parent'], flush=True)


def main():
    require_compute(); p = argparse.ArgumentParser(); p.add_argument('stage', choices=['prepare'])
    p.add_argument('--field', type=int, default=int(os.getenv('SLURM_ARRAY_TASK_ID', 0)))
    a = p.parse_args(); prepare(a.field)


if __name__ == '__main__': main()
