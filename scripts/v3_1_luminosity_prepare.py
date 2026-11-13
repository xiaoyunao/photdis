#!/usr/bin/env python3
"""Freeze calibration sky roles and native likelihoods for one scale candidate."""
import concurrent.futures
import json
import multiprocessing as mp
import os
import joblib
import numpy as np
from scipy.special import logsumexp
from v2_1_near import ROOT, load, sha, save_json, atomic_npz, csv_rows
from v3_0_response import observation
from v3_0_cohorts import INPUT_KEYS
from photdis.stage1 import require_compute
from photdis.v3.data import stable_rank
from photdis.v3.model import inference_model
from photdis.v3.inference import kernel_loglike
from photdis.v3.systematics import calibration_vectors

_TABLE=_MODEL=_CAL=_OUT=_INPUTS=_AUDIT=None


def one(row):
    identity = int(_TABLE['identity'][row]); path = _OUT/f'object_{identity}.npz'; marker = path.with_suffix('.json')
    if marker.exists():
        done = json.loads(marker.read_text()); assert done['inputs'] == _INPUTS and sha(path) == done['sha256']
        return marker.name
    frequency = _MODEL.weights@_MODEL.classes
    conditional = _MODEL.weights[:, None]*_MODEL.classes/frequency
    binary = np.array([1-_MODEL.binary_reference_fraction, _MODEL.binary_reference_fraction])
    arrays = dict(identity=_TABLE['identity'][[row]], TYPE=_TABLE['TYPE'][[row]], class_frequency=frequency)
    for tag, step in [('coarse', .02)]+([('fine', .01)] if identity in _AUDIT else []):
        grid = np.linspace(-20, 50, round(70/step)+1)
        lk, _ = kernel_loglike(_MODEL, observation(_TABLE, row), grid, calibration_mag=_CAL[row])
        marginal = logsumexp(lk+np.log(binary)[:, None, None], axis=0)
        values = np.array([logsumexp(marginal+np.log(conditional[:,c,None]), axis=0) for c in range(3)])
        arrays.update({tag+'_grid':grid, tag+'_class_loglike':values})
    atomic_npz(path, **arrays)
    save_json(marker, dict(inputs=_INPUTS, sha256=sha(path), job=os.getenv('SLURM_JOB_ID'), identity=identity))
    return marker.name


def main():
    global _TABLE,_MODEL,_CAL,_OUT,_INPUTS,_AUDIT
    require_compute(); d = ROOT/'data/v3_0'; _OUT = ROOT/'data/v3_1/luminosity_calibration'; _OUT.mkdir(exist_ok=True)
    roles = ROOT/'data/v3_1/validation_roles/complete.json'; frozen = json.loads(roles.read_text()); assert frozen['roles_frozen']
    paths = [d/'references.npz', d/'uncertainty_refit/draw_12/model.joblib', d/'uncertainty/shared_draws.npz',
        d/'calibration.npz', roles, ROOT/'docs/v3_1_luminosity_correction_protocol.md',
        ROOT/'scripts/v3_1_luminosity_prepare.py', ROOT/'src/photdis/v3_1/luminosity.py',
        ROOT/'src/photdis/v3/inference.py', ROOT/'src/photdis/v3/fast.py', ROOT/'src/photdis/v3/model.py']
    _INPUTS = {str(p.relative_to(ROOT)):sha(p) for p in paths}; freeze = _OUT/'inputs.json'
    if freeze.exists(): assert json.loads(freeze.read_text()) == _INPUTS
    else: save_json(freeze, _INPUTS)
    if (_OUT/'complete.json').exists():
        done = json.loads((_OUT/'complete.json').read_text()); assert done['inputs'] == _INPUTS
        for name, digest in done['outputs'].items(): assert sha(_OUT/name) == digest
        for name, digest in done['objects'].items():
            assert sha(_OUT/name) == digest
            record = json.loads((_OUT/name).read_text()); assert sha((_OUT/name).with_suffix('.npz')) == record['sha256']
        return
    t = load(paths[0]); labels = t['class_label']; ref = t['parallax_eligible'] | t['specdis_valid'] | t['spdist_valid']
    base = (t['role'] == 3) & t['reference_stellar_eligible'] & ref & (labels >= 0)
    assert (t['TYPE'][base] == 'PSF').all() and not t['historical_fit'][base].any()
    partition = (stable_rank(t['cell'], salt=318101) % 2).astype('i1')
    rows = []; inclusion = []; counts = []
    for part in range(2):
        for c in range(3):
            eligible = np.flatnonzero(base & (partition == part) & (labels == c))
            rank = stable_rank(t['identity'][eligible], salt=318102)
            chosen = eligible[np.argsort(rank, kind='stable')[:512]]
            rows.extend(chosen); inclusion.extend([len(chosen)/len(eligible)]*len(chosen) if len(eligible) else [])
            counts.append(dict(partition='fit' if part == 0 else 'guard', reference_class=c,
                eligible=len(eligible), retained=len(chosen), cells=len(np.unique(t['cell'][chosen]))))
    rows = np.asarray(rows, int); inclusion = np.asarray(inclusion)
    assert len(rows) and len(np.unique(t['identity'][rows])) == len(rows)
    _TABLE = {k:t[k][rows] for k in INPUT_KEYS}; _TABLE['field'] = np.full(len(rows), -1, 'i8')
    _TABLE['design_weight'] = 1/inclusion
    assert not np.intersect1d(t['cell'][rows][partition[rows] == 0], t['cell'][rows][partition[rows] == 1]).size
    for name in ('field', 'common'):
        v = load(roles.parent/f'{name}_photometry.npz')
        assert not np.intersect1d(v['identity'], _TABLE['identity']).size
        assert not np.intersect1d(v['ls_id'], _TABLE['ls_id']).size
    evidence_keys = ['identity','TYPE','cell','class_label','parallax_eligible','p_corr','gaia_parallax_error',
        'specdis_valid','specdis_distance','specdis_sigma_mu','spdist_valid','spdist_distance','spdist_sigma_mu']
    evidence = {k:t[k][rows] for k in evidence_keys}; evidence.update(partition=partition[rows], inclusion_probability=inclusion)
    # One reference channel for fitting; overlapping source diagnostics remain possible.
    evidence['fit_channel'] = np.where(evidence['parallax_eligible'], 0, np.where(evidence['specdis_valid'], 1, 2)).astype('i1')
    audit = []
    for part in (0,1):
        use = np.flatnonzero(evidence['partition'] == part)
        rank = stable_rank(evidence['identity'][use], salt=318104)
        audit.extend(use[np.argsort(rank, kind='stable')[:16]])
    _AUDIT = set(int(x) for x in evidence['identity'][audit]); evidence['quadrature_audit'] = np.isin(evidence['identity'], list(_AUDIT))
    atomic_npz(_OUT/'photometry.npz', **_TABLE); atomic_npz(_OUT/'evidence.npz', **evidence); csv_rows(_OUT/'admission.csv', counts)
    _MODEL = inference_model(joblib.load(paths[1])); assert _MODEL.converged
    assert not np.intersect1d(_TABLE['identity'], _MODEL.training_identity).size
    _CAL, _ = calibration_vectors(_TABLE, load(paths[2]), 12)
    atomic_npz(_OUT/'calibration_vectors.npz', identity=_TABLE['identity'], TYPE=_TABLE['TYPE'], calibration_mag=_CAL)
    with concurrent.futures.ProcessPoolExecutor(max_workers=32, mp_context=mp.get_context('fork')) as pool:
        for i, _ in enumerate(pool.map(one, range(len(rows)))):
            if i % 128 == 0: print('LUMINOSITY CALIBRATION CURVES', i, flush=True)
    save_json(_OUT/'complete.json', dict(inputs=_INPUTS, job=os.getenv('SLURM_JOB_ID'), n=len(rows), counts=counts,
        outputs={name:sha(_OUT/name) for name in ['photometry.npz','evidence.npz','admission.csv','calibration_vectors.npz']},
        objects={p.name:sha(p) for p in sorted(_OUT.glob('object_*.json'))}, quadrature_audit_identities=sorted(_AUDIT),
        native_nonPSF=0, native_missing_TYPE=0, original_parent_exclusions='data/v3_0/preparation.json',
        scope='Original hyperparameter role3, sky-separated fit/guard, unused prospective identities excluded. Likelihood curves only; no candidate fit or final residual opening.'))


if __name__ == '__main__': main()
