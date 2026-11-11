#!/usr/bin/env python3
"""Freeze development inputs, original application sample, independent SED bank.

Prospective 3250 and new-sky reference outcomes are deliberately not accessed.
Inference inputs and evaluation evidence are written separately.
"""
import json
import os
import numpy as np
from v2_1_near import ROOT, load, sha, atomic_npz, save_json
from photdis.stage1 import require_compute
from photdis.reference_v1_4 import unique_join
from photdis.audit import rvs_metallicity
from photdis.v3.data import stable_rank
from photdis.v3.controls import optical_magnitudes

D = ROOT/'data/v3_0'
INPUT_KEYS = ['identity', 'ls_id', 'TYPE', 'RA', 'DEC', 'EBV', 'cell',
              'flux', 'flux_error', 'transmission', 'band_state', 'band_mask',
              'photometric_system']
EVIDENCE_KEYS = ['identity', 'TYPE', 'role', 'historical_fit', 'class_label',
    'primary_channel', 'reference_stellar_eligible', 'rvs_feh', 'rvs_feh_err',
    'rvs_teff', 'rvs_logg', 'rvs_parameter_quality', 'specdis_valid',
    'specdis_distance', 'specdis_sigma_mu', 'specdis_raw_binary_flag',
    'spdist_valid', 'spdist_distance', 'spdist_sigma_mu', 'parallax_eligible',
    'p_corr', 'gaia_parallax_error', 'nonstellar_risk', 'redrock_nonstellar_evidence']


def main():
    require_compute(); inputs_dir = D/'cohorts'; evidence_dir = D/'evaluation'
    inputs_dir.mkdir(exist_ok=True); evidence_dir.mkdir(exist_ok=True)
    paths = [D/'references.npz', D/'sed_cloud.npz', D/'calibration.npz',
             ROOT/'data/v2_2/application_inputs.npz', ROOT/'scripts/v3_0_cohorts.py']
    paths.extend(sorted((D/'application_inputs').glob('field_*.json')))
    frozen = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    marker = inputs_dir/'complete.json'
    freeze = inputs_dir/'inputs.json'
    if freeze.exists(): assert json.loads(freeze.read_text()) == frozen
    else: save_json(freeze, frozen)
    if marker.exists():
        record = json.loads(marker.read_text()); assert record['inputs'] == frozen
        for name, digest in record['outputs'].items(): assert sha(D/name) == digest
        return
    t = load(D/'references.npz'); cal = load(D/'calibration.npz')
    with np.load(D/'sed_cloud.npz') as z: fit_identity = z['identity']
    files = []; roles = {}
    for role, label in [(3, 'hyper'), (4, 'development')]:
        use = t['role'] == role
        assert not np.intersect1d(t['identity'][use], fit_identity).size
        assert (t['TYPE'][use] == 'PSF').all()
        out = {k: t[k][use] for k in INPUT_KEYS}
        out['field'] = np.full(use.sum(), -1, 'i2'); out['design_weight'] = np.ones(use.sum())
        path = inputs_dir/(label+'.npz'); atomic_npz(path, **out); files.append(path)
        path = evidence_dir/(label+'.npz'); atomic_npz(path, **{k: t[k][use] for k in EVIDENCE_KEYS}); files.append(path)
        roles[label] = int(use.sum())
    # More independent real source templates, rather than more noise copies of224.
    mu = np.full(len(t['identity']), np.nan); sigma = mu.copy()
    for c, source in [(1, 'specdis'), (2, 'spdist')]:
        use = t['primary_channel'] == c
        mu[use] = 10+5*np.log10(t[source+'_distance'][use])+cal['map_parameters'][c-1, 0]
        sigma[use] = np.hypot(t[source+'_sigma_mu'][use], cal['map_parameters'][c-1, 1])
    gaia = (t['primary_channel'] == 3) & t['parallax_eligible'] & (t['p_corr'] > 10*t['gaia_parallax_error'])
    mu[gaia] = 10-5*np.log10(t['p_corr'][gaia])
    sigma[gaia] = 5/np.log(10)*t['gaia_parallax_error'][gaia]/t['p_corr'][gaia]
    mag, error = optical_magnitudes(t)
    eligible = ((t['role'] == 4) & t['reference_stellar_eligible'] & np.isfinite(mu)
                & (sigma < .3) & (error.max(axis=1) < .03))
    chosen = []; bank_population = {}
    for c, cap in [(0, 2560), (1, 2560), (2, 2560), (-1, 512)]:
        rows = np.flatnonzero(eligible & (t['class_label'] == c))
        rows = rows[np.argsort(stable_rank(t['identity'][rows], 301901))]
        chosen.extend(rows[:cap]); bank_population[str(c)] = len(rows)
    # Fill unused class quota with independent known-class sources; all IDs fixed.
    remaining = np.flatnonzero(eligible & (t['class_label'] >= 0))
    remaining = remaining[~np.isin(remaining, chosen)]
    remaining = remaining[np.argsort(stable_rank(t['identity'][remaining], 301902))]
    chosen.extend(remaining[:max(8192-len(chosen), 0)]); rows = np.sort(chosen)
    assert len(np.unique(t['identity'][rows])) == len(rows)
    assert not np.intersect1d(t['identity'][rows], fit_identity).size
    feh, feh_valid = rvs_metallicity(t['rvs_feh'][rows], t['rvs_teff'][rows], t['rvs_logg'][rows])
    scale = 10.**(.4*(mu[rows]-10.))[:, None]/t['transmission'][rows]
    # Weak optional-band measurements are evidence, not asserted exact true SEDs.
    # The simulation module samples their positive-flux measurement posterior.
    bank = {k: t[k][rows] for k in INPUT_KEYS}
    bank.update(class_label=t['class_label'][rows], reference_mu=mu[rows], reference_sigma=sigma[rows],
        flux_at_1kpc=t['flux'][rows]*scale, flux_error_at_1kpc=t['flux_error'][rows]*scale,
        primary_channel=t['primary_channel'][rows], corrected_feh=feh,
        corrected_feh_valid=feh_valid & t['rvs_parameter_quality'][rows], original_rows=rows)
    path = D/'empirical_transfer_bank.npz'; atomic_npz(path, **bank); files.append(path)
    del t
    # Original historical application sample only; no prospective IDs are read.
    with np.load(ROOT/'data/v2_2/application_inputs.npz') as z:
        ids = z['identity']
        weight_key = 'sample_weight' if 'sample_weight' in z.files else 'weight'
        if weight_key not in z.files: raise ValueError('Original application design weight missing')
        weights = z[weight_key]
    pieces = []; matched = np.zeros(len(ids), bool)
    for field in range(14):
        parent = load(D/'application_inputs'/f'field_{field:02}.npz')
        j = unique_join(ids, parent['identity']); use = j >= 0
        if (matched & use).any(): raise ValueError('Application identity has multiple field owners')
        matched |= use
        if not use.any(): continue
        part = {k: parent[k][j[use]] for k in INPUT_KEYS}
        part['field'] = np.full(use.sum(), field, 'i2')
        part['design_weight'] = weights[use]; part['original_sample_row'] = np.flatnonzero(use)
        pieces.append(part)
    if not matched.all(): raise ValueError('Original application sample not fully recovered')
    result = {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
    order = np.argsort(result['original_sample_row'])
    result = {k: v[order] for k, v in result.items()}
    np.testing.assert_array_equal(result['identity'], ids)
    path = inputs_dir/'application.npz'; atomic_npz(path, **result); files.append(path)
    save_json(marker, dict(inputs=frozen, outputs={str(p.relative_to(D)): sha(p) for p in files},
        roles=roles, application=len(ids), bank_sources=len(rows), bank_class_population=bank_population,
        bank_missing_class=int((bank['class_label'] < 0).sum()),
        bank_caveat='Independent source IDs from seen development; teacher distance scale and template flux uncertainty remain. Not blind or exact physical SED truth.',
        prospective_outcomes_opened=False, job=os.getenv('SLURM_JOB_ID')))
    print('DEVELOPMENT COHORTS', roles, 'APPLICATION', len(ids), 'EMPIRICAL BANK', len(rows), flush=True)


if __name__ == '__main__': main()
