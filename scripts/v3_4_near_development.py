#!/usr/bin/env python3
"""Compare fixed distance views on a photometric dwarf selection and opened references."""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path

import joblib
import numpy as np
from scipy.special import logsumexp

from photdis.stage1 import galactic, require_compute
from photdis.v3.controls import apply_temperature, optical_magnitudes, probe_features
from photdis.v3.metrics import distance_summary, histogram_parallax, observed_parallax
from photdis.v3.model import inference_model
from photdis.v3.data import stable_rank
from photdis.v3_3 import class_log_likelihood, frozen_qrf, reference_log_curves
from v2_1_near import ROOT, atomic_npz, csv_rows, load, save_json, sha
import v3_0_evaluate as historical
from v3_0_response import observation
from v3_3_catalog import FREEZE, joint_summary, read_frozen
from v3_3_sed import BASE, MANIFESTS, verified
from v3_4_near_development_inputs import CONFIG, OUT as INPUTS, cells

OUT = INPUTS.parent/'comparison'
MU = np.linspace(-5., 30., 1401)
DISTANCE_EDGES = np.array([0., 2., 5., 10., 20., 50., np.inf])
VIEWS = ('nominal', 'candidate', 'qrf')
_STATE = None


def one(i):
    photo, models, priors, evidence, qrf = _STATE
    use = evidence['parallax_eligible'][i]
    parallax = evidence['p_corr'][i]
    error = evidence['gaia_parallax_error'][i]
    o = observation(photo, i)
    values = []
    for name in VIEWS[:2]:
        ll = class_log_likelihood(models[name], o, MU)
        p = joint_summary(MU, ll, priors[name])
        scores = observed_parallax(MU, logsumexp(reference_log_curves(ll, MU, priors[name]), axis=0),
                                   parallax, error) if use else {}
        values.append(dict(quantiles_kpc=p['distance_quantiles_kpc'], qnear=p['qnear'],
                           mass=p['class_distance_bins'].sum(axis=0), class_probability=p['class_probability'],
                           numerical_flag=p['numerical_flag'], gaia_log_score=scores.get('log_score', np.nan)))
    edges, mass = qrf['mu_edges'][i], qrf['mass'][i]
    bounds = np.r_[-np.inf, 10+5*np.log10(DISTANCE_EDGES[1:])]
    cdf = np.clip((bounds[:, None]-edges[:-1])/np.diff(edges), 0, 1) @ mass
    scores = histogram_parallax(edges, mass, parallax, error, order=32) if use else {}
    values.append(dict(quantiles_kpc=qrf['distance'][i], qnear=cdf[3], mass=np.diff(cdf),
                       class_probability=qrf['weights'][i], numerical_flag=False,
                       gaia_log_score=scores.get('log_score', np.nan)))
    return values


def align_evidence(photo, source):
    ix = photo['opened_reference_row']
    known = ix >= 0
    if (ix[known] >= len(source['identity'])).any():
        raise ValueError('Reference lookup exceeds historical table')
    np.testing.assert_array_equal(photo['identity'][known], source['identity'][ix[known]])
    if not (source['TYPE'][ix[known]] == 'PSF').all() or not (source['role'][ix[known]] == 4).all():
        raise ValueError('Expected opened strict-PSF reference rows')
    data = {}
    for key, value in source.items():
        fill = False if value.dtype.kind == 'b' else np.nan if value.dtype.kind == 'f' else '' if value.dtype.kind in 'SU' else -1
        a = np.full((len(ix),)+value.shape[1:], fill, dtype=value.dtype)
        a[known] = value[ix[known]]
        data[key] = a
    return data


def append_opened_photo(photo, opened, cfg):
    # The archived application frame can have no common identities with the
    # development split. Apply the same observational cuts to available
    # development photometry and keep its application design weight at zero.
    optical = [0, 1, 3]
    valid = ((opened['TYPE'] == 'PSF') & (opened['photometric_system'] == 'DECam_DR11')
             & opened['band_mask'][:, optical].all(axis=1)
             & (opened['flux'][:, optical] > 0).all(axis=1))
    rows = np.flatnonzero(valid)
    base = {k: v[rows] for k, v in opened.items()}
    _, errs = optical_magnitudes(base)
    observed_r = 22.5-2.5*np.log10(base['flux'][:, 1])
    latitude = galactic(base['RA'], base['DEC'])[1]
    use = ((observed_r >= cfg['observed_r_edges'][0]) & (observed_r < cfg['observed_r_edges'][-1])
           & (errs.max(axis=1) < .1) & (np.abs(latitude) > 30)
           & (base['EBV'] >= 0) & (base['EBV'] < .03)
           & ~np.isin(base['identity'], photo['identity']))
    rows = rows[use]
    selected = {k: v[rows] for k, v in opened.items()}
    group, _, _ = cells(selected, cfg)
    # Stable cap within each previously opened sky block and observable cell.
    # Class/distance availability does not enter these ranks.
    block = np.stack([selected['cell'], group], axis=1)
    _, block_id = np.unique(block, axis=0, return_inverse=True)
    block_id = block_id.ravel()
    order = np.lexsort((stable_rank(selected['identity'], cfg['sample_salt']), block_id))
    groups = np.split(order, np.flatnonzero(np.diff(block_id[order]))+1)
    rows = rows[np.concatenate([g[:cfg['sample_cap_per_field_cell']] for g in groups])]
    other = {k: v[rows] for k, v in opened.items() if k in photo}
    g, r, c = cells({k: v[rows] for k, v in opened.items()}, cfg)
    other.update(field=np.full(len(rows), -1, 'i2'), observed_r=r, dereddened_g_r=c, stratum=g,
                 application_sample=np.zeros(len(rows), bool), application_weight=np.zeros(len(rows)),
                 opened_reference_row=rows, parent_row=np.full(len(rows), -1, 'i8'))
    return {k: np.concatenate((v, other[k])) for k, v in photo.items()}


def summarize(photo, evidence, center, width, dwarf_score, values, qrf_supported, near_mu):
    # These are development views. Their use does not assign release eligibility.
    nominal_q = values['nominal']['qnear']
    masks = {'photometric_envelope': qrf_supported,
             'dwarf_score_080': qrf_supported & (dwarf_score >= .80),
             'near_dwarf_comparison': qrf_supported & (dwarf_score >= .80) & (nominal_q >= .90),
             'strict_near_dwarf_comparison': qrf_supported & (dwarf_score >= .95) & (nominal_q >= .97)}
    label = evidence['class_label']
    is_giant = label == 1
    available = np.isfinite(center)
    known_near = available & (center+1.96*width < near_mu)
    known_far = available & (center-1.96*width > near_mu)
    r, color = photo['observed_r'], photo['dereddened_g_r']
    groups = {'all': np.ones(len(r), bool),
              'r_16_18': (r >= 16) & (r < 18), 'r_18_20': (r >= 18) & (r < 20),
              'red': color >= .8, 'blue': color < .8}
    groups.update({f'field_{field:02d}': photo['field'] == field for field in np.unique(photo['field']) if field >= 0})
    groups['opened_separate_frame'] = photo['field'] < 0
    weight = photo['application_weight']
    distance_rows, contamination_rows, distributions = [], [], []
    for selection, include in masks.items():
        for group, mask in groups.items():
            use = include & mask
            if not use.any():
                continue
            matches = use & (photo['opened_reference_row'] >= 0)
            n_label = int((use & (label >= 0)).sum())
            n_giant = int((use & is_giant).sum())
            n_far = int((use & is_giant & known_far).sum())
            n_near = int((use & is_giant & known_near).sum())
            contamination_rows.append(dict(selection=selection, group=group,
                parent_estimate=float(weight[use].sum()), sample_n=int((use & photo['application_sample']).sum()),
                reference_n=int(matches.sum()), labeled_n=n_label, dwarf=int((use & (label == 0)).sum()),
                giant=n_giant, other=int((use & (label == 2)).sum()), without_class=int(matches.sum())-n_label,
                giant_near_core=n_near, giant_far_core=n_far, giant_distance_unresolved=n_giant-n_far-n_near))
            for key, v in values.items():
                use_score = use & np.isfinite(v['gaia_log_score'])
                for name, subset in [('all_labels', use), ('dwarf', use & (label == 0)),
                                     ('giant', use & is_giant), ('other', use & (label == 2))]:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        q = 10+5*np.log10(v['quantiles_kpc'][subset])
                    distance_rows.append(dict(selection=selection, group=group, view=key, reference_class=name,
                        gaia_n=int((subset & use_score).sum()),
                        mean_gaia_score=float(v['gaia_log_score'][subset & use_score].mean())
                            if (subset & use_score).any() else np.nan,
                        **distance_summary(q, center[subset])))
                estimated = weight[use] @ v['mass'][use]
                np.testing.assert_allclose(estimated.sum(), weight[use].sum(), atol=1e-6, rtol=0)
                for j, count in enumerate(estimated):
                    distributions.append(dict(selection=selection, group=group, view=key,
                        lower_kpc=DISTANCE_EDGES[j], upper_kpc=DISTANCE_EDGES[j+1], expected_n=count))
    return masks, distance_rows, contamination_rows, distributions


def main():
    require_compute()
    cfg = json.loads(CONFIG.read_text())
    marker = INPUTS/'complete.json'
    index = json.loads(marker.read_text())
    photo_path = verified(INPUTS/'photometry.npz', index['outputs']['photometry.npz'])
    photo = load(photo_path)
    if not (photo['TYPE'] == 'PSF').all() or len(np.unique(photo['identity'])) != len(photo['identity']):
        raise ValueError('Expected a strict unique PSF roster')
    cohort = json.loads((MANIFESTS/'cohorts.json').read_text())['outputs']
    ref_path = verified(BASE/'evaluation/development.npz', cohort['evaluation/development.npz'])
    opened_path = verified(BASE/'cohorts/development.npz', cohort['cohorts/development.npz'])
    opened_table, opened_reference = load(opened_path), load(ref_path)
    np.testing.assert_array_equal(opened_table['identity'], opened_reference['identity'])
    photo = append_opened_photo(photo, opened_table, cfg)
    if len(np.unique(photo['identity'])) != len(photo['identity']):
        raise ValueError('Duplicate sources after adding opened development data')
    evidence = align_evidence(photo, opened_reference)
    print('OBSERVABLE DEVELOPMENT ROSTER', len(photo['identity']),
          int((photo['opened_reference_row'] >= 0).sum()), flush=True)
    cal_path = verified(BASE/'calibration.npz', json.loads((MANIFESTS/'calibration.json').read_text())['sha256'])
    historical._CAL = load(cal_path)
    center, width, *_ = historical.evidence_views(evidence)['primary']
    controls = json.loads((MANIFESTS/'controls/complete.json').read_text())
    probe_path = verified(BASE/'controls/probes.joblib', controls['outputs']['probes.joblib'])
    probe = joblib.load(probe_path)['grz']
    np.testing.assert_array_equal(probe['model'].classes_, [0, 1, 2])
    cls = apply_temperature(probe['model'].predict_proba(probe_features(photo, 'grz')), probe['temperature'])
    paths = [verified(ROOT/rec['path'], rec['sha256']) for rec in
             (read_frozen()['models'][name] for name in VIEWS)]
    models = dict(zip(VIEWS, map(joblib.load, paths)))
    opened = photo['opened_reference_row'] >= 0
    for name, model in models.items():
        if np.intersect1d(model.training_identity, photo['identity'][opened]).size:
            raise ValueError(f'Training identities entered opened validation for {name}')
    mag, err = optical_magnitudes(photo)
    qrf = frozen_qrf(models.pop('qrf'), mag, err, photo['identity'])
    priors = {name: model.weights @ model.classes for name, model in models.items()}
    global _STATE
    _STATE = photo, {name: inference_model(model) for name, model in models.items()}, priors, evidence, qrf
    with ProcessPoolExecutor(max_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', '16')), mp_context=mp.get_context('fork')) as pool:
        result = list(pool.map(one, range(len(photo['identity'])), chunksize=12))
    values = {name: {k: np.asarray([r[i][k] for r in result]) for k in result[0][i]}
              for i, name in enumerate(VIEWS)}
    near_mu = 10+5*np.log10(cfg['near_boundary_kpc'])
    masks, metrics, contamination, distances = summarize(photo, evidence, center, width, cls[:, 0],
                                                          values, qrf['supported'], near_mu)
    OUT.mkdir(parents=True, exist_ok=False)
    keys = ('identity', 'ls_id', 'TYPE', 'RA', 'DEC', 'cell', 'field', 'observed_r',
            'dereddened_g_r', 'band_state', 'application_sample', 'application_weight')
    atomic_npz(OUT/'roster.npz', **{k: photo[k] for k in keys}, reference_class=evidence['class_label'],
               reference_mu=center, reference_sigma_mu=width,
               opened_match=opened, signed_gaia_eligible=evidence['parallax_eligible'])
    for name, view in values.items():
        atomic_npz(OUT/f'{name}.npz', identity=photo['identity'], TYPE=photo['TYPE'],
                   qrf_supported=qrf['supported'], dwarf_selection_score=cls[:, 0], **masks, **view)
    csv_rows(OUT/'matched_distances.csv', metrics)
    csv_rows(OUT/'reference_composition.csv', contamination)
    csv_rows(OUT/'distance_distributions.csv', distances)
    sources = [FREEZE, CONFIG, marker, photo_path, opened_path, ref_path, cal_path, probe_path, *paths, Path(__file__)]
    save_json(OUT/'complete.json', dict(job=os.getenv('SLURM_JOB_ID'), n=len(photo['identity']),
        application_n=int(photo['application_sample'].sum()), opened_n=int(opened.sum()),
        all_PSF=True, new_outcomes_read=False, inputs={str(p.relative_to(ROOT)): sha(p) for p in sources},
        outputs={p.name: sha(p) for p in sorted(OUT.iterdir())},
        interpretation='Development comparison. QRF class weights include unknown labels. Reference counts describe the separately identified opened comparison cohort. Giant distance split uses the calibrated reference core 95% interval.'))


if __name__ == '__main__':
    main()
