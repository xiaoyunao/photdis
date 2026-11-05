#!/usr/bin/env python3
"""One expanded-A QRF contrast, preserving original training and scoring identities."""
import concurrent.futures
import json
import multiprocessing as mp
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone

from baseline_v1_4_experiments import (
    ROOT, OUT, KEYS, arrays, groups, load, save_json, sha, subset, truth, atomic_npz,
)
from photdis.audit import rvs_metallicity
from photdis.baseline_v0 import HistogramQRF
from photdis.baseline_v1 import integrate
from photdis.diagnostics_v1_4 import paired_stats, residual_stats
from photdis.gaia_numerics_v1_4 import refined_gaia
from photdis.stage1 import require_compute

DEST = ROOT / 'data/baseline_v1_5_r0'
EXPERIMENTS = OUT / 'experiments'
BASELINE = EXPERIMENTS / 'expanded_A_distance'
REFINED = OUT / 'gaia_refined/expanded_A'
SPECTRAL = ['rvs_teff', 'rvs_logg', 'rvs_feh', 'rvs_feh_err', 'rvs_parameter_quality']
_T = _MODEL = None


def evaluate_chunk(rows):
    path = DEST / 'chunks' / f'{rows[0]:08d}.npz'
    marker = path.with_suffix('.json')
    if marker.exists():
        assert sha(path) == json.loads(marker.read_text())['sha256']
        assert np.array_equal(load(path)['rows'], rows)
        return path
    mags, errors, ids = arrays(_T, rows)
    p = integrate(_MODEL, mags, errors, ids, draws=128)['wide']
    assert np.isfinite(p['pdf']).all() and np.allclose(p['pdf'].sum(axis=1), 1, atol=2e-5)
    g = _T['parallax_eligible'][rows]
    rr = rows[g]
    gaia = refined_gaia(p['pdf'][g], _MODEL.edges, _T['g0'][rr],
                       _T['p_corr'][rr], _T['gaia_parallax_error'][rr]) if len(rr) else {
                           'full_log_density': np.empty(0), 'full_pit': np.empty(0),
                           'log_order': np.empty(0, dtype='i2'), 'pit_adaptive': np.empty(0, bool),
                       }
    atomic_npz(path, rows=rows, identity=ids, mu=p['mu'],
               supported=p['supported'], ambiguous=p['ambiguous'], gaia_rows=rr,
               **{k: gaia[k] for k in ['full_log_density', 'full_pit', 'log_order', 'pit_adaptive']})
    save_json(marker, dict(sha256=sha(path)))
    return path


def metal_layers(t):
    quality = t['rvs_parameter_quality'] & np.isfinite(t['rvs_feh_err']) & (t['rvs_feh_err'] > 0)
    dwarf = (t['class_label'] == 0) & quality
    raw = t['rvs_feh']
    corrected, domain = rvs_metallicity(raw, t['rvs_teff'], t['rvs_logg'])
    return {
        'dwarf_raw_feh_lt_m1p5': dwarf & (raw < -1.5),
        'dwarf_raw_feh_lt_m1p5_in_domain': dwarf & domain & (raw < -1.5),
        'dwarf_corrected_feh_lt_m1p5': dwarf & domain & (corrected < -1.5),
    }


def comparison(t, old, new, old_gaia):
    layers = groups(t)
    layers.update(metal_layers(t))
    for i in (-1, 0, 1, 2):
        layers[f'class{i}:r18_20:no_W1'] = (t['class_label'] == i) & layers['r18_20:no_W1']
    gaia = t['parallax_eligible']
    report = {}
    for split, label in [(1, 'cal'), (2, 'development_test')]:
        report[label] = {}
        for name, mask in layers.items():
            use = mask & (t['split'] == split)
            d = {}
            for kind in ('chosen', 'Gaia', 'specdis', 'spdist'):
                # Brightness/source and sky/source comparisons share the same rows.
                ref = truth(t, kind)[use]
                a, b = old['mu'][use], new['mu'][use]
                d[kind] = dict(old=residual_stats(a, ref), new=residual_stats(b, ref),
                               paired=paired_stats(a[:, 2]-ref, b[:, 2]-ref, t['cell'][use]))
            q = use[gaia]
            gaia_result = dict(n=int(q.sum()), negative=int((t['p_corr'][use & gaia]<0).sum()))
            for arm, values in [('old', old_gaia), ('new', new)]:
                logd, pit = values['full_log_density'][q], values['full_pit'][q]
                gaia_result[arm] = dict(mean_log_density=np.mean(logd) if len(logd) else None,
                                        pit_histogram=np.histogram(pit, np.linspace(0, 1, 11))[0],
                                        pit_extreme=np.mean((pit<.05)|(pit>.95)) if len(pit) else None)
            d['full_Gaia'] = gaia_result
            report[label][name] = d
    return report


def main():
    global _T, _MODEL
    require_compute()
    (DEST / 'chunks').mkdir(parents=True, exist_ok=True)
    ids_path = EXPERIMENTS / 'expanded_A_identities.npz'
    legacy_path = EXPERIMENTS / 'expanded_A_legacy_training.npz'
    model_path = EXPERIMENTS / 'expanded_A_distance.joblib'
    old_ref = ROOT / 'data/baseline_v1_3/references.npz'
    inputs = [OUT/'references.npz', old_ref, ids_path, legacy_path, model_path,
              BASELINE/'evaluation.npz', BASELINE/'scores.json',
              REFINED/'evaluation.npz', REFINED/'scores.json']
    code = [Path(__file__), *[ROOT/'src/photdis'/f'{m}.py' for m in
            ('baseline_v0', 'baseline_v1', 'gaia_numerics_v1_4', 'diagnostics_v1_4', 'audit')]]
    fingerprint = {str(p.relative_to(ROOT)): sha(p) for p in inputs + code}
    freeze = DEST / 'freeze.json'
    if freeze.exists():
        assert json.loads(freeze.read_text()) == fingerprint
    else:
        save_json(freeze, fingerprint)
    if (DEST/'result.json').exists():
        saved = json.loads((DEST/'result.json').read_text())
        assert sha(DEST/'evaluation.npz') == saved['evaluation_sha256']
        print('Verified completed experiment')
        return
    assembly = json.loads((OUT/'assembly.json').read_text())
    assert fingerprint[str(inputs[0].relative_to(ROOT))] == assembly['reference_sha256']
    for directory in (BASELINE, REFINED):
        marker = json.loads((directory/'scores.json').read_text())
        assert fingerprint[str((directory/'evaluation.npz').relative_to(ROOT))] == marker['evaluation_sha256']
    with np.load(inputs[0]) as z:
        _T = {k: z[k] for k in KEYS+SPECTRAL}
    ids, legacy_ids = load(ids_path), load(legacy_path)
    old, old_gaia = load(BASELINE/'evaluation.npz'), load(REFINED/'evaluation.npz')
    rows = old['rows']
    assert np.array_equal(_T['identity'][rows], old['identity'])
    assert np.array_equal(old['identity'], ids['distance_eval_identity'])
    assert np.array_equal(rows[_T['parallax_eligible'][rows]], old_gaia['rows'])
    assert np.array_equal(_T['identity'][old_gaia['rows']], old_gaia['identity'])
    assert np.isin(_T['split'][rows], [1, 2]).all()
    dtrain = np.flatnonzero((_T['split'] == 0) & _T['fit_eligible'])
    assert np.array_equal(_T['identity'][dtrain], ids['distance_train_identity'])
    with np.load(old_ref) as z:
        legacy_rows = np.flatnonzero((z['split']==0) & z['fit_eligible'])
        assert np.array_equal(z['identity'][legacy_rows], legacy_ids['distance_train_identity'])
        mags = np.column_stack([z[k][legacy_rows] for k in ('g0','r0','z0','w10','w20')])
        target, labels = z['Mg'][legacy_rows], z['class_label'][legacy_rows]
    train_identity = np.r_[legacy_ids['distance_train_identity'], ids['distance_train_identity']]
    assert len(np.unique(train_identity)) == len(train_identity)
    assert not np.intersect1d(train_identity, old['identity']).size
    mags = np.concatenate([mags, arrays(_T, dtrain)[0]])
    target = np.r_[target, _T['Mg'][dtrain]]
    labels = np.r_[labels, _T['class_label'][dtrain]]
    assert np.isfinite(mags[:, :3]).all() and np.isfinite(target).all()
    baseline = joblib.load(model_path)
    assert baseline.dim == 2 and baseline.forest.n_features_in_ == 2
    assert baseline.forest.estimators_[0].tree_.n_node_samples[0] == len(target)
    if (DEST/'model.joblib').exists():
        _MODEL = joblib.load(DEST/'model.joblib')
    else:
        _MODEL = HistogramQRF(3, bin_width=baseline.bin_width, feature_schema='grz_r0')
        _MODEL.forest = clone(baseline.forest)
        _MODEL.forest.n_jobs = 32
        _MODEL.fit(mags, target, labels)
        _MODEL.forest.n_jobs = baseline.forest.n_jobs
        joblib.dump(_MODEL, DEST/'model.tmp.joblib')
        (DEST/'model.tmp.joblib').replace(DEST/'model.joblib')
    assert _MODEL.feature_schema == 'grz_r0' and _MODEL.forest.n_features_in_ == 3
    assert _MODEL.forest.get_params() == baseline.forest.get_params()
    assert np.array_equal(_MODEL.edges, baseline.edges)
    print('Fitted grz_r0, training:',len(target),'evaluation:',len(rows),flush=True)
    batches = [rows[j:j+512] for j in range(0, len(rows), 512)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=24, mp_context=mp.get_context('fork')) as pool:
        outputs = []
        for j, path in enumerate(pool.map(evaluate_chunk, batches), 1):
            outputs.append(path)
            if j % 20 == 0 or j == len(batches):
                print(f'grz_r0 chunks {j}/{len(batches)}', flush=True)
    chunks = [load(p) for p in outputs]
    new = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    assert np.array_equal(new['identity'], old['identity']) and np.array_equal(new['rows'], rows)
    assert np.array_equal(new['gaia_rows'], old_gaia['rows'])
    assert np.isfinite(new['mu']).all() and (np.diff(new['mu'], axis=1)>=0).all()
    atomic_npz(DEST/'evaluation.npz', **new)
    report = comparison(subset(_T, rows), old, new, old_gaia)
    assert all(sha(ROOT/p) == h for p,h in fingerprint.items())
    save_json(DEST/'result.json', dict(job_id=os.getenv('SLURM_JOB_ID'), train=len(target),
              evaluation=len(rows), gaia=len(new['gaia_rows']), model_sha256=sha(DEST/'model.joblib'),
              evaluation_sha256=sha(DEST/'evaluation.npz'), parameters=baseline.forest.get_params(),
              feature_schema=['g0-r0','r0-z0','r0'], inputs=fingerprint, scores=report))
    print('COMPLETE grz_r0', flush=True)


if __name__ == '__main__':
    main()
