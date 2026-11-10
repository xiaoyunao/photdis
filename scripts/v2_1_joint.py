#!/usr/bin/env python3
"""Checkpointed full-PDF frozen joint audit; never refit or change deployment."""
import argparse
import concurrent.futures
import multiprocessing as mp
import os
import time
import json
import joblib
import numpy as np
from v2_1_near import DEST, TABLES, ROOT, load, subset, csv_rows, sha, save_json, atomic_npz, strata, reference
from baseline_v1_4_experiments import arrays
from photdis.stage1 import require_compute
from photdis.reference_v1_4 import unique_join
from photdis.baseline_v0 import BANDS, ERR
from photdis.baseline_v1 import integrate, quantiles_interpolated
from photdis.probability_v2_1 import JointReferenceForest, near_mass, migration, psf_admission, interval_score

_J = _T = None


def inputs():
    model_path = ROOT/'data/baseline_v1_4/experiments/expanded_A_distance.joblib'
    saved = json.loads(model_path.with_suffix('.json').read_text())
    assert sha(model_path) == saved['sha256']
    paths = dict(model=model_path, development=DEST/'development.npz',
                 old=ROOT/'data/baseline_v1_3/references.npz',
                 new=ROOT/'data/baseline_v1_4/references.npz',
                 identities=ROOT/'data/baseline_v1_4/experiments/expanded_A_identities.npz',
                 legacy=ROOT/'data/baseline_v1_4/experiments/expanded_A_legacy_training.npz',
                 core=ROOT/'src/photdis/probability_v2_1.py', runner=ROOT/'scripts/v2_1_joint.py',
                 protocol=ROOT/'docs/v2_1_protocol.md')
    freeze = dict(inputs={k: sha(v) for k, v in paths.items()}, draws=128, chunk_size=128,
                  model='expanded_A', class_channels=['dwarf', 'evolved', 'other', 'unlabeled'])
    return paths, freeze


def prepare(paths, freeze):
    model = joblib.load(paths['model'])
    pieces = []
    keys = list(dict.fromkeys([*BANDS, *ERR, 'identity', 'ls_id', 'Mg', 'class_label', 'label_source']))
    for which, idname in [('old', 'legacy'), ('new', 'identities')]:
        ids = load(paths[idname])['distance_train_identity']
        with np.load(paths[which]) as z:
            index = unique_join(ids, z['identity'])
            assert (index >= 0).all()
            pieces.append({k: z[k][index] for k in keys})
    train = {k: np.concatenate([p[k] for p in pieces]) for k in keys}
    assert len(np.unique(train['identity'])) == len(train['identity'])
    m, e, ids = arrays(train, np.arange(len(train['identity'])))
    # Includes order, not just set equality: frozen NN fit retained original design.
    np.testing.assert_allclose((model.feature_matrix(m)-model.mean)/model.scale,
                               model.nn._fit_X, rtol=0, atol=1e-12)
    native = load(ROOT/'data/v2_0/native_report.npz')
    j = unique_join(train['ls_id'], native['ls_id'])
    types = np.full(len(j), '', dtype='U4')
    types[j >= 0] = native['TYPE'][j[j >= 0]]
    good, morphology = psf_admission(types)
    train['TYPE'] = types
    atomic_npz(DEST/'frozen_training_audit.npz', **train)
    joint = JointReferenceForest(model, m, train['Mg'], train['class_label'], train['label_source'])
    save_json(DEST/'joint_reconstruction.json', dict(training_n=len(ids), morphology=morphology,
              training_order_verified=True, max_marginal_errors=joint.reconstruction_max_errors,
              source_codes=joint.source_values,
              historical_exception='Exact original members retained only for frozen audit; new probe must require verified PSF'))
    return joint


def chunk(rows):
    number = int(rows[0])
    path = DEST/'joint_chunks'/f'{number:07}.npz'
    marker = path.with_suffix('.json')
    ids = _T['identity'][rows]
    if marker.exists():
        assert sha(path) == json.loads(marker.read_text())['sha256']
        old = load(path)
        assert np.array_equal(old['identity'], ids)
        return str(path)
    m, e, _ = arrays(_T, rows)
    p = _J.predict(m, e, ids)
    # The new joint marginal is checked against actual stored frozen outputs for
    # every object, and fresh scalar integration on the first objects per chunk.
    mu_error = float(np.max(abs(p['mu']-_T['mu'][rows])))
    scalar = integrate(_J.model, m[:2], e[:2], ids[:2], draws=128)['wide']
    pdf_error = float(np.max(abs(p['pdf'][:2]-scalar['pdf'])))
    class_error = float(np.max(abs(p['joint'][:2, :4].sum(axis=2)-scalar['weights'])))
    if mu_error > 5e-4 or max(pdf_error, class_error) > 3e-6:
        raise ValueError(f'Frozen reproduction failed: {mu_error}, {pdf_error}, {class_error}')
    edges = _J.model.edges
    qchannels = np.column_stack([near_mass(p['joint'][:, k], edges, m[:, 0])
                                for k in range(_J.nchannel)])
    nnom = near_mass(p['nominal'][:, :4].sum(axis=1), edges, m[:, 0])
    atomic_npz(path, identity=ids, TYPE=_T['TYPE'][rows], rows=rows, mu=p['mu'],
               pdf=p['pdf'].astype('f4'), joint=p['joint'].astype('f4'),
               nominal=p['nominal'].astype('f4'), qnear=qchannels[:, :4].sum(axis=1),
               joint_near=qchannels, nominal_qnear=nnom)
    save_json(marker, dict(sha256=sha(path), n=len(rows), mu_error=mu_error,
                           pdf_error=pdf_error, class_error=class_error))
    return str(path)


def run(workers):
    global _J, _T
    require_compute()
    paths, fingerprint = inputs()
    freeze = DEST/'joint_freeze.json'
    if freeze.exists():
        assert json.loads(freeze.read_text()) == fingerprint, 'Changed inputs/code: explicit migration required'
    else:
        save_json(freeze, fingerprint)
    _T = load(paths['development'])
    assert (_T['TYPE'] == 'PSF').all()
    _J = prepare(paths, fingerprint)
    assert not np.intersect1d(_T['identity'], load(DEST/'frozen_training_audit.npz')['identity']).size
    (DEST/'joint_chunks').mkdir(exist_ok=True)
    atomic_npz(DEST/'joint_grid.npz', edges=_J.model.edges, source_codes=_J.source_values)
    jobs = [np.arange(i, min(i+128, len(_T['identity']))) for i in range(0, len(_T['identity']), 128)]
    start = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork')) as pool:
        for i, path in enumerate(pool.map(chunk, jobs), 1):
            if i == 1 or i % 16 == 0:
                print('JOINT', i, '/', len(jobs), 'seconds', round(time.time()-start, 1), flush=True)
    save_json(DEST/'joint_complete.json', dict(n=len(_T['identity']), chunks=len(jobs),
              seconds=time.time()-start, job_id=os.getenv('SLURM_JOB_ID'), freeze_sha256=sha(freeze)))
    summarize()


def summarize():
    require_compute()
    t = load(DEST/'development.npz')
    grid = load(DEST/'joint_grid.npz')
    edges = grid['edges']
    predictions = []
    probability_rows, reliability, risks, classes = [], [], [], []
    for path in sorted((DEST/'joint_chunks').glob('*.npz')):
        marker = json.loads(path.with_suffix('.json').read_text())
        assert sha(path) == marker['sha256']
        p = load(path)
        rows = p['rows']
        q = subset(t, rows)
        assert np.array_equal(p['identity'], q['identity']) and (p['TYPE'] == 'PSF').all()
        out = {k: p[k] for k in ('identity', 'qnear', 'nominal_qnear', 'joint_near')}
        out['class_mass'] = p['joint'][:, :4].sum(axis=2)
        out['source_mass'] = p['joint'][:, 4:].sum(axis=2)
        for factor in [.5, 2.]:
            joint = p['joint'][:, :4].copy()
            joint[:, 1] *= factor
            joint /= joint.sum(axis=(1, 2))[:, None, None]
            out['qnear_stress_'+str(factor)] = near_mass(joint.sum(axis=1), edges, q['g0'])
        for kind in ('chosen', 'Gaia', 'specdis', 'spdist'):
            ref, sigma = reference(q, kind)
            target = q['g0']-ref
            finite = np.isfinite(target)
            idx = np.searchsorted(edges, np.where(finite, target, edges[0]-1), side='right')-1
            inside = finite & (idx >= 0) & (idx < len(edges)-1)
            density = np.zeros(len(idx))
            density[inside] = p['pdf'][np.flatnonzero(inside), idx[inside]]/np.diff(edges)[idx[inside]]
            with np.errstate(divide='ignore'):
                logscore = np.log(density)
            logscore[~finite] = np.nan
            out['log_teacher_'+kind] = logscore
            out['interval_score_'+kind] = interval_score(p['mu'], ref)
        for k, label in [(0, 'dwarf'), (1, 'evolved'), (2, 'other')]:
            mass = out['class_mass'][:, k]
            available = (q['class_label'] == k) & (mass > 0)
            cp = p['joint'][available, k]/mass[available, None]
            cmu = q['g0'][available, None]-quantiles_interpolated(cp, edges)[:, ::-1]
            ref = q['ref_chosen'][available]
            for j, index in enumerate(np.flatnonzero(available)):
                classes.append(dict(identity=int(q['identity'][index]), split=int(q['split'][index]),
                                    group=label, reference_mu=ref[j], mu50=cmu[j, 2],
                                    coverage68=bool(cmu[j, 1] <= ref[j] <= cmu[j, 3]),
                                    width68=cmu[j, 3]-cmu[j, 1], class_mass=mass[index]))
        predictions.append(out)
    p = {k: np.concatenate([a[k] for a in predictions]) for k in predictions[0]}
    assert np.array_equal(p['identity'], t['identity'])
    atomic_npz(DEST/'probability.npz', TYPE=t['TYPE'], **p)
    for split, role in [(1, 'cal'), (2, 'development_test')]:
        for name, group in strata(t).items():
            base = (t['split'] == split) & group
            for kind in ('chosen', 'Gaia', 'specdis', 'spdist'):
                ref, sigma = reference(t, kind)
                valid = base & np.isfinite(ref)
                certain = valid & np.isfinite(sigma) & (sigma > 0) & ((ref+sigma < 15) | (ref-sigma > 15))
                for certainty, mask in [('point_reference', valid), ('certain_1sigma', certain)]:
                    near = ref[mask] < 15
                    qn = p['qnear'][mask]
                    ls = p['log_teacher_'+kind][mask]
                    probability_rows.append(dict(split=role, group=name, reference=kind, boundary=certainty,
                        n=int(mask.sum()), brier=float(np.mean((qn-near)**2)) if len(qn) else None,
                        zero_teacher_density=int(np.isneginf(ls).sum()),
                        mean_teacher_log_score=float(np.mean(ls)) if len(ls) and np.isfinite(ls).all() else None,
                        mean_log_score_status='minus_infinity' if np.isneginf(ls).any() else 'finite_or_empty',
                        interval_score68=float(np.mean(p['interval_score_'+kind][mask])) if len(ls) else None))
                    for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
                        use = (qn >= lo) & ((qn < hi) if hi < 1 else (qn <= hi))
                        reliability.append(dict(split=role, group=name, reference=kind, boundary=certainty,
                            low=lo, high=hi, n=int(use.sum()), mean_q=float(np.mean(qn[use])) if use.any() else None,
                            observed_near=float(np.mean(near[use])) if use.any() else None))
                    for factor, key in [(1., 'qnear'), (.5, 'qnear_stress_0.5'), (2., 'qnear_stress_2.0')]:
                        for threshold in [.5, .8, .95]:
                            risks.append(dict(split=role, group=name, reference=kind, boundary=certainty,
                                evolved_weight_stress=factor, threshold=threshold,
                                **migration(ref[mask], p[key][mask] > threshold, sigma[mask])))
    csv_rows(TABLES/'probability_scores.csv', probability_rows)
    csv_rows(TABLES/'reliability.csv', reliability)
    csv_rows(TABLES/'risk_retention.csv', risks)
    csv_rows(TABLES/'class_conditional.csv', classes)
    # Select only on cal and conservative point-reference risk, never test tuning.
    choices = [r for r in risks if r['split'] == 'cal' and r['group'] == 'all'
               and r['reference'] == 'chosen' and r['boundary'] == 'point_reference'
               and r['evolved_weight_stress'] == 1 and r['far_contamination'] is not None
               and r['far_contamination'] <= .05 and r['near_selected']+r['far_selected'] >= 100]
    rule = max(choices, key=lambda r: r['completeness']) if choices else None
    save_json(DEST/'near_rule.json', dict(cal_selected_threshold=None if rule is None else rule['threshold'],
               target_reference_contamination=.05, minimum_selected=100,
               claim='Reference-development rule only; not independent validation or application contamination guarantee'))
    print('PROBABILITY_SUMMARY_COMPLETE', len(t['identity']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args()
    summarize() if args.summarize else run(args.workers)
