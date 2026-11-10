#!/usr/bin/env python3
"""One class-partition distance candidate and same-PSF-identity QRF control."""
import concurrent.futures
import multiprocessing as mp
import os
import json
import joblib
import numpy as np
from v2_1_near import ROOT, DEST, TABLES, load, subset, csv_rows, save_json, sha, atomic_npz
from baseline_v1_4_experiments import arrays
from photdis.stage1 import require_compute
from photdis.baseline_v0 import HistogramQRF, metrics
from photdis.baseline_v1 import integrate
from photdis.baseline_v1_2 import full_parallax_log_density
from photdis.probability_v2_1 import near_mass, migration, interval_score
from photdis.partition_v2_1 import class_partition_candidate

_T = _MODEL = _NAME = None
OUT = DEST/'partition_candidate'


def chunk(rows):
    path = OUT/f'{_NAME}_{rows[0]:07}.npz'
    marker = path.with_suffix('.json')
    if marker.exists():
        assert sha(path) == json.loads(marker.read_text())['sha256']
        return path
    m, e, ids = arrays(_T, rows)
    p = integrate(_MODEL, m, e, ids, draws=128)['wide']
    ga = _T['parallax_eligible'][rows]
    log = np.full(len(rows), np.nan)
    log[ga] = full_parallax_log_density(p['pdf'][ga], _MODEL.edges, m[ga, 0],
              _T['p_corr'][rows][ga], _T['gaia_parallax_error'][rows][ga], order=16)
    atomic_npz(path, identity=ids, TYPE=_T['TYPE'][rows], rows=rows, mu=p['mu'], pdf=p['pdf'],
               qnear=near_mass(p['pdf'], _MODEL.edges, m[:, 0]), gaia_log=log)
    save_json(marker, dict(sha256=sha(path)))
    return path


def main():
    global _T, _MODEL, _NAME
    require_compute()
    OUT.mkdir(exist_ok=True)
    train = load(DEST/'information_train.npz')
    t = load(DEST/'development.npz')
    _T = subset(t, t['C'] & t['i_usable'])
    assert (train['TYPE'] == 'PSF').all() and (_T['TYPE'] == 'PSF').all()
    assert not np.intersect1d(train['identity'], _T['identity']).size
    freeze = {str(p.relative_to(ROOT)): sha(p) for p in [DEST/'information_train.npz', DEST/'development.npz',
              ROOT/'docs/v2_1_partition_candidate.md', ROOT/'src/photdis/partition_v2_1.py', ROOT/'scripts/v2_1_partition_candidate.py']}
    marker = OUT/'freeze.json'
    if marker.exists():
        assert json.loads(marker.read_text()) == freeze
    else:
        save_json(marker, freeze)
    m = arrays(train, np.arange(len(train['identity'])))[0]
    if (OUT/'candidate.joblib').exists():
        control, candidate = [joblib.load(OUT/(name+'.joblib')) for name in ['control', 'candidate']]
    else:
        control = HistogramQRF(3)
        control.forest.n_jobs = 16
        control.fit(m, train['Mg'], train['class_label'])
        candidate = class_partition_candidate(control, m, train['Mg'], train['class_label'])
        for name, model in [('control', control), ('candidate', candidate)]:
            model.forest.n_jobs = 1
            joblib.dump(model, OUT/(name+'.joblib'))
    atomic_npz(OUT/'identities.npz', train_identity=train['identity'], train_TYPE=train['TYPE'],
               evaluation_identity=_T['identity'], evaluation_TYPE=_T['TYPE'])
    report = []
    for name, model in [('control', control), ('candidate', candidate)]:
        _NAME, _MODEL = name, model
        jobs = [np.arange(i, min(i+256, len(_T['identity']))) for i in range(0, len(_T['identity']), 256)]
        with concurrent.futures.ProcessPoolExecutor(max_workers=16, mp_context=mp.get_context('fork')) as pool:
            paths = []
            for i, path in enumerate(pool.map(chunk, jobs), 1):
                paths.append(path)
                if i % 32 == 0:
                    print('PARTITION', name, i, '/', len(jobs), flush=True)
        parts = []
        for path in paths:
            with np.load(path) as z:
                parts.append({k: z[k] for k in ['identity', 'mu', 'qnear', 'gaia_log']})
        p = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
        assert np.array_equal(p['identity'], _T['identity'])
        atomic_npz(OUT/(name+'_evaluation.npz'), TYPE=_T['TYPE'], **p)
        for split, role in [(1, 'cal'), (2, 'development_test')]:
            for c, group in [(None, 'all'), (0, 'dwarf'), (1, 'evolved'), (2, 'other'), (-1, 'unknown')]:
                use = _T['split'] == split
                if c is not None:
                    use &= _T['class_label'] == c
                valid = use & np.isfinite(_T['ref_chosen'])
                ga = use & _T['parallax_eligible']
                ref = _T['ref_chosen'][valid]
                report.append(dict(model=name, split=role, group=group,
                    **metrics(p['mu'][valid], ref),
                    brier=float(np.mean((p['qnear'][valid]-(ref < 15))**2)),
                    interval_score68=float(np.mean(interval_score(p['mu'][valid], ref))),
                    gaia_n=int(ga.sum()), gaia_negative=int((ga & (_T['p_corr'] < 0)).sum()),
                    gaia_log=float(p['gaia_log'][ga].mean()),
                    **{'near95_'+k: v for k, v in migration(ref, p['qnear'][valid] > .95).items()}))
    csv_rows(TABLES/'partition_candidate.csv', report)
    def row(model, group):
        return next(r for r in report if r['model'] == model and r['split'] == 'cal' and r['group'] == group)
    a, b = row('control', 'all'), row('candidate', 'all')
    guards = dict(gaia_log=b['gaia_log'] >= a['gaia_log'], all_tail=b['tail_1'] <= a['tail_1'],
        evolved_tail=row('candidate', 'evolved')['tail_1'] <= row('control', 'evolved')['tail_1'],
        dwarf_tail=row('candidate', 'dwarf')['tail_1'] <= row('control', 'dwarf')['tail_1']+.005,
        near_contamination=b['near95_far_contamination'] <= a['near95_far_contamination']+.005,
        near_completeness=b['near95_completeness'] >= a['near95_completeness']-.02)
    save_json(DEST/'partition_candidate_complete.json', dict(job_id=os.getenv('SLURM_JOB_ID'), guards=guards,
        cal_pass=all(guards.values()), n_train=len(m), n_evaluation=len(_T['identity']),
        frozen_training=freeze, candidate='Balanced class split loss, natural leaf distance histogram',
        deployment='unchanged', independent_validation='not established'))


if __name__ == '__main__':
    main()
