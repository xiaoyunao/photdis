#!/usr/bin/env python3
"""Same-star frozen band probability audit; separate band from training scope."""
import concurrent.futures
import multiprocessing as mp
import os
import joblib
import numpy as np
from v2_1_near import ROOT, DEST, TABLES, load, subset, csv_rows, save_json, sha, atomic_npz
from v2_1_information import sample_indices
from baseline_v1_4_experiments import arrays
from photdis.stage1 import require_compute
from photdis.reference_v1_4 import unique_join
from photdis.baseline_v0 import distance_quantiles, metrics
from photdis.baseline_v1 import integrate
from photdis.probability_v2_1 import psf_admission, near_mass, migration, interval_score

_T = _MODEL = _NAME = None
NAMES = ['matched_grz', 'matched_griz', 'expanded_A', 'expanded_B', 'expanded_C']


def predict(rows):
    optical = _NAME.startswith('matched_')
    m, e, ids = arrays(_T, rows, optical)
    if optical:
        p = _MODEL.predict(m, e, ids, 128)
        mu = distance_quantiles(m[:, 0], p['q_Meffective'])[0]
    else:
        p = integrate(_MODEL, m, e, ids, draws=128)['wide']
        mu = p['mu']
    qnear = near_mass(p['pdf'], _MODEL.edges, m[:, 0])
    return dict(identity=ids, rows=rows, mu=mu, qnear=qnear, pdf=p['pdf'])


def main():
    global _T, _MODEL, _NAME
    require_compute()
    base = load(DEST/'development.npz')
    eligible = (base['split'] == 2) & base['C'] & base['i_usable']
    # Fixed identity sample shared by every pair, never chosen by outcomes.
    rows = np.flatnonzero(eligible)
    rows = rows[sample_indices(base['identity'][rows], 4000)]
    _T = subset(base, rows)
    assert (_T['TYPE'] == 'PSF').all()
    scores, check = [], {}
    for name in NAMES:
        _NAME = name
        path = ROOT/'data/baseline_v1_4/experiments'/f'{name}_distance.joblib'
        _MODEL = joblib.load(path)
        tasks = [np.arange(i, min(i+256, len(rows))) for i in range(0, len(rows), 256)]
        with concurrent.futures.ProcessPoolExecutor(max_workers=8, mp_context=mp.get_context('fork')) as pool:
            parts = list(pool.map(predict, tasks))
        p = {k: np.concatenate([x[k] for x in parts]) for k in parts[0]}
        old = load(ROOT/'data/v2_0'/f'chosen_{name}.npz')
        j = unique_join(_T['identity'], old['identity'])
        assert (j >= 0).all()
        error = float(np.max(abs(old['mu'][j]-p['mu'])))
        assert error < 1e-4
        atomic_npz(DEST/f'band_probability_{name}.npz', **p, TYPE=_T['TYPE'], edges=_MODEL.edges)
        check[name] = dict(mu_max_error=error, model_sha256=sha(path), n=len(j))
        groups = {'all': np.ones(len(j), bool), 'dwarf': _T['class_label'] == 0,
                  'evolved': _T['class_label'] == 1, 'other': _T['class_label'] == 2,
                  'unknown': _T['class_label'] == -1, 'r18_20': (_T['r'] >= 18) & (_T['r'] < 20)}
        for group, use in groups.items():
            valid = use & np.isfinite(_T['ref_chosen'])
            ref = _T['ref_chosen'][valid]
            qnear = p['qnear'][valid]
            ga = use & _T['parallax_eligible']
            for threshold in [.5, .8, .95]:
                scores.append(dict(model=name, group=group, threshold=threshold,
                    brier=float(np.mean((qnear-(ref < 15))**2)) if len(ref) else None,
                    interval_score68=float(np.mean(interval_score(p['mu'][valid], ref))) if len(ref) else None,
                    gaia_n=int(ga.sum()), gaia_negative=int((ga & (_T['p_corr'] < 0)).sum()),
                    gaia_full_log_score=float(np.mean(old['full_log_density'][j][ga])) if ga.any() else None,
                    **migration(ref, qnear > threshold)))
        print('BAND_PDF', name, len(rows), 'reproduction', error, flush=True)
    csv_rows(TABLES/'band_probability.csv', scores)
    save_json(DEST/'band_probability_complete.json', dict(job_id=os.getenv('SLURM_JOB_ID'), checks=check,
        pairings=[['matched_grz', 'matched_griz'], ['expanded_A', 'expanded_B'], ['expanded_B', 'expanded_C']],
        sampling='Identity-hash4000 development objects with i,W1,W2, verified PSF; all pairs identical objects',
        propagation='Shared grz Sobol draws within each declared pair;128draws',
        interpretation='Matched grz/griz has identical original training IDs. Expanded A/B/C have different training-band eligibility: distance contrasts mix band and training-pool effects. The separate classification probe controls training IDs. No new distance model fitted.'))


if __name__ == '__main__':
    main()
