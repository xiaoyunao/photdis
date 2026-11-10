#!/usr/bin/env python3
"""Verify the one candidate's actual artifacts and final diagnostic exports."""
import csv
import json
import os
import joblib
import numpy as np
from v2_1_near import ROOT, DEST, TABLES, load, save_json, sha, subset
from v2_1_validate import main as validate_base
from photdis.probability_v2_1 import near_mass
from photdis.baseline_v1_2 import full_parallax_log_density


def main():
    validate_base()
    checks = {}
    def check(name, value):
        checks[name] = bool(value)
        if not value:
            raise AssertionError(name)
    out = DEST/'partition_candidate'
    ids = load(out/'identities.npz')
    t = load(DEST/'development.npz')
    t = subset(t, t['C'] & t['i_usable'])
    train = load(DEST/'information_train.npz')
    for role in ('train', 'evaluation'):
        check(role+'_PSF', (ids[role+'_TYPE'] == 'PSF').all())
        check(role+'_unique', len(np.unique(ids[role+'_identity'])) == len(ids[role+'_identity']))
    check('candidate_original_train_ids', np.array_equal(train['identity'], ids['train_identity']))
    check('candidate_original_eval_ids', np.array_equal(t['identity'], ids['evaluation_identity']))
    check('candidate_roles_disjoint', not np.intersect1d(ids['train_identity'], ids['evaluation_identity']).size)
    freeze = json.loads((out/'freeze.json').read_text())
    for name, expected in freeze.items():
        check('frozen_'+name, sha(ROOT/name) == expected)
    models = [joblib.load(out/(name+'.joblib')) for name in ('control', 'candidate')]
    for key in ('n_estimators', 'min_samples_leaf', 'max_depth', 'max_features', 'bootstrap', 'random_state'):
        check('same_'+key, models[0].forest.get_params()[key] == models[1].forest.get_params()[key])
    check('same_histogram_edges', np.array_equal(models[0].edges, models[1].edges))
    quadrature = {}
    for name, model in zip(('control', 'candidate'), models):
        combined = load(out/(name+'_evaluation.npz'))
        check(name+'_combined_ids', np.array_equal(combined['identity'], t['identity']))
        check(name+'_combined_PSF', (combined['TYPE'] == 'PSF').all())
        covered = np.zeros(len(t['identity']), int)
        for path in sorted(out.glob(name+'_[0-9]*.npz')):
            marker = json.loads(path.with_suffix('.json').read_text())
            check(path.stem+'_sha', sha(path) == marker['sha256'])
            q = load(path)
            rows = q['rows']
            covered[rows] += 1
            check(path.stem+'_ids_PSF', np.array_equal(q['identity'], t['identity'][rows]) and (q['TYPE'] == 'PSF').all())
            check(path.stem+'_PDF', np.isfinite(q['pdf']).all() and (q['pdf'] >= 0).all() and np.allclose(q['pdf'].sum(axis=1), 1, atol=2e-6))
            check(path.stem+'_near', np.allclose(q['qnear'], near_mass(q['pdf'], model.edges, t['g0'][rows]), atol=1e-7))
            check(path.stem+'_combined', np.array_equal(q['mu'], combined['mu'][rows]) and np.array_equal(q['qnear'], combined['qnear'][rows]))
            if name not in quadrature:
                ga = t['parallax_eligible'][rows]
                refined = full_parallax_log_density(q['pdf'][ga], model.edges, t['g0'][rows][ga],
                    t['p_corr'][rows][ga], t['gaia_parallax_error'][rows][ga], order=32)
                delta = abs(refined-q['gaia_log'][ga])
                quadrature[name] = dict(n=int(ga.sum()), max_difference=float(delta.max()),
                                        mean_difference=float(delta.mean()), comparison='16 versus32 nodes, first identity block')
                check(name+'_bounded_quadrature', delta.max() < .001)
        check(name+'_exactly_once', (covered == 1).all())
        # The mean log score is dominated by unlabeled extreme observations.
        # Check their integration explicitly; a first-block check alone cannot
        # establish numerical stability of that tail.
        valid = np.flatnonzero(t['parallax_eligible'])
        extreme = valid[np.argsort(combined['gaia_log'][valid])[:32]]
        tail16, tail32, tail64 = [], [], []
        for block in np.unique(extreme//256):
            q = load(out/f'{name}_{block*256:07}.npz')
            selected = np.isin(q['rows'], extreme)
            rows = q['rows'][selected]
            tail16.extend(q['gaia_log'][selected])
            for order, dest in [(32, tail32), (64, tail64)]:
                dest.extend(full_parallax_log_density(q['pdf'][selected], model.edges, t['g0'][rows],
                    t['p_corr'][rows], t['gaia_parallax_error'][rows], order=order))
        delta16 = abs(np.asarray(tail16)-tail64)
        delta32 = abs(np.asarray(tail32)-tail64)
        quadrature[name]['worst32'] = dict(n=len(extreme), max_16_64=float(delta16.max()),
            max_32_64=float(delta32.max()), minimum_log=float(combined['gaia_log'][extreme].min()),
            mean_log=float(combined['gaia_log'][valid].mean()),
            worst32_share_negative_log=float(np.minimum(combined['gaia_log'][extreme], 0).sum()/np.minimum(combined['gaia_log'][valid], 0).sum()))
        # Record the accuracy of16 nodes; do not silently rewrite frozen scores.
        check(name+'_extreme_quadrature32_64', delta32.max() < .001)
    for filename in ('color_neighbors_broad', 'template_degradation_broad', 'partition_candidate'):
        with open(TABLES/(filename+'.csv')) as stream:
            rows = list(csv.DictReader(stream))
        check(filename+'_nonempty', len(rows) > 0)
        if filename == 'color_neighbors_broad':
            check(filename+'_PSF', all(r['TYPE'] == 'PSF' for r in rows))
            check(filename+'_noWISE_present', any(r['W1'] == 'False' for r in rows))
    status = json.loads((DEST/'partition_candidate_complete.json').read_text())
    check('candidate_six_fixed_guards', len(status['guards']) == 6 and status['cal_pass'] == all(status['guards'].values()))
    base = json.loads((DEST/'validation.json').read_text())
    save_json(DEST/'final_validation.json', dict(job_id=os.getenv('SLURM_JOB_ID'),
        base_checks=base['actual_checks'], additional_checks=len(checks),
        total_actual_checks=base['actual_checks']+len(checks), all_passed=all(checks.values()),
        checks=checks, base_validation_sha256=sha(DEST/'validation.json'),
        quadrature=quadrature,
        limitations=base['caveats']))
    print('FINAL VALIDATION', base['actual_checks']+len(checks), 'actual checks passed', flush=True)


if __name__ == '__main__':
    main()
