#!/usr/bin/env python3
"""Actual frozen-cache, probability, identity, route and repeated-noise checks."""
import os
import csv
import json
from collections import defaultdict
import numpy as np
from v2_1_near import ROOT, DEST as OLD, load, sha, save_json
from v2_2_application import DEST
from photdis.stage1 import require_compute
from photdis.probability_v2_1 import near_mass


def main():
    require_compute(); checks = {}
    def check(name, ok):
        checks[name] = bool(ok)
        assert ok, name
    def read(name):
        with open(DEST/'tables'/(name+'.csv')) as f:
            return list(csv.DictReader(f))
    for filename in ('audit_complete', 'application_complete', 'noise_complete'):
        meta = json.loads((DEST/(filename+'.json')).read_text())
        for path, expected in meta['frozen_inputs'].items():
            check(filename+':'+path, sha(ROOT/path) == expected)
    counts = defaultdict(int)
    for r in read('reliability'):
        key = tuple(r[k] for k in ('split', 'group', 'reference', 'boundary'))
        counts[key] += int(r['n'])
    for r in read('reliability_denominators'):
        key = tuple(r[k] for k in ('split', 'group', 'reference', 'boundary'))
        check('reliability:'+str(key), counts[key] == int(r['n']))
    t = load(OLD/'development.npz'); old = load(OLD/'probability.npz'); new = load(DEST/'probability_canonical.npz')
    check('canonical_identity', np.array_equal(t['identity'], new['identity']))
    check('canonical_finite_range', np.isfinite(new['qnear']).all() and ((new['qnear'] >= 0) & (new['qnear'] <= 1)).all())
    for threshold in (.5, .8, .95):
        check('threshold_unchanged'+str(threshold), np.array_equal(new['qnear'] > threshold, old['qnear'] > threshold))
    for split, lost in [(1, 308), (2, 379)]:
        valid = (t['split'] == split) & np.isfinite(t['ref_chosen'])
        check('actual_missing_endpoints_'+str(split), ((old['qnear'] > 1) & valid).sum() == lost)
    app = load(DEST/'application_inputs.npz')
    check('parent_count_weight', len(app['identity']) == 26000 and np.isclose(app['sample_weight'].sum(), 8899334))
    check('app_PSF', (app['TYPE'] == 'PSF').all())
    grids = load(DEST/'application_grids.npz')
    for model in ('expanded_A', 'baseline_v1'):
        combined = load(DEST/(model+'_application.npz')); ids = []
        for path in sorted((DEST/'application_chunks').glob(model+'_*.npz')):
            p = load(path); ids.extend(p['identity'])
            check(path.stem+':sha', sha(path) == json.loads(path.with_suffix('.json').read_text())['sha256'])
            check(path.stem+':PSF_PDF', (p['TYPE'] == 'PSF').all() and np.isfinite(p['pdf']).all() and (p['pdf'] >= 0).all() and np.allclose(p['pdf'].sum(axis=1), 1, atol=2e-6))
            check(path.stem+':class_mass', np.allclose(p['weights'].sum(axis=1), 1, atol=2e-6))
            start = int(path.stem.rsplit('_', 1)[1]); rows = np.arange(start, start+len(p['identity']))
            for code in np.unique(p['route']):
                use = p['route'] == code; edges = grids['expanded_A' if model == 'expanded_A' else 'ABC'[code]]
                bins = len(edges)-1
                check(path.stem+':native_q_'+str(code), np.allclose(p['qnear'][use], near_mass(p['pdf'][use, :bins], edges, app['g0'][rows][use]), atol=1e-7))
                check(path.stem+':padding_'+str(code), (p['pdf'][use, bins:] == 0).all())
            check(path.stem+':q_range', ((p['qnear'] >= 0) & (p['qnear'] <= 1)).all())
        check(model+':exact_once', np.array_equal(ids, app['identity']))
        check(model+':combined_order', np.array_equal(combined['identity'], app['identity']))
        if model == 'baseline_v1':
            check('frozen_routes', np.array_equal(combined['route'], app['route']))
            check('catalog_quantiles', np.max(abs(combined['mu']-app['mu'])) < 5e-6)
    for r in read('application_summary'):
        if r['population'] == 'application':
            check('evidence:'+str((r['model'], r['threshold'], r['group'])), np.isclose(sum(float(r[k]) for k in ('evidence_zero_weight', 'evidence_sparse_weight', 'evidence_20plus_weight')), float(r['weighted_n'])))
    d = load(DEST/'noise_design.npz')
    probe = load(OLD/'optical_probe.npz'); train = load(ROOT/'data/v2_0/training_expanded_A.npz')
    check('noise_disjoint_distance', not np.intersect1d(d['identity'], train['identity']).size)
    check('noise_disjoint_probe', not np.intersect1d(d['identity'], probe['train_identity']).size)
    check('noise_PSF_one_role', (d['TYPE'] == 'PSF').all() and (d['split'] == 2).all())
    check('noise_variance', np.allclose(d['template_e'][:, :3]**2+d['added_sigma'][:, :3]**2, d['matched_e'][:, :3]**2))
    check('noise_matching_radius', np.max(d['color_separation']) <= .3)
    for path in sorted((DEST/'noise_chunks').glob('*.npz')):
        p = load(path)
        check(path.stem+':sha', sha(path) == json.loads(path.with_suffix('.json').read_text())['sha256'])
        check(path.stem+':ids', np.array_equal(p['identity'], d['identity']) and (p['TYPE'] == 'PSF').all())
        check(path.stem+':pdf', np.isfinite(p['pdf']).all() and (p['pdf'] >= 0).all() and np.allclose(p['pdf'].sum(axis=1), 1, atol=2e-6))
        for k in ('point_class', 'averaged_class', 'prior_class'):
            check(path.stem+':'+k, np.isfinite(p[k]).all() and np.allclose(p[k].sum(axis=1), 1, atol=2e-6))
    check('noise_four_plus_control', len(list((DEST/'noise_chunks').glob('*.npz'))) == 5)
    far = load(DEST/'far_shape.npz'); check('far_ids_PSF', np.array_equal(far['identity'], t['identity']) and (far['TYPE'] == 'PSF').all())
    reservation = load(DEST/'identity_unused_reservation.npz')
    check('reservation_no_old_evaluation', not np.intersect1d(reservation['identity'], t['identity']).size)
    check('reservation_no_expanded_training', not np.intersect1d(reservation['identity'], train['identity']).size)
    check('reservation_PSF', (reservation['TYPE'] == 'PSF').all())
    check('reservation_old5_unchanged', sha(OLD/'future_validation_reservation.npz') == json.loads((OLD/'final_validation_reservation.json').read_text())['reservation_sha256'])
    save_json(DEST/'validation.json', dict(job_id=os.getenv('SLURM_JOB_ID'), actual_checks=len(checks), all_passed=all(checks.values()), checks=checks))
    print('VALIDATION', len(checks), 'checks passed', flush=True)


if __name__ == '__main__':
    main()
