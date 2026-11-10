#!/usr/bin/env python3
"""Held-out identity, observed-target matched, four-replicate noise transfer."""
import json
import os
import joblib
import numpy as np
from v2_1_near import ROOT, DEST as OLD, load, subset, csv_rows, save_json, sha, atomic_npz
from baseline_v1_common import arrays
from photdis.baseline_v0 import metrics
from photdis.baseline_v1 import integrate, qmc_noise
from photdis.probability_v2_1 import near_mass, interval_score, migration
from photdis.selection_v2_2 import canonical_probability
from photdis.information_v2_1 import calibrated, class_score
from photdis.stage1 import require_compute

DEST = ROOT/'data/v2_2'


def main():
    require_compute(); (DEST/'noise_chunks').mkdir(parents=True, exist_ok=True)
    t = load(OLD/'development.npz'); target = load(OLD/'application_psf.npz')
    train = load(ROOT/'data/v2_0/training_expanded_A.npz')
    probe = load(OLD/'optical_probe.npz')
    train_ids = train['identity'] if 'identity' in train else train['train_identity']
    eligible = ((t['split'] == 2) & (t['TYPE'] == 'PSF') & np.isfinite(t['ref_chosen']) &
                (t['optical_error'] < .02) & ~np.isin(t['identity'], train_ids) &
                ~np.isin(t['identity'], probe['train_identity']))
    rng = np.random.default_rng(20261110)
    # Balance only template coverage; report these are not population fractions.
    candidates = []
    for c in (-1, 0, 1, 2):
        for wise in range(3):
            ix = np.flatnonzero(eligible & (t['class_label'] == c) & (t['B'].astype(int)+t['C'].astype(int) == wise))
            candidates.extend(rng.permutation(ix)[:22])
    ix = np.array(candidates, int)[:256]
    t = subset(t, ix)
    assert (t['TYPE'] == 'PSF').all() and (target['TYPE'] == 'PSF').all()
    assert not np.intersect1d(t['identity'], train_ids).size
    assert not np.intersect1d(t['identity'], probe['train_identity']).size
    m, e, ids = arrays(t, np.arange(len(ix)))
    tm, te, tids = arrays(target, np.arange(len(target['identity'])))
    colors = np.column_stack((m[:, 0]-m[:, 1], m[:, 1]-m[:, 2]))
    tc = np.column_stack((tm[:, 0]-tm[:, 1], tm[:, 1]-tm[:, 2]))
    rows, misses = [], []
    for lo, hi in [(18, 20), (20, 22), (22, 25)]:
        for i in range(len(ids)):
            use = ((target['r'] >= lo) & (target['r'] < hi) & (target['r'] >= t['r'][i]) &
                   (target['B'] == t['B'][i]) & (target['C'] == t['C'][i]) &
                   np.isfinite(te[:, :3]).all(axis=1) & (te[:, :3] >= e[i, :3]).all(axis=1) &
                   (np.linalg.norm(tc-colors[i], axis=1) <= .3))
            choices = np.flatnonzero(use)
            if not len(choices):
                misses.append(dict(template_identity=int(ids[i]), r_low=lo, class_label=int(t['class_label'][i]), W1=bool(t['B'][i])))
                continue
            weights = target['sample_weight'][choices]; j = rng.choice(choices, p=weights/weights.sum())
            rows.append((i, j, lo, hi))
    assert rows, 'No matched targets'
    source = np.array([r[0] for r in rows]); targ = np.array([r[1] for r in rows]); low = np.array([r[2] for r in rows])
    shift = target['r'][targ]-t['r'][source]
    clean_m = m[source]+shift[:, None]; clean_e = e[source].copy()
    new_e = clean_e.copy(); new_e[:, :3] = te[targ, :3]
    added = np.sqrt(np.maximum(new_e**2-clean_e**2, 0))
    # Only optical error is migrated: no WISE model is evaluated in this contrast.
    added[:, 3:] = 0
    truth = t['ref_chosen'][source]+shift
    base = dict(identity=ids[source], TYPE=t['TYPE'][source], split=t['split'][source],
        template_row=source, target_identity=tids[targ], target_row=targ, r_low=low,
        class_label=t['class_label'][source], W1=t['B'][source], W2=t['C'][source],
        truth_mu=truth, magnitude_shift=shift, template_m=m[source], template_e=e[source],
        matched_m=clean_m, matched_e=new_e, added_sigma=added,
        color_separation=np.linalg.norm(tc[targ]-colors[source], axis=1))
    atomic_npz(DEST/'noise_design.npz', **base)
    model_path = ROOT/'data/baseline_v1_4/experiments/expanded_A_distance.joblib'
    model = joblib.load(model_path); clf = joblib.load(OLD/'probe_broad_grz.joblib')
    params = json.loads((OLD/'optical_probe_complete.json').read_text())['calibration']
    tr = load(OLD/'frozen_training_audit.npz')
    labels = tr['class_label'][np.isin(tr['identity'], probe['train_identity'])]
    assert len(labels) == len(probe['train_identity'])
    prior = np.bincount(labels.astype(int), minlength=3)/len(labels)
    freeze = {str(p.relative_to(ROOT)): sha(p) for p in [model_path, OLD/'probe_broad_grz.joblib',
        OLD/'optical_probe_complete.json', OLD/'application_psf.npz', OLD/'development.npz',
        DEST/'noise_design.npz', ROOT/'scripts/v2_2_noise.py', ROOT/'docs/v2_2_protocol.md']}
    lock = DEST/'noise_freeze.json'
    if lock.exists():
        assert json.loads(lock.read_text()) == freeze
    else:
        save_json(lock, freeze)
    records = []
    for replica in (-1, 0, 1, 2, 3):
        path = DEST/'noise_chunks'/f'replica_{replica}.npz'
        meta = path.with_suffix('.json')
        if path.exists() and meta.exists():
            assert sha(path) == json.loads(meta.read_text())['sha256']
            p = load(path)
        else:
            # One shared band perturbation per template/target/replica, not independent color errors.
            z = np.random.default_rng(20261110+max(replica, 0)).standard_normal(clean_m.shape)
            mag = clean_m.copy() if replica == -1 else clean_m+z*added
            err = clean_e if replica == -1 else new_e
            p = integrate(model, mag, err, base['identity'], 128, keep_pdf=True)['wide']
            p['qnear'] = canonical_probability(near_mass(p['pdf'], model.edges, mag[:, 0]))
            point = clf.predict_proba(mag)
            # Frozen cal map is applied per prediction before averaging, matching a
            # marginal of calibrated probabilities rather than fitting on replicas.
            averaged_cal = np.zeros_like(point)
            for draw in range(128):
                value = clf.predict_proba(mag+qmc_noise(base['identity'], draw)*np.nan_to_num(err, nan=0))
                averaged_cal += calibrated(value, params['fixed_forest'])/128
            p.update(identity=base['identity'], TYPE=base['TYPE'], observed_m=mag, error=err,
                point_class=calibrated(point, params['fixed_forest']), averaged_class=averaged_cal,
                prior_class=calibrated(np.broadcast_to(prior, point.shape), params['no_color']))
            atomic_npz(path, **p); save_json(meta, dict(sha256=sha(path), n=len(source)))
        masks = dict(all=np.ones(len(source), bool), no_W1=~base['W1'], W1=base['W1'])
        masks.update({name: base['class_label'] == c for c, name in [(-1, 'unknown'), (0, 'dwarf'), (1, 'evolved'), (2, 'other')]})
        for rlo in (18, 20, 22):
            for group, mask in masks.items():
                use = (low == rlo) & mask
                if not use.any():
                    continue
                truth0 = truth[use]; labeled = use & np.isin(base['class_label'], [0, 1, 2])
                movement = migration(truth0, p['qnear'][use] > .95)
                row = dict(replica=replica, condition='shifted_clean' if replica == -1 else 'target_noise',
                    r_low=rlo, group=group, independent_templates=len(np.unique(base['identity'][use])),
                    **metrics(p['mu'][use], truth0), interval_score68=float(interval_score(p['mu'][use], truth0).mean()),
                    **{'q95_'+k: v for k, v in movement.items()},
                    mean_optical_error=float(np.max(p['error'][use, :3], axis=1).mean()))
                for mode in ('point', 'averaged'):
                    score = class_score(p[mode+'_class'][labeled], base['class_label'][labeled].astype(int), p['prior_class'][labeled])
                    row.update({mode+'_'+k: v for k, v in score.items()})
                records.append(row)
        print('NOISE REPLICA', replica, 'COMPLETE', len(source), flush=True)
    csv_rows(DEST/'tables/noise_transfer.csv', records)
    csv_rows(DEST/'tables/noise_unmatched.csv', misses)
    # Per-replica variability is not a new independent sample size.
    stability = []
    for rlo in (18, 20, 22):
        for group in ('all', 'no_W1', 'W1', 'dwarf', 'evolved', 'other', 'unknown'):
            rr = [r for r in records if r['r_low'] == rlo and r['group'] == group and r['replica'] >= 0]
            for key in ('median_relative', 'tail_1', 'coverage68', 'interval_score68', 'q95_far_contamination', 'point_log_gain_over_prior', 'averaged_log_gain_over_prior'):
                vv = np.array([r.get(key, np.nan) if r.get(key) is not None else np.nan for r in rr])
                if np.isfinite(vv).any():
                    stability.append(dict(r_low=rlo, group=group, metric=key, repetitions=int(np.isfinite(vv).sum()),
                        mean=float(np.nanmean(vv)), std=float(np.nanstd(vv)), minimum=float(np.nanmin(vv)), maximum=float(np.nanmax(vv))))
    csv_rows(DEST/'tables/noise_repeat_stability.csv', stability)
    save_json(DEST/'noise_complete.json', dict(job_id=os.getenv('SLURM_JOB_ID'), frozen_inputs=freeze,
        templates=len(ids), matched_pairs=len(source), unique_matched_templates=len(np.unique(base['identity'])),
        unmatched=len(misses), replicas=4, distance_draws=128, template_train_overlap=0,
        role='Previously inspected development identities; disjoint from fitting, not independent final validation',
        interpretation='Conditional optical noise-transfer experiment, with WISE state matched. Class/WISE-balanced templates do not represent target populations. Shifted teacher distance is not new faint truth. Four repetitions are clustered by source identity. No model or calibration refit.'))
    assert all(sha(ROOT/k) == v for k, v in freeze.items())
    print('NOISE COMPLETE', flush=True)


if __name__ == '__main__':
    main()
