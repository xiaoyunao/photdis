#!/usr/bin/env python3
"""Cache-only endpoint repair, selected-distance audit and validation inventory."""
import csv
import json
import os
import numpy as np
from v2_1_near import ROOT, DEST as OLD, load, subset, csv_rows, save_json, sha, atomic_npz, strata, reference, DISTANCES
from photdis.stage1 import require_compute
from photdis.baseline_v0 import metrics
from photdis.audit import rvs_metallicity
from photdis.probability_v2_1 import interval_score, migration, psf_admission
from photdis.selection_v2_2 import canonical_probability, reliability_bins, wilson, evidence_cells

DEST = ROOT/'data/v2_2'
TABLES = DEST/'tables'


def repair(t, p):
    raw = p['qnear']
    q = canonical_probability(raw)
    audit = dict(n=len(raw), finite=int(np.isfinite(raw).sum()), minimum=float(raw.min()),
        maximum=float(raw.max()), below_zero=int((raw < 0).sum()), above_one=int((raw > 1).sum()),
        max_change=float(np.max(abs(q-raw))), input_sha256=sha(OLD/'probability.npz'))
    audit['by_split'] = {str(s): dict(n=int((t['split'] == s).sum()), above_one=int(((t['split'] == s) & (raw > 1)).sum())) for s in (1, 2)}
    for threshold in (.5, .8, .95):
        assert np.array_equal(raw > threshold, q > threshold)
    atomic_npz(DEST/'probability_canonical.npz', identity=t['identity'], TYPE=t['TYPE'], qnear=q,
               qnear_raw=raw, endpoint_changed=(q != raw))
    rows, scores = [], []
    for split, role in [(1, 'cal'), (2, 'development_test')]:
        for group, mask in strata(t).items():
            base = (t['split'] == split) & mask
            for kind in ('chosen', 'Gaia', 'specdis', 'spdist'):
                ref, sigma = reference(t, kind)
                valid = base & np.isfinite(ref)
                certain = valid & np.isfinite(sigma) & (sigma > 0) & ((ref+sigma < 15) | (ref-sigma > 15))
                for boundary, use in [('point_reference', valid), ('certain_1sigma', certain)]:
                    key = dict(split=role, group=group, reference=kind, boundary=boundary)
                    bins = reliability_bins(q[use], ref[use] < 15)
                    rows.extend(dict(**key, **r) for r in bins)
                    assert sum(r['n'] for r in bins) == int(use.sum())
                    scores.append(dict(**key, n=int(use.sum()), clipped=int((use & (q != raw)).sum()),
                        brier=float(np.mean((q[use]-(ref[use] < 15))**2)) if use.any() else None,
                        brier_change=float(np.mean((q[use]-(ref[use] < 15))**2-(raw[use]-(ref[use] < 15))**2)) if use.any() else None))
    csv_rows(TABLES/'reliability.csv', rows)
    csv_rows(TABLES/'reliability_denominators.csv', scores)
    save_json(DEST/'endpoint_audit.json', audit)
    return q


def selected(t, p, q):
    masks = strata(t, cross=False)
    for lo, hi in [(-np.inf, .3), (.3, .8), (.8, 1.3), (1.3, np.inf)]:
        masks[f'gr_{lo}_{hi}'] = (t['g0']-t['r0'] >= lo) & (t['g0']-t['r0'] < hi)
    feh, domain = rvs_metallicity(t['rvs_feh'], t['rvs_teff'], t['rvs_logg'])
    quality = domain & t['rvs_parameter_quality'] & (t['rvs_feh_err'] > 0) & np.isfinite(t['rvs_feh_err'])
    for lo, hi in [(-np.inf, -1.5), (-1.5, -.5), (-.5, np.inf)]:
        masks[f'dwarf_corrected_FeH_{lo}_{hi}'] = quality & (t['class_label'] == 0) & (feh >= lo) & (feh < hi)
    rows = []
    for split, role in [(1, 'cal'), (2, 'development_test')]:
        for kind in ('chosen', 'Gaia', 'specdis', 'spdist'):
            ref, sigma = reference(t, kind)
            d = 10**((ref-10)/5)
            for threshold in (.5, .8, .95):
                for group, mask in masks.items():
                    for lo, hi in [(0, np.inf)]+DISTANCES:
                        base = (t['split'] == split) & mask & np.isfinite(ref) & (d >= lo) & (d < hi)
                        use = base & (q > threshold)
                        ls = p['log_teacher_'+kind][use]
                        row = dict(split=role, model='expanded_A', threshold=threshold, reference=kind,
                            group=group, low_kpc=lo, high_kpc=hi, eligible_n=int(base.sum()),
                            selected_n=int(use.sum()), **metrics(t['mu'][use], ref[use]),
                            interval_score68=float(interval_score(t['mu'][use], ref[use]).mean()) if use.any() else None,
                            teacher_zero=int(np.isneginf(ls).sum()),
                            teacher_log=float(ls.mean()) if len(ls) and np.isfinite(ls).all() else None,
                            teacher_log_status='minus_infinity' if np.isneginf(ls).any() else 'finite_or_empty')
                        rows.append(row)
    csv_rows(TABLES/'selected_distance.csv', rows)
    risk = []
    rng = np.random.default_rng(20261110)
    for split, role in [(1, 'cal'), (2, 'development_test')]:
        valid = (t['split'] == split) & np.isfinite(t['ref_chosen'])
        for threshold in (.5, .8, .95):
            r = migration(t['ref_chosen'][valid], q[valid] > threshold)
            low, high = wilson(r['far_selected'], r['near_selected']+r['far_selected'])
            cells = np.unique(t['cell'][valid])
            cc = [migration(t['ref_chosen'][valid & (t['cell'] == c)], q[valid & (t['cell'] == c)] > threshold) for c in cells]
            counts = np.array([[x['near_selected'], x['near_rejected'], x['far_selected'], x['far_rejected']] for x in cc])
            draws = counts[rng.integers(0, len(cells), (2000, len(cells)))].sum(axis=1)
            risks = draws[:, 2]/np.maximum(draws[:, 0]+draws[:, 2], 1)
            risk.append(dict(split=role, threshold=threshold, **r, wilson_low=low, wilson_high=high,
                spatial_low=float(np.quantile(risks, .025)), spatial_high=float(np.quantile(risks, .975)),
                cells=len(cells), far_prevalence=(r['far_selected']+r['far_rejected'])/r['n'],
                false_near_rate=r['far_selected']/(r['far_selected']+r['far_rejected'])))
    for r in risk:
        cal = next(x for x in risk if x['split'] == 'cal' and x['threshold'] == r['threshold'])
        prevalence = r['far_prevalence']
        r['cal_acceptance_at_this_prevalence'] = prevalence*cal['false_near_rate']/(prevalence*cal['false_near_rate']+(1-prevalence)*cal['completeness'])
    csv_rows(TABLES/'selection_risk.csv', risk)


def branch(t, p):
    cell = evidence_cells(t)
    mass = p['class_mass']
    known = mass[:, :3].sum(axis=1)
    prob = np.divide(mass[:, 1], known, out=np.full(len(known), np.nan), where=known > 0)
    rows = []
    for split, role in [(1, 'cal'), (2, 'development_test')]:
        for key in np.unique(cell[t['split'] == split]):
            use = (t['split'] == split) & (cell == key)
            labeled = use & np.isin(t['class_label'], [0, 1, 2]) & np.isfinite(prob)
            n = int(labeled.sum()); giants = int((labeled & (t['class_label'] == 1)).sum())
            low, high = wilson(giants, n)
            rows.append(dict(split=role, observable_cell=int(key), n=int(use.sum()), labeled_n=n,
                observed_evolved=giants/n if n else None, model_evolved_known_conditional=float(prob[labeled].mean()) if n else None,
                model_unlabeled_mass=float(mass[use, 3].mean()), wilson_low=low, wilson_high=high))
    csv_rows(TABLES/'observable_branch_frequency.csv', rows)
    train = load(OLD/'frozen_training_audit.npz')
    psf, audit = psf_admission(train['TYPE'])
    result = []
    for c, label in [(-1, 'unlabeled'), (0, 'dwarf'), (1, 'evolved'), (2, 'other')]:
        for lo, hi in zip([-np.inf, -2, 0, 2, 4, 6, 8, 10], [-2, 0, 2, 4, 6, 8, 10, np.inf]):
            use = psf & (train['class_label'] == c) & (train['Mg'] >= lo) & (train['Mg'] < hi)
            result.append(dict(group=label, Mg_low=lo, Mg_high=hi, n=int(use.sum()), Gaia_labels=int((use & (train['label_source'] == 1)).sum()), SpecDis_labels=int((use & (train['label_source'] == 2)).sum())))
    csv_rows(TABLES/'training_luminosity_support.csv', result)
    # Contribution to TOTAL W1 gain, not per-class gain mistaken as total.
    with open(ROOT/'manifests/v2_1/information_scores.csv') as stream:
        scores = list(csv.DictReader(stream))
    result = []
    rr = [r for r in scores if r['bands'] == 'grzW1' and r['split'] == 'development_test' and r['raw_or_calibrated'] == 'calibrated']
    total = int(next(r['n'] for r in rr if r['group'] == 'all'))
    for group in ('dwarf', 'evolved', 'other'):
        a = next(r for r in rr if r['group'] == group and r['method'] == 'fixed_forest')
        b = next(r for r in rr if r['group'] == group and r['method'] == 'frozen_partitions_same_labels')
        result.append(dict(group=group, n=int(a['n']), log_gain=float(a['log_score'])-float(b['log_score']),
            total_contribution=int(a['n'])/total*(float(a['log_score'])-float(b['log_score'])), brier_change=float(a['brier'])-float(b['brier'])))
    csv_rows(TABLES/'W1_information_contribution.csv', result)
    save_json(DEST/'branch_audit.json', dict(training_morphology=audit,
        interpretation='Known-class conditioning is explicit; reference-selected frequencies are not target-population truth. No class/far reweighting fitted.',
        independent_distance_evidence='Existing Gaia/SpecDis/SPDist are shared teachers; no new independent far shape confirmation'))


def inventory():
    from v2_1_reserve import tiles
    pop = load(ROOT/'data/v2_0/population.npz')
    old = json.loads((OLD/'final_validation_reservation.json').read_text())
    used = []
    for path in old['inputs']:
        with np.load(ROOT/path) as z:
            used.extend(z[k] for k in z.files if 'identity' in k)
    for filename in ('information_train', 'information_eval', 'optical_probe', 'nonstellar_predictions', 'development'):
        q = load(OLD/(filename+'.npz'))
        used.append(q['identity'])
        if 'train_identity' in q:
            used.append(q['train_identity'])
    ids = np.unique(np.concatenate(used))
    psf, audit = psf_admission(pop['TYPE'])
    occupied = np.isin(pop['identity'], ids) | np.isin(pop['split'], [1, 2]) | pop['historical_identity']
    unused = psf & pop['photometric_eligible'] & (pop['split'] == 0) & ~occupied
    candidate = subset(pop, unused)
    assert (candidate['TYPE'] == 'PSF').all() and not np.intersect1d(candidate['identity'], ids).size
    path = DEST/'identity_unused_reservation.npz'
    if path.exists():
        assert np.array_equal(load(path)['identity'], candidate['identity'])
    else:
        atomic_npz(path, **candidate)
    rows = []
    for lo, hi in [(-np.inf, 18), (18, 20), (20, 22), (22, 25), (25, np.inf)]:
        for wise in (False, True):
            mask = (candidate['r'] >= lo) & (candidate['r'] < hi) & (candidate['B'] == wise)
            rows.append(dict(r_low=lo, r_high=hi, W1=wise, n=int(mask.sum()), class_eligible=int((mask & candidate['class_eligible']).sum()), distance_eligible=int((mask & candidate['fit_eligible']).sum()), parallax_eligible=int((mask & candidate['parallax_eligible']).sum())))
    csv_rows(TABLES/'validation_inventory.csv', rows)
    save_json(DEST/'validation_inventory.json', dict(identity_only_unused=int(unused.sum()),
        buffered_new_sky_original=old['reserved_identities'], morphology=audit,
        reservation_sha256=sha(path), purpose='New identities in existing sky, pending reference/selection/system-membership evidence; no outcome predictions inspected',
        limitations=['No new distance-scale independence certified', 'Shared systems require membership audit', 'Existing inspected cal/test not repurposed', 'Original five-ID novel-sky reservation unchanged']))


def main():
    require_compute(); TABLES.mkdir(parents=True, exist_ok=True)
    frozen = {str(p.relative_to(ROOT)): sha(p) for p in [OLD/'probability.npz', OLD/'development.npz', ROOT/'docs/v2_1_protocol.md', ROOT/'scripts/v2_1_joint.py', ROOT/'src/photdis/probability_v2_1.py']}
    t, p = load(OLD/'development.npz'), load(OLD/'probability.npz')
    assert np.array_equal(t['identity'], p['identity']) and (t['TYPE'] == 'PSF').all()
    q = repair(t, p); selected(t, p, q); branch(t, p); inventory()
    assert all(sha(ROOT/name) == h for name, h in frozen.items())
    save_json(DEST/'audit_complete.json', dict(job_id=os.getenv('SLURM_JOB_ID'), frozen_inputs=frozen,
        old_outputs_unchanged=True, no_refit=True))
    print('AUDIT COMPLETE', flush=True)


if __name__ == '__main__':
    main()
