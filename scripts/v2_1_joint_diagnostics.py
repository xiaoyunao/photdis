#!/usr/bin/env python3
"""Per-star branch attribution and fixed, explicitly selected PDF illustrations."""
import json
import csv
import numpy as np
from v2_1_near import ROOT, DEST, TABLES, DISTANCES, CLASSES, load, subset, csv_rows, save_json, sha, atomic_npz
from v2_1_information import sample_indices
from photdis.stage1 import require_compute


def main():
    require_compute()
    t = load(DEST/'development.npz')
    edges = load(DEST/'joint_grid.npz')['edges']
    source_codes = load(DEST/'joint_grid.npz')['source_codes']
    p = load(DEST/'probability.npz')
    assert np.array_equal(t['identity'], p['identity'])
    ref = t['ref_chosen']
    valid = (t['split'] == 2) & np.isfinite(ref)
    categories = dict(
        near_dwarf_success=valid & (ref < 15) & (t['class_label'] == 0) & (abs(t['mu'][:, 2]-ref) < .2),
        far_giant_median_failure=valid & (ref >= 15) & (t['class_label'] == 1) & (t['mu'][:, 2] < 15),
        far_giant_recovered=valid & (ref >= 15) & (t['class_label'] == 1) & (t['mu'][:, 2] >= 15),
        poor_metal_dwarf=valid & (t['class_label'] == 0) & t['rvs_parameter_quality'] & (t['rvs_feh'] < -1.5),
        no_WISE_multimodal=valid & ~t['B'] & t['ambiguous'],
        unlabeled=valid & (t['class_label'] == -1))
    chosen = {}
    occupied = []
    for category, mask in categories.items():
        possible = np.flatnonzero(mask & ~np.isin(np.arange(len(mask)), occupied))
        if len(possible):
            row = possible[sample_indices(t['identity'][possible], 1)][0]
            chosen[int(row)] = category
            occupied.append(row)
    full_correct = np.full(len(ref), np.nan)
    nominal_correct = full_correct.copy()
    joint_correct = np.full((len(ref), 4+len(source_codes)), np.nan)
    examples = []
    checks = []
    for path in sorted((DEST/'joint_chunks').glob('*.npz')):
        marker = json.loads(path.with_suffix('.json').read_text())
        assert sha(path) == marker['sha256']
        q = load(path)
        rows = q['rows']
        low = t['g0'][rows]-ref[rows]-1
        high = t['g0'][rows]-ref[rows]+1
        overlap = np.maximum(0, np.minimum(edges[1:][None], high[:, None])-np.maximum(edges[:-1][None], low[:, None]))/np.diff(edges)
        joint_correct[rows] = np.einsum('ncb,nb->nc', q['joint'], overlap)
        full_correct[rows] = joint_correct[rows, :4].sum(axis=1)
        nominal_correct[rows] = np.einsum('nb,nb->n', q['nominal'][:, :4].sum(axis=1), overlap)
        checks.append(dict(path=path.name, **marker))
        for i, row in enumerate(rows):
            if int(row) not in chosen:
                continue
            mu_edges = t['g0'][row]-edges[::-1]
            joint = q['joint'][i, :, ::-1]/np.diff(mu_edges)[None]
            nominal = q['nominal'][i, :4, ::-1].sum(axis=0)/np.diff(mu_edges)
            for b in range(len(edges)-1):
                record = dict(identity=int(t['identity'][row]), category=chosen[int(row)],
                    mu_low=mu_edges[b], mu_high=mu_edges[b+1], reference_mu=ref[row],
                    model_median=t['mu'][row, 2], qnear=p['qnear'][row],
                    nominal_density=nominal[b], total_density=joint[:4, b].sum(),
                    dwarf=joint[0, b], evolved=joint[1, b], other=joint[2, b], unlabeled=joint[3, b])
                for s, code in enumerate(source_codes):
                    record['source_'+str(code)] = joint[4+s, b]
                examples.append(record)
    atomic_npz(DEST/'branch_attribution.npz', identity=t['identity'], TYPE=t['TYPE'],
               correct_window_mass=full_correct, nominal_correct_window_mass=nominal_correct,
               joint_correct_window=joint_correct, source_codes=source_codes)
    summary = []
    distance = 10**((ref-10)/5)
    for c, name in [(None, 'all')]+CLASSES:
        for lo, hi in DISTANCES:
            use = valid & (distance >= lo) & (distance < hi)
            if c is not None:
                use &= t['class_label'] == c
            n = int(use.sum())
            row = dict(group=name, low_kpc=lo, high_kpc=hi, n=n)
            if n:
                row.update(mean_near_probability=float(p['qnear'][use].mean()),
                    median_near_probability=float(np.median(p['qnear'][use])),
                    median_correct_window_mass=float(np.median(full_correct[use])),
                    mean_correct_window_mass=float(np.mean(full_correct[use])),
                    mean_nominal_correct_window_mass=float(np.mean(nominal_correct[use])),
                    correct_window_below_1pct=float(np.mean(full_correct[use] < .01)),
                    far_probability_at_least_10pct=float(np.mean(p['qnear'][use] <= .9)),
                    nominal_to_noise_near_change=float(np.mean(p['qnear'][use]-p['nominal_qnear'][use])),
                    giant_near_joint_mean=float(np.mean(p['joint_near'][use, 1])),
                    giant_near_independent_product_mean=float(np.mean(p['qnear'][use]*p['class_mass'][use, 1])))
                for k, label in enumerate(['dwarf', 'evolved', 'other', 'unlabeled']):
                    row['correct_window_'+label] = float(np.mean(joint_correct[use, k]))
                for k, code in enumerate(source_codes):
                    row['correct_window_source_'+str(code)] = float(np.mean(joint_correct[use, 4+k]))
            summary.append(row)
    csv_rows(TABLES/'branch_attribution.csv', summary)
    csv_rows(TABLES/'pdf_examples.csv', examples)
    # Per-star conditional export stays on NERSC. Only finite reference targets
    # enter conditional performance; missing-reference placeholder flags do not.
    with open(TABLES/'class_conditional.csv') as stream:
        conditional = list(csv.DictReader(stream))
    conditional_summary = []
    for split in (1, 2):
        for label in ('dwarf', 'evolved', 'other'):
            rr = [r for r in conditional if int(r['split']) == split and r['group'] == label
                  and np.isfinite(float(r['reference_mu']))]
            delta = np.array([float(r['mu50'])-float(r['reference_mu']) for r in rr])
            conditional_summary.append(dict(split=split, group=label, n=len(rr),
                median_delta_mu=float(np.median(delta)) if len(rr) else None,
                severe_fraction=float(np.mean(abs(delta) > 1)) if len(rr) else None,
                coverage68=float(np.mean([r['coverage68'] == 'True' for r in rr])) if rr else None,
                median_width68=float(np.median([float(r['width68']) for r in rr])) if rr else None))
    csv_rows(TABLES/'class_conditional_summary.csv', conditional_summary)
    save_json(DEST/'joint_diagnostics_complete.json', dict(examples={str(t['identity'][row]): category for row, category in chosen.items()},
        example_selection='One deterministic identity-hash choice per declared outcome stratum; illustrative, not representative validation',
        checks_n=len(checks), max_pdf_error=max(r['pdf_error'] for r in checks),
        max_class_error=max(r['class_error'] for r in checks), max_mu_error=max(r['mu_error'] for r in checks),
        terminology='Joint reference contribution; unknown denotes missing class evidence; no Galactic prior or physical posterior calibration implied'))


if __name__ == '__main__':
    main()
