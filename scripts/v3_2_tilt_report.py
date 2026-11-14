#!/usr/bin/env python3
"""Summarize the four-arm opened-field experiment with paired anchors and counts."""
import csv
import json
import os
from pathlib import Path

import numpy as np

from photdis.stage1 import require_compute
from photdis.v3_1.validation_report import field_statistic
from v2_1_near import ROOT, atomic_npz, csv_rows, load, save_json, sha
from v3_2_tilt_contrasts import ARMS, contrast_rows, matched_rows, number
from v3_2_tilt_inputs import verified

ANCHORS = ('reference/nominal', 'reference/response12', 'old_target/old_target',
           'candidate/zero', 'candidate/point', 'candidate/bootstrap_evidence')
ROLES = ('new_sky_field', 'new_source_seen_sky_field')


def joint_moments(p):
    """Integrate one shared population draw per catalog."""
    if (p.ndim != 3 or p.shape[-1] != 18 or not np.isfinite(p).all() or
            (p < 0).any() or not np.allclose(p.sum(axis=2), 1., rtol=0, atol=1e-10)):
        raise ValueError('Expected draw by PSF object by 18 normalized probabilities')
    n = p.shape[1]
    centers = p.sum(axis=1)
    mean = centers.mean(axis=0)
    conditional = np.diag(mean) - np.einsum('sni,snj->ij', p, p) / len(p)
    diff = centers - mean
    shared = diff.T @ diff / len(p)
    total = conditional + shared
    if not (abs(mean.sum() - n) < 1e-8 and abs(total.sum(axis=0)).max() < 1e-8
            and np.linalg.eigvalsh(total).min() > -1e-8):
        raise ValueError('Joint count moments fail conservation or covariance check')
    return mean, conditional, shared


def giant_strata(rows):
    """Predeclared bins for opened same-object decomposition."""
    giant = np.array([int(r['class_label']) == 1 for r in rows])
    yield 'all', giant
    for key, boundaries in [('g_r', (-np.inf, 0.3, 0.8, 1.3, np.inf)),
                            ('r0', (-np.inf, 16., 18., 20., np.inf)),
                            ('feh', (-np.inf, -1.5, -.5, np.inf))]:
        value = np.array([number(r, key) for r in rows])
        for lo, hi in zip(boundaries, boundaries[1:]):
            yield f'{key}/{lo:g}_{hi:g}', giant & (value >= lo) & (value < hi)
        yield f'{key}/missing', giant & ~np.isfinite(value)
    wise = np.array([number(r, 'w1_state') for r in rows])
    for state in np.unique(wise[np.isfinite(wise)]):
        yield f'W1_state/{int(state)}', giant & (wise == state)


def giant_decomposition(rows, cells, role, arm):
    fields = ('reference_p_giant', 'p_giant', 'reference_oracle_far_mass', 'oracle_far_mass',
              'reference_oracle_far_peak_mu', 'oracle_far_peak_mu', 'reference_oracle_median_mu',
              'oracle_median_mu', 'reference_oracle_gaia_score', 'oracle_gaia_score',
              'reference_oracle_astronn_score', 'oracle_astronn_score')
    values = {key: np.array([number(r, key) for r in rows]) for key in fields}
    for key in ('p_giant', 'oracle_far_mass', 'oracle_gaia_score', 'oracle_astronn_score', 'oracle_median_mu'):
        left, right = values[key], values['reference_' + key]
        present = ~np.isnan(left) & ~np.isnan(right)
        with np.errstate(invalid='ignore'):
            delta = left - right
        delta[present & np.isnan(delta)] = np.inf
        values[key + '_minus_reference'] = delta
    for stratum, mask in giant_strata(rows):
        for name, value in values.items():
            yield dict(role=role, arm=arm, stratum=stratum, metric=name, group_n=int(mask.sum()),
                       **field_statistic(value[mask], cells[mask]))


def entry_sensitivity(role, arm, rows, values):
    # Alternatives share the observed likelihood, objects and underlying draws.
    if (values.shape != (len(rows), 3, 3, 6) or not np.isfinite(values).all()
            or (values < 0).any() or not np.allclose(values.sum(axis=(2, 3)), 1., rtol=0, atol=1e-10)):
        raise ValueError('Expected three normalized entry levels per observed object')
    summaries, counts = [], []
    totals = values.sum(axis=0)
    for level in (0, 2):
        delta = values[:, level]-values[:, 1]
        p = float(abs(delta).max())
        count_delta = totals[level]-totals[1]
        c = float(abs(count_delta).max())
        stats = {}
        for key, tol in [('gaia_score', .02), ('gaia_pit', .003), ('teacher_score', .02)]:
            field = f'entry_{level}_{key}_delta'
            use = np.array([float(r[field]) for r in rows if r.get(field) not in (None, '')])
            finite = np.isfinite(use)
            stats[f'{key}_n'] = len(use)
            stats[f'{key}_max_finite_abs_delta'] = float(abs(use[finite]).max()) if finite.any() else None
            stats[f'{key}_nonfinite'] = int((~finite).sum())
            stats[f'{key}_failed'] = int((~finite | (abs(use) >= tol)).sum())
        summaries.append(dict(role=role, arm=arm, level=level, n=len(rows),
                              maximum_joint_probability_delta=p, maximum_mean_count_delta=c,
                              probability_pass=p < .003, count_pass=c < .5, **stats))
        for c_index in range(3):
            for distance_bin in range(6):
                counts.append(dict(role=role, arm=arm, level=level, class_index=c_index,
                                   distance_bin=distance_bin,
                                   expected_count=float(totals[level, c_index, distance_bin]),
                                   base_expected_count=float(totals[1, c_index, distance_bin]),
                                   count_delta=float(count_delta[c_index, distance_bin])))
    return summaries, counts


def historical_decomposition(groups, anchors):
    """Decompose the old-target to candidate-zero gap on common support."""
    records = []
    for role in ROLES:
        selected = [g for g in groups if g[ARMS[0]]['role'] == role
                    and g[ARMS[0]]['candidate_admission'] == 'True']
        for stratum, label in [('all', None), ('dwarf', 0), ('giant', 1)]:
            same = [g for g in selected if label is None or int(g[ARMS[0]]['class_label']) == label]
            cells = np.array([int(g[ARMS[0]]['cell']) for g in same])
            for metric in ('signed_gaia_log_score', 'class_log_score', 'p_giant', 'qnear'):
                values = []
                for group in same:
                    row = group[ARMS[0]]
                    identity = int(row['identity'])
                    old, new = (anchors[identity, name] for name in ('old_target/old_target', 'candidate/zero'))
                    for anchor in (old, new):
                        if (anchor['TYPE'] != 'PSF' or anchor['role'] != role or anchor['cell'] != row['cell']
                                or anchor['reference_class'] != row['class_label']):
                            raise ValueError('Historical decomposition identity mismatch')
                    values.append([number(dict(old, class_label=row['class_label']), metric),
                                   *(number(group[arm], metric) for arm in ARMS),
                                   number(dict(new, class_label=row['class_label']), metric)])
                matrix = np.asarray(values, dtype=float).reshape(-1, 6)
                available = ~np.isnan(matrix).any(axis=1)
                finite = np.isfinite(matrix[available]).all(axis=1)
                historic_old, bz, bf, cz, cf, historic_zero = matrix[available][finite].T
                components = {
                    'historical_gap': historic_zero-historic_old,
                    'old_target_to_matched_broad_free': bf-historic_old,
                    'remove_tilt_in_broad': bz-bf,
                    'change_population_at_zero_tilt': cz-bz,
                    'matched_candidate_zero_to_historical': historic_zero-cz,
                    'population_by_tilt_interaction': (cf-cz)-(bf-bz),
                }
                np.testing.assert_allclose(
                    components['historical_gap'],
                    sum(components[k] for k in list(components)[1:5]), rtol=0, atol=1e-10)
                for name, delta in components.items():
                    full = np.full(len(finite), np.inf)
                    full[finite] = delta
                    records.append(dict(role=role, stratum=stratum, metric=metric, component=name,
                                        group_n=len(same), missing_n=int((~available).sum()),
                                        **field_statistic(full, cells[available])))
    return records


def summarize(groups, anchors, forecast, manifest):
    absolute, alignment, confusion, counts, gates = [], [], [], [], []
    covariances, decomposition, entry_checks, entry_counts = {}, [], [], []
    for role in ROLES:
        selected = [g for g in groups if g[ARMS[0]]['role'] == role and
                    g[ARMS[0]]['candidate_admission'] == 'True']
        if not selected:
            raise ValueError(f'No candidate rows for {role}')
        identity = [int(g[ARMS[0]]['identity']) for g in selected]
        cells = [int(g[ARMS[0]]['cell']) for g in selected]
        paths = [forecast / f'object_{i}.npz' for i in identity]
        payloads = [load(verified(p, manifest['objects'][p.name])) for p in paths]
        for i, payload in zip(identity, payloads):
            np.testing.assert_array_equal(payload['identity'], [i])
            np.testing.assert_array_equal(payload['TYPE'], ['PSF'])
        for arm in ARMS:
            rows = [g[arm] for g in selected]
            decomposition.extend(giant_decomposition(rows, np.asarray(cells), role, arm))
            states = [v[arm + '_draw_class_distance'].reshape(-1, 18) for v in payloads]
            p = np.stack(states, axis=1)
            mean, conditional, shared = joint_moments(p)
            averaged = p.mean(axis=0)
            saved = np.array([[number(r, f'c{c}_bin{b}') for c in range(3) for b in range(6)]
                              for r in rows])
            if not np.allclose(averaged, saved, rtol=0, atol=1e-10):
                raise ValueError('Reported joint probabilities differ from saved states')
            entry = np.stack([v[arm + '_entry_class_distance'] for v in payloads])
            np.testing.assert_allclose(entry[:, 1].reshape(-1, 18), saved, rtol=0, atol=1e-10)
            sensitivity, alternatives = entry_sensitivity(role, arm, rows, entry)
            entry_checks.extend(sensitivity)
            entry_counts.extend(alternatives)
            covariances.update({f'{role}_{arm}_conditional': conditional,
                                f'{role}_{arm}_shared': shared})
            for j in range(18):
                counts.append(dict(role=role, arm=arm, n=len(rows), draws=len(p), class_index=j//6,
                                   distance_bin=j % 6, expected_count=mean[j],
                                   conditional_variance=conditional[j, j], shared_variance=shared[j, j]))
            for label in (0, 1):
                ref = [r for r in rows if int(r['class_label']) == label]
                for predicted in (0, 1, 2):
                    confusion.append(dict(role=role, arm=arm, class_label=label, predicted=predicted,
                                          count=sum(int(r['predicted_class']) == predicted for r in ref),
                                          denominator=len(ref)))
            for stratum, class_label in [('all', None), ('dwarf', 0), ('giant', 1)]:
                mask = np.array([class_label is None or int(r['class_label']) == class_label for r in rows])
                for metric in ('signed_gaia_log_score', 'class_log_score', 'p_giant', 'qnear',
                               'reference_p_giant', 'oracle_gaia_score', 'reference_oracle_gaia_score',
                               'oracle_far_mass', 'reference_oracle_far_mass'):
                    values = np.array([number(r, metric) for r in rows])
                    absolute.append(dict(role=role, arm=arm, stratum=stratum, metric=metric,
                                         **field_statistic(values[mask], np.asarray(cells)[mask])))
                for anchor in ANCHORS:
                    reference = [anchors[(i, anchor)] for i in identity]
                    for left, right in zip(rows, reference):
                        if (left['role'] != right['role'] or left['cell'] != right['cell']
                                or left['class_label'] != right['reference_class'] or right['TYPE'] != 'PSF'):
                            raise ValueError(f'Anchor identity mismatch: {anchor}')
                    for metric in ('signed_gaia_log_score', 'class_log_score', 'p_giant', 'qnear'):
                        values = np.array([number(r, metric) for r in rows])
                        old = np.array([number(dict(r, class_label=r['reference_class']), metric)
                                        for r in reference])
                        use = mask & ~np.isnan(values) & ~np.isnan(old)
                        with np.errstate(invalid='ignore'):
                            delta = values[use] - old[use]
                        undefined = np.isnan(delta)
                        delta[undefined] = np.inf
                        alignment.append(dict(role=role, arm=arm, anchor=anchor, metric=metric,
                                              stratum=stratum, group_n=int(mask.sum()),
                                              undefined_nonfinite=int(undefined.sum()),
                                              **field_statistic(delta, np.asarray(cells)[use])))
            gates.append(dict(role=role, arm=arm, n=len(rows),
                              grid_flags=sum(r['grid_passed'] != 'True' for r in rows),
                              **{f'{channel}_grid_flags': sum(
                                  r.get(f'coarse_{channel}_score_passed') == 'False' for r in rows)
                                 for channel in ('gaia', 'astronn', 'class', 'specdis', 'spdist', 'mws_class')},
                              sampler_flags=sum(r['sampler_passed'] != 'True' for r in rows),
                              probability_identity_max=max(float(r['joint_delta']) for r in rows),
                              marginal_half_max=max(float(r['marginal_half_delta']) for r in rows)))
    return dict(absolute=absolute, historical_anchor=alignment, confusion=confusion,
                joint_counts=counts, numerical=gates, giant_decomposition=decomposition,
                entry_sensitivity=entry_checks, entry_counts=entry_counts), covariances


def main():
    require_compute()
    root = ROOT / 'data/v3_2'
    forecast = root / 'tilt_predictions/initial'
    marker_path = forecast / 'complete.json'
    marker = json.loads(marker_path.read_text())
    source = verified(forecast / 'objects.csv', marker['outputs']['objects.csv'])
    base = ROOT / 'manifests/v3_2/opened_diagnostics'
    anchor_marker = base / 'complete.json'
    anchor_path = verified(base / 'objects.csv', json.loads(anchor_marker.read_text())['outputs']['objects.csv'])
    with source.open(newline='') as stream:
        groups = matched_rows(csv.DictReader(stream))
    if len(groups) != marker['n']:
        raise ValueError('Prediction identity count mismatch')
    with anchor_path.open(newline='') as stream:
        anchors = {}
        for r in csv.DictReader(stream):
            key = int(r['identity']), r['view']
            if key in anchors:
                raise ValueError('Duplicate anchor row')
            anchors[key] = r
    tables, covariance = summarize(groups, anchors, forecast, marker)
    tables['factorial'] = contrast_rows(groups)
    tables['historical_decomposition'] = historical_decomposition(groups, anchors)
    out = root / 'tilt_report/initial'
    out.mkdir(parents=True, exist_ok=False)
    for name, rows in tables.items():
        csv_rows(out / f'{name}.csv', rows)
    atomic_npz(out / 'joint_covariance.npz', **covariance)
    save_json(out / 'complete.json', dict(
        inputs={str(p.relative_to(ROOT)): sha(p) for p in
                (marker_path, anchor_marker, source, anchor_path, Path(__file__).resolve(),
                 ROOT / 'scripts/v3_2_tilt_contrasts.py')},
        outputs={p.name: sha(p) for p in sorted(out.iterdir())},
        job=os.getenv('SLURM_JOB_ID'), n=marker['n'], all_PSF=True,
        scope='Opened-sky conditional diagnostic, with shared population draw covariance.'))


if __name__ == '__main__':
    main()
