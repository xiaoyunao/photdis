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
              'oracle_median_mu', 'reference_oracle_gaia_score', 'oracle_gaia_score')
    values = {key: np.array([number(r, key) for r in rows]) for key in fields}
    for key in ('p_giant', 'oracle_far_mass', 'oracle_gaia_score', 'oracle_median_mu'):
        values[key + '_minus_reference'] = values[key] - values['reference_' + key]
    for stratum, mask in giant_strata(rows):
        for name, value in values.items():
            yield dict(role=role, arm=arm, stratum=stratum, metric=name, group_n=int(mask.sum()),
                       **field_statistic(value[mask], cells[mask]))


def summarize(groups, anchors, forecast, manifest):
    absolute, alignment, confusion, counts, gates = [], [], [], [], []
    covariances, decomposition = {}, []
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
                              sampler_flags=sum(r['sampler_passed'] != 'True' for r in rows),
                              probability_identity_max=max(float(r['joint_delta']) for r in rows),
                              marginal_half_max=max(float(r['marginal_half_delta']) for r in rows)))
    return dict(absolute=absolute, historical_anchor=alignment, confusion=confusion,
                joint_counts=counts, numerical=gates, giant_decomposition=decomposition), covariances


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
