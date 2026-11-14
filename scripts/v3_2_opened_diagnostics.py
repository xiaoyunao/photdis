#!/usr/bin/env python3
"""Compare frozen v3.1 views on the already-opened field cohort."""
import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np

from photdis.stage1 import require_compute
from photdis.v3_1.validation_report import field_statistic, paired_field_statistic
from v2_1_near import ROOT, csv_rows, save_json, sha
from v3_1_validation_report import read_cohort

PATH = (
    'reference/nominal', 'reference/response12', 'old_target/old_target',
    'candidate/zero', 'candidate/point', 'candidate/bootstrap_evidence',
)
PAIRS = (*zip(PATH, PATH[1:]), (PATH[2], PATH[-1]))
METRICS = ('gaia/log_score', 'class/log_score', 'class/correct', 'qnear')
ROLES = ('new_sky_field', 'new_source_seen_sky_field')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    require_compute()
    output = args.output.resolve()
    allowed = ROOT / 'data/v3_2'
    if not output.is_relative_to(allowed) or output == allowed:
        parser.error('Output must be a new directory below data/v3_2')
    if output.exists():
        parser.error('Choose a new output directory to preserve prior work')

    root = ROOT / 'data/v3_1'
    report = root / 'final_validation/report'
    report_marker = report / 'complete.json'
    inputs = {str(path.relative_to(ROOT)): sha(path)
              for path in (report_marker, Path(__file__).resolve(),
                           ROOT / 'scripts/v3_1_validation_report.py',
                           ROOT / 'src/photdis/v3_1/validation_report.py',
                           ROOT / 'src/photdis/v3_1/comparison.py')}
    photo, evidence, records, views, metrics = read_cohort(root, 'field', inputs)
    if not set(PATH) <= set(views):
        raise ValueError('Frozen comparison path is incomplete')

    paired, reliability, objects = [], [], []
    for role in ROLES:
        eligible = photo['candidate_admission'] & (photo['role'] == role)
        labels = evidence['class_label']
        sky = photo['cell']
        for stratum, mask in [('all', eligible), ('dwarf', eligible & (labels == 0)),
                               ('giant', eligible & (labels == 1))]:
            for baseline, view in PAIRS:
                left, left_ok = metrics[view]
                right, right_ok = metrics[baseline]
                for metric in METRICS:
                    if not np.array_equal(np.isnan(left[metric][mask]),
                                          np.isnan(right[metric][mask])):
                        raise ValueError(f'Mismatched reference availability: {view} {metric}')
                    paired.append(dict(role=role, stratum=stratum, baseline=baseline, view=view,
                                       metric=metric, group_n=int(mask.sum()),
                                       inherited_or_score_flags=int((mask & ~(left_ok & right_ok)).sum()),
                                       **paired_field_statistic(left[metric][mask], right[metric][mask], sky[mask])))
        labeled = eligible & (labels >= 0)
        for view in views:
            probability = np.array([r['views'][view]['class_probability'] for r in records])
            if (probability < 0).any() or not np.allclose(probability.sum(axis=1), 1, atol=1e-8, rtol=0):
                raise ValueError(f'Invalid class probabilities: {view}')
            for c, name in [(0, 'dwarf'), (1, 'giant')]:
                p = probability[:, c]
                for index in range(10):
                    lower, upper = index / 10, (index + 1) / 10
                    mask = labeled & (p >= lower) & ((p < upper) if index < 9 else (p <= upper))
                    stats = field_statistic((labels[mask] == c).astype(float), sky[mask])
                    reliability.append(dict(role=role, view=view, target_class=name,
                                            bin_lower=lower, bin_upper=upper,
                                            mean_probability=float(p[mask].mean()) if mask.any() else None,
                                            **stats))
            for i in np.flatnonzero(eligible):
                value = records[i]['views'][view]
                objects.append(dict(identity=records[i]['identity'], TYPE='PSF', role=role,
                                    cell=int(sky[i]), view=view, reference_class=int(labels[i]),
                                    predicted_class=int(probability[i].argmax()),
                                    p_dwarf=probability[i, 0], p_giant=probability[i, 1],
                                    p_other=probability[i, 2], qnear=value['qnear'],
                                    signed_gaia_log_score=value.get('gaia', {}).get('log_score'),
                                    class_log_score=value.get('class', {}).get('log_score'),
                                    accepted=bool(metrics[view][1][i])))

    rows = [r for r in paired if r['role'] == ROLES[0] and r['stratum'] == 'all']
    for row in rows:
        expected_n = {'gaia/log_score': 166, 'class/log_score': 157, 'class/correct': 157,
                      'qnear': 167}[row['metric']]
        if row['n'] != expected_n:
            raise ValueError(f'Unexpected frozen sample size: {row}')
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in [('score_path.csv', paired), ('reliability.csv', reliability),
                       ('objects.csv', objects)]:
        csv_rows(output / name, rows)
    frozen = json.loads(report_marker.read_text())['outputs']
    for role in ROLES:
        names = [f'{role}_confusion.csv']
        names += [f'{role}_summary_class_{selection}.csv'
                  for selection in ('all_objects', 'numerically_accepted')]
        for name in names:
            source = report / name
            if sha(source) != frozen[name]:
                raise ValueError(f'Frozen report hash mismatch: {name}')
            inputs[str(source.relative_to(ROOT))] = frozen[name]
            shutil.copyfile(source, output / name)
    save_json(output / 'complete.json', dict(
        inputs=inputs, outputs={p.name: sha(p) for p in sorted(output.glob('*.csv'))},
        job=os.environ.get('SLURM_JOB_ID'), all_PSF=True, n=len(records),
        scope='Already-opened reference-availability selected field cohort. Sky-block intervals '
              'are conditional on this cohort. All-object statistics retain numerical flags.'
    ))


if __name__ == '__main__':
    main()
