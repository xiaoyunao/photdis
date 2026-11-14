#!/usr/bin/env python3
"""Audit missing distance labels in the frozen, opened high-q samples."""
import csv
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROLES = ('new_sky_field', 'new_source_seen_sky_field')
MAIN = 'candidate/bootstrap_evidence'


def missing_label_bounds(selected, qualified, far):
    if not 0 <= far <= qualified <= selected or selected == 0:
        raise ValueError('Require positive selection with nested reference counts')
    missing = selected-qualified
    return far/selected, (far+missing)/selected


def main():
    report = ROOT/'manifests/v3_1/final_validation/report'
    marker = report/'complete.json'
    source = json.loads(marker.read_text())['outputs']
    paths = [marker, Path(__file__).resolve()]
    rows = []
    for role in ROLES:
        path = report/f'{role}_highq.csv'
        if sha256(path.read_bytes()).hexdigest() != source[path.name]:
            raise ValueError('Opened source hash mismatch')
        paths.append(path)
        with path.open(newline='') as stream:
            selected = [r for r in csv.DictReader(stream) if r['view'] == MAIN and r['role'] == role]
        if {r['reference_qualification'] for r in selected if r['reference_channel']=='astronn'} != {'available','precise'}:
            raise ValueError('Missing AstroNN qualifications')
        for row in selected:
            n, k, f = [int(row[key]) for key in ('n', 'reference_n', 'reference_center_far')]
            lower, upper = missing_label_bounds(n, k, f)
            rows.append(dict(role=role, reference_channel=row['reference_channel'],
                             reference_qualification=row['reference_qualification'],
                             highq_n=n, reference_n=k, teacher_far_n=f, missing_n=n-k,
                             teacher_labeled_far_fraction=f/k if k else None,
                             missing_label_lower=lower, missing_label_upper=upper,
                             numerical_failed=int(row['main_numerical_failed'])))
    out = ROOT/'manifests/v3_2/contamination_support'
    out.mkdir(exist_ok=True)
    path = out/'opened_highq_bounds.csv'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    (out/'complete.json').write_text(json.dumps(dict(
        inputs={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in paths},
        outputs={path.name:sha256(path.read_bytes()).hexdigest()}, n_rows=len(rows),
        scope='Opened selected reference cohorts. Bounds treat labeled teacher centers as fixed and allow every missing label to be near or far. Separate from distance uncertainties and full-parent contamination.'), indent=2)+'\n')


if __name__ == '__main__':
    main()
