#!/usr/bin/env python3
"""Run the v3.2 delivery audits without turning numerical failures into passes."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re

from v3_2_branch_audit import audit as branch
from v3_2_collect_radial_results import archive_verified, collect, completed_jobs
from v3_2_external_audit import audit as external
from v3_2_factorial_audit import audit as factorial
from v3_2_figure_delivery import audit as figures
from v3_2_opened_audit import audit as opened
from v3_2_preservation_check import examine as preservation
from v3_2_radial_report_audit import audit as radial
from v3_2_radial_report_audit import audit_available as radial_available
from v3_2_structure_distance_audit import audit as structure_distance
from v3_2_structure_joint_audit import audit as structure


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'manifests/v3_2'
PENDING = {
    'radial_report/complete.json': 'radial recovery report',
    'structure_joint/complete.json': 'NGC288 joint quadrature',
    'radial_report/figure.json': 'radial recovery plates',
    'structure_joint/figure.json': 'NGC288 joint plate',
}
DOCUMENTS = (
    'pro_review_7699299.md', 'v3_2_result.md', 'v3_2_requirements_result.md',
    'v3_2_atlas.md', 'v3_2_protocol.md', 'v3_2_external_result.md',
    'v3_2_opened_result.md', 'v3_2_factorial_result.md', 'v3_2_contamination_support.md',
    'v3_2_radial_protocol.md', 'v3_2_structure_protocol.md',
)
REPORTS = {
    '61266374': ('radial_report', ('numerical.csv', 'recovery.csv', 'coverage.csv',
                 'paired_count_loss.csv', 'paired_count_summary.csv', 'shrinkage.csv',
                 'shrinkage_summary.csv', 'empirical_minus_model.csv', 'empirical_minus_model_summary.csv')),
    '61263908': ('structure_joint', ('input_checks.csv', 'checks.csv', 'summary.csv',
                 'membership.csv', 'count_covariance.csv', 'branch_comparison.json')),
}


def collect_reports():
    finished = completed_jobs(tuple(REPORTS))
    result = {}
    for job, (folder, tables) in REPORTS.items():
        if job not in finished:
            result[folder] = 'no successful Slurm completion'
            continue
        root = f'data/v3_2/{folder}'
        filenames = ('complete.json', *tables)
        files = collect([f'{root}/{name}' for name in filenames], max_file_bytes=6_000_000)
        payload = {name: files[f'{root}/{name}'] for name in filenames}
        receipt = json.loads(payload['complete.json'])
        if (receipt['job'] != job or receipt['all_PSF'] is not True
                or any(sha256(payload[name]).hexdigest() != receipt['outputs'][name] for name in tables)):
            raise ValueError(f'Invalid Slurm identity or compact report digest: {folder}')
        archive_verified(BASE/folder, payload)
        result[folder] = dict(job=job, receipt_sha256=sha256(payload['complete.json']).hexdigest())
    return result


def audit(*, completed_only=False):
    pending = [label for path, label in PENDING.items() if not (BASE/path).is_file()]
    if pending and not completed_only:
        raise FileNotFoundError('Pending deliverables: ' + ', '.join(pending))
    sources = {}
    for name in DOCUMENTS:
        path = ROOT/'docs'/name
        raw = path.read_bytes()
        sources[str(path.relative_to(ROOT))] = sha256(raw).hexdigest()
        for target in re.findall(r'\]\(([^)]+)\)', raw.decode()):
            if '://' in target or target.startswith('#'):
                continue
            if not (path.parent/target.split('#')[0]).exists():
                raise ValueError(f'Missing report link {target} in {name}')
    protected = preservation()
    opened_result = opened()
    external_result = external()
    factorial_result = factorial()
    figure_result = figures(completed_only=completed_only)
    if (protected['passed'] is not True or opened_result['passed'] is not True
            or external_result['passed'] is not True or figure_result['passed'] is not True):
        raise ValueError('A completed delivery audit did not pass')
    result = dict(passed=True, final=not completed_only, pending=pending,
                  preservation=protected, opened=opened_result, external=external_result,
                  factorial=factorial_result, branch=branch(), figures=figure_result, documents=sources)
    if completed_only:
        result['structure_distance'] = structure_distance()
        result['radial_available'] = radial_available()
    else:
        result['radial'] = radial()
        result['structure'] = structure()
        if (result['radial']['n_catalogs'] != 288 or result['radial']['n_arms'] != 576
                or result['structure']['n_states'] != 32 or result['structure']['n'] != 512
                or figure_result['plates'] != 11):
            raise ValueError('Incomplete registered production census')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--completed-only', action='store_true', help='Run audits available before the two final computations')
    mode.add_argument('--collect-reports', action='store_true', help='Archive compact outputs from successful report jobs')
    args = parser.parse_args()
    print(json.dumps(collect_reports() if args.collect_reports else audit(completed_only=args.completed_only),
                     indent=2))
