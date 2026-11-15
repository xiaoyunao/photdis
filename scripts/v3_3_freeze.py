#!/usr/bin/env python3
"""Bind the candidate, validation roster and evaluation code before opening Gaia."""
import argparse
import json
from pathlib import Path

from photdis.stage1 import require_compute
from v2_1_near import ROOT, sha
from v3_3_catalog import FREEZE
from v3_3_catalog_inputs import OUTPUT as CATALOG
from v3_3_development import OUT as DEVELOPMENT
from v3_3_development_report import OUT as REPORT
from v3_3_prospective_inputs import OUT as PROSPECTIVE
from v3_3_sed import BASE, MANIFESTS, OUT as INITIAL, verified


def read(path):
    return json.loads(path.read_text())


def rel(path):
    return str(path.relative_to(ROOT))


def model(path, digest):
    return dict(path=rel(verified(path, digest)), sha256=digest)


def main():
    require_compute()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--decision', type=Path, required=True)
    args = parser.parse_args()
    decision_path = args.decision.resolve()
    decision = read(decision_path)
    research_model = decision['research_catalog_model']
    if research_model not in ('nominal', 'candidate') or not decision['development_rationale'].strip():
        raise ValueError('A development-supported research model and rationale are required')
    if decision['candidate_fit'] != 'initial':
        raise ValueError('This freeze entry point expects the completed initial fit')
    if (PROSPECTIVE.parent/'score/started.json').exists():
        raise ValueError('A prospective outcome opening already exists')
    dev = DEVELOPMENT/'complete.json'
    report = REPORT/'complete.json'
    cat = CATALOG/'complete.json'
    prospect = PROSPECTIVE/'complete.json'
    initial = INITIAL/'complete.json'
    for path in (dev, report, cat, prospect):
        if not read(path)['all_PSF']:
            raise ValueError(f'Expected strictly native PSF records in {path}')
    if read(prospect)['outcomes_read']:
        raise ValueError('Prospective receipt is exposed')
    if read(report)['inputs'][rel(dev)] != sha(dev):
        raise ValueError('Development report and paired scoring receipt diverged')
    nominal = BASE/'sed_fit.joblib'
    qrf = BASE/'controls/qrf_grz.joblib'
    models = dict(
        nominal=model(nominal, read(MANIFESTS/'sed_fit.json')['sha256']),
        candidate=model(INITIAL/'model.joblib', read(initial)['sha256']),
        qrf=model(qrf, read(MANIFESTS/'controls/complete.json')['outputs'][qrf.name]))
    inputs = [dev, report, cat, prospect, initial, decision_path,
              *sorted((ROOT/'scripts').glob('v3_3_*.py')),
              *sorted((ROOT/'scripts').glob('run_v3_3_*.sh')),
              *sorted((ROOT/'configs').glob('v3_3_*.json'))]
    sources = ROOT/'src/photdis'
    inputs += [sources/name for name in ('v3_3.py', 'gaia.py', 'baseline_v1.py', 'v3/fast.py',
        'v3/inference.py', 'v3/likelihood.py', 'v3/metrics.py', 'v3/selection.py',
        'v3_1/gaussian_response.py', 'v3_1/selection_dust.py', 'v3_1/structure.py',
        'v3_1/structure_posterior.py', 'v3_1/validation_report.py')]
    freeze = dict(models=models, research_catalog_model=research_model,
        fitted_candidate_converged=read(initial)['converged'],
        catalog_inputs_sha256=sha(cat), prospective_inputs_sha256=sha(prospect),
        scoring_target='Corrected signed Gaia parallax with measured uncertainty on previously unused PSFs.',
        decision_sha256=sha(decision_path), inputs={rel(path): sha(path) for path in inputs})
    with FREEZE.open('x') as stream:
        json.dump(freeze, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
