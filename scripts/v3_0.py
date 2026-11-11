#!/usr/bin/env python3
"""Unified v3 workflow entry; heavy stages require an allocated compute node.

Examples: v3_0.py status; v3_0.py sed fit; v3_0.py predict development --view grz.
The stage drivers share photdis.v3 likelihood, roles, selection and count code.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STAGES = {
    'freeze': 'v3_0_freeze.py', 'delivery': 'v3_0_delivery.py', 'prepare': 'v3_0_prepare.py', 'calibrate': 'v3_0_calibrate.py',
    'native': 'v3_0_native.py', 'sed': 'v3_0_sed.py',
    'controls': 'v3_0_controls.py', 'cohorts': 'v3_0_cohorts.py',
    'uncertainty-audit': 'v3_0_uncertainty_audit.py',
    'uncertainty': 'v3_0_uncertainty.py', 'benchmark': 'v3_0_response.py',
    'predict': 'v3_0_predict.py', 'independent': 'v3_0_independent.py',
    'selection': 'v3_0_selection.py', 'target': 'v3_0_target.py',
    'qrf': 'v3_0_qrf_predict.py', 'evaluate': 'v3_0_evaluate.py',
    'selection-audit': 'v3_0_selection_audit.py',
    'simulate': 'v3_0_simulate.py', 'uncertainty-refit': 'v3_0_uncertainty_refit.py',
    'hierarchy-benchmark': 'v3_0_hierarchy_benchmark.py',
    'support-audit': 'v3_0_support_audit.py',
    'catalog': 'v3_0_catalog.py', 'catalog-audit': 'v3_0_catalog_audit.py', 'catalog-numeric-audit': 'v3_0_catalog_numeric_audit.py', 'scale-example': 'v3_0_scale_example.py',
    'recovery-report': 'v3_0_recovery_report.py', 'recovery-precision': 'v3_0_recovery_precision.py', 'final-figures': 'v3_0_final_figures.py',
    'validation-inputs': 'v3_0_validation_inputs.py', 'validation-audit': 'v3_0_validation_audit.py', 'candidate-predict': 'v3_0_candidate_predict.py',
    'recovery-batch': 'v3_0_recovery_batch.py',
    'recovery': 'v3_0_recovery.py',
    'development-figures': 'v3_0_development_figures.py',
    'population-projection': 'v3_0_projection.py', 'transfer-plan': 'v3_0_transfer_plan.py',
    'planned-projection': 'v3_0_planned_projection.py', 'catalog-summary': 'v3_0_catalog_summary.py', 'catalog-export': 'v3_0_catalog_export.py', 'design-summary': 'v3_0_design_summary.py',
    'validation-figures': 'v3_0_validation_figures.py', 'transfer-evaluate': 'v3_0_transfer_evaluate.py',
    'population-quadrature': 'v3_0_population_quadrature.py',
    'native-response': 'v3_0_native_response.py',
    'evidence-refine': 'v3_0_evidence_refine.py', 'evidence-parallel': 'v3_0_evidence_parallel.py',
    'shared-recovery': 'v3_0_shared_recovery.py', 'shared-recovery-refine': 'v3_0_shared_recovery_refine.py',
    'shared-recovery-report': 'v3_0_shared_recovery_report.py',
    'shared-recovery-batch': 'v3_0_shared_recovery_batch.py',
    'dust-selection-audit': 'v3_0_dust_selection_audit.py',
    'diagnostics': 'v3_0_diagnostics.py',
    'ensemble-products': 'v3_0_ensemble.py',
    'shared-products': 'v3_0_shared_products.py',
    'refine-target': 'v3_0_refine_target.py',
    'shared-selection': 'v3_0_shared_selection.py', 'shared-target': 'v3_0_shared_target.py',
    'shared-predict': 'v3_0_shared_predict.py', 'application-evidence': 'v3_0_application_evidence.py',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['status']+list(STAGES))
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.stage == 'status':
        names = ['preparation.json', 'calibration.json', 'sed_cloud.json',
                 'sed_fit.json', 'controls/complete.json', 'cohorts/complete.json',
                 'uncertainty/complete.json']
        for name in names:
            path = ROOT/'data/v3_0'/name
            if not path.exists():
                path = ROOT/'manifests/v3_0'/name
            if path.exists():
                record = json.loads(path.read_text())
                print(name, json.dumps({k: record[k] for k in
                    ('job', 'jobs', 'n', 'sources', 'admitted', 'converged', 'rows', 'elapsed_seconds') if k in record}))
            else: print(name, 'NO COMPLETION ARTIFACT')
        print('Scientific acceptance and A–F completion require the final delivery audit; file presence is insufficient.')
        return
    from photdis.stage1 import require_compute
    require_compute()
    completed = subprocess.run([sys.executable, '-u', str(ROOT/'scripts'/STAGES[args.stage]), *args.arguments])
    raise SystemExit(completed.returncode)


if __name__ == '__main__': main()
