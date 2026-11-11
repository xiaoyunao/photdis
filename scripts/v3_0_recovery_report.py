#!/usr/bin/env python3
"""Aggregate all frozen recovery catalogs without hiding failed numerical cases."""
import csv
import json
import os
import numpy as np
from v2_1_near import ROOT, sha, save_json, csv_rows
from photdis.stage1 import require_compute
D = ROOT/'data/v3_0'


def main():
    require_compute(); rows = []; metrics = []; numeric = []; inputs = {}
    for view in ('grz', 'multiband'):
        for catalog in range(112):
            folder = D/'recovery'/view/f'catalog_{catalog:03}_s2048_cached'
            path = folder/'complete.json'; record = json.loads(path.read_text())
            inputs[str(path.relative_to(ROOT))] = sha(path)
            for name, digest in record['outputs'].items(): assert sha(folder/name) == digest
            tag = dict(view=view, catalog=catalog, scenario=record['scenario'], numerical_passed=record['passed'])
            numeric.append(dict(**tag, max_rhat=record['max_rhat'], minimum_ess=record['minimum_ess'],
                marginal_half_delta=record['marginal_half_delta'], direct_normalization=record['max_direct_normalization_error'],
                omitted_bound=record['max_omitted_bound']))
            for name, dest in [('recovery.csv', rows), ('distance.csv', metrics)]:
                with (folder/name).open() as stream:
                    dest.extend(dict(**tag, **r) for r in csv.DictReader(stream))
    out = ROOT/'manifests/v3_0/recovery'; out.mkdir(parents=True, exist_ok=True)
    csv_rows(out/'counts.csv', rows); csv_rows(out/'distance.csv', metrics); csv_rows(out/'numerical.csv', numeric)
    summary = []
    for view in ('grz', 'multiband'):
        scenarios = sorted({r['scenario'] for r in rows})
        for scenario in scenarios:
            for method in ('qrf_point', 'qrf_pdf', 'v3_reference', 'v3_target'):
                selected = [r for r in rows if r['view'] == view and r['scenario'] == scenario and r['method'] == method and r['axis'] == 'distance']
                catalogs = sorted({r['catalog'] for r in selected}); errors = []
                for catalog in catalogs:
                    group = [r for r in selected if r['catalog'] == catalog]
                    errors.append(sum(abs(float(r['mean'])-int(r['truth'])) for r in group)/128)
                summary.append(dict(view=view, scenario=scenario, method=method, catalogs=len(catalogs),
                    mean_distance_count_L1=float(np.mean(errors)), median_distance_count_L1=float(np.median(errors)),
                    numerical_passed_catalogs=len({r['catalog'] for r in selected if r['numerical_passed']})))
    csv_rows(out/'summary.csv', summary)
    coverage = []
    for view in ('grz', 'multiband'):
        for scenario in sorted({r['scenario'] for r in rows}):
            selected = [r for r in rows if r['view'] == view and r['scenario'] == scenario and r['method'] == 'v3_target']
            for axis in ('distance', 'class_distance'):
                for b in range(6 if axis == 'distance' else 18):
                    g = [r for r in selected if r['axis'] == axis and int(r['bin']) == b]
                    coverage.append(dict(view=view, scenario=scenario, axis=axis, bin=b, catalogs=len(g),
                        coverage95=float(np.mean([r['covered95']=='True' for r in g])),
                        mean_bias_count=float(np.mean([float(r['mean'])-int(r['truth']) for r in g])),
                        mean_width95_count=float(np.mean([float(r['upper95'])-float(r['lower95']) for r in g])),
                        mean_truth_count=float(np.mean([int(r['truth']) for r in g]))))
    csv_rows(out/'coverage.csv', coverage)
    save_json(out/'complete.json', dict(inputs=inputs, catalogs=224, numerically_passed=sum(r['numerical_passed'] for r in numeric),
        maximum_final_halfdelta=max(r['marginal_half_delta'] for r in numeric),
        outputs={p.name: sha(p) for p in out.glob('*.csv')}, job=os.getenv('SLURM_JOB_ID'),
        scope='Observed catalog counts; conditional fitted SED; eight catalogs per scenario share a source library and are not new independent real calibration stars. Numerical failures retained and flagged. Shared-response recovery remains required.'))
    print('RECOVERY REPORT', sum(r['numerical_passed'] for r in numeric), 'of', len(numeric), flush=True)


if __name__ == '__main__': main()
