#!/usr/bin/env python3
"""Reconcile completed ensemble tables; scientific failures remain failures."""
import argparse
import csv
import json
import math
from pathlib import Path
from v2_1_near import ROOT, sha, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', required=True, choices=['broad', 'candidate', 'holdout'])
    case = parser.parse_args().case
    folder = ROOT / 'manifests/v3_1/target_ensemble' / case
    marker = folder / 'complete.json'
    done = json.loads(marker.read_text())
    inputs = {str(marker.relative_to(ROOT)): sha(marker)}
    checks, roundoff = [], []

    def check(name, condition):
        checks.append(dict(check=name, passed=bool(condition)))
        if not condition:
            raise AssertionError(name)

    check('case34PSF', done['case'] == case and done['states'] == 34 and done['all_PSF'])
    tables = {}
    for name in ['inventory.csv', 'weights.csv', 'checks.csv', 'counts.csv']:
        path = folder / name
        check('source_hash:' + name, sha(path) == done['outputs'][name])
        inputs[str(path.relative_to(ROOT))] = sha(path)
        with path.open() as stream:
            tables[name] = list(csv.DictReader(stream))
    inventory = tables['inventory.csv']
    check('34_unique_states', len(inventory) == 34 and {int(r['correction']) for r in inventory} == set(range(34)))
    for r in inventory:
        i = int(r['correction'])
        path = ROOT / f'manifests/v3_1/target_refinement/{case}/correction_{i:02}/complete.json'
        check(f'refinement_identity:{i}', sha(path) == done['inputs'][f'data/v3_1/target_refinement/{case}/correction_{i:02}/complete.json'])
        inputs[str(path.relative_to(ROOT))] = sha(path)
        ref = json.loads(path.read_text())
        check(f'refinement_outcome:{i}', ref['case'] == case and ref['correction'] == i and
              ref['selected_draws'] == int(r['draws']) and ref['passed'] == (r['passed'] == 'True'))
    views = done['views']
    expected = {'zero': [0], 'point': [1], 'bootstrap_evidence': list(range(2, 34)), 'bootstrap_cut': list(range(2, 34))}
    check('four_named_views', len(views) == 4 and {r['view'] for r in views} == set(expected))
    n = {'broad': 26000, 'candidate': 3540, 'holdout': 20000}[case]
    for view in views:
        label = view['view']
        check('fixed_state_set:' + label, view['states'] == expected[label])
        check('all_required_outcomes:' + label, view['inputs_passed'] ==
              all(r['passed'] == 'True' for r in inventory if int(r['correction']) in expected[label]))
        selected = {name: [r for r in rows if r['view'] == label] for name, rows in tables.items() if name != 'inventory.csv'}
        if not view['inputs_passed']:
            check('no_survivor_normalization:' + label, not view['accepted'] and all(not r for r in selected.values()))
            continue
        check('cohort_size:' + label, view['n'] == n)
        numeric = selected['checks.csv']
        check('four_numerical_variants:' + label, len(numeric) == 4 and
              {r['variant'] for r in numeric} == {'base', 'entry_seed', 'entry_level', 'native_fine'})
        for row in numeric:
            variant = row['variant']
            name = label + '/' + variant
            weighted = label != 'bootstrap_cut'
            passed = (float(row['maximum_probability_delta']) < .003 and
                      float(row['maximum_count_mean_delta']) < .5 and
                      float(row['maximum_state_weight_delta']) < .01 and
                      (abs(float(row['log_evidence_delta'])) < .05 or not weighted))
            check('frozen_numeric_gate:' + name, (row['evidence_weighted'] == 'True') == weighted and
                  (row['passed'] == 'True') == passed)
            weights = [r for r in selected['weights.csv'] if r['variant'] == variant]
            check('complete_weight_states:' + name, len(weights) == len(expected[label]) and
                  {int(r['correction']) for r in weights} == set(expected[label]))
            w = [float(r['weight']) for r in weights]
            # Subtracting large absolute log evidences sets the normalization
            # precision. Propagate that roundoff; never renormalize saved data.
            weight_tolerance = max(1e-12, 4 * max(math.ulp(float(r['log_evidence'])) for r in weights))
            weight_error = math.fsum(w) - 1
            check('normalized_nonnegative_weights:' + name, all(math.isfinite(v) and v >= 0 for v in w) and abs(weight_error) < weight_tolerance)
            if variant == 'base':
                check('reported_ESS_and_max:' + label, math.isclose(1 / sum(v*v for v in w), view['state_ESS'], rel_tol=1e-12) and
                      math.isclose(max(w), view['maximum_state_weight'], rel_tol=1e-12))
            rows = [r for r in selected['counts.csv'] if r['variant'] == variant]
            check('18_joint_cells:' + name, len(rows) == 18 and
                  {(int(r['class_index']), int(r['distance_bin'])) for r in rows} == {(c, d) for c in range(3) for d in range(6)})
            check('count_conservation:' + name, all(int(r['n']) == n for r in rows) and
                  abs(math.fsum(float(r['mean']) for r in rows) - n) < 1e-7 + n * weight_tolerance)
            weighted_residual = math.fsum(float(r['mean']) for r in rows) - n * math.fsum(w)
            check('weighted_count_conservation:' + name, abs(weighted_residual) < 1e-7)
            roundoff.append(dict(view=label, variant=variant, weight_sum_minus_one=weight_error,
                weight_roundoff_tolerance=weight_tolerance,
                weighted_count_conservation_residual=weighted_residual,
                count_sum_minus_n=math.fsum(float(r['mean']) for r in rows) - n))
            check('variance_components:' + name, all(math.isclose(float(r['variance']),
                  sum(float(r[k]) for k in ['conditional_variance', 'within_state_shared_variance', 'between_state_variance']),
                  rel_tol=1e-9, abs_tol=1e-7) for r in rows))
            check('ordered_integer_intervals:' + name, all(0 <= int(r['lower95']) <= int(r['median']) <= int(r['upper95']) <= n for r in rows))
        check('view_acceptance_preserved:' + label, view['accepted'] == all(r['passed'] == 'True' for r in numeric))
    main_view = next(v for v in views if v['view'] == 'bootstrap_evidence')
    check('main_flag_preserved', done['main_candidate_accepted'] == main_view['accepted'])
    check('overall_flag_preserved', done['passed'] == all(v['accepted'] for v in views))
    inputs[str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    save_json(folder / 'compact_integrity.json', dict(inputs=inputs, checks=checks,
        actual_checks=len(checks), passed=True, case=case, roundoff=roundoff,
        original_main_accepted=main_view['accepted'], original_overall_passed=done['passed'],
        scope='Actual compact sources, complete state identities, weights, count moments and unchanged numerical flags. Conditional count payload hashes were authenticated in the allocated ensemble job; this compact check does not reread fit population arrays or assert scientific calibration.'))
    print(json.dumps(dict(case=case, actual_checks=len(checks), compact_passed=True,
                         main_numerical_accepted=main_view['accepted'])))


if __name__ == '__main__':
    main()
