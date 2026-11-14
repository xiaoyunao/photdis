#!/usr/bin/env python3
"""Reconcile compact factorial fit receipts with the verified input shards."""
import argparse
import csv
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path

from v3_2_tilt_input_audit import audit as audit_inputs


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'manifests/v3_2'
MAP_COLUMNS = ('maximum_tilt_delta', 'maximum_joint_basis_delta',
               'maximum_expected_basis_count_delta', 'maximum_fixed_parameter_entry_delta',
               'total_fixed_parameter_log_entry_delta')
DRAW_COLUMNS = ('maximum_fixed_draw_entry_delta', 'absolute_log_normalizer_delta',
                'relative_log_weight_span', 'weight_ess', 'total_variation',
                'maximum_joint_basis_probability_delta', 'maximum_expected_basis_count_delta')
SIGNED_COLUMNS = ('total_fixed_parameter_log_entry_delta', 'absolute_log_normalizer_delta')


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def sensitivity(folder, name, checksum, columns, *, draws=None):
    path = folder/name
    if digest(path) != checksum:
        raise ValueError(f'Sensitivity table digest mismatch: {path}')
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    expected = {'level', *columns}
    if draws is not None:
        expected.add('draws')
    if len(rows) != 2 or {row['level'] for row in rows} != {'0', '2'}:
        raise ValueError(f'Expected two entry alternatives: {path}')
    result = []
    for row in rows:
        if (set(row) != expected or None in row.values()
                or (draws is not None and int(row['draws']) != draws)):
            raise ValueError(f'Invalid sensitivity record: {path}')
        values = {key: float(row[key]) for key in columns}
        if any(not isfinite(v) or (v < 0 and k not in SIGNED_COLUMNS)
               for k, v in values.items()):
            raise ValueError(f'Invalid numerical diagnostic: {path}')
        if draws is not None and (not 1-1e-9 <= values['weight_ess'] <= draws+1e-9
                                  or values['total_variation'] > 1):
            raise ValueError(f'Invalid posterior reweighting diagnostic: {path}')
        result.append(dict(level=int(row['level']), **values))
    return sorted(result, key=lambda row: row['level'])


def audit(*, complete=False):
    inputs = audit_inputs(complete=True)
    if inputs['remaining_flags'] or inputs['pending_flags']:
        raise ValueError('Unresolved input contexts in the primary factorial')
    markers = {}
    recipe = None
    for index in range(len(inputs['shards'])):
        for level, name in [('initial', 'shard'), ('refined', 'refined_shard')]:
            relative = f'manifests/v3_2/tilt_inputs/{name}_{index:02}/complete.json'
            markers[f'data/v3_2/tilt_inputs/{level}/shard_{index:02}/complete.json'] = inputs['inputs'][relative]
            if level == 'initial':
                source = json.loads((ROOT/relative).read_text())['inputs']
                shared = {k: v for k, v in source.items() if k != 'scripts/v3_2_tilt_inputs.py'}
                if recipe is not None and shared != recipe:
                    raise ValueError(f'Scientific recipe varies between initial shards: {index}')
                recipe = shared
    pilot_path = 'manifests/v3_2/tilt_pilot_refined/complete.json'
    pilot = json.loads((ROOT/pilot_path).read_text())
    remote_sources = ('data/v3_1/target_native_pilot/roles.npz',
                      'data/v3_0/uncertainty_refit/draw_12/model.joblib')
    fit_sources = (pilot_path, 'scripts/v3_2_tilt_fit.py', 'src/photdis/v3/fast_hierarchy.py',
                   'src/photdis/v3/hierarchy.py', 'src/photdis/v3/sampling.py',
                   'src/photdis/v3_1/fixed_population.py', 'src/photdis/v3_2/tilt_transfer.py')
    source_digests = {k: pilot['inputs'][k] for k in remote_sources}
    source_digests.update({k: digest(ROOT/k) for k in fit_sources})
    records = []
    for case in ('broad', 'candidate'):
        for mode in ('zero', 'free'):
            folder = BASE/'tilt_fits/initial'/f'{case}_{mode}'
            if not complete and not folder.exists():
                continue
            path = folder/'complete.json'
            receipt = json.loads(path.read_text())
            if ((receipt['case'], receipt['tilt'], receipt['n']) != (case, mode, inputs[case])
                    or receipt['all_PSF'] is not True or receipt['refined_inputs'] is not True
                    or receipt['diagnostic_with_flags'] is not False
                    or receipt['retained_context_flags'] != dict(broad=0, candidate=0)
                    or receipt['shard_markers'] != markers or receipt['shard_inputs'] != recipe
                    or receipt['inputs'] != source_digests):
                raise ValueError(f'Fit identity or input ancestry mismatch: {path}')
            passed = (receipt['max_rhat'] < 1.05 and receipt['minimum_ess'] >= 100
                      and receipt['divergences'] == 0)
            if (receipt['draws_per_chain'] != 512 or receipt['warmup'] != 512
                    or type(receipt['divergences']) is not int or receipt['divergences'] < 0
                    or passed is not receipt['sampler_passed']):
                raise ValueError(f'Invalid sampler verdict: {path}')
            entry = sensitivity(folder, 'entry_sensitivity.csv',
                                receipt['outputs']['entry_sensitivity.csv'], MAP_COLUMNS)
            posterior = sensitivity(folder, 'posterior_entry_sensitivity.csv',
                                     receipt['outputs']['posterior_entry_sensitivity.csv'],
                                     DRAW_COLUMNS, draws=4*receipt['draws_per_chain'])
            if mode == 'zero' and any(r['maximum_tilt_delta'] != 0 for r in entry):
                raise ValueError(f'Unexpected fitted tilt in zero arm: {path}')
            records.append(dict(arm=f'{case}/{mode}', n=receipt['n'], job=receipt['job'],
                                receipt_sha256=digest(path), sampler_passed=passed,
                                map_entry=entry, posterior_entry=posterior))
    return dict(registered=4, verified=len(records), records=records,
                scope='Compact receipts and sensitivity tables. Posterior payload validation '
                      'and opened class-distance score checks run on allocated compute.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--complete', action='store_true', help='Require all four fitted arms')
    print(json.dumps(audit(complete=parser.parse_args().complete), indent=2))
