#!/usr/bin/env python3
"""Audit/aggregate small frozen result summaries; no catalogs or refitting."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from photdis.giant_information import field_bootstrap, purity, wilson


def aggregate(result):
    cfg = result['config']
    output = {'scope': 'Post-fit arithmetic on frozen whole-field results; no reselection or refitting.',
              'systems': {}, 'validation': {}}
    for system in cfg['release_system'].values():
        cells = [r for r in result['folds'] if r['system']==system]
        fields = [f['systems'][system]['total'] for f in result['inventory']['fields'] if system in f['systems']]
        total = {k: sum(f[k] for f in fields) for k in fields[0]} if fields else {}
        system_out = {'availability': total, 'panels': {}}
        for panel in ('optical', 'common5', 'common20'):
            selected = [r for r in cells if r['panel']==panel]
            tab = {'classification': {}, 'abundance': {}, 'extra_abundance_vs_A': {}}
            for model in ('A', 'B', 'C'):
                available = [r['models'][model]['strata'] for r in selected if model in r['models']]
                strata = sorted(set().union(*(s.keys() for s in available))) if available else []
                tab['classification'][model] = {}
                for s in strata:
                    rows = [a[s] for a in available if s in a]
                    rates = {c: wilson(sum(v[c]['success'] for v in rows), sum(v[c]['n'] for v in rows))
                             for c in ('giants', 'dwarfs')}
                    rates['purity_if_two_classes'] = purity(rates['giants']['rate'], rates['dwarfs']['rate'], cfg['purity_giant_fractions'])
                    tab['classification'][model][s] = rates
            for model in ('marginal', 'A', 'B', 'C'):
                rows = [r['abundance'][model]['strata']['all'] for r in selected if model in r['abundance']]
                if not rows:
                    continue
                n = sum(r['n'] for r in rows)
                tab['abundance'][model] = {'n': n,
                    'mean_log_density': sum(r['n']*r['mean_log_density'] for r in rows)/n,
                    'coverage68': wilson(sum(r['coverage68']['success'] for r in rows), n),
                    'coverage95': wilson(sum(r['coverage95']['success'] for r in rows), n),
                    'field_median_latent_width68_range': [min(r['latent_width68'] for r in rows), max(r['latent_width68'] for r in rows)]}
            for model in ('B', 'C'):
                pairs = [r for r in selected if model in r['abundance'] and 'A' in r['abundance']
                         and r['abundance_test_n']>=cfg['minimum_informative_giants_per_field']]
                delta = [r['abundance'][model]['paired_log_gain']-r['abundance']['A']['paired_log_gain'] for r in pairs]
                tab['extra_abundance_vs_A'][model] = field_bootstrap(delta, cfg['bootstrap_draws'], cfg['seed'])
            system_out['panels'][panel] = tab
        output['systems'][system] = system_out

    checks = {}
    checks['42_registered_cells'] = len(result['folds'])==42
    checks['native_camera_isolation'] = all(
        all(r['system'] in result['inventory']['fields'][i]['systems'] for i in r['training_fields'])
        for r in result['folds'])
    checks['heldout_field_not_in_training'] = all(r['field_index'] not in r['training_fields'] for r in result['folds'])
    checks['same_class_test_counts'] = all(
        len({(v['strata']['all']['giants']['n'], v['strata']['all']['dwarfs']['n']) for v in r['models'].values()})<=1
        for r in result['folds'])
    checks['same_abundance_test_counts'] = all(
        len({v['strata']['all']['n'] for v in r['abundance'].values()})<=1 for r in result['folds'])
    checks['same_native_training_class_counts'] = all(r['train_giants']>=cfg['min_train_giants'] and r['train_dwarfs']>=cfg['min_train_dwarfs']
                                                    for r in result['folds'] if r['status']=='scored')
    checks['coverage_histograms_account_for_tails'] = all(
        sum(s['histograms'][variable][tier])+sum(s['outside_histograms'][variable][tier].values())==n
        for f in result['inventory']['fields'] for s in f['systems'].values()
        for variable in s['histograms'] for tier,n in s['total'].items())
    checks['unique_identity_inventory'] = result['inventory']['n_spectral_records']==result['inventory']['n_spectral_identity_groups']
    checks['no_conflicting_labels'] = result['inventory']['n_conflicting_groups']==0
    assert all(checks.values()), checks
    output['validation'] = checks
    return output


def main():
    source = ROOT/'manifests/giant_information_v1.json'
    result = aggregate(json.loads(source.read_text()))
    result['source_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    result['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    path = ROOT/'manifests/giant_information_decision_v1.json'
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result['validation'], indent=2))


if __name__ == '__main__':
    main()
