#!/usr/bin/env python3
"""Frozen whole-field class/abundance comparisons, with no distance inference."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'.deps')]
import joblib
import numpy as np
from photdis.giant_information import (
    BANDS, ERRORS, DIMENSIONS, fit_classifier, predict_classifier, class_metrics,
    colors_and_covariance, fit_abundance, predict_abundance, abundance_metrics,
    field_bootstrap, purity)
from photdis.stage1 import require_compute, write_json

CONFIG = Path('configs/giant_information_v1.json')


def strata(t, indices, cfg):
    n = len(indices)
    result = {'all': np.ones(n, bool)}
    teff = t['desi_teff'][indices]
    result['teff4200_4500'] = teff<4500
    result['teff4500_5000'] = teff>=4500
    values = {'r': t['r'][indices],
              'optical_error': np.maximum.reduce([t[k][indices] for k in ERRORS[:3]])}
    for key, value in values.items():
        for lo, hi in zip(cfg[key+'_bins'][:-1], cfg[key+'_bins'][1:]):
            result[f'{key}_{lo:g}_{hi:g}'] = (value>=lo) & (value<hi)
    minimum = np.minimum(t['snr_w1'][indices], t['snr_w2'][indices])
    result['wise5_20'] = (minimum>=5) & (minimum<20)
    result['wise20_plus'] = minimum>=20
    return result


def basic_selection(t, cfg):
    finite = np.all(np.isfinite(np.column_stack([t[k] for k in BANDS[:3]+ERRORS[:3]])), axis=1)
    return (t['representative'] & ~t['identity_conflict'] & (t['label']>=0) & finite
            & (t['desi_teff']>=cfg['teff_bounds'][0]) & (t['desi_teff']<cfg['teff_bounds'][1]))


def panel_masks(t, cfg):
    finite = np.all(np.isfinite(np.column_stack([t[k] for k in BANDS+ERRORS])), axis=1)
    wise = finite & t['w1_clean'] & t['w2_clean']
    minimum = np.minimum(t['snr_w1'], t['snr_w2'])
    return {'optical': np.ones(len(wise), bool),
            'common5': wise & (minimum>=cfg['wise_snr_min']),
            'common20': wise & (minimum>=cfg['wise_high_snr'])}


def fit_field(t, inventory, cfg, field, system, panel, mask, out):
    camera = np.array([cfg['release_system'][str(v)] for v in t['release']])
    selected = basic_selection(t, cfg) & mask & (camera==system)
    test = np.flatnonzero(selected & (t['field_index']==field))
    train = np.flatnonzero(selected & (t['field_index']!=field))
    assert not set(t['identity_group'][train]) & set(t['identity_group'][test])
    # Identity assignment has reduced all groups to at most one representative.
    assert len(set(t['identity_group'][np.r_[train, test]]))==len(train)+len(test)
    ng = int((t['label'][train]==1).sum()); nd = int((t['label'][train]==0).sum())
    record = {'field_index': field, 'field': inventory['fields'][field]['field'],
              'system': system, 'panel': panel, 'train_giants': ng, 'train_dwarfs': nd,
              'n_test': len(test), 'models': {}, 'abundance': {},
              'training_fields': np.unique(t['field_index'][train]).tolist()}
    if not len(test):
        record['status'] = 'no_test_labels'; return record
    bands = np.column_stack([t[k] for k in BANDS])
    errors = np.column_stack([t[k] for k in ERRORS])
    models = ['A'] if panel=='optical' else ['A', 'B', 'C']
    masks = strata(t, test, cfg)
    prefix = out/f'{system}_{field}_{panel}'
    predictions = {'test_rows': test, 'ls_id': t['ls_id'][test],
                   'identity_group': t['identity_group'][test], 'label': t['label'][test]}
    if ng>=cfg['min_train_giants'] and nd>=cfg['min_train_dwarfs'] and len(record['training_fields'])>=cfg['min_train_fields']:
        record['status'] = 'scored'
        for name in models:
            model = fit_classifier(bands[train], errors[train], t['label'][train], DIMENSIONS[name],
                                   C=cfg['logistic_C'], draws=cfg['band_draws'], seed=cfg['seed']+field)
            score, score_error, outside = predict_classifier(
                model, bands[test], errors[test], draws=cfg['band_draws'], seed=cfg['seed']+100+field)
            by = {k: class_metrics(t['label'][test][m], score[m], cfg['score_threshold'])
                  for k, m in masks.items() if m.any()}
            by['color_extrapolation'] = class_metrics(t['label'][test][outside], score[outside], cfg['score_threshold'])
            by['color_in_box'] = class_metrics(t['label'][test][~outside], score[~outside], cfg['score_threshold'])
            main = by['all']
            main['purity_if_two_classes'] = purity(main['giants']['rate'], main['dwarfs']['rate'], cfg['purity_giant_fractions'])
            chosen = test[(score>=cfg['score_threshold']) & (t['label'][test]==1)]
            population = {}
            for quantity in ('desi_teff', 'desi_logg', 'feh_corrected'):
                value = t[quantity][chosen]
                value = value[np.isfinite(value)]
                population[quantity] = {'n': len(value), 'q16_50_84': np.quantile(value, [.16, .5, .84]).tolist() if len(value) else None}
            record['models'][name] = {'strata': by, 'selected_labeled_giant_population': population}
            predictions['score_'+name] = score
            predictions['score_noise_sd_'+name] = score_error
            predictions['color_extrapolation_'+name] = outside
            joblib.dump(model, str(prefix)+f'_class_{name}.joblib')
    else:
        record['status'] = 'insufficient_training_classes_or_fields'

    valid = (t['abundance_calibration_valid'] & (t['label']==1)
             & (t['desi_feh_err']>0) & (t['desi_feh_err']<=cfg['feh_sigma_max'])
             & np.isfinite(t['feh_corrected']) & np.isfinite(t['feh_sigma']))
    ztrain, ztest = train[valid[train]], test[valid[test]]
    record['abundance_train_n'] = len(ztrain); record['abundance_test_n'] = len(ztest)
    zfields = np.unique(t['field_index'][ztrain])
    if len(ztrain)>=cfg['min_feh_train'] and len(zfields)>=cfg['min_train_fields'] and len(ztest):
        record['abundance_status'] = 'scored'
        predictions['abundance_test_rows'] = ztest
        for name in ['marginal']+models:
            dim = 0 if name=='marginal' else DIMENSIONS[name]
            if dim:
                colors, cov = colors_and_covariance(bands, errors, dim)
            else:
                colors, cov = np.empty((len(bands), 0)), np.empty((len(bands), 0, 0))
            model = fit_abundance(colors[ztrain], cov[ztrain], t['feh_corrected'][ztrain],
                                  t['feh_sigma'][ztrain], ridge=cfg['feh_ridge'], floor=cfg['feh_intrinsic_floor'])
            mean, sigma = predict_abundance(model, colors[ztest], cov[ztest])
            by = {}
            for key, m in strata(t, ztest, cfg).items():
                if m.any():
                    by[key], _ = abundance_metrics(t['feh_corrected'][ztest][m], t['feh_sigma'][ztest][m], mean[m], sigma[m])
            main, logp = abundance_metrics(t['feh_corrected'][ztest], t['feh_sigma'][ztest], mean, sigma)
            nofloor, _ = abundance_metrics(t['feh_corrected'][ztest], t['feh_sigma_random'][ztest], mean, sigma)
            record['abundance'][name] = {'strata': by, 'fitted_sigma': model['sigma'],
                                        'no_label_floor_fixed_model': nofloor,
                                        'gradient_max': model['gradient_max']}
            predictions['feh_mean_'+name] = mean
            predictions['feh_sigma_'+name] = sigma
            predictions['feh_logp_'+name] = logp
            joblib.dump(model, str(prefix)+f'_feh_{name}.joblib')
        baseline = predictions['feh_logp_marginal']
        for name in models:
            record['abundance'][name]['paired_log_gain'] = float((predictions['feh_logp_'+name]-baseline).mean())
    else:
        record['abundance_status'] = 'insufficient_training_giants_or_fields_or_test'
    np.savez_compressed(str(prefix)+'_scores.npz', **predictions)
    return record


def transfer_summary(records, cfg):
    result = {}
    for system in cfg['release_system'].values():
        result[system] = {}
        for panel in ('optical', 'common5', 'common20'):
            cells = [r for r in records if r['system']==system and r['panel']==panel]
            summary = {'abundance': {}, 'classification': {}}
            for model in ('A', 'B', 'C'):
                usable = [r for r in cells if model in r['abundance'] and
                          r['abundance_test_n']>=cfg['minimum_informative_giants_per_field']]
                value = field_bootstrap([r['abundance'][model]['paired_log_gain'] for r in usable], cfg['bootstrap_draws'], cfg['seed'])
                value['stable_gain'] = bool(value['n_fields']>=cfg['minimum_informative_fields']
                                            and value['interval95'][0]>0 and value['positive_fraction']>=.75)
                summary['abundance'][model] = value
            for model in ('B', 'C'):
                for cls in ('giants', 'dwarfs'):
                    usable = [r for r in cells if model in r['models'] and
                              r['models']['A']['strata']['all'][cls]['n']>=10]
                    values = [r['models'][model]['strata']['all'][cls]['rate']-
                              r['models']['A']['strata']['all'][cls]['rate'] for r in usable]
                    summary['classification'][model+'_'+cls+'_rate_change'] = field_bootstrap(values, cfg['bootstrap_draws'], cfg['seed'])
            result[system][panel] = summary
    return result


def main():
    require_compute()
    cfg = json.loads(CONFIG.read_text()); out = Path(cfg['output_dir'])
    inventory = json.loads((out/'inventory.json').read_text())
    assert inventory['config_sha256']==hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    if (out/'results.json').exists():
        raise FileExistsError(out/'results.json')
    with np.load(out/'labels.npz', allow_pickle=False) as source:
        t = {k: source[k] for k in source.files}
    masks = panel_masks(t, cfg)
    records = []
    for item in inventory['fields']:
        for system in item['systems']:
            for panel, mask in masks.items():
                result = fit_field(t, inventory, cfg, item['field_index'], system, panel, mask, out)
                records.append(result)
                write_json(out/'progress.json', {'completed_folds': len(records), 'latest': result})
                print(json.dumps({k:result[k] for k in ('field', 'system', 'panel', 'status', 'n_test')}), flush=True)
    write_json(out/'results.json', {'config': cfg, 'inventory': inventory, 'folds': records,
                                  'transfer': transfer_summary(records, cfg),
                                  'source_hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                                    [CONFIG, Path(__file__), ROOT/'src/photdis/giant_information.py']}})


if __name__ == '__main__':
    main()
