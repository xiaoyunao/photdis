#!/usr/bin/env python3
"""Once-only registered NGC 4147 transfer; fits remain immutable."""
import hashlib
import json
from pathlib import Path
import socket

import fitsio
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from photdis.cluster_rgb_conditional import grid_from_bins
from photdis.cluster_rgb_lower import probabilities, slope
from photdis.cluster_rgb_external import (
    verify_hashes, identity_hash, abundance_conditions, observe, score,
    prior_only, target_recovery)
from photdis.stage1 import require_compute, write_json

CONFIG = Path('configs/cluster_rgb_ngc4147_v1.json')
COVERAGE = Path('manifests/cluster_rgb_ngc4147_blind_coverage_v1.json')
OUT = Path('manifests/cluster_rgb_ngc4147_result_v1.json')


def main():
    require_compute()
    if OUT.exists(): raise FileExistsError(OUT)
    cfg = json.loads(CONFIG.read_text())
    verify_hashes(cfg['model_sha256'])
    cover = json.loads(COVERAGE.read_text())
    verify_hashes({cover['data_file']: cover['data_sha256']})
    a = json.loads(Path('manifests/cluster_rgb_lower_model_v2.json').read_text())
    b = json.loads(Path('manifests/cluster_rgb_lower_robust_v2.json').read_text())
    fits = a['results'] | b['results']
    grid = grid_from_bins(a['support'], .02)
    whole = fitsio.read(cover['data_file'])
    mg = whole['g0']-cfg['external_mu']
    phase = whole['phase'].astype('U')
    base = ((whole['pmember'] > .9) & (phase == 'lower_rgb')
            & (whole['release'] == 11010) & (mg >= .5) & (mg < 3))
    common = base & whole['grz_clean_same_row']
    common &= np.isfinite(sum(whole[k] for k in (
        'g0', 'gr0', 'rz0', 'sigma_g', 'sigma_r', 'sigma_z')))
    tab, ref = whole[common], mg[common]
    if identity_hash(tab['source_id']) != cover['common_ids_sha256']:
        raise ValueError('Pre-score identity changed')
    if len(tab) == 0: raise ValueError('No eligible independent test stars')
    colors = np.column_stack([tab['gr0'], tab['rz0']])
    errors = np.column_stack([tab['sigma_g'], tab['sigma_r'], tab['sigma_z']])
    conditions = abundance_conditions(cfg)
    baseline, prior_q = prior_only(ref)
    results, rows, quantiles, pair = {}, {}, {}, {}
    for name, fit in fits.items():
        nc = 2 if name.endswith('grz') else 1
        results[name] = {}
        for label, (zs, zw) in conditions.items():
            p = probabilities(fit['models'], grid, zs, zw, colors[:, :nc], errors)
            stats, q, r = score(grid, p, ref)
            stats['log_score_gain_over_prior'] = stats['overall']['log_density']+np.log(2.5)
            stats['width68_fraction_of_prior'] = stats['overall']['width68']/1.7
            results[name][label] = stats
            rows[name+'__'+label], quantiles[name+'__'+label] = r, q
        print(name, results[name]['reference']['overall'], flush=True)
    nmad = lambda x: 1.4826*np.median(abs(x-np.median(x, axis=-1, keepdims=True)), axis=-1)
    ix = np.random.default_rng(4147).integers(len(ref), size=(2000, len(ref)))
    for kind in ('quadratic', 'monotone', 'robust'):
        r1, r2 = rows[kind+'_gr__reference'], rows[kind+'_grz__reference']
        pair[kind] = dict(gain=float(nmad(r1)-nmad(r2)),
                         source_bootstrap95=np.quantile(nmad(r1[ix])-nmad(r2[ix]), [.025,.975]).tolist())

    primary = fits[cfg['primary']]['models']
    zs, zw = conditions['reference']
    sensitivities = {}
    scenarios = {}
    for sign in (-1, 1):
        scenarios[f'mu_{sign:+}sigma'] = dict(dmu=sign*cfg['external_mu_sigma'])
        scenarios[f'dust_{sign:+}sigma'] = dict(dust=sign*cfg['dust_scale_sigma_ebv'])
        for j, band in enumerate('grz'):
            zero = np.zeros(3); zero[j] = sign*cfg['zero_point_sigma_mag']
            scenarios[f'zero_{band}_{sign:+}sigma'] = dict(zero=zero)
    for label, distance in cfg['distance_alternatives_kpc'].items():
        scenarios['distance_'+label] = dict(dmu=5*np.log10(distance)+10-cfg['external_mu'])
    for ebv in cfg['dust_reference_alternatives']:
        scenarios[f'dust_literature_{ebv:.2f}'] = dict(dust=ebv-float(np.median(tab['ebv'])))
    nominal_r = rows[cfg['primary']+'__reference']
    for label, params in scenarios.items():
        shifted, shifted_ref = observe(colors, ref, np.zeros_like(errors), **params)
        p = probabilities(primary, grid, zs, zw, shifted, errors)
        stats, _, r = score(grid, p, shifted_ref, bin_reference=ref)
        stats['median_change_paired'] = float(np.median(r-nominal_r))
        stats['difference_of_medians'] = float(np.median(r)-np.median(nominal_r))
        sensitivities[label] = stats

    extra = base & whole['gr_clean'] & ~common
    extra &= np.isfinite(whole['g0']+whole['gr0']+whole['sigma_g']+whole['sigma_r'])
    gr_extra = {'n': int(extra.sum())}
    if extra.any():
        et = whole[extra]
        # The unused z band is not an input to the one-color likelihood.
        ep = probabilities(fits['robust_gr']['models'], grid, zs, zw,
                           et['gr0'][:, None], np.column_stack([
                               et['sigma_g'], et['sigma_r'], np.zeros(len(et))]))
        gr_extra, _, _ = score(grid, ep, mg[extra])
    print('observed scores and shared-scale scenarios complete', flush=True)

    geometry = []
    central = primary[0]
    cz = np.asarray(central['beta'])[:, -1]
    inv = np.linalg.inv(central['intrinsic'])
    for m in (.875, 1.625, 2.5):
        cm = slope(central, np.array([m]))[0]
        geometry.append(dict(mg=m, dc_dmg=cm.tolist(), dc_dfeh=cz.tolist(),
            d_mg_per_d_feh=float(-(cm@inv@cz)/(cm@inv@cm)),
            covariance_cosine=float((cm@inv@cz)/np.sqrt((cm@inv@cm)*(cz@inv@cz)))))
    # Save observed results before the longer simulation, without changing any fit.
    result = dict(primary=cfg['primary'], config_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
                  model_sha256=cfg['model_sha256'], coverage_sha256=hashlib.sha256(COVERAGE.read_bytes()).hexdigest(),
                  identity_sha256=identity_hash(tab['source_id']), n_common=len(tab),
                  prior_only=baseline, results=results, paired=pair, shared_scale=sensitivities,
                  gr_extra=gr_extra, geometry=geometry, compute_host=socket.gethostname())
    data = Path('data/cluster_rgb_ngc4147_v1'); data.mkdir(parents=True, exist_ok=True)
    write_json(data/'observed_scores.json', result)
    np.savez(data/'scores.npz', source_id=tab['source_id'], mg_ref=ref, **rows)
    recovery = target_recovery(primary, grid, cfg, ref, errors)
    sim_path = Path('manifests/cluster_rgb_ngc4147_recovery_v1.json')
    write_json(sim_path, dict(config_sha256=result['config_sha256'],
                             model_sha256=cfg['model_sha256'], results=recovery))
    result['recovery_manifest'] = str(sim_path)
    sim = recovery['observed_mg_shared']
    result['scale_prediction95'] = sim['bands_true_mg']['overall']['median']
    mid = results[cfg['primary']]['reference']['overall']['median']
    result['center_inside_registered_prediction95'] = bool(
        result['scale_prediction95'][0] <= mid <= result['scale_prediction95'][2])
    trend = {}
    for label, actual in results[cfg['primary']]['reference']['bins'].items():
        expected = sim['bands_proxy_mg'][label]['median']
        trend[label] = dict(n=actual['n'], observed=actual.get('median'),
                            prediction95=expected,
                            inside=None if not actual['n'] or expected is None else bool(expected[0] <= actual['median'] <= expected[2]))
    result['trend_comparison'] = trend
    make_figures(whole, cfg, common, ref, results, baseline, rows, recovery)
    verify_hashes(cfg['model_sha256'])
    write_json(OUT, result)
    print('completed', OUT, flush=True)


def make_figures(whole, cfg, common, ref, results, baseline, rows, recovery):
    out = Path('figures/cluster_rgb'); out.mkdir(exist_ok=True, parents=True)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    high = whole['pmember'] > .9
    ax[0].scatter(whole['bp_rp'][high], whole['gaia_g'][high]-cfg['external_mu'],
                  s=5, color='.65', label='P>0.9 members')
    ax[0].scatter(whole['bp_rp'][common], whole['gaia_g'][common]-cfg['external_mu'],
                  s=12, label='fixed lower-RGB common sample')
    ax[0].set(xlabel='Gaia BP−RP', ylabel='Gaia G − reference modulus', ylim=(5,-2))
    for name in ('quadratic_grz','monotone_grz','robust_grz'):
        ax[1].scatter(ref, rows[name+'__reference'], s=13, alpha=.65, label=name)
    ax[1].axhline(0, color='k', lw=.7)
    ax[1].set(xlabel='Reference Mg', ylabel='Predicted − reference modulus')
    bins = list(baseline['bins']); x = np.arange(.75,3,.5)
    for name, vals in [('prior only',baseline)] + [(n,results[n]['reference']) for n in ('robust_gr','robust_grz')]:
        ax[2].plot(x,[vals['bins'][k].get('median',np.nan) for k in bins], 'o-', label=name)
    ax[2].set(xlabel='Reference Mg bin', ylabel='Median modulus residual')
    for a in ax: a.legend(fontsize=7); a.grid(alpha=.2)
    fig.savefig(out/'08_ngc4147_external_v1.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for mode, label in [('grid_nominal','nominal scales, true Mg'),('grid_shared','shared scales, true Mg')]:
        band = recovery[mode]['bands_true_mg']
        v = np.array([band[k]['median'] for k in bins])
        ax[0].plot(x,v[:,1],'o-',label=label)
        ax[0].fill_between(x,v[:,0],v[:,2],alpha=.12)
    ax[0].set(xlabel='True Mg bin',ylabel='Simulated median modulus residual')
    b = recovery['observed_mg_shared']['bands_proxy_mg']
    v = np.array([b[k]['median'] if b[k]['median'] is not None else [np.nan]*3 for k in bins])
    ax[1].fill_between(x,v[:,0],v[:,2],alpha=.2,label='95% synthetic-cluster envelope')
    ax[1].plot(x,v[:,1],'o-',label='simulation median')
    ax[1].plot(x,[results['robust_grz']['reference']['bins'][k].get('median',np.nan) for k in bins], 'o-',label='observed fixed primary')
    ax[1].set(xlabel='Reference Mg bin',ylabel='Median modulus residual')
    labels=list(results['robust_grz'])
    centers=[results['robust_grz'][k]['overall']['median'] for k in labels]
    widths=[results['robust_grz'][k]['overall']['width68'] for k in labels]
    ax[2].barh(labels,centers,label='center'); ax[2].scatter(np.array(widths)/2,np.arange(len(labels)),color='k',marker='x',label='half median 68% width')
    ax[2].set(xlabel='Modulus residual / interval half-width [mag]')
    for a in ax: a.axhline(0,color='.5',lw=.5) if a != ax[2] else a.axvline(0,color='.5',lw=.5); a.legend(fontsize=7); a.grid(alpha=.2)
    fig.savefig(out/'09_ngc4147_recovery_v1.png',dpi=160); plt.close(fig)


if __name__ == '__main__': main()
