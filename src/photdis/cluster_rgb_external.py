"""Frozen RGB external-transfer diagnostics and target-matched generators."""
import hashlib
from pathlib import Path

import numpy as np

from photdis.cluster_rgb_conditional import RG, RR, RZ
from photdis.cluster_rgb_lower import mean, probabilities, summarize

EXTINCTION = np.array([RG, RR, RZ])


def verify_hashes(expected):
    for name, digest in expected.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen input changed: {name}")


def identity_hash(ids):
    ids = np.asarray(ids, dtype='<i8')
    if len(np.unique(ids)) != len(ids):
        raise ValueError('Repeated source identities')
    return hashlib.sha256(np.sort(ids).tobytes()).hexdigest()


def abundance_conditions(cfg):
    def nodes(z, sigma):
        return ([z-np.sqrt(2)*sigma, z, z+np.sqrt(2)*sigma], [.25, .5, .25])
    out = {'reference': nodes(cfg['feh'], cfg['feh_sigma']),
           'exact_reference': ([cfg['feh']], [1.])}
    for label, (z, sigma) in cfg['feh_alternatives'].items():
        out[label] = nodes(z, sigma)
    for label, weights in cfg['unknown_weights'].items():
        out['unknown_'+label] = (cfg['unknown_feh_components'], weights)
    return out


def observe(colors, mg, band_noise, *, zero=None, dust=0., dmu=0.):
    """Same observed g in color and final modulus, including coherent shifts.

    Positive dmu means a reference modulus larger than the generator truth.
    Positive dust means an additional applied E(B-V) correction.
    """
    shift = np.asarray(band_noise) + (np.zeros(3) if zero is None else zero)
    shift = shift - EXTINCTION*dust
    dc = np.column_stack([shift[:, 0]-shift[:, 1], shift[:, 1]-shift[:, 2]])
    return np.asarray(colors)+dc, np.asarray(mg)+shift[:, 0]-dmu


def score(grid, prob, ref, bin_reference=None):
    stats, q, residual = summarize(grid, prob, ref, bin_reference=bin_reference)
    bins = np.asarray(ref if bin_reference is None else bin_reference)
    masks = {'overall': np.ones(len(ref), bool)}
    masks.update({f'{lo:.1f}_{lo+.5:.1f}': (bins >= lo) & (bins < lo+.5)
                  for lo in np.arange(.5, 3, .5)})
    for label, mask in masks.items():
        s = stats['overall'] if label == 'overall' else stats['bins'][label]
        if mask.any():
            s['width95'] = float(np.median(q[mask, 4]-q[mask, 0]))
    return stats, q, residual


def prior_only(ref):
    ref = np.asarray(ref)
    q = np.tile([.5625, .9, 1.75, 2.6, 2.9375], (len(ref), 1))
    def one(mask):
        if not mask.any(): return {'n': 0}
        r = ref[mask]-1.75
        outside = (ref[mask] < .5) | (ref[mask] >= 3)
        return dict(n=int(mask.sum()), median=float(np.median(r)),
                    nmad=float(1.4826*np.median(abs(r-np.median(r)))),
                    tail=float(np.mean(abs(r) > .5)),
                    cover68=float(np.mean((ref[mask] >= .9) & (ref[mask] <= 2.6))),
                    cover95=float(np.mean((ref[mask] >= .5625) & (ref[mask] <= 2.9375))),
                    width68=1.7, width95=2.375, outside_support_n=int(outside.sum()),
                    log_density=None if outside.any() else float(-np.log(2.5)))
    return {'overall': one(np.ones(len(ref), bool)),
            'bins': {f'{lo:.1f}_{lo+.5:.1f}': one((ref >= lo) & (ref < lo+.5))
                     for lo in np.arange(.5, 3, .5)}}, q


def matched_errors(true_mg, observed_mg, errors, rng):
    """Match the empirical error vectors by fixed half-mag bins."""
    actual_bin = np.clip(((np.asarray(observed_mg)-.5)/.5).astype(int), 0, 4)
    target_bin = np.clip(((np.asarray(true_mg)-.5)/.5).astype(int), 0, 4)
    populated = np.unique(actual_bin)
    indices = []
    for b in target_bin:
        nearest = populated[np.argmin(abs(populated-b))]
        indices.append(rng.choice(np.flatnonzero(actual_bin == nearest)))
    return np.asarray(errors)[indices]


def cluster_bands(per_cluster):
    """Predictive bands across whole realizations, not independent stars."""
    out = {}
    for label in ['overall'] + list(per_cluster[0]['bins']):
        values = [v['overall'] if label == 'overall' else v['bins'][label]
                  for v in per_cluster]
        out[label] = {}
        for key in ('median', 'nmad', 'cover68', 'cover95', 'width68', 'width95'):
            v = [s[key] for s in values if s['n']]
            out[label][key] = np.quantile(v, [.025, .5, .975]).tolist() if v else None
    return out


def target_recovery(models, grid, cfg, observed_mg, errors):
    """A grid shrinkage control and two fully shared cluster generators."""
    sim = cfg['simulation']
    zs, zw = abundance_conditions(cfg)['reference']
    out = {}
    for mode in ('grid_nominal', 'grid_shared', 'observed_mg_shared'):
        rng = np.random.default_rng(sim['seed'])
        records, proxy_records, nuisances = [], [], []
        for cid in range(sim['clusters']):
            m = (np.asarray(observed_mg).copy() if mode == 'observed_mg_shared'
                 else np.repeat(sim['grid_mg'], sim['repeats_per_grid']))
            err = matched_errors(m, observed_mg, errors, rng)
            shared = mode != 'grid_nominal'
            index = int(rng.integers(len(models))) if shared else 0
            z = rng.normal(cfg['feh'], cfg['feh_sigma']) if shared else cfg['feh']
            dmu = rng.normal(0, cfg['external_mu_sigma']) if shared else 0.
            dust = rng.normal(0, cfg['dust_scale_sigma_ebv']) if shared else 0.
            zero = rng.normal(0, cfg['zero_point_sigma_mag'], 3) if shared else np.zeros(3)
            generator = models[index]
            color = mean(generator, m, z)
            scatter = rng.multivariate_normal(np.zeros(2), generator['intrinsic'], len(m))
            mix = generator['contamination']
            component = rng.choice(len(mix['weights']), len(m), p=mix['weights'])
            scatter *= np.sqrt(np.asarray(mix['scales'])[component])[:, None]
            noisy, ref = observe(color+scatter, m, rng.normal(size=err.shape)*err,
                                 zero=zero, dust=dust, dmu=dmu)
            prob = probabilities(models, grid, zs, zw, noisy, err)
            stats, _, _ = score(grid, prob, ref, bin_reference=m)
            proxy_stats, _, _ = score(grid, prob, ref)
            records.append(stats)
            proxy_records.append(proxy_stats)
            nuisances.append(dict(model_index=index, feh=float(z), dmu=float(dmu),
                                  dust=float(dust), zero=zero.tolist()))
        out[mode] = dict(per_cluster=records, per_cluster_proxy_bins=proxy_records,
                         bands_true_mg=cluster_bands(records),
                         bands_proxy_mg=cluster_bands(proxy_records), nuisances=nuisances,
                         n_per_cluster=len(m), synthetic_clusters=sim['clusters'])
        print('simulation completed', mode, flush=True)
    return out
