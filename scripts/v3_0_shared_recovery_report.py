#!/usr/bin/env python3
"""Recover realized counts with finite shared-response and latent uncertainty."""
import argparse
import json
import os
import numpy as np
from scipy.special import logsumexp, softmax
from v2_1_near import ROOT, load, sha, save_json, atomic_npz, csv_rows
from photdis.stage1 import require_compute
from photdis.v3.compression import DISTANCE_MU_BOUNDARIES
from photdis.v3.ensemble_counts import combine_counts, combine_star_means
from photdis.v3.metrics import distance_summary

D = ROOT/'data/v3_0'


def quantile(values, weights):
    order = np.argsort(values)
    return np.asarray(values)[order][np.searchsorted(np.cumsum(weights[order])/weights.sum(), [.025, .975])]


def main():
    require_compute()
    p = argparse.ArgumentParser()
    p.add_argument('--draw-ids', default=','.join(map(str, range(32))))
    p.add_argument('--draws', type=int, default=2048)
    p.add_argument('--catalogs', type=int, default=112)
    p.add_argument('--name', default='all32_native')
    a = p.parse_args()
    ids = list(map(int, a.draw_ids.split(',')))
    assert len(ids) == len(set(ids)) and len(ids) >= 2
    out = D/'shared_recovery_summary'/a.name
    out.mkdir(parents=True, exist_ok=True)
    rows, distances, numerical, weight_rows, inputs = [], [], [], [], {}
    for catalog in range(a.catalogs):
        records, stars, counts = [], [], []
        for draw in ids:
            folder = D/'shared_recovery'/f'refit_{draw:02}'/f'catalog_{catalog:03}_s{a.draws}_native'
            marker = folder/'complete.json'
            record = json.loads(marker.read_text())
            inputs[str(marker.relative_to(ROOT))] = sha(marker)
            for name, digest in record['outputs'].items():
                assert sha(folder/name) == digest
            records.append(record)
            stars.append(load(folder/'per_star.npz'))
            counts.append(load(folder/'counts.npz'))
        for star in stars:
            for key in ('identity', 'TYPE', 'truth_mu', 'truth_class'):
                np.testing.assert_array_equal(star[key], stars[0][key])
            assert (star['TYPE'] == 'PSF').all()
        logz = np.array([r['bridge']['log_evidence'] for r in records])
        weight = softmax(logz)
        combined = combine_counts(counts, logz)
        mean, delta = combine_star_means(
            np.array([s['probability'].reshape(128, 18) for s in stars]),
            np.array([s['half_a'].reshape(128, 18) for s in stars]),
            np.array([s['half_b'].reshape(128, 18) for s in stars]), logz)
        np.testing.assert_allclose(mean.sum(axis=0), combined['mean'], atol=1e-8)
        mu, pit, logscore = [], [], []
        for i in range(128):
            curves, densities = [], []
            for star in stars:
                lo, hi = star['cdf_offset'][i:i+2]
                grid = star['cdf_grid'][lo:hi]
                curves.append((grid, star['class_cdf'][:, lo:hi].sum(axis=0)))
                densities.append(np.interp(star['truth_mu'][i], grid, star['full_logpdf'][lo:hi]))
            grid = np.unique(np.concatenate([x for x, y in curves]))
            cdf = sum(w*np.interp(grid, x, y) for w, (x, y) in zip(weight, curves))
            mu.append(np.interp([.025, .16, .5, .84, .975], cdf, grid))
            pit.append(np.interp(stars[0]['truth_mu'][i], grid, cdf))
            with np.errstate(divide='ignore'):
                logscore.append(logsumexp(np.log(weight)+densities))
        mu = np.array(mu)
        truth_mu, truth_class = stars[0]['truth_mu'], stars[0]['truth_class']
        truth_bin = np.searchsorted(DISTANCE_MU_BOUNDARIES, truth_mu, side='right')
        truth_counts = np.bincount(truth_class*6+truth_bin, minlength=18).reshape(3, 6)
        scenario = records[0]['scenario']
        passed = all(r['passed'] for r in records) and delta < .005
        numeric = dict(catalog=catalog, scenario=scenario, numerical_passed=passed,
            final_marginal_half_delta=delta, response_ess=combined['response_effective_sample_size'],
            failed_response_draws=','.join(str(i) for i, r in zip(ids, records) if not r['passed']))
        numerical.append(numeric)
        for draw, w, record in zip(ids, weight, records):
            weight_rows.append(dict(catalog=catalog, response=draw, weight=w, log_evidence=record['bridge']['log_evidence']))
        realizations = combined['count_draws'].reshape(-1, 3, 6)
        for axis in ('distance', 'class_distance'):
            value = realizations.sum(axis=1) if axis == 'distance' else realizations.reshape(-1, 18)
            truth = truth_counts.sum(axis=0) if axis == 'distance' else truth_counts.ravel()
            expected = mean.reshape(128, 3, 6).sum(axis=(0, 1)) if axis == 'distance' else mean.sum(axis=0)
            for b in range(len(truth)):
                lo, hi = quantile(value[:, b], combined['draw_weights'])
                rows.append(dict(catalog=catalog, scenario=scenario, axis=axis, bin=b, truth=int(truth[b]),
                    mean=expected[b], lower95=int(lo), upper95=int(hi), covered95=bool(lo <= truth[b] <= hi), numerical_passed=passed))
        for group, use in [('all', np.ones(128, bool)), ('2_5kpc', truth_bin == 1)]+[(f'class_{i}', truth_class == i) for i in range(3)]:
            distances.append(dict(catalog=catalog, scenario=scenario, group=group, numerical_passed=passed,
                **distance_summary(mu[use], truth_mu[use]), mean_logscore=float(np.mean(np.asarray(logscore)[use])) if use.any() else np.nan,
                pit_low=float(np.mean(np.asarray(pit)[use] < .05)) if use.any() else np.nan,
                pit_high=float(np.mean(np.asarray(pit)[use] > .95)) if use.any() else np.nan))
        atomic_npz(out/f'catalog_{catalog:03}_counts.npz', **combined)
        atomic_npz(out/f'catalog_{catalog:03}_stars.npz', identity=stars[0]['identity'], TYPE=stars[0]['TYPE'],
            probability=mean.reshape(128, 3, 6), mu=mu, truth_mu=truth_mu, truth_class=truth_class, logscore=logscore, pit=pit)
    csv_rows(out/'counts.csv', rows)
    csv_rows(out/'distance.csv', distances)
    csv_rows(out/'numerical.csv', numerical)
    csv_rows(out/'response_weights.csv', weight_rows)
    summary = []
    for scenario in sorted({r['scenario'] for r in rows}):
        for axis in ('distance', 'class_distance'):
            for b in range(6 if axis == 'distance' else 18):
                r = [x for x in rows if x['scenario'] == scenario and x['axis'] == axis and x['bin'] == b]
                summary.append(dict(scenario=scenario, axis=axis, bin=b, catalogs=len(r),
                    coverage95=float(np.mean([x['covered95'] for x in r])),
                    mean_bias_count=float(np.mean([x['mean']-x['truth'] for x in r])),
                    mean_width95_count=float(np.mean([x['upper95']-x['lower95'] for x in r]))))
    csv_rows(out/'coverage.csv', summary)
    save_json(out/'complete.json', dict(inputs=inputs, response_ids=ids, catalogs=a.catalogs,
        numerically_passed=sum(r['numerical_passed'] for r in numerical),
        outputs={p.name: sha(p) for p in out.iterdir() if p.suffix in ('.csv', '.npz')}, job=os.getenv('SLURM_JOB_ID'),
        scope='Finite response-prior inference, observed latent counts; repeated simulated catalogs share a template library. Low response ESS is unresolved finite-prior integration, not scientific certainty. Failed numerical cases remain visible.'))


if __name__ == '__main__':
    main()
