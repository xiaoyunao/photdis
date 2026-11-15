#!/usr/bin/env python3
"""Audit the exported v3.3 arrays and collect compact science products."""
import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from photdis.stage1 import require_compute
from photdis.v3.likelihood import trapezoid_weights
from v2_1_near import ROOT, csv_rows, load, save_json, sha
from v3_3_catalog import EXPORT, FREEZE, OUTPUT, read_frozen
from v3_3_prospective import FORECAST, SCORES
from v3_3_sed import verified
from v3_3_structure import OUT as STRUCTURE, score

OUT = ROOT/'data/v3_3/delivery'
VIEWS = ('nominal', 'candidate', 'qrf')


def read(path):
    return json.loads(path.read_text())


def authenticate(folder):
    record = read(folder/'complete.json')
    for name, digest in record['outputs'].items():
        verified(folder/name, digest)
    return record


def roster(expected, actual):
    np.testing.assert_array_equal(expected['identity'], actual['identity'])
    np.testing.assert_array_equal(expected['TYPE'], actual['TYPE'])
    if len(np.unique(actual['identity'])) != len(actual['identity']) or not (actual['TYPE'] == 'PSF').all():
        raise ValueError('Unique native PSF sources required')


def validate_sed(t):
    mu = t['mu']
    p = t['joint_density_per_mu']
    if not np.isfinite(p).all() or (p < 0).any() or not np.all(np.diff(mu) > 0):
        raise ValueError('Invalid joint density or distance grid')
    np.testing.assert_allclose(p.sum(axis=1), t['distance_marginal_per_mu'], atol=1e-8, rtol=0)
    np.testing.assert_allclose(p @ trapezoid_weights(mu), t['class_probability'], atol=1e-6, rtol=0)
    np.testing.assert_allclose(t['class_probability'].sum(axis=1), 1, atol=1e-8, rtol=0)
    np.testing.assert_allclose(t['class_distance_bins'].sum(axis=2), t['class_probability'], atol=1e-8, rtol=0)
    near = mu <= 15
    if mu[near][-1] != 15 or mu[-1] <= 10+5*np.log10(50):
        raise ValueError('The grid must cover both sides of 10 kpc and extend beyond 50 kpc')
    np.testing.assert_allclose(t['distance_marginal_per_mu'][:, near] @ trapezoid_weights(mu[near]),
                               t['qnear'], atol=1e-6, rtol=0)
    np.testing.assert_allclose(t['qnear'] + t['qfar'], 1, atol=1e-8, rtol=0)
    flags = ((t['coarse_probability_delta'] > .003) | (t['coarse_quantile_delta_mag'] > .03)
             | (t['edge_mass'].max(axis=1) > 1e-5))
    np.testing.assert_array_equal(flags, t['numerical_flag'])
    for i, m in enumerate((mu < 15, mu >= 15)):
        d = 10**((mu[m]-10)/5)
        predicted = d[(t['distance_marginal_per_mu'][:, m]/d).argmax(axis=1)]
        q = t['qnear' if i == 0 else 'qfar']
        np.testing.assert_allclose(t['distance_mode_kpc'][q >= 1e-12, i], predicted[q >= 1e-12])


def catalog():
    admission = authenticate(OUTPUT)
    export = authenticate(EXPORT)
    rows, seen = [], []
    for fname in admission['outputs']:
        photo = load(OUTPUT/fname)
        field = Path(fname).stem
        seen.extend(photo['identity'].tolist())
        groups = {'all': np.ones(len(photo['identity']), bool)}
        mag = photo['observed_r']
        groups.update({f'r {lo} to {hi}': (mag >= lo) & (mag < hi)
                       for lo, hi in ((16, 17), (17, 18), (18, 19), (19, 20))})
        groups.update({f'W1 state {i}': photo['band_state'][:, 4] == i for i in np.unique(photo['band_state'][:, 4])})
        for view in VIEWS:
            t = load(EXPORT/f'{field}_{view}.npz')
            roster(photo, t)
            if view == 'qrf':
                np.testing.assert_allclose(t['mass'].sum(axis=1), 1, atol=1e-8, rtol=0)
                edges = t['mu_edges']
                qnear = np.sum(t['mass']*np.clip((15-edges[:, :-1])/np.diff(edges, axis=1), 0, 1), axis=1)
                np.testing.assert_allclose(qnear, t['qnear'], atol=1e-8, rtol=0)
                mask = t['supported']
            else:
                validate_sed(t)
                mask = ~t['numerical_flag']
            for group, g in groups.items():
                row = dict(field=field, view=view, group=group, n=int(g.sum()), supported_or_unflagged=int((g & mask).sum()),
                           near_expected=t['qnear'][g].sum(), far_expected=(1-t['qnear'][g]).sum(),
                           qnear_gt_095=int((g & (t['qnear'] > .95)).sum()))
                if view != 'qrf':
                    bins = t['class_distance_bins'][g].sum(axis=0)
                    for c, name in enumerate(('dwarf', 'giant', 'other')):
                        for j, interval in enumerate(('0_2', '2_5', '5_10', '10_20', '20_50', '50_plus')):
                            row[f'{name}_{interval}_expected'] = bins[c, j]
                rows.append(row)
    if len(set(seen)) != len(seen) or len(seen) != export['n'] or export['n'] != admission['n']:
        raise ValueError('Catalog roster is incomplete or contains duplicate identities')
    return rows, export['n']


def structure():
    record = authenticate(STRUCTURE)
    target = read(ROOT/'configs/v3_0_independent_scale.json')
    response = load(STRUCTURE/'response.npz')
    with (STRUCTURE/'distance_score.csv').open() as stream:
        published = {(r['branch'], r['variant']): r for r in csv.DictReader(stream)}
    score_rows = []
    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    for branch, color in [('full', '#156082'), ('gaussian', '#c07025')]:
        for variant in ('fine', 'coarse_posterior', 'coarse_likelihood', 'second_entry_seed'):
            t = load(STRUCTURE/f'{branch}_{variant}.npz')
            roster(response, t)
            p = t['common_mu_probability']
            np.testing.assert_allclose(p.sum(), 1, atol=1e-9, rtol=0)
            mu = t['common_mu']
            values = score(mu, p, target)
            saved = published[branch, variant]
            for key, value in values.items():
                if key == 'external_center_in_68':
                    if str(value) != saved[key]:
                        raise ValueError('Dynamical scale inclusion differs from published value')
                else:
                    np.testing.assert_allclose(value, float(saved[key]), atol=1e-10, rtol=0)
            score_rows.append(dict(branch=branch, variant=variant, **values))
            if variant == 'fine':
                d = 10**((mu-10)/5)
                counts = p/trapezoid_weights(d)
                ax.plot(d, counts, color=color, label='Full likelihood' if branch == 'full' else 'Gaussian summary')
    d = target['distance_kpc']
    ax.axvline(d, color='0.3', ls='--', label='Dynamical distance')
    ax.axvspan(d-target['distance_lower_error_kpc'], d+target['distance_upper_error_kpc'], color='0.7', alpha=.3)
    ax.set(title='NGC 288 common distance', xlabel='Distance (kpc)', ylabel='Posterior density (per kpc)')
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(OUT/f'ngc288_distance.{ext}', dpi=220)
    plt.close(fig)
    return score_rows, record['n']


def prospective():
    forecast, scores = authenticate(FORECAST), authenticate(SCORES)
    if forecast['model_freeze'] != sha(FREEZE) or scores['freeze']['forecast'] != sha(FORECAST/'complete.json'):
        raise ValueError('Broken prediction and opening chain')
    base = load(SCORES/'nominal.npz')
    for view in VIEWS:
        t = load(SCORES/f'{view}.npz')
        roster(base, t)
        roster(t, load(FORECAST/f'{view}.npz'))
        np.testing.assert_array_equal(base['eligible'], t['eligible'])
        np.testing.assert_array_equal(base['corrected_parallax'], t['corrected_parallax'])
        use = t['eligible']
        if not np.isfinite(t['log_score'][use]).all() or not ((t['pit'][use] >= 0) & (t['pit'][use] <= 1)).all():
            raise ValueError('Invalid signed Gaia forecast scores')
    n = int(base['eligible'].sum())
    if n != scores['signed_gaia']:
        raise ValueError('Inconsistent signed Gaia sample size')
    return n


def main():
    require_compute()
    read_frozen()
    OUT.mkdir(parents=True, exist_ok=False)
    census, n = catalog()
    s, cluster = structure()
    gaia = prospective()
    csv_rows(OUT/'catalog_counts.csv', census)
    csv_rows(OUT/'structure_score_recomputed.csv', s)
    parents = (FREEZE, OUTPUT/'complete.json', EXPORT/'complete.json', STRUCTURE/'complete.json',
               FORECAST/'complete.json', SCORES/'complete.json', Path(__file__))
    save_json(OUT/'complete.json', dict(job=os.getenv('SLURM_JOB_ID'), catalog=n, signed_gaia=gaia,
        structure=cluster, inputs={str(p.relative_to(ROOT)): sha(p) for p in parents},
        outputs={p.name: sha(p) for p in sorted(OUT.iterdir())}))


if __name__ == '__main__':
    main()
