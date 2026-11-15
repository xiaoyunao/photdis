#!/usr/bin/env python3
"""Build the matched development diagnostics from frozen reference and prediction rows."""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from photdis.stage1 import require_compute
from photdis.v3.controls import optical_magnitudes
from photdis.v3.data import SIGNIFICANT, WEAK
from photdis.v3.metrics import distance_summary
from v2_1_near import ROOT, atomic_npz, csv_rows, load, save_json, sha
from v3_3_development import OUT as DEVELOPMENT
from v3_3_sed import BASE, MANIFESTS, verified

OUT = DEVELOPMENT.parent/'initial_report'
VIEWS = ('nominal', 'candidate', 'qrf')
LABELS = ('Nominal SED', 'Class-conditional SED', 'QRF')
COLORS = ('#365a7d', '#b04335', '#52764d')


def reliability(probability, label, use):
    for low, high in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        m = use & (probability >= low)
        m &= probability < high if high < 1. else probability <= high
        n = int(m.sum())
        if n:
            yield dict(lower=low, upper=high, n=n, predicted=float(probability[m].mean()),
                       observed=float(label[m].mean()))


def figure_distance(views, ref, giant, support):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), layout='constrained')
    lo, hi = np.nanpercentile(ref, [1, 99])
    for name, label, color in zip(VIEWS, LABELS, COLORS):
        data = views[name]
        valid = giant & np.isfinite(ref) & (support if name == 'qrf' else True)
        q = data['quantiles']
        if name != 'qrf':
            condition = data['oracle_quantiles']
            axes[0, 0].scatter(ref[valid], condition[valid, 2], s=6, alpha=.32, color=color, label=label)
            axes[1, 0].scatter(ref[valid], condition[valid, 2]-ref[valid], s=6, alpha=.32, color=color)
        axes[0, 1].scatter(ref[valid], q[valid, 2], s=6, alpha=.26, color=color, label=label)
        axes[1, 1].scatter(ref[valid], q[valid, 2]-ref[valid], s=6, alpha=.26, color=color)
    for ax, title in zip(axes[0], ('Given reference giant class', 'Class-marginalized predictions')):
        ax.plot([lo, hi], [lo, hi], ':', color='0.4')
        ax.set_title(title)
        ax.set_ylabel('Predicted distance modulus (mag)')
        ax.legend(frameon=False, fontsize=8)
    for ax in axes[1]:
        ax.axhline(0, color='0.4', ls=':')
        ax.set(xlabel='Reference distance modulus (mag)', ylabel='Predicted − reference (mag)')
    fig.suptitle('Opened development giants')
    save_plate(fig, 'giant_distance')


def save_plate(figure, name):
    for extension in ('png', 'pdf'):
        figure.savefig(OUT/f'{name}.{extension}', dpi=220)
    plt.close(figure)


def report(views, photo):
    candidate = views['candidate']
    ref = candidate['reference_mu']
    label = candidate['reference_class']
    valid = np.isfinite(ref)
    giant = label == 1
    support = views['qrf']['supported']
    mag, _ = optical_magnitudes(photo)
    apparent_r = 22.5-2.5*np.log10(photo['flux'][:, 1])
    r_width = candidate['quantiles'][:, 3]-candidate['quantiles'][:, 1]
    w1 = photo['band_state'][:, 4]
    scopes = {'all': valid, 'giant': giant & valid, 'dwarf': (label == 0) & valid,
              'other': (label == 2) & valid, 'weak_W1': (w1 == WEAK) & valid,
              'significant_W1': (w1 == SIGNIFICANT) & valid,
              'unavailable_W1': ~photo['band_mask'][:, 4] & valid,
              'metal_poor_giant': giant & valid & (candidate['corrected_feh'] < -1.5)}
    scopes.update({f'r_{l}_{h}': valid & (apparent_r >= l) & (apparent_r < h)
                   for l, h in ((0, 18), (18, 20), (20, 22), (22, np.inf))})
    scopes.update({f'distance_{l}_{h}': valid & (10**((ref-10.)/5.) >= l) & (10**((ref-10.)/5.) < h)
                   for l, h in ((0, 2), (2, 5), (5, 10), (10, 20), (20, 50), (50, np.inf))})
    distance, curve = [], []
    for name, values in views.items():
        for group, mask in scopes.items():
            for role, use in [('all_development', mask), ('matched_QRF_support', mask & support)]:
                distance.append(dict(view=name, group=group, role=role,
                                     **distance_summary(values['quantiles'][use], ref[use])))
        roles = [('Nearer than 10 kpc', values['qnear'], ref < 15., valid & support)]
        if name != 'qrf':
            roles.append(('Giant class', values['class_probability'][:, 1], giant, label >= 0))
        for target, p, truth, use in roles:
            curve.extend(dict(view=name, target=target, **row) for row in reliability(p, truth, use))
    csv_rows(OUT/'matched_distance.csv', distance)
    csv_rows(OUT/'reliability.csv', curve)
    figure_distance(views, ref, giant, support)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout='constrained')
    for ax, target in zip(axes, ('Giant class', 'Nearer than 10 kpc')):
        for name, title, color in zip(VIEWS, LABELS, COLORS):
            rows = [row for row in curve if row['view'] == name and row['target'] == target]
            if rows:
                ax.plot([r['predicted'] for r in rows], [r['observed'] for r in rows], 'o-', ms=4,
                        lw=1, color=color, label=title)
        ax.plot([0, 1], [0, 1], ':', color='0.4')
        ax.set(xlim=(0, 1), ylim=(0, 1), title=target, xlabel='Mean predicted probability',
               ylabel='Fraction in development references')
        ax.legend(frameon=False, fontsize=8)
    save_plate(fig, 'reference_reliability')
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), layout='constrained')
    g_r, r_z = mag[:, 0]-mag[:, 1], mag[:, 1]-mag[:, 2]
    g_abs = mag[:, 0]-ref
    residual = candidate['quantiles'][:, 2]-ref
    for ax, data, title, colormisc in (
        (axes[0, 0], residual, 'Distance-modulus residual', dict(vmin=-2, vmax=2, cmap='coolwarm')),
        (axes[0, 1], candidate['qnear'], 'Probability below 10 kpc', dict(vmin=0, vmax=1, cmap='viridis')),
        (axes[1, 0], r_width, 'Central 68% interval width', dict(vmin=0, vmax=3, cmap='magma')),
        (axes[1, 1], candidate['class_probability'][:, 1], 'Giant probability', dict(vmin=0, vmax=1, cmap='viridis'))):
        m = valid & np.isfinite(data)
        y = g_abs if ax in axes[0] else r_z
        h = ax.scatter(g_r[m], y[m], c=data[m], s=8, alpha=.75, **colormisc)
        if ax in axes[0]:
            ax.invert_yaxis()
        ax.set(title=title, xlabel='Dereddened g − r (mag)',
               ylabel='Reference absolute g (mag)' if ax in axes[0] else 'Dereddened r − z (mag)')
        fig.colorbar(h, ax=ax, shrink=.85)
    save_plate(fig, 'color_distance_diagnostics')
    fig, ax = plt.subplots(figsize=(7.5, 4.5), layout='constrained')
    lower, upper = np.arange(5)*.6, (np.arange(5)+1)*.6
    upper[-1] = np.inf
    width_table = []
    for name, title, color in zip(VIEWS, LABELS, COLORS):
        q = views[name]['quantiles']
        w = q[:, 3]-q[:, 1]
        use = valid & support
        for l, h in zip(lower, upper):
            k = use & (w >= l) & (w < h)
            if k.any():
                width_table.append(dict(view=name, n=int(k.sum()), median_width=float(np.median(w[k])),
                    agreement=float(((ref[k] >= q[k, 1]) & (ref[k] <= q[k, 3])).mean()), lower=l, upper=h))
        a = [row for row in width_table if row['view'] == name]
        ax.plot([row['median_width'] for row in a], [row['agreement'] for row in a],
                'o-', color=color, label=title)
    ax.axhline(.68, color='0.4', ls=':')
    ax.set(xlabel='Median fitted 68% interval width (mag)', ylabel='Fraction containing reference',
           title='Interval width and agreement on matched development references', ylim=(0, 1))
    ax.legend(frameon=False, fontsize=8)
    csv_rows(OUT/'interval_agreement.csv', width_table)
    save_plate(fig, 'interval_agreement')
    atomic_npz(OUT/'plot_values.npz', identity=candidate['identity'], TYPE=candidate['TYPE'],
               observed_r=apparent_r, g_minus_r=g_r, r_minus_z=r_z, absolute_g=g_abs)


def main():
    require_compute()
    marker = DEVELOPMENT/'complete.json'
    receipt = json.loads(marker.read_text())
    record = json.loads((MANIFESTS/'cohorts.json').read_text())
    source = verified(BASE/'cohorts/development.npz', record['outputs']['cohorts/development.npz'])
    photo = load(source)
    views = {name: load(verified(DEVELOPMENT/f'{name}.npz', receipt['outputs'][f'{name}.npz'])) for name in VIEWS}
    ids = views['candidate']['identity']
    order = np.argsort(photo['identity'])
    ix = order[np.searchsorted(photo['identity'][order], ids)]
    photo = {key: value[ix] for key, value in photo.items()}
    for data in views.values():
        np.testing.assert_array_equal(data['identity'], photo['identity'])
        np.testing.assert_array_equal(data['TYPE'], photo['TYPE'])
    if not (photo['TYPE'] == 'PSF').all():
        raise ValueError('Expected entirely native PSF development diagnostic')
    OUT.mkdir(parents=True, exist_ok=False)
    report(views, photo)
    save_json(OUT/'complete.json', dict(n=len(ids), all_PSF=True, development_only=True,
        inputs={str(p.relative_to(ROOT)): sha(p) for p in (marker, source, Path(__file__).resolve())},
        outputs={p.name: sha(p) for p in sorted(OUT.iterdir())}, job=os.getenv('SLURM_JOB_ID')))


if __name__ == '__main__':
    main()
