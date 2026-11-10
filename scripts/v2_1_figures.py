#!/usr/bin/env python3
"""Compact scientific plates rendered from audited CSVs, without catalog scans."""
import csv
import json
import hashlib
import re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
from photdis_figure_style import STYLE, BLUE, ORANGE, GREEN, GRAY

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT/'manifests/v2_1'
FIG = ROOT/'figures/v2_1'
COLORS = [BLUE, ORANGE, GREEN, GRAY]
CLASSES = ['dwarf', 'evolved', 'other', 'unknown']
LABELS = ['Dwarf', 'Giant reference', 'Other stellar', 'Unlabeled']
DIST = ['0–2', '2–5', '5–10', '10–20', '20–50', '50+']


def read(name):
    with open(TABLES/(name+'.csv')) as f:
        return list(csv.DictReader(f))


def select(rows, **conditions):
    return [r for r in rows if all(r.get(k) == str(v) for k, v in conditions.items())]


def number(row, key):
    return float(row.get(key) or 'nan')


def save(fig, name, rows, sources, caption):
    FIG.mkdir(parents=True, exist_ok=True)
    dest = TABLES/'figures'
    dest.mkdir(exist_ok=True)
    for artist in fig.findobj(match=Text):
        if re.search(r'[\u3400-\u9fff]', artist.get_text()):
            raise ValueError('Non-English figure text')
    fig.savefig(FIG/(name+'.png'), dpi=300, bbox_inches='tight')
    fig.savefig(FIG/(name+'.pdf'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(dest/(name+'.csv'), 'w') as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)
    metadata = dict(name=name, caption=caption, selection='Verified DR11 PSF; development test; no display clipping',
                    rows=len(rows), model='expanded_A',
                    inputs={p: hashlib.sha256((TABLES/(p+'.csv')).read_bytes()).hexdigest() for p in sources})
    (dest/(name+'.json')).write_text(json.dumps(metadata, indent=2)+'\n')


def distance():
    rows = select(read('distance_class'), split='development_test', reference='chosen')
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    selected = []
    for i, (name, label, color) in enumerate(zip(CLASSES, LABELS, COLORS)):
        data = sorted(select(rows, group=name), key=lambda r: number(r, 'low_kpc'))
        selected += data
        n = np.array([number(r, 'n') for r in data])
        axes[0, 0].bar(np.arange(6)+(i-1.5)*.18, n, width=.18, color=color, label=label)
        for ax, key, scale in [(axes[0, 1], 'median_relative', 100), (axes[1, 0], 'tail_1', 100),
                               (axes[1, 1], 'coverage68', 100)]:
            value = np.array([number(r, key)*scale for r in data])
            ax.plot(np.arange(6), value, 'o-', color=color, label=label)
            small = n < 20
            ax.scatter(np.arange(6)[small], value[small], marker='x', s=80, color='black', zorder=4)
    axes[0, 0].set(yscale='log', ylabel='Reference count')
    axes[0, 0].legend(fontsize=10)
    axes[0, 1].set(ylabel='Median distance residual (%)')
    axes[0, 1].axhline(0, color='black', lw=1, ls='--')
    axes[1, 0].set(ylabel=r'Severe fraction: $|\Delta\mu|>1$ (%)', ylim=(-3, 103))
    axes[1, 1].set(ylabel='68% interval coverage (%)', ylim=(-3, 103))
    axes[1, 1].axhline(68, color='black', lw=1, ls='--')
    for ax in axes.ravel():
        ax.set_xticks(np.arange(6), DIST)
        ax.set_xlabel('Reference heliocentric distance (kpc)')
        ax.axvline(2.5, color=GRAY, ls=':', lw=1.5)
    fig.suptitle('Full PSF development sample · expanded A · chosen reference\nBlack crosses: fewer than20 objects', fontsize=16)
    save(fig, '01_distance_population', selected, ['distance_class'],
         'Reference-conditioned performance, not actual selected-near sample or independent accuracy. Sparse bins are descriptive.')


def movement():
    rows = select(read('migration'), split='development_test', reference='chosen')
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    selected = []
    for ax, name, label in zip(axes.ravel(), ['all']+CLASSES, ['All']+LABELS):
        r = select(rows, group=name)[0]
        selected.append(r)
        matrix = np.array([[number(r, 'near_selected'), number(r, 'near_rejected')],
                           [number(r, 'far_selected'), number(r, 'far_rejected')]])
        fraction = matrix/np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        ax.imshow(fraction, vmin=0, vmax=1, cmap='Blues', aspect='auto')
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, f'{int(value):,}\n{fraction[i,j]*100:.1f}%', ha='center', va='center',
                    color='white' if fraction[i,j] > .65 else 'black')
        ax.set_xticks([0, 1], ['Selected near', 'Rejected'])
        ax.set_yticks([0, 1], ['Reference near', 'Reference far'])
        ax.grid(False)
        ax.set_title(label+f"\nSelected far fraction {100*number(r, 'far_contamination'):.1f}%", fontsize=13)
    ax = axes.ravel()[-1]
    vals = [number(select(rows, group=c)[0], 'far_selected') for c in CLASSES]
    ax.bar(np.arange(4), vals, color=COLORS)
    ax.set_xticks(np.arange(4), LABELS, rotation=25, ha='right')
    ax.set_ylabel('Far references selected as near')
    ax.set_title('Composition of far contamination', fontsize=13)
    fig.suptitle(r'Actual median selection: $d_{50}<10$ kpc · PSF development test', fontsize=17)
    save(fig, '02_near_far_migration', selected, ['migration'],
         'Row percentages and counts. Far contamination denominator is all selected objects within each stated reference class. Reference point boundaries; uncertainty separately tabulated.')


def conditions():
    rows = select(read('distance_class'), split='development_test', reference='chosen')
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    selected = []
    for name, label, color in zip(['dwarf_FeH_-inf_-1.5', 'dwarf_FeH_-1.5_-0.5', 'dwarf_FeH_-0.5_inf'],
                                  ['[Fe/H] < -1.5', '-1.5 to -0.5', '[Fe/H] >= -0.5'], COLORS):
        data = sorted(select(rows, group=name), key=lambda r: number(r, 'low_kpc'))
        selected += data
        axes[0].plot(np.arange(6), [100*number(r, 'median_relative') for r in data], 'o-', color=color, label=label)
    axes[0].axhline(0, color='black', ls='--', lw=1)
    axes[0].set(ylabel='Dwarf median distance residual (%)', title='Metallicity strata')
    axes[0].legend(fontsize=9)
    for name, color in zip(['error_0_0.03', 'error_0.03_0.1', 'error_0.1_inf'], COLORS):
        data = sorted(select(rows, group=name), key=lambda r: number(r, 'low_kpc'))
        selected += data
        axes[1].plot(np.arange(6), [100*number(r, 'tail_1') for r in data], 'o-', color=color, label=name.replace('error_', 'mag error ').replace('_', ' to '))
    axes[1].set(ylabel='Severe distance error (%)', title='Optical quality')
    axes[1].legend(fontsize=9)
    for ax in axes[:2]:
        ax.set_xticks(np.arange(6), DIST)
        ax.set_xlabel('Reference distance (kpc)')
        ax.axvline(2.5, color=GRAY, ls=':')
    mig = select(read('migration'), split='development_test', reference='chosen')
    sky = [r for r in mig if r['group'].startswith('cell_')]
    selected += sky
    axes[2].scatter([number(r, 'n') for r in sky], [100*number(r, 'far_contamination') for r in sky], color=BLUE)
    for r in sky:
        axes[2].annotate(r['group'].replace('cell_', ''), (number(r, 'n'), 100*number(r, 'far_contamination')), fontsize=9)
    axes[2].set(xlabel='Chosen references per sky cell', ylabel='Selected far contamination (%)', title='Development sky variation', xscale='log')
    fig.suptitle('Full PSF development strata · no display clipping', fontsize=17)
    save(fig, '06_distance_quality_sky', selected, ['distance_class', 'migration'],
         'Observed reference strata only. Metallicity is diagnostic, never application input. Sparse and selection-dependent evidence; no final-validation claim.')


def information():
    rows = select(read('information_scores'), split='development_test', group='all', raw_or_calibrated='calibrated')
    neighbors = read('color_neighbors')
    degraded = read('template_degradation')
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    names = ['grz', 'griz', 'grzW1', 'grzW1W2']
    selected = list(rows)+degraded
    for method, label, color in [('frozen_partitions_same_labels', 'Frozen partitions, common labels', BLUE),
                                  ('fixed_forest', 'Fixed classification forest', ORANGE)]:
        data = [select(rows, bands=name, method=method)[0] for name in names]
        axes[0, 0].plot(np.arange(4), [number(r, 'log_gain_over_prior') for r in data], 'o-', color=color, label=label)
        axes[0, 1].plot(np.arange(4), [number(r, 'brier') for r in data], 'o-', color=color, label=label)
    axes[0, 0].axhline(0, color=GRAY, ls='--')
    axes[0, 0].set(ylabel='Log score gain over no-color prior (nats)', title='Held-out development class information')
    axes[0, 0].legend(fontsize=10)
    axes[0, 1].set(ylabel='Multiclass Brier score', title='Calibrated probabilities; lower is better')
    for c, label, color in [(0, 'Dwarf', BLUE), (1, 'Giant reference', ORANGE), (2, 'Other stellar', GREEN)]:
        values = []
        for name in names:
            group = select(neighbors, bands=name, class_label=c)
            v = np.array([number(r, 'same_class_mass') for r in group])
            v = v[np.isfinite(v)]
            values.append(float(np.mean(v)) if len(v) else np.nan)
            selected.append(dict(panel='local_same_class', bands=name, class_label=c, n=len(v), mean=values[-1]))
        axes[1, 0].plot(np.arange(4), values, 'o-', color=color, label=label)
    axes[1, 0].set(ylabel='Local same-class reference mass', ylim=(0, 1), title='Full color covariance; bounded neighbor search')
    axes[1, 0].legend(fontsize=10)
    for condition, label, color in [('high_quality', 'Original bright templates', GREEN), ('target_noise', 'Same templates + target noise', GRAY)]:
        data = [select(degraded, bands=name, condition=condition)[0] for name in names]
        axes[1, 1].plot(np.arange(4), [number(r, 'log_gain_over_prior') for r in data], 'o-', color=color, label=label)
    axes[1, 1].set(ylabel='Template-conditional log score gain', title='Shared band noise and common target routes')
    axes[1, 1].legend(fontsize=10)
    for ax in axes.ravel():
        ax.set_xticks(np.arange(4), names)
    fig.suptitle('Matched PSF band controls · identical labels and objects\nFixed forest calibration uses cal only; original QRF partition training differs', fontsize=15)
    save(fig, '03_color_information', selected, ['information_scores', 'color_neighbors', 'template_degradation'],
         'Single fixed classifier framework. Information gain is reference/model conditional, not a physical upper bound. Common i+W1+W2 sample has narrower coverage than optical parent. Degraded training templates are not independent validation.')


if __name__ == '__main__':
    plt.rcParams.update(STYLE)
    distance()
    movement()
    conditions()
    if (TABLES/'information_scores.csv').exists():
        information()
