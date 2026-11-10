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
        writer = csv.DictWriter(f, fields, lineterminator='\n')
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
    corrected = (TABLES/'metallicity_quality.csv').exists()
    metal = select(read('metallicity_quality'), split='development_test', abundance='corrected') if corrected else []
    for name, flo, label, color in zip(['dwarf_FeH_-inf_-1.5', 'dwarf_FeH_-1.5_-0.5', 'dwarf_FeH_-0.5_inf'],
                                  ['-inf', '-1.5', '-0.5'], ['[Fe/H] < -1.5', '-1.5 to -0.5', '[Fe/H] >= -0.5'], COLORS):
        data = sorted(select(metal, feh_low=flo) if corrected else select(rows, group=name), key=lambda r: number(r, 'low_kpc'))
        selected += data
        values = np.array([100*number(r, 'median_relative') for r in data])
        axes[0].plot(np.arange(6), values, 'o-', color=color, label=label)
        sparse = np.array([number(r, 'n') < 20 for r in data])
        axes[0].scatter(np.arange(6)[sparse], values[sparse], color='black', marker='x', s=65, zorder=4)
    axes[0].axhline(0, color='black', ls='--', lw=1)
    axes[0].set(ylabel='Dwarf median distance residual (%)', title='Corrected [Fe/H], valid domain' if corrected else 'Raw RVS metallicity strata')
    axes[0].legend(fontsize=9)
    for name, color in zip(['error_0_0.03', 'error_0.03_0.1', 'error_0.1_inf'], COLORS):
        data = sorted(select(rows, group=name), key=lambda r: number(r, 'low_kpc'))
        selected += data
        values = np.array([100*number(r, 'tail_1') for r in data])
        axes[1].plot(np.arange(6), values, 'o-', color=color, label=name.replace('error_', 'mag error ').replace('_', ' to '))
        sparse = np.array([number(r, 'n') < 20 for r in data])
        axes[1].scatter(np.arange(6)[sparse], values[sparse], color='black', marker='x', s=65, zorder=4)
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
    fig.suptitle('Full PSF development strata · no display clipping · crosses: N<20', fontsize=17)
    save(fig, '06_distance_quality_sky', selected, ['distance_class', 'migration']+(['metallicity_quality'] if corrected else []),
         'Observed reference strata only. Corrected FeH uses author domain and positive formal errors; diagnostic, never application input. Sparse and selection-dependent evidence; no final-validation claim.')


def pdf_examples():
    rows = read('pdf_examples')
    ids = list(dict.fromkeys(r['identity'] for r in rows))
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for ax, identity in zip(axes.ravel(), ids):
        data = select(rows, identity=identity)
        x = np.array([(number(r, 'mu_low')+number(r, 'mu_high'))/2 for r in data])
        total = np.array([number(r, 'total_density') for r in data])
        for key, label, color in zip(CLASSES, LABELS, COLORS):
            field = 'unlabeled' if key == 'unknown' else key
            ax.plot(x, [number(r, field) for r in data], color=color, label=label)
        ax.plot(x, total, color='black', lw=1.4, label='Total with noise')
        ax.plot(x, [number(r, 'nominal_density') for r in data], color='black', lw=1, ls='--', label='Total, nominal colors')
        ref = number(data[0], 'reference_mu')
        ax.axvline(ref, color='#9467bd', ls=':', label='Chosen reference')
        ax.axvline(15, color=GRAY, ls='--', alpha=.5)
        significant = total > total.max()*.001
        lo = min(x[significant].min(), ref, 15)-.5
        hi = max(x[significant].max(), ref, 15)+.5
        ax.set(xlim=(lo, hi), xlabel='Distance modulus (mag)', ylabel='Reference mass density / mag',
               title=data[0]['category'].replace('_', ' ')+f"\nq(near)={number(data[0], 'qnear'):.3f}")
        ax.set_title(ax.get_title(), fontsize=12)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Frozen QRF joint reference contributions · illustrative PSF development cases\nClass channels sum to the original PDF; unlabeled is not a physical population', fontsize=15)
    save(fig, '04_joint_reference_pdf', rows, ['pdf_examples'],
         'Deterministic identity-hash examples within stated outcome strata. Full exported grid retained despite display limits. Solid class curves include shared-g Sobol128 noise. Reference contribution is not calibrated physical class-distance posterior.')


def near_probability():
    rel = select(read('reliability'), split='development_test', reference='chosen', boundary='point_reference')
    risk = select(read('risk_retention'), split='development_test', reference='chosen', boundary='point_reference')
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    selected = []
    for name, label, color in zip(['all', 'dwarf', 'evolved', 'unknown'], ['All', 'Dwarf', 'Giant reference', 'Unlabeled'], ['black', BLUE, ORANGE, GRAY]):
        data = select(rel, group=name)
        selected += data
        axes[0].plot([number(r, 'mean_q') for r in data], [number(r, 'observed_near') for r in data], 'o-', color=color, label=label)
        small = [r for r in data if 0 < number(r, 'n') < 20]
        axes[0].scatter([number(r, 'mean_q') for r in small], [number(r, 'observed_near') for r in small], marker='x', color='black')
        data = select(risk, group=name, evolved_weight_stress='1.0')
        selected += data
        axes[1].plot([number(r, 'completeness') for r in data], [number(r, 'far_contamination') for r in data], 'o-', color=color, label=label)
        if name == 'all':
            for r in data:
                axes[1].annotate('q>'+r['threshold'], (number(r, 'completeness'), number(r, 'far_contamination')), fontsize=9)
    axes[0].plot([0, 1], [0, 1], ls='--', color=GRAY)
    axes[0].set(xlim=(0, 1), ylim=(0, 1), xlabel='Mean predicted near probability', ylabel='Observed reference-near fraction', title='Reference-event reliability')
    axes[0].legend(fontsize=9)
    axes[1].set(xlabel='Reference-near completeness', ylabel='Far fraction among selected', title='Risk versus retention', ylim=(0, 1))
    for factor, label, color in [('0.5', 'Giant mass x0.5', BLUE), ('1.0', 'Original mass', GRAY), ('2.0', 'Giant mass x2', ORANGE)]:
        data = select(risk, group='all', evolved_weight_stress=factor)
        selected += data
        axes[2].plot([number(r, 'threshold') for r in data], [number(r, 'far_contamination') for r in data], 'o-', color=color, label=label)
    axes[2].set(xlabel='Near-probability threshold', ylabel='Selected far contamination', title='Branch-weight stress, not calibrated priors')
    axes[2].legend(fontsize=9)
    fig.suptitle('expanded A · full PSF development reference sample · full histogram probabilities', fontsize=15)
    save(fig, '05_near_probability', selected, ['reliability', 'risk_retention'],
         'Thresholds fixed at0.5/0.8/0.95; cal selects rule, development test evaluates. Class strata are diagnostics, not an equal-per-class calibration requirement. Stress multipliers are not real-population prior estimates. Boundary-safe rows tabulated separately.')


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
    if (TABLES/'pdf_examples.csv').exists():
        pdf_examples()
    if (TABLES/'reliability.csv').exists():
        near_probability()
