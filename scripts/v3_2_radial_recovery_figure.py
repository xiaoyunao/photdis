#!/usr/bin/env python3
"""Plot narrow-structure recovery coverage with its registered denominators."""
import csv
from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'manifests/v3_2/radial_report'
OUTPUT = ROOT/'figures/v3_2/radial_recovery'
METRICS = {
    'mu_location': 'Location',
    'mu_width': 'Width',
    'pre_entry_fraction': 'Pre-entry fraction',
    'selected_members': 'Selected member count',
}
GROUPS = (
    ('model', 'near', 'Model SEDs, near background'),
    ('model', 'deep', 'Model SEDs, deep background'),
    ('empirical', 'near', 'Empirical SEDs, near background'),
    ('empirical', 'deep', 'Empirical SEDs, deep background'),
)
FRACTIONS = (.1, .3, .6, .9)
WIDTHS = (.08, .25)
SUBSETS = ('all_fitted', 'relative_and_counts_accepted')
LABELS = ('All fits', 'Accepted fits')


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read_coverage(path):
    with path.open(newline='') as stream:
        rows = [r for r in csv.DictReader(stream) if r['metric'] in METRICS and r['arm'] == 'local']
    fixed, prior = {}, {}
    for r in rows:
        k, n, total = (int(r[name]) for name in ('covered', 'fitted', 'registered'))
        if not 0 <= k <= n <= total or r['subset'] not in SUBSETS:
            raise ValueError('Invalid radial coverage denominator')
        if n:
            estimate, lo, hi = [float(r[key]) for key in ('coverage', 'coverage_lo95', 'coverage_hi95')]
            if not (0 <= lo <= estimate <= hi <= 1 and abs(estimate-k/n) < 1e-12):
                raise ValueError('Invalid fitted coverage interval')
        else:
            estimate = lo = hi = np.nan
        if abs(float(r['coverage_lower_all'])-k/total) > 1e-12 or abs(
                float(r['coverage_upper_all'])-(k+total-n)/total) > 1e-12:
            raise ValueError('Invalid registered coverage bounds')
        if r['design'] == 'fixed':
            key = (r['subset'], r['metric'], r['source'], r['background'], float(r['width']), float(r['fraction']))
            table, size = fixed, 8
        elif (r['design'], r['source'], r['background']) == ('prior_predictive', 'model', 'prior'):
            key = r['subset'], r['metric']
            table, size = prior, 32
        else:
            raise ValueError('Unknown coverage cohort')
        if total != size or key in table:
            raise ValueError('Unexpected coverage stratum or duplicate')
        table[key] = k, n, total, estimate, lo, hi
    keys = {(subset, metric, source, background, width, fraction)
            for subset in SUBSETS for metric in METRICS for source, background, _ in GROUPS
            for width in WIDTHS for fraction in FRACTIONS}
    if set(fixed) != keys or set(prior) != {(subset, metric) for subset in SUBSETS for metric in METRICS}:
        raise ValueError('Coverage is missing a registered design cell')
    return fixed, prior


def fixed_figure(fixed):
    subset = SUBSETS[1]
    fig, axes = plt.subplots(4, 4, figsize=(12.4, 7.8), sharex=True, sharey=True)
    cmap = plt.get_cmap('Blues').with_extremes(bad='#eeeeee')
    scale = Normalize(0, 1)
    for i, (metric, name) in enumerate(METRICS.items()):
        axes[i, 0].set_ylabel(name + '\nWidth (mag)')
        for j, (source, background, title) in enumerate(GROUPS):
            ax = axes[i, j]
            pairs = [[fixed[subset, metric, source, background, w, f] for f in FRACTIONS] for w in WIDTHS]
            values = np.array([[p[3] for p in row] for row in pairs])
            heat = ax.imshow(values, cmap=cmap, norm=scale, aspect='auto')
            for y, row in enumerate(pairs):
                for x, (k, n, total, p, _, _) in enumerate(row):
                    label = f'{k}/{n}' if n else '—'
                    ax.text(x, y-.15, label, va='center', ha='center', fontweight='bold',
                            color='white' if p > .62 else '#1c2932')
                    ax.text(x, y+.19, f'{n}/{total} accepted', va='center', ha='center', fontsize=7,
                            color='white' if p > .62 else '#1c2932')
            ax.set_yticks(range(2), [f'{w:.2f}' for w in WIDTHS])
            if i == 0:
                ax.set_title(title, fontsize=10)
            ax.set_xticks(range(4), [f'{f:.0%}' for f in FRACTIONS])
    fig.suptitle('Fixed-truth recovery after numerical and sampler gates', fontweight='bold')
    fig.supxlabel('Generating pre-entry member fraction', y=.055)
    fig.subplots_adjust(left=.14, right=.89, top=.89, bottom=.135, wspace=.15, hspace=.32)
    color_axis = fig.add_axes([.912, .135, .014, .755])
    fig.colorbar(heat, cax=color_axis, label='95% interval coverage among accepted fits',
                 format=PercentFormatter(1))
    return fig


def prior_figure(prior):
    fig, ax = plt.subplots(figsize=(8.7, 4))
    nominal = ax.axvline(.95, color='.45', linestyle='--', lw=1, label='Nominal 95% coverage')
    colors = ('#355f80', '#aa502d')
    for subset, color, shift in zip(SUBSETS, colors, (-.14, .14)):
        for i, metric in enumerate(METRICS):
            k, n, total, p, lo, hi = prior[subset, metric]
            if n:
                ax.hlines(i+shift, lo, hi, color=color, lw=1.5)
                ax.scatter(p, i+shift, color=color, s=25, zorder=2)
            ax.text(1.03, i+shift, f'{k}/{n} covered, {n}/{total} evaluated',
                    color=color, va='center', fontsize=9)
    ax.set(xlim=(-.025, 1.02), ylim=(3.6, -.6), xlabel='95% interval coverage')
    ax.set_yticks(range(len(METRICS)), list(METRICS.values()))
    ax.set_xticks((0, .25, .5, .75, 1))
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis='x', alpha=.18)
    fig.suptitle('Prior-predictive recovery with random truth', fontweight='bold')
    handles = [nominal, *(Line2D([], [], marker='o', linestyle='', color=color, label=label)
                          for color, label in zip(colors, LABELS))]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(.62, -.26), ncol=3, frameon=False, fontsize=8)
    fig.subplots_adjust(left=.22, right=.71, bottom=.26, top=.86)
    return fig


def main():
    receipt, table = SOURCE/'complete.json', SOURCE/'coverage.csv'
    record = json.loads(receipt.read_text())
    if (record['all_PSF'] is not True or record['n_catalogs'] != 288 or record['n_arms'] != 576
            or record['outputs'][table.name] != digest(table)):
        raise ValueError('Complete authenticated radial recovery required')
    fixed, prior = read_coverage(table)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False,
                         'axes.spines.right': False, 'font.size': 10})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    files = []
    for name, fig in (('fixed_coverage', fixed_figure(fixed)), ('prior_coverage', prior_figure(prior))):
        for ext in ('png', 'pdf'):
            path = OUTPUT/f'{name}.{ext}'
            fig.savefig(path, dpi=300)
            files.append(path)
        plt.close(fig)
    sources = (receipt, table, Path(__file__).resolve(), *files)
    (SOURCE/'figure.json').write_text(json.dumps({str(p.relative_to(ROOT)): digest(p)
                                                 for p in sources}, indent=2)+'\n')


if __name__ == '__main__':
    main()
