#!/usr/bin/env python3
"""Plot the generating fraction and realized membership of the frozen mocks."""
import csv
from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'manifests/v3_2/radial_design'
OUTPUT = ROOT/'figures/v3_2/radial_design'
FRACTIONS = (.1, .3, .6, .9)
WIDTHS = (.08, .25)
COLORS = ('#29617d', '#b74e2e')


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read_design(receipt, design):
    record = json.loads(receipt.read_text())
    if (record['all_PSF'] is not True or record['n_catalogs'] != 288
            or digest(design) != record['outputs'][design.name]):
        raise ValueError('Frozen radial design verification failed')
    with design.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    fixed, prior, ids = {}, {}, set()
    for row in rows:
        catalog, repeat, n, members = (int(row[k]) for k in (
            'catalog', 'repeat', 'n', 'selected_members'))
        fraction, width = float(row['fraction']), float(row['width'])
        if catalog in ids or n != 128 or not 0 <= members <= n or not 0 < fraction < 1:
            raise ValueError(f'Invalid catalog {catalog}')
        ids.add(catalog)
        if row['design'] == 'fixed':
            key = row['source'], row['background'], fraction, width, repeat
            if key in fixed:
                raise ValueError(f'Duplicate fixed design cell: {key}')
            fixed[key] = members/n
        elif (row['design'], row['source'], row['background']) == ('prior_predictive', 'model', 'prior'):
            if repeat in prior:
                raise ValueError(f'Duplicate prior draw: {repeat}')
            prior[repeat] = (fraction, members/n)
        else:
            raise ValueError(f'Unknown design: {row}')
    expected = {(source, background, fraction, width, repeat)
                for source in ('model', 'empirical') for background in ('near', 'deep')
                for fraction in FRACTIONS for width in WIDTHS for repeat in range(8)}
    if ids != set(range(288)) or set(fixed) != expected or set(prior) != set(range(32)):
        raise ValueError('Incomplete fixed or prior-predictive design')
    return fixed, prior


def main():
    design, receipt = SOURCE/'design.csv', SOURCE/'complete.json'
    fixed, prior = read_design(receipt, design)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False,
                         'axes.spines.right': False, 'font.size': 10})
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.7), sharex=True, sharey=True)
    markers = ('o', 's')
    panels = (
        ('model', 'near', 'Model SEDs · Near background', axes[0, 0]),
        ('model', 'deep', 'Model SEDs · Deep background', axes[0, 1]),
        ('empirical', 'near', 'Empirical SEDs · Near background', axes[1, 0]),
        ('empirical', 'deep', 'Empirical SEDs · Deep background', axes[1, 1]),
    )
    prior_axis = axes[0, 2]
    for ax in [p[3] for p in panels] + [prior_axis]:
        ax.plot([0, 1], [0, 1], '--', color='.52', lw=1)
        ax.set(xlim=(0, 1), ylim=(0, 1), xticks=(0, .25, .5, .75, 1),
               yticks=(0, .25, .5, .75, 1))
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.grid(alpha=.18)
        ax.set_axisbelow(True)
    for source, background, title, ax in panels:
        for width, color, marker in zip(WIDTHS, COLORS, markers):
            values = np.array([[fixed[source, background, fraction, width, repeat]
                                for repeat in range(8)] for fraction in FRACTIONS])
            ax.scatter(np.repeat(FRACTIONS, 8), values.ravel(), color=color, marker=marker,
                       facecolors='none', s=22, alpha=.48)
            ax.plot(FRACTIONS, np.median(values, axis=1), color=color, marker=marker,
                    lw=1.3, markersize=4)
        ax.set_title(title, loc='left', fontsize=10, fontweight='bold')
    x, y = zip(*(prior[r] for r in sorted(prior)))
    prior_axis.scatter(x, y, color='#407668', s=17, alpha=.8)
    prior_axis.set_title('Model SEDs · Prior-predictive draws', loc='left',
                         fontsize=10, fontweight='bold')
    prior_axis.tick_params(labelbottom=True)
    legend = axes[1, 2]
    legend.axis('off')
    handles = [Line2D([], [], color=c, marker=m, label=f'{w:.2f} mag width')
               for w, c, m in zip(WIDTHS, COLORS, markers)]
    handles.append(Line2D([], [], color='.52', ls='--', label='Equal fractions'))
    legend.legend(handles=handles, loc='upper left', frameon=False)
    legend.text(.035, .52, 'Open symbols show fixed-truth replicates.\nSolid lines join their medians.',
                transform=legend.transAxes, fontsize=9, linespacing=1.45)
    fig.suptitle('Realized membership after photometric selection', fontweight='bold')
    fig.supxlabel('Generating member fraction')
    fig.supylabel('Realized selected member fraction', x=.018)
    fig.subplots_adjust(left=.087, right=.981, bottom=.11, top=.877, hspace=.41, wspace=.23)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    outputs = [OUTPUT/f'selected_membership.{suffix}' for suffix in ('png', 'pdf')]
    for file in outputs:
        fig.savefig(file, dpi=300)
    plt.close(fig)
    files = [receipt, design, Path(__file__).resolve(), *outputs]
    (SOURCE/'figure.json').write_text(json.dumps(
        {str(p.relative_to(ROOT)): digest(p) for p in files}, indent=2)+'\n')


if __name__ == '__main__':
    main()
