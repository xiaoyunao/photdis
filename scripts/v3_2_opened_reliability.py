#!/usr/bin/env python3
"""Render the frozen field-cohort one-vs-rest reliability comparisons."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'manifests/v3_2/opened_diagnostics'
VIEWS = {
    'reference/nominal': 'Nominal reference',
    'reference/response12': 'Response12 reference',
    'old_target/old_target': 'Old target',
    'candidate/zero': 'Candidate zero',
    'candidate/point': 'Candidate point',
    'candidate/bootstrap_evidence': 'Candidate bootstrap',
}


def main():
    source = SOURCE / 'reliability.csv'
    marker = json.loads((SOURCE / 'complete.json').read_text())
    if hashlib.sha256(source.read_bytes()).hexdigest() != marker['outputs'][source.name]:
        raise ValueError('Reliability table hash mismatch')
    with source.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.8), sharex=True, sharey=True)
    for ax, (view, title) in zip(axes.flat, VIEWS.items()):
        ax.plot([0, 1], [0, 1], color='.63', linewidth=0.9)
        for target, color, symbol in [('giant', '#a45013', 'o'), ('dwarf', '#23699b', 's')]:
            bins = [r for r in rows if r['role'] == 'new_sky_field' and
                    r['view'] == view and r['target_class'] == target and int(r['n'])]
            if sum(int(r['n']) for r in bins) != 157:
                raise ValueError(f'Reliability bins do not conserve labels: {view}, {target}')
            for index, row in enumerate(bins):
                x, y = float(row['mean_probability']), float(row['mean'])
                if row['lo95']:
                    ax.vlines(x, float(row['lo95']), float(row['hi95']), color=color, alpha=.7)
                ax.scatter(x, y, s=16 + 2 * int(row['n']), marker=symbol, color=color,
                           alpha=.74, label=target.title() if index == 0 else None)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=.15)
        ax.set_xlim(-.055, 1.055)
        ax.set_ylim(-.055, 1.055)
        ax.set_xticks([0, .25, .5, .75, 1])
        ax.set_yticks([0, .25, .5, .75, 1])
    fig.supxlabel('Mean predicted class probability', y=.084)
    fig.supylabel('Observed class fraction', x=.027)
    fig.suptitle('Class reliability in the opened new-sky cohort', y=.974, fontsize=17)
    fig.text(.5, .927, '157 labeled PSF objects: 142 giants and 15 dwarfs', ha='center')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .01), ncol=2, frameon=False)
    fig.text(.5, .064, 'Marker area increases with bin occupancy. Lines show 95% sky-block intervals where available.',
             ha='center', fontsize=9)
    fig.subplots_adjust(left=.09, right=.98, top=.86, bottom=.14, wspace=.13, hspace=.26)
    output = ROOT / 'figures/v3_2/opened_diagnostics'
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ('png', 'pdf'):
        fig.savefig(output / f'class_reliability.{suffix}', dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    main()
