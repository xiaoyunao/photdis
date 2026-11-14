#!/usr/bin/env python3
"""Plot the paired Gaia and giant-class results of the registered size control."""
import csv
from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    base = ROOT/'manifests/v3_2/tilt_size_control/comparison'
    table = base/'paired.csv'
    receipt = base/'complete.json'
    record = json.loads(receipt.read_text())
    if (record['outputs'][table.name] != digest(table) or record['all_PSF'] is not True
            or record['new_outcomes_read'] is not False or record['sample_psf'] != 3540):
        raise ValueError('Invalid opened-only size-control receipt')
    with table.open(newline='') as stream:
        rows = list(csv.DictReader(stream))

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False,
                         'axes.spines.right': False, 'font.size': 10})
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(9.5, 4))
    comparisons = (
        ('broad_zero_3540', 'broad_zero', 'Broad 3,540 vs. broad 26,000', '#24597c'),
        ('candidate_zero', 'broad_zero_3540', 'Candidate vs. broad 3,540', '#a94e22'),
        ('broad_free_3540', 'broad_free', 'Broad 3,540 vs. broad 26,000', '#24597c'),
        ('candidate_free', 'broad_free_3540', 'Candidate vs. broad 3,540', '#a94e22'),
    )
    for ax, title, metric, stratum, n in zip(
        axes, ('Signed Gaia (166 pairs)', 'Giant class (142 pairs)'),
        ('signed_gaia_log_score', 'class_log_score'), ('all', 'giant'), (166, 142)
    ):
        ax.axvline(0, color='.45', linestyle='--', lw=1)
        ax.axhline(1.5, color='.84', lw=.8)
        for y, (left, right, _, color) in enumerate(comparisons):
            hits = [r for r in rows if r['role'] == 'new_sky_field' and r['stratum'] == stratum
                    and r['metric'] == metric and (r['left'], r['right']) == (left, right)]
            if len(hits) != 1 or int(hits[0]['n']) != n or hits[0]['status'] != 'sky_block_interval':
                raise ValueError(f'Invalid comparison: {metric} {left} {right}')
            mean, lo, hi = (float(hits[0][key]) for key in ('mean', 'lo95', 'hi95'))
            ax.hlines(y, lo, hi, color=color, lw=2)
            ax.scatter(mean, y, s=28, color=color, zorder=2)
        ax.set(title=title, xlabel='Paired log-score change (nat)', xlim=(-.85, 1.42), ylim=(3.6, -.6))
        ax.grid(axis='x', alpha=.17)
    axes[0].set_yticks(range(4), [label for _, _, label, _ in comparisons])
    axes[0].text(-.61, .5, 'Zero tilt', transform=axes[0].get_yaxis_transform(), va='center', ha='right')
    axes[0].text(-.61, 2.5, 'Free tilt', transform=axes[0].get_yaxis_transform(), va='center', ha='right')
    fig.suptitle('Opened-sky scores with a size-matched broad sample', fontweight='bold')
    fig.subplots_adjust(left=.34, right=.98, top=.80, bottom=.19, wspace=.16)

    output = ROOT/'figures/v3_2/tilt_size_control'
    output.mkdir(parents=True, exist_ok=True)
    files = [output/f'size_control.{ext}' for ext in ('png', 'pdf')]
    for path in files:
        fig.savefig(path, dpi=300)
    plt.close(fig)
    sources = (table, receipt, Path(__file__).resolve(), *files)
    (base/'figure.json').write_text(json.dumps({str(p.relative_to(ROOT)): digest(p) for p in sources}, indent=2)+'\n')


if __name__ == '__main__':
    main()
