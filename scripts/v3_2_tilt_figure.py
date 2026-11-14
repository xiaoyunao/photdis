#!/usr/bin/env python3
"""Render the same-object population and tilt contrasts in the opened field."""
import csv
from hashlib import sha256
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'manifests/v3_2/tilt_report/initial'
OUTPUT = ROOT/'figures/v3_2/tilt_factorial'
CONTRASTS = {
    'broad_free_minus_zero': 'Free tilt in broad population',
    'candidate_free_minus_zero': 'Free tilt in candidate population',
    'candidate_minus_broad_zero': 'Candidate vs. broad at zero tilt',
    'candidate_minus_broad_free': 'Candidate vs. broad with free tilt',
    'population_by_tilt_interaction': 'Population × tilt interaction',
}
PANELS = (
    ('signed_gaia_log_score', 'all', 'Signed Gaia parallax prediction', 166, 9),
    ('class_log_score', 'all', 'Dwarf and giant classification', 157, 8),
    ('class_log_score', 'giant', 'Giant classification', 142, 8),
)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def select_rows(folder):
    marker = folder/'complete.json'
    source = folder/'factorial.csv'
    receipt = json.loads(marker.read_text())
    if receipt['all_PSF'] is not True or digest(source) != receipt['outputs'][source.name]:
        raise ValueError('Invalid opened factorial source')
    expected = {(metric, stratum, contrast): (n, cells)
                for metric, stratum, _, n, cells in PANELS for contrast in CONTRASTS}
    rows = {}
    with source.open(newline='') as stream:
        for row in csv.DictReader(stream):
            key = tuple(row[k] for k in ('metric', 'stratum', 'contrast'))
            if row['role'] != 'new_sky_field' or key not in expected:
                continue
            n, cells, nonfinite = (int(row[k]) for k in ('n', 'cells', 'nonfinite'))
            if key in rows or (n, cells) != expected[key] or not 0 <= nonfinite <= n:
                raise ValueError(f'Mismatched paired support: {key}')
            stats = [float(row[k]) if row[k] else None for k in ('mean', 'lo95', 'hi95')]
            if nonfinite:
                valid = all(x is None for x in stats)
            else:
                valid = all(x is not None and math.isfinite(x) for x in stats) and stats[1] <= stats[2]
            if not valid:
                raise ValueError(f'Invalid paired interval: {key}')
            rows[key] = row
    if rows.keys() != expected.keys():
        raise ValueError('Incomplete factorial contrast source')
    return rows


def render(source, output):
    rows = select_rows(source)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.6), sharey=True)
    for ax, (metric, stratum, title, n, cells) in zip(axes, PANELS):
        for y, contrast in enumerate(CONTRASTS):
            row = rows[metric, stratum, contrast]
            if row['mean']:
                color = '#a65333' if contrast == 'population_by_tilt_interaction' else '#285c80'
                ax.hlines(y, float(row['lo95']), float(row['hi95']), color=color, lw=1.6)
                ax.scatter(float(row['mean']), y, color=color, s=27, zorder=3)
            else:
                ax.text(.04, y, f"Nonfinite ({row['nonfinite']} pairs)",
                        transform=ax.get_yaxis_transform(), va='center', color='.36')
        ax.axvline(0, color='.45', lw=1, zorder=0)
        ax.set_title(f'{title}\n{n} objects across {cells} sky blocks', fontsize=9.5, loc='left')
        ax.set_yticks(range(len(CONTRASTS)), list(CONTRASTS.values()))
        ax.grid(axis='x', alpha=.19)
    axes[0].set_ylim(len(CONTRASTS)-.4, -.6)
    fig.suptitle('Matched population and SED tilt contrasts', fontweight='bold')
    fig.supxlabel('Mean paired log-score change (nat) with 95% sky-block interval', fontsize=9)
    fig.subplots_adjust(left=.235, right=.975, bottom=.17, top=.77, wspace=.27)
    output.mkdir(parents=True, exist_ok=True)
    figures = [output/f'opened_factorial.{ext}' for ext in ('png', 'pdf')]
    for path in figures:
        fig.savefig(path, dpi=300)
    plt.close(fig)
    return figures


def main():
    figures = render(SOURCE, OUTPUT)
    paths = [SOURCE/'complete.json', SOURCE/'factorial.csv', Path(__file__).resolve(), *figures]
    (SOURCE/'figure.json').write_text(json.dumps(
        {str(p.relative_to(ROOT)): digest(p) for p in paths}, indent=2)+'\n')


if __name__ == '__main__':
    main()
