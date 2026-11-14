#!/usr/bin/env python3
"""Render opened classification comparisons and the frozen count transfer audit."""
import csv
from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT/'manifests/v3_2'
OUTPUT = ROOT/'figures/v3_2/opened_audits'
COLORS = ('#285c80', '#287c79', '#b46323')


def table(folder, name):
    root = INPUTS/folder
    source = root/name
    marker = json.loads((root/'complete.json').read_text())
    if sha256(source.read_bytes()).hexdigest() != marker['outputs'][name]:
        raise ValueError(f'Source hash mismatch: {source}')
    with source.open(newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    paired = table('opened_probes', 'paired.csv')
    counts = table('transfer_count_audit', 'conditional_counts.csv')
    anchors = ('reference/nominal', 'old_target/old_target', 'candidate/zero')
    labels = ('Nominal reference', 'Old target', 'Candidate zero')
    selected = {(r['anchor'], r['stratum']): r for r in paired
                if r['role'] == 'new_sky_field' and r['probe'] == 'multiband'
                and r['metric'] == 'log_score' and r['stratum'] in ('all', 'giant')}
    plotted = []
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False,
                         'axes.spines.right': False, 'font.size': 10})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={'width_ratios': [1.03, 1]})
    ax = axes[0]
    for i, (anchor, label) in enumerate(zip(anchors, labels)):
        for j, (stratum, n, symbol, color) in enumerate(
                [('all', 157, 'o', COLORS[0]), ('giant', 142, 's', COLORS[1])]):
            r = selected[anchor, stratum]
            if int(r['n']) != n or int(r['cells']) != 8:
                raise ValueError('Probe panel has changed its frozen reference set')
            y = i+(-.14 if j == 0 else .14)
            x, lo, hi = [float(r[k]) for k in ('mean', 'lo95', 'hi95')]
            ax.hlines(y, lo, hi, color=color, lw=1.7)
            ax.scatter(x, y, color=color, marker=symbol, s=35, zorder=3,
                       label=f'{"All labeled" if j == 0 else "Giants"} (n={n})' if i == 0 else None)
            plotted.append(dict(panel='class_probe', anchor=label, stratum=stratum, n=n,
                                mean=x, lo95=lo, hi95=hi))
    ax.axvline(0, color='.48', lw=1, zorder=0)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlabel('Multiband minus comparator log score (nat)')
    ax.set_title('Opened class-probe comparison', loc='left', fontweight='bold')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.23), frameon=False, ncol=2)
    ax.grid(axis='x', alpha=.2)

    ax = axes[1]
    rows = sorted((r for r in counts if r['envelope'] == 'broad' and r['variant'] == 'entry_level'),
                  key=lambda r: int(r['state']))
    if len(rows) != 34 or sum(r['v31_passed'] == 'False' for r in rows) != 33:
        raise ValueError('Count panel requires the frozen 34-state transfer')
    for r in rows:
        state = int(r['state'])
        weight, response = float(r['weight_effect']), float(r['response_effect'])
        np.testing.assert_allclose(weight+response, float(r['count_delta']), atol=1e-10, rtol=0)
        plotted.append(dict(panel='count_transfer', state=state, weight_effect=weight,
                            response_effect=response, passed=r['v31_passed']))
    x = [int(r['state']) for r in rows]
    ax.plot(x, [float(r['weight_effect']) for r in rows], 'o', ms=4.5, color=COLORS[2],
            label='Population weights')
    ax.plot(x, [float(r['response_effect']) for r in rows], '|', ms=6, color=COLORS[0],
            label='Conditional response')
    ax.axhline(-.5, color='.45', lw=1, linestyle='--', label='Negative tolerance edge')
    ax.set(xlabel='Correction state index', ylabel='Signed shift at the largest-change bin (objects)',
           ylim=(-.74, .10))
    ax.set_title('Broad-envelope expected count transfer', loc='left', fontweight='bold')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.23), frameon=False, ncol=2)
    ax.grid(axis='y', alpha=.2)
    fig.subplots_adjust(left=.134, right=.975, top=.90, bottom=.26, wspace=.37)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(OUTPUT/f'probe_and_count_audit.{ext}', dpi=300)
    plt.close(fig)
    source = INPUTS/'opened_audits_figure'
    source.mkdir(exist_ok=True)
    fields = list(dict.fromkeys(k for row in plotted for k in row))
    with (source/'plotted.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(plotted)
    files = [INPUTS/'opened_probes/paired.csv', INPUTS/'transfer_count_audit/conditional_counts.csv',
             Path(__file__).resolve(), *sorted(OUTPUT.glob('probe_and_count_audit.*')), source/'plotted.csv']
    (source/'complete.json').write_text(json.dumps(
        {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in files}, indent=2)+'\n')


if __name__ == '__main__':
    main()
