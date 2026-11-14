#!/usr/bin/env python3
"""Plot the authenticated fixed-distance NGC288 convergence checks."""
import csv
from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
CHECKS = (
    ('maximum_member_probability_delta', 'Maximum membership change', .003),
    ('maximum_mean_count_delta', 'Mean count change (objects)', .5),
    ('maximum_common_mu_quantile_delta', 'Modulus quantile change (mag)', .03),
    ('log_evidence_delta', 'Absolute log evidence change (nat)', .05),
)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    source = ROOT/'manifests/v3_2/structure_odds_audit'
    receipt = source/'complete.json'
    checks = source/'checks.csv'
    record = json.loads(receipt.read_text())
    if (digest(checks) != record['outputs'][checks.name] or not record['all_PSF']
            or record['n'] != 512 or record['responses'] != 32):
        raise ValueError('Invalid fixed-distance NGC288 receipt')
    with checks.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if (len(rows) != 4 or {r['distance_order'] for r in rows} != {'64'}
            or {(r['branch'], int(r['previous_odds_order']), int(r['odds_order'])) for r in rows}
            != {(branch, lo, hi) for branch in ('full', 'gaussian') for lo, hi in ((256, 512), (512, 1024))}):
        raise ValueError('Expected both odds refinements for each likelihood')
    for row in rows:
        passed = all(abs(float(row[key])) < limit for key, _, limit in CHECKS)
        if str(passed) != row['passed']:
            raise ValueError('Recorded decision differs from registered tolerances')

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False,
                         'axes.spines.right': False, 'font.size': 10})
    fig, axes = plt.subplots(2, 2, figsize=(8.7, 6.5))
    for ax, (key, title, limit) in zip(axes.flat, CHECKS):
        ax.axhline(limit, color='.35', linestyle='--', lw=1, label='Tolerance')
        for branch, label, shift, color, marker in (
            ('full', 'Full likelihood', -.12, '#285c80', 'o'),
            ('gaussian', 'Gaussian summary', .12, '#b46323', 's'),
        ):
            values = sorted((r for r in rows if r['branch'] == branch), key=lambda r: int(r['odds_order']))
            ax.scatter([shift, 1+shift], [abs(float(r[key])) for r in values],
                       label=label, color=color, marker=marker, clip_on=False, zorder=3)
        ax.set(title=title, xlim=(-.4, 1.4), ylim=(0, None))
        ax.set_xticks([0, 1], ['256 to 512', '512 to 1024'])
        ax.grid(axis='y', alpha=.22)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle('NGC288 odds quadrature at 64 distance nodes', fontweight='bold')
    fig.supxlabel('Spatial odds quadrature order', y=.092)
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .015), ncol=3, frameon=False)
    fig.subplots_adjust(bottom=.165, top=.875, hspace=.48, wspace=.32)

    out = ROOT/'figures/v3_2/structure_odds_audit'
    out.mkdir(parents=True, exist_ok=True)
    files = [out/f'fixed_distance_odds.{ext}' for ext in ('png', 'pdf')]
    for file in files:
        fig.savefig(file, dpi=300)
    plt.close(fig)
    inventory = source/'figure.json'
    sources = [checks, receipt, Path(__file__).resolve(), *files]
    inventory.write_text(json.dumps({str(p.relative_to(ROOT)): digest(p) for p in sources}, indent=2)+'\n')


if __name__ == '__main__':
    main()
