#!/usr/bin/env python3
"""Plot signed Gaia score and HST scale results from authenticated tables."""
import csv
from hashlib import sha256
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'manifests/v3_2'


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read_verified(folder, table):
    receipt = MANIFEST / folder / 'complete.json'
    source = receipt.with_name(table)
    record = json.loads(receipt.read_text())
    if not record['all_PSF'] or record['outputs'][table] != digest(source):
        raise ValueError(f'Invalid source receipt: {source}')
    with source.open(newline='') as stream:
        return list(csv.DictReader(stream)), (receipt, source)


def select(rows, n, key, **filters):
    matches = [row for row in rows if all(row[field] == val for field, val in filters.items())]
    if len(matches) != 1 or int(matches[0]['n']) != n:
        raise ValueError(f'Unexpected row count for {filters}')
    center, lower, upper = (float(matches[0][field]) for field in (key, 'lo95', 'hi95'))
    if not all(map(math.isfinite, (center, lower, upper))) or lower > upper:
        raise ValueError(f'Invalid interval for {filters}')
    return center, lower, upper


def signed_score_panel(ax, title, rows, n, contrasts, anchor_key, **filters):
    ax.axvline(0, color='.45', lw=1, linestyle='--')
    for y, (anchor, _, color) in enumerate(contrasts):
        center, lower, upper = select(rows, n, 'mean', **filters, **{anchor_key: anchor})
        ax.hlines(y, lower, upper, color=color, lw=2)
        ax.scatter(center, y, color=color, s=28, zorder=3)
    ax.set(title=title, xlabel='Candidate minus comparator (nat)', ylim=(2.5, -.5))
    ax.set_yticks(range(len(contrasts)), [label for _, label, _ in contrasts])
    ax.grid(axis='x', alpha=.17)


def reliability_plate(rows, out):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.8), sharey=True)
    for ax, target, title, n in zip(axes, ('near', 'giant'),
                                     ('Nearer than 10 kpc', 'Giant class'), (982, 676)):
        ax.plot([0, 1], [0, 1], color='.5', lw=1, linestyle='--')
        for view, label, color in (
            ('candidate', 'Candidate', '#22587d'), ('qrf', 'QRF', '#287252'),
            ('old_target', 'Old target', '#a85321'), ('nominal_reference', 'Nominal reference', '#615d72'),
        ):
            bins = [r for r in rows if r['view'] == view and r['target'] == target
                    and r['admission_scope'] == 'matched_candidate_admission' and r['numerical_only'] == 'False']
            expected = 0 if (view, target) == ('qrf', 'giant') else n
            if len(bins) != 10 or sum(int(r['n']) for r in bins) != expected:
                raise ValueError(f'Unexpected reliability denominator: {view}, {target}')
            supported = [r for r in bins if int(r['n'])]
            if not supported:
                continue
            x, y, count = ([float(r[key]) for r in supported] for key in ('mean_probability', 'mean', 'n'))
            intervals = [r for r in supported if r['lo95'] and r['hi95']]
            ax.vlines([float(r['mean_probability']) for r in intervals],
                       [float(r['lo95']) for r in intervals], [float(r['hi95']) for r in intervals],
                       color=color, lw=1, alpha=.6)
            ax.scatter(x, y, s=[10 + 4 * math.sqrt(k) for k in count], color=color, label=label, alpha=.75)
        ax.set(title=f'{title} ({n} references)', xlabel='Mean predicted probability',
               xlim=(-.035, 1.035), ylim=(-.035, 1.035))
        ax.grid(alpha=.16)
    axes[0].set_ylabel('Observed reference fraction')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .025), ncol=4,
               frameon=False, markerscale=.75)
    fig.suptitle('DR20 near-distance and giant-class reliability', fontweight='bold')
    fig.subplots_adjust(left=.09, right=.98, top=.83, bottom=.22, wspace=.19)
    files = [out / f'dr20_reliability.{ext}' for ext in ('png', 'pdf')]
    for path in files:
        fig.savefig(path, dpi=300)
    plt.close(fig)
    return files


def main():
    development, development_sources = read_verified('branch_development', 'paired.csv')
    dr20, dr20_sources = read_verified('final_validation_dr20/scores', 'paired.csv')
    reliability, (_, reliability_source) = read_verified('final_validation_dr20/scores', 'reliability.csv')
    hst, hst_sources = read_verified('final_validation_common', 'scale_agreement.csv')
    blue, orange, green, gray = '#22587d', '#a85321', '#287252', '#615d72'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False,
                         'axes.spines.right': False, 'font.size': 10})
    fig, (dev_ax, dr20_ax, hst_ax) = plt.subplots(1, 3, figsize=(13, 4.2))
    signed_score_panel(
        dev_ax, 'Opened development (166 pairs)', development, 166,
        (('candidate_free', 'Candidate/free', blue), ('old_target/old_target', 'Old target', orange),
         ('reference/nominal', 'Nominal reference', gray)),
        'baseline', role='new_sky_field', scope='candidate_admitted', label='-1', metric='gaia_score',
    )
    signed_score_panel(
        dr20_ax, 'DR20 field (1,099 pairs)', dr20, 1099,
        (('qrf', 'QRF', green), ('old_target', 'Old target', orange),
         ('nominal_reference', 'Nominal reference', gray)),
        'anchor', admission_scope='matched_candidate_admission', group='all',
        metric='gaia_log_score', numerical_only='False',
    )
    hst_ax.axhline(0, color='.45', lw=1, linestyle='--')
    for offset, (view, label, color) in enumerate(
        (('candidate', 'Candidate', blue), ('qrf', 'QRF', green),
         ('old_target', 'Old target', orange), ('nominal_reference', 'Nominal reference', gray))
    ):
        for i, (system, n) in enumerate((('NGC_104_47Tuc', 77), ('NGC_362', 26))):
            center, lower, upper = select(
                hst, n, 'median', system=system, view=view, candidate_minus_view='False',
                scope='native_photometric', numerical_only='False', metric='median_relative_residual',
            )
            x = i + .14 * (offset - 1.5)
            hst_ax.vlines(x, 100 * lower, 100 * upper, color=color, lw=2)
            hst_ax.scatter(x, 100 * center, color=color, s=28, zorder=3, label=label if i == 0 else None)
    hst_ax.set(title='HST common distances', ylabel='Median distance residual (%)', xlim=(-.4, 1.4))
    hst_ax.set_xticks([0, 1], ['47 Tuc (77)', 'NGC 362 (26)'])
    hst_ax.grid(axis='y', alpha=.17)
    hst_ax.legend(loc='lower center', bbox_to_anchor=(.5, -.34), ncol=2, frameon=False, fontsize=8)
    fig.suptitle('Photdis v3.2 signed Gaia scores and common distance scales', fontweight='bold')
    fig.subplots_adjust(left=.116, right=.985, top=.81, bottom=.24, wspace=.78)

    out = ROOT / 'figures/v3_2/external_validation'
    out.mkdir(parents=True, exist_ok=True)
    files = [out / f'gaia_and_hst.{ext}' for ext in ('png', 'pdf')]
    for path in files:
        fig.savefig(path, dpi=300)
    plt.close(fig)
    sources = (*development_sources, *dr20_sources, *hst_sources, Path(__file__).resolve(),
               reliability_source, *files, *reliability_plate(reliability, out))
    inventory = MANIFEST / 'final_validation_dr20/figure.json'
    inventory.write_text(json.dumps({str(p.relative_to(ROOT)): digest(p) for p in sources}, indent=2) + '\n')


if __name__ == '__main__':
    main()
