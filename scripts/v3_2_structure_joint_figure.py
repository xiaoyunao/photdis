#!/usr/bin/env python3
"""Plot the paired membership and covariance from the final NGC288 grid."""
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from v3_2_structure_joint_audit import BASE, ROOT, audit, digest


def render(membership, covariance, summary, *, subtitle):
    fig, (member_ax, variance_ax) = plt.subplots(1, 2, figsize=(9.5, 4.1))
    full, gaussian = ([float(row[key]) for row in membership] for key in ('full', 'gaussian'))
    member_ax.plot([0, 1], [0, 1], '--', color='.5', lw=1)
    member_ax.scatter(gaussian, full, s=12, alpha=.45, color='#255c7c', linewidths=0)
    member_ax.set(title=f'Paired membership ({len(membership)} PSFs)',
                  xlabel='Gaussian summary probability', ylabel='Full likelihood probability',
                  xlim=(-.025, 1.025), ylim=(-.025, 1.025), aspect='equal')
    member_ax.grid(alpha=.17)

    final = {row['branch']: row for row in summary if row['variant'] == 'm256_z1024'}
    variances = {(r['branch'], r['component']): float(r['value'])
                 for r in covariance if r['row'] == r['column'] == 'member'}
    branches = ('full', 'gaussian')
    conditional = [variances[b, 'conditional'] for b in branches]
    shared = [variances[b, 'shared'] for b in branches]
    variance_ax.bar(branches, conditional, width=.52, color='#467393', label='Conditional variance')
    variance_ax.bar(branches, shared, width=.52, bottom=conditional, color='#b85d32',
                    label='Shared-state variance')
    maximum = max(1., *(variances[b, 'total'] for b in branches))
    for x, branch in enumerate(branches):
        mean = float(final[branch]['mean_members'])
        variance_ax.text(x, variances[branch, 'total'] + .035*maximum, f'Mean {mean:.2f}',
                         ha='center', va='bottom')
    variance_ax.set_xticks(range(2), ['Full likelihood', 'Gaussian summary'])
    variance_ax.set(title='Member-count variance', ylabel='Variance (objects squared)', ylim=(0, maximum*1.20))
    variance_ax.grid(axis='y', alpha=.17)
    variance_ax.set_axisbelow(True)
    variance_ax.legend(loc='upper center', bbox_to_anchor=(.5, -.15), ncol=2, frameon=False, fontsize=9)
    fig.suptitle('NGC288 likelihood comparison on the final quadrature grid', fontweight='bold')
    fig.text(.5, .875, subtitle, ha='center', fontsize=9)
    fig.subplots_adjust(left=.09, right=.98, bottom=.22, top=.77, wspace=.38)
    return fig


def main():
    verdict = audit()
    tables = [BASE/name for name in ('membership.csv', 'count_covariance.csv', 'summary.csv')]
    records = []
    for path in tables:
        with path.open(newline='') as stream:
            records.append(list(csv.DictReader(stream)))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False, 'axes.spines.right': False})
    subtitle = 'Registered quadrature checks passed' if verdict['passed'] else 'Registered quadrature failures retained'
    fig = render(*records, subtitle=subtitle)
    out = ROOT/'figures/v3_2/structure_joint'
    out.mkdir(parents=True, exist_ok=True)
    files = [out/f'likelihood_contrast.{ext}' for ext in ('png', 'pdf')]
    for path in files:
        fig.savefig(path, dpi=300)
    plt.close(fig)
    sources = (BASE/'complete.json', *tables, Path(__file__).resolve(), *files)
    (BASE/'figure.json').write_text(json.dumps(dict(
        quadrature_passed=verdict['passed'],
        sources={str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in sources}),
        indent=2)+'\n')


if __name__ == '__main__':
    main()
