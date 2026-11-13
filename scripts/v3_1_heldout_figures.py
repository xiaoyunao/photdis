#!/usr/bin/env python3
"""Plot authenticated complete-state sky transfer without implying distance truth."""
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from v2_1_near import ROOT, sha, save_json, csv_rows
from photdis_figure_style import STYLE, BLUE, ORANGE, GREEN
from photdis.v3.compression import DISTANCE_MU_BOUNDARIES


def main():
    source = ROOT / 'manifests/v3_1/heldout_transfer/report'
    marker = source / 'complete.json'
    done = json.loads(marker.read_text())
    assert done['states'] == 102 and done['all_PSF'] and not done['references_read']
    inputs = {str(marker.relative_to(ROOT)): sha(marker)}
    tables = {}
    for name in ['inventory.csv', 'predictive_summary.csv', 'counts.csv',
                 'field_summary.csv', 'checks.csv', 'paired.csv']:
        path = source / name
        assert sha(path) == done['outputs'][name]
        inputs[str(path.relative_to(ROOT))] = sha(path)
        with path.open() as stream:
            tables[name] = list(csv.DictReader(stream))
    inventory = tables['inventory.csv']
    assert len(inventory) == 102
    assert {(r['case'], int(r['correction'])) for r in inventory} == {
        (case, i) for case in ['broad', 'candidate', 'holdout'] for i in range(34)}
    view = 'bootstrap_evidence'
    cases = [('broad', 'Broad fit (in sample)', BLUE),
             ('candidate', 'Candidate fit (in sample)', ORANGE),
             ('holdout', 'Held-out fit', GREEN)]
    envelopes = [('broad', 'Broad selection'), ('candidate', 'Candidate selection')]
    distance_edges = 10 ** ((DISTANCE_MU_BOUNDARIES - 10) / 5)
    np.testing.assert_allclose(distance_edges, [2, 5, 10, 20, 50], rtol=1e-12)
    distance_labels = [f'{lo:g}–{hi:g}' for lo, hi in zip(np.r_[0, distance_edges[:-1]], distance_edges)]
    distance_labels.append(f'>={distance_edges[-1]:g}')
    status = {}
    for case, _, _ in cases:
        rows = [r for r in done['views'] if r['case'] == case and r['view'] == view]
        assert rows
        for envelope, _ in envelopes:
            matches = [r for r in rows if r.get('envelope') == envelope]
            if not matches:
                assert len(rows) == 1 and not rows[0]['available']
                matches = rows
            assert len(matches) == 1
            status[case, envelope] = matches[0]
    out = ROOT / 'figures/v3_1/heldout_transfer'
    meta = ROOT / 'manifests/v3_1/heldout_transfer_figures'
    out.mkdir(parents=True, exist_ok=True)
    meta.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(STYLE)
    outputs, plotted = {}, []

    def select(name, case, envelope):
        rows = [r for r in tables[name] if r['case'] == case and
                r['view'] == view and r['envelope'] == envelope and r['variant'] == 'base']
        if not status[case, envelope]['available']:
            assert not rows
        return rows

    def finish(fig, name):
        fig.canvas.draw()
        for label in fig.findobj(matplotlib.text.Text):
            assert not any('\u4e00' <= c <= '\u9fff' for c in label.get_text())
        for ext in ['png', 'pdf']:
            path = out / f'{name}.{ext}'
            fig.savefig(path, dpi=300)
            outputs[str(path.relative_to(ROOT))] = sha(path)
        plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(21, 13))
    fig.subplots_adjust(left=.07, right=.97, bottom=.20, top=.79, wspace=.35, hspace=.55)
    for ei, (envelope, envelope_label) in enumerate(envelopes):
        absent = []
        for ci, (case, label, color) in enumerate(cases):
            rows = select('counts.csv', case, envelope)
            if not rows:
                absent.append(label)
                continue
            assert len(rows) == 18
            assert {(int(r['class_index']), int(r['distance_bin'])) for r in rows} == {
                (c, d) for c in range(3) for d in range(6)}
            total = {int(r['n']) for r in rows}
            assert len(total) == 1 and abs(sum(float(r['mean']) for r in rows) - next(iter(total))) < 1e-7
            for c in range(3):
                ax = axes[ei, c]
                subset = sorted([r for r in rows if int(r['class_index']) == c],
                                key=lambda r: int(r['distance_bin']))
                for r in subset:
                    x = int(r['distance_bin']) + (ci - 1) * .22
                    ax.plot([x, x], [float(r['lower95']), float(r['upper95'])], color=color, lw=1.8)
                    ax.scatter(x, float(r['mean']), color=color, marker=['o', 's', '^'][ci], s=22,
                               label=label if int(r['distance_bin']) == 0 else None)
                    plotted.append(dict(figure='heldout_joint_counts', panel=f'{ei}:{c}',
                                        source='counts.csv', **r))
        for c, class_label in enumerate(['Dwarf', 'Evolved', 'Other / unknown']):
            ax = axes[ei, c]
            ax.set(title=f'{envelope_label} | {class_label}', ylabel='Objects in the selected sample', ylim=(0, None))
            ax.set_xticks(range(6), distance_labels, rotation=25, ha='right')
            ax.set_xlabel('Distance interval (kpc)')
            if c == 0 and ax.get_legend_handles_labels()[0]:
                ax.legend(frameon=False, fontsize=9)
            if absent:
                ax.text(.02, .97, 'Unavailable: ' + ', '.join(absent), transform=ax.transAxes,
                        va='top', fontsize=9, wrap=True)
    fig.suptitle('Fixed sky fields: shared class × distance count transfer', y=.97, fontsize=21)
    fig.text(.5, .92, 'Fields 12 / 4 / 6 | broad n=6000, candidate n=935 | same native-PSF identities | main evidence-weighted view', ha='center', fontsize=12)
    fig.text(.5, .86, 'Points: model expected counts. Lines: shared realized-count 95% intervals. No true-distance counts supplied.', ha='center', fontsize=12)
    fig.text(.5, .11, 'Only the held-out population fit excludes these fields. The broad and candidate fits include their photometry.', ha='center', fontsize=11)
    fig.text(.5, .06, 'Unavailable conditional states are not renormalized. Inherited ensemble failures remain; counts are model-conditional.', ha='center', fontsize=11)
    for ax in axes.flat:
        ax.title.set_fontsize(13)
        ax.xaxis.label.set_fontsize(11)
        ax.yaxis.label.set_fontsize(11)
        ax.tick_params(labelsize=10)
    finish(fig, 'heldout_joint_counts')

    fig, axes = plt.subplots(2, 3, figsize=(21, 13))
    fig.subplots_adjust(left=.075, right=.97, bottom=.23, top=.79, wspace=.38, hspace=.7)
    half_values = [float(r[prefix + '_half_log_density_delta'])
                   for r in tables['predictive_summary.csv']
                   if r['view'] == view and r['variant'] == 'base'
                   for prefix in ['joint', 'generalized_design_joint']]
    half_upper = max(1., 1.2 * max(v for v in half_values if np.isfinite(v)))
    status_lines = []
    for ei, (envelope, envelope_label) in enumerate(envelopes):
        flags = []
        for ci, (case, label, color) in enumerate(cases):
            rows = select('predictive_summary.csv', case, envelope)
            state = status[case, envelope]
            flags.append(f"{case}: available={state['available']}, conditional numeric={state['accepted']}")
            if not rows:
                for ax in axes[ei]:
                    ax.text(ci, .5, 'Unavailable', rotation=90, transform=ax.get_xaxis_transform(), ha='center', va='center', fontsize=9)
                continue
            assert len(rows) == 1
            r = rows[0]
            assert (r['population_fit_excludes_heldout'] == 'True') == (case == 'holdout')
            axes[ei, 0].scatter(ci, float(r['design_mean_pointwise_logscore']), c=color, s=40)
            for prefix, offset, symbol in [('joint', -.1, 'o'), ('generalized_design_joint', .1, 's')]:
                for col, key in [(1, 'importance_ess'), (2, 'half_log_density_delta')]:
                    value = float(r[f'{prefix}_{key}'])
                    if np.isfinite(value):
                        axes[ei, col].scatter(ci + offset, value, c=color, marker=symbol, s=40)
                    else:
                        axes[ei, col].text(ci + offset, .9, 'Nonfinite', rotation=90,
                                          transform=axes[ei, col].get_xaxis_transform(), ha='center', fontsize=9)
            for col in range(3):
                plotted.append(dict(figure='heldout_predictive_support', panel=f'{ei}:{col}',
                                    source='predictive_summary.csv', **r))
        status_lines.append(envelope_label + ' — ' + '; '.join(flags))
        axes[ei, 0].set(title=envelope_label + ' | Pointwise score', ylabel='Design-weighted mean log density')
        axes[ei, 1].set(title=envelope_label + ' | Shared support', ylabel='Joint predictive weight ESS', yscale='log')
        axes[ei, 1].axhline(100, color='.4', ls='--', lw=1)
        axes[ei, 2].set(title=envelope_label + ' | Half-draw sensitivity',
                        ylabel='Absolute joint log-density difference\n(symmetric log scale)', ylim=(0, half_upper))
        axes[ei, 2].set_yscale('symlog', linthresh=.01)
        half_ticks = [v for v in [0, .01, .05, .2, 1, 5, 20] if v <= half_upper]
        axes[ei, 2].set_yticks(half_ticks, [f'{v:g}' for v in half_ticks])
        axes[ei, 2].axhline(.2, color='.4', ls='--', lw=1)
        for ax in axes[ei]:
            ax.set_xticks(range(3), ['Broad\n(in sample)', 'Candidate\n(in sample)', 'Held out'])
            ax.set_xlim(-.5, 2.5)
    for ax in axes.flat:
        ax.title.set_fontsize(13)
        ax.yaxis.label.set_fontsize(11)
        ax.tick_params(labelsize=10)
    fig.suptitle('Photometric predictive support across frozen sky fields', y=.97, fontsize=21)
    fig.text(.5, .92, 'Main evidence-weighted view | broad n=6000, candidate n=935 | matching selection normalization | no distance references read', ha='center', fontsize=12)
    fig.text(.5, .86, 'Circles: unit-object joint prediction. Squares: generalized design-weighted joint prediction (not a physical catalog density).', ha='center', fontsize=11)
    for i, line in enumerate(status_lines):
        fig.text(.5, .15 - .035 * i, line, ha='center', fontsize=10)
    fig.text(.5, .065, 'Support rule: weight ESS ≥ 100 AND half log-density change < 0.2. Weight ESS is not autocorrelation-adjusted.', ha='center', fontsize=11)
    fig.text(.5, .025, 'Photometric support does not prove full convergence, true-distance accuracy, population recovery or independent contamination.', ha='center', fontsize=11)
    finish(fig, 'heldout_predictive_support')
    csv_rows(meta / 'plotted.csv', plotted)
    outputs[str((meta / 'plotted.csv').relative_to(ROOT))] = sha(meta / 'plotted.csv')
    for path in [ROOT / 'scripts/v3_1_heldout_figures.py', ROOT / 'scripts/photdis_figure_style.py',
                 ROOT / 'src/photdis/v3/compression.py',
                 ROOT / 'docs/v3_1_heldout_transfer_protocol.md']:
        inputs[str(path.relative_to(ROOT))] = sha(path)
    save_json(meta / 'complete.json', dict(inputs=inputs, outputs=outputs, figures=2, panels=12,
        plotted_rows=len(plotted), states=102, english_text_verified=True,
        visual_review_pending=True, scientific_acceptance=False,
        scope='Actual complete-state transfer report only. Main view, all cases, missing and failed states retained. No synthetic fallback or true-distance validation claim.'))


if __name__ == '__main__':
    main()
