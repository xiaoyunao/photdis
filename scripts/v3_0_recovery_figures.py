#!/usr/bin/env python3
"""Conditional continuous-recovery atlas from verified small result tables."""
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from v2_1_near import ROOT, sha, csv_rows, save_json
from photdis_figure_style import STYLE, BLUE, GREEN, ORANGE, GRAY

SOURCE = ROOT/'manifests/v3_0/recovery_final'
OUT = ROOT/'figures/v3_0/recovery'
TABLE = ROOT/'manifests/v3_0/figures/recovery'


def read(name):
    return list(csv.DictReader((SOURCE/(name+'.csv')).open()))


def save(fig, name, rows, source, caption):
    OUT.mkdir(parents=True, exist_ok=True)
    TABLE.mkdir(parents=True, exist_ok=True)
    assert not any(any('\u4e00' <= c <= '\u9fff' for c in x.get_text()) for x in fig.findobj(matplotlib.text.Text))
    fig.savefig(OUT/(name+'.png'), dpi=300)
    fig.savefig(OUT/(name+'.pdf'))
    plt.close(fig)
    csv_rows(TABLE/(name+'.csv'), rows)
    save_json(TABLE/(name+'.json'), dict(caption=caption, source_sha256=sha(SOURCE/(source+'.csv')),
        png_sha256=sha(OUT/(name+'.png')), pdf_sha256=sha(OUT/(name+'.pdf')), csv_sha256=sha(TABLE/(name+'.csv')),
        english_text_verified=True, png_dpi=300, scope='Continuous simulated observed catalogs; conditional on nominal fitted SED, not final response-marginal recovery.'))


def main():
    plt.rcParams.update(STYLE)
    record = json.loads((SOURCE/'complete.json').read_text())
    for name, digest in record['outputs'].items():
        assert sha(SOURCE/name) == digest
    summary = read('summary')
    scenarios = sorted({r['scenario'] for r in summary})
    labels = [s.replace('_', ' ').capitalize().replace('wise','WISE').replace('4kpc','4 kpc') for s in scenarios]
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    methods = [('qrf_point', 'QRF point', GRAY), ('qrf_pdf', 'QRF PDF', BLUE),
               ('v3_reference', 'v3 reference prior', GREEN), ('v3_target', 'v3 target prior', ORANGE)]
    for ax, view, title in zip(axes, ('grz', 'multiband'), ('Optical grz', 'Available multiband')):
        for j, (method, label, color) in enumerate(methods):
            values = [100*float(next(r for r in summary if r['view'] == view and r['scenario'] == s and r['method'] == method)['mean_distance_count_L1']) for s in scenarios]
            ax.plot(values, np.arange(len(scenarios))+(j-1.5)*.13, 'o', ms=5, color=color, label=label)
        ax.set(title=title, yticks=range(len(labels)), yticklabels=labels, xlabel='Mean distance-count L1 error (%)')
        ax.invert_yaxis()
        ax.tick_params(axis='y', labelsize=11)
        ax.grid(axis='x', alpha=.15)
        ax.legend(fontsize=10, frameon=False)
    fig.suptitle('Target population inference: gains and transfer failures', fontsize=18)
    fig.subplots_adjust(left=.18, right=.98, top=.9, bottom=.09, wspace=.68)
    save(fig, 'distance_count_recovery', summary, 'summary', 'Eight 128-object realizations per scenario. L1 is the sum of absolute distance-bin fraction errors, not per-star distance precision. All 224 conditional numerical cases passed after explicit refinements.')
    coverage = read('coverage')
    selected = [r for r in coverage if r['axis'] == 'distance']
    fig, axes = plt.subplots(1, 2, figsize=(13, 8))
    for ax, view, title in zip(axes, ('grz', 'multiband'), ('Optical grz', 'Available multiband')):
        matrix = np.array([[float(next(r for r in selected if r['view'] == view and r['scenario'] == s and int(r['bin']) == b)['coverage95']) for b in range(6)] for s in scenarios])
        im = ax.imshow(100*matrix, vmin=0, vmax=100, cmap='viridis', aspect='auto')
        for i in range(len(scenarios)):
            for b in range(6):
                ax.text(b, i, f'{100*matrix[i,b]:.0f}', ha='center', va='center', color='white' if matrix[i,b] < .55 else 'black', fontsize=9)
        ax.set(title=title, xticks=range(6), xticklabels=['0–2', '2–5', '5–10', '10–20', '20–50', '>50'], yticks=range(len(labels)), yticklabels=labels, xlabel='Distance bin (kpc)')
        ax.tick_params(axis='both', labelsize=10)
    fig.colorbar(im, ax=axes, fraction=.025, pad=.03, label='Empirical coverage (%)')
    fig.suptitle('Observed-count intervals: nominal 95%, eight catalogs per cell', fontsize=17)
    # Explicit positions avoid colorbar overlap with long scenario labels.
    axes[0].set_position([.17, .1, .25, .79]); axes[1].set_position([.60, .1, .25, .79])
    fig.axes[-1].set_position([.88, .1, .015, .79])
    save(fig, 'distance_count_coverage', selected, 'coverage', 'Realized latent counts with shared population uncertainty, conditional on nominal response. Coverage is discrete in steps of 12.5%; repeated realizations share a template library. Zero-true-count cells are retained.')
    chosen = [r for r in coverage if r['axis'] == 'distance' and int(r['bin']) == 1]
    fig, ax = plt.subplots(figsize=(11, 8))
    for j, (view, color, label) in enumerate([('grz', BLUE, 'grz'), ('multiband', ORANGE, 'Multiband')]):
        r = [next(x for x in chosen if x['view'] == view and x['scenario'] == s) for s in scenarios]
        ax.plot([float(x['mean_bias_count']) for x in r], np.arange(len(r))+(j-.5)*.2, 'o', color=color, label=label)
    ax.axvline(0, color='black', lw=.8, ls='--')
    ax.set(yticks=range(len(labels)), yticklabels=labels, xlabel='Mean inferred minus realized count (objects)', title='The 2–5 kpc bin remains a distinct recovery test')
    ax.invert_yaxis(); ax.legend(frameon=False); ax.tick_params(axis='y', labelsize=12)
    fig.subplots_adjust(left=.28, right=.96, bottom=.1, top=.9)
    save(fig, 'middle_distance_bias', chosen, 'coverage', 'Signed count bias for the predeclared 2–5 kpc failure stratum; no scenarios removed after inspecting results.')
    save_json(TABLE/'complete.json', dict(figures=3, inputs_sha256=sha(SOURCE/'complete.json'), files={p.name: sha(p) for p in TABLE.glob('*.json') if p.name not in ('complete.json', 'validation.json')}))


if __name__ == '__main__':
    main()
