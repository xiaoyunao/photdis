#!/usr/bin/env python3
"""English paired-diagnostic plates from already computed source-backed tables."""
import csv
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from v2_1_near import ROOT, sha, save_json, csv_rows
from photdis.stage1 import require_compute
from photdis_figure_style import STYLE, BLUE, ORANGE, GRAY


def main():
    require_compute()
    source = ROOT/'data/v3_1/target_prior/diagnostic/paired_report'
    done = json.loads((source/'complete.json').read_text())
    assert done['passed'] and sha(source/'all.csv') == done['outputs']['all.csv']
    with (source/'all.csv').open() as stream: rows = list(csv.DictReader(stream))
    out = ROOT/'data/v3_1/paired_figures'; out.mkdir(exist_ok=True)
    plt.rcParams.update(STYLE)
    groups = [('all', 'All', GRAY), ('0', 'Dwarf', BLUE), ('1', 'Evolved', ORANGE)]
    plates = [
        ('target_prior_transfer', [('response12_target', 'response12', 'Target - reference prior'),
            ('response12_target', 'nominal', 'Target - nominal model')], 'Shared target-prior transfer on seen development stars'),
        ('class_metallicity_contrasts', [('class_feh', 'class', 'Class + FeH - class'),
            ('class_feh', 'feh', 'Class + FeH - FeH')], 'Class and metallicity effects on identical eligible stars'),
    ]
    panels = [('primary', 'absolute_mu_error_change', 1., 'Change in absolute distance-modulus error (mag)', 'Positive = worse'),
        ('primary', 'reference_logscore_gain', 1., 'Change in source-error-convolved log score', 'Positive = better'),
        ('signed_gaia', 'signed_gaia_PIT_tail_change', 100., 'Change in signed-Gaia PIT tails (percentage points)', 'Positive = worse'),
        ('primary', 'reference_point_agreement68_change', 100., 'Change in 68% reference-point agreement (percentage points)', 'Positive = better')]
    outputs = {}; counts = {}
    for stem, contrasts, title in plates:
        fig, axes = plt.subplots(2, 2, figsize=(18, 10))
        fig.subplots_adjust(left=.19, right=.98, bottom=.12, top=.86, hspace=.60, wspace=.83)
        selected = []
        for panel, (ax, spec) in enumerate(zip(axes.flat, panels)):
            ref, metric, scale, xlabel, direction = spec
            labels = []
            for number, (control, baseline, label) in enumerate(contrasts):
                for j, (group, name, color) in enumerate(groups):
                    match = [r for r in rows if r['control'] == control and r['baseline'] == baseline and r['reference_class'] == group
                        and r['source'] == ref and r['metric'] == metric]
                    assert len(match) == 1
                    r = match[0]; y = 3*number+j
                    lo, mean, hi = [scale*float(r[k]) for k in ['lo95', 'mean', 'hi95']]
                    ax.plot([lo, hi], [y, y], color=color, lw=2.4)
                    ax.scatter([mean], [y], color=color, s=38, zorder=3)
                    labels.append(f"{label}\n{name}: N={int(r['n']):,}, sky blocks={r['cells']}")
                    selected.append(dict(plate=stem, panel=panel, **r))
            ax.set_yticks(range(6), labels, fontsize=9)
            ax.set_ylim(5.65, -.65); ax.axvline(0, color='black', lw=.8, ls='--')
            ax.axhline(2.5, color='.85', lw=.8)
            ax.set_xlabel(xlabel, fontsize=10); ax.set_title(f'{chr(65+panel)} | {direction}', loc='left', fontsize=12)
            ax.grid(axis='x', color='.92', linewidth=.6); ax.set_axisbelow(True)
        fig.suptitle(title, fontsize=19, y=.97)
        fig.text(.5, .915, 'Native DR11 PSF only | original design weights | 95% sky-block bootstrap intervals', ha='center', fontsize=12)
        footer = 'Shared target prior: 1,542 new-cell transfers; one original numerical flag retained.' if stem.startswith('target') else 'Truth-assisted class/FeH conditions are diagnostic controls; they are not production inputs.'
        fig.text(.5, .058, footer, ha='center', fontsize=11)
        fig.text(.5, .03, 'Seen development evidence. Reference-point agreement is not true coverage; source teachers can share systematics.', ha='center', fontsize=11)
        texts = [x.get_text() for x in fig.findobj(matplotlib.text.Text)]
        assert not any(any('\u4e00' <= c <= '\u9fff' for c in text) for text in texts)
        for extension in ['png', 'pdf']: fig.savefig(out/(stem+'.'+extension), dpi=300)
        plt.close(fig); csv_rows(out/(stem+'.csv'), selected)
        for extension in ['png', 'pdf', 'csv']: outputs[stem+'.'+extension] = sha(out/(stem+'.'+extension))
        counts[stem] = len(selected)
    save_json(out/'complete.json', dict(inputs={str(p.relative_to(ROOT)): sha(p) for p in [source/'complete.json', source/'all.csv',
        ROOT/'scripts/v3_1_paired_figures.py', ROOT/'scripts/photdis_figure_style.py']}, outputs=outputs,
        job=os.getenv('SLURM_JOB_ID'), plates=2, plotted_rows=counts, png_dpi=300, english_text_verified=True,
        visual_review_pending=True, scope='Two seen-development paired diagnostic plates. All plotted rows preserve exact source denominators and original numerical flags; no new scientific sample or outcome.'))
    print('PAIRED FIGURES COMPLETE', counts, flush=True)


if __name__ == '__main__': main()
