#!/usr/bin/env python3
"""Render final-product disposition separately from conditional model ambiguity."""
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
from v2_1_near import ROOT, sha, save_json, csv_rows
from photdis_figure_style import STYLE
from v3_1_domain_figures import WISE, R_LABELS, COLOR_LABELS


def main():
    source = ROOT / 'manifests/v3_1/domain_adjudication'
    marker = source / 'complete.json'
    done = json.loads(marker.read_text())
    assert done['all_PSF'] and done['cells'] == 75 and done['candidate_n'] == 3540
    assert done['calibrated_cells'] == done['scientifically_accepted_ambiguous_cells'] == 0
    path = source / 'cells.csv'
    assert sha(path) == done['outputs']['cells.csv']
    inputs = {str(marker.relative_to(ROOT)): sha(marker), str(path.relative_to(ROOT)): sha(path)}
    with path.open() as stream:
        rows = sorted(csv.DictReader(stream), key=lambda r: int(r['cell']))
    assert [int(r['cell']) for r in rows] == list(range(75))
    assert sum(int(r['candidate_n']) for r in rows) == 3540
    out = ROOT / 'figures/v3_1/domain_adjudication'
    meta = ROOT / 'manifests/v3_1/domain_adjudication_figures'
    out.mkdir(parents=True, exist_ok=True)
    meta.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(2, 3, figsize=(17, 11))
    fig.subplots_adjust(left=.07, right=.88, bottom=.20, top=.78, wspace=.25, hspace=.62)
    categorical = ListedColormap(['#dddddd', '#a63603'])
    category_norm = BoundaryNorm([-.5, .5, 1.5], 2)
    continuous = plt.get_cmap('viridis').copy()
    continuous.set_bad('#dddddd')
    fraction_norm = Normalize(0, 1)
    plotted = []
    for w in range(3):
        group = [r for r in rows if int(r['wise_bin']) == w]
        assert len(group) == 25
        category, fraction = np.zeros((5, 5)), np.full((5, 5), np.nan)
        for r in group:
            rb, cb = int(r['r_bin']), int(r['color_bin'])
            assert int(r['cell']) == (rb * 5 + cb) * 3 + w
            n, multi = int(r['candidate_n']), int(r['multimodal_n'])
            assert 0 <= multi <= n
            assert r['scientific_disposition'] == ('failed_main_numerical_gate' if n else 'outside_candidate')
            category[rb, cb] = int(n > 0)
            if n:
                fraction[rb, cb] = multi / n
        top = axes[0, w].imshow(category, origin='lower', cmap=categorical, norm=category_norm, aspect='auto')
        bottom = axes[1, w].imshow(np.ma.masked_invalid(fraction), origin='lower', cmap=continuous, norm=fraction_norm, aspect='auto')
        for r in group:
            rb, cb = int(r['r_bin']), int(r['color_bin'])
            n, multi = int(r['candidate_n']), int(r['multimodal_n'])
            top_text = str(n) if n else '—'
            bottom_text = f'{100 * multi / n:.0f}%' if n else '—'
            axes[0, w].text(cb, rb, top_text, ha='center', va='center', fontsize=11, color='white' if n else 'black')
            value = multi / n if n else None
            rgba = continuous(fraction_norm(value)) if n else (.87, .87, .87, 1)
            luminance = .2126 * rgba[0] + .7152 * rgba[1] + .0722 * rgba[2]
            axes[1, w].text(cb, rb, bottom_text, ha='center', va='center', fontsize=11,
                            color='black' if luminance > .5 else 'white')
            for panel, display, plotted_value in [('disposition', top_text, int(n > 0)),
                                                  ('conditional_multimodality', bottom_text, value)]:
                plotted.append(dict(source='cells.csv', panel=panel, display=display, plotted_value=plotted_value, **r))
        for i in range(2):
            ax = axes[i, w]
            ax.set_title(('Scientific-use status' if i == 0 else 'Conditional multimodality') + ' | ' + WISE[w], fontsize=11)
            ax.set_xticks(range(5), COLOR_LABELS, fontsize=10)
            ax.set_yticks(range(5), R_LABELS, fontsize=10)
            ax.set_xlabel('Extinction-corrected g - r', fontsize=11)
            if w == 0:
                ax.set_ylabel('Observed r magnitude', fontsize=11)
            ax.grid(False)
    bar = fig.colorbar(top, cax=fig.add_axes([.90, .55, .015, .19]), ticks=[0, 1])
    bar.ax.set_yticklabels(['No candidate', 'Failed\nnumerical gate'], fontsize=9)
    bar = fig.colorbar(bottom, cax=fig.add_axes([.90, .23, .015, .19]), ticks=[0, .25, .5, .75, 1])
    bar.set_label('Fraction with multiple modes', fontsize=10)
    fig.suptitle('Final candidate: scientific-use disposition and retained ambiguity', y=.97, fontsize=21)
    fig.text(.5, .92, 'Same 3540 native-PSF candidate identities | 30 occupied cells | no calibrated scientific-use domain established', ha='center', fontsize=12)
    fig.text(.5, .86, 'Top: object counts; every occupied cell inherits the main ensemble evidence failure. Bottom: model-conditional information only.', ha='center', fontsize=11)
    fig.text(.5, .12, 'Multiple modes means at least two distance basins with posterior mass > 0.05. All weaker modes remain in the full product.', ha='center', fontsize=11)
    fig.text(.5, .075, 'Ambiguity does not erase numerical failure or establish physical classes / binaries. Full posteriors remain for conditional diagnostics.', ha='center', fontsize=11)
    fig.text(.5, .03, 'Grey —: no candidate object. Precision/reference gaps, correlated external failures and global support limits remain in separate evidence tables.', ha='center', fontsize=10)
    fig.canvas.draw()
    assert not any('\u4e00' <= c <= '\u9fff' for t in fig.findobj(matplotlib.text.Text) for c in t.get_text())
    outputs = {}
    for ext in ['png', 'pdf']:
        path = out / ('scientific_use_and_ambiguity.' + ext)
        fig.savefig(path, dpi=300)
        outputs[str(path.relative_to(ROOT))] = sha(path)
    plt.close(fig)
    csv_rows(meta / 'plotted.csv', plotted)
    outputs[str((meta / 'plotted.csv').relative_to(ROOT))] = sha(meta / 'plotted.csv')
    for path in [ROOT / 'scripts/v3_1_domain_adjudication_figures.py', ROOT / 'scripts/v3_1_domain_figures.py',
                 ROOT / 'scripts/photdis_figure_style.py']:
        inputs[str(path.relative_to(ROOT))] = sha(path)
    save_json(meta / 'complete.json', dict(inputs=inputs, outputs=outputs, figures=1, panels=6,
        plotted_rows=len(plotted), english_text_verified=True, visual_review_pending=True,
        scientific_domain_adjudication_complete=False,
        scope='Actual complete-evidence synthesis; disposition and model ambiguity remain distinct. Final written review and image inspection pending.'))


if __name__ == '__main__':
    main()
