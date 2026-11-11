#!/usr/bin/env python3
"""English scientific development atlas; display cuts never change metrics."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from scipy.stats import binned_statistic_2d
from v2_1_near import ROOT, load, sha, save_json, csv_rows
from photdis.stage1 import require_compute
from photdis.v3.controls import optical_magnitudes
from photdis_figure_style import STYLE, BLUE, GREEN, ORANGE, GRAY

D = ROOT/'data/v3_0'; OUT = ROOT/'figures/v3_0/development'; TABLE = ROOT/'manifests/v3_0/figures/development'
MODELS = [('qrf', 'Matched QRF grz', GRAY), ('grz_fit', 'v3 grz', BLUE),
          ('griz_fit', 'v3 griz', GREEN), ('multiband_fit', 'v3 multiband', ORANGE)]
REFERENCES = [('primary', 'Primary reference'), ('gaia_precise', 'Gaia (S/N > 10)'), ('specdis', 'SpecDis'), ('spdist', 'SPDist')]
GROUPS = [('all', 'All reference objects', None), ('dwarf', 'Conservative dwarfs', 0),
          ('evolved', 'Evolved-star reference', 1), ('other', 'Other spectral stars', 2), ('unknown', 'No reliable spectral class', -1)]


def clip_pair(x, y):
    good = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0); keep = good.copy(); record = dict(valid=int(good.sum()))
    for label, values in [('x', x), ('y', y)]:
        mean = values[good].mean() if good.any() else np.nan
        sigma = values[good].std() if good.any() else np.nan
        keep &= (values >= mean-3*sigma) & (values <= mean+3*sigma)
        record[label+'_mean_kpc'] = float(mean); record[label+'_sigma_kpc'] = float(sigma)
    record.update(kept=int(keep.sum()), clipped=int(good.sum()-keep.sum()))
    return keep, record


def save(fig, name, rows, metadata):
    OUT.mkdir(parents=True, exist_ok=True); TABLE.mkdir(parents=True, exist_ok=True)
    # Inspect every textual artist before output; science figures are English only.
    text = [x.get_text() for x in fig.findobj(matplotlib.text.Text)]
    assert not any(any('\u4e00' <= c <= '\u9fff' for c in s) for s in text)
    fig.savefig(OUT/(name+'.png'), dpi=300); fig.savefig(OUT/(name+'.pdf'))
    plt.close(fig); csv_rows(TABLE/(name+'.csv'), rows)
    metadata.update(png_sha256=sha(OUT/(name+'.png')), pdf_sha256=sha(OUT/(name+'.pdf')),
        csv_sha256=sha(TABLE/(name+'.csv')), png_dpi=300, english_text_verified=True,
        stage='Seen development, reference-prior prediction; not final independent acceptance', job=os.getenv('SLURM_JOB_ID'))
    save_json(TABLE/(name+'.json'), metadata)
    print('FIGURE', name, flush=True)


def distances(t, model, label):
    for group, group_name, cls in GROUPS:
        population = np.ones(len(t['identity']), bool) if cls is None else t['class_label'] == cls
        fig = plt.figure(figsize=(20, 8.8))
        outer = fig.add_gridspec(2, 4, height_ratios=[5, 1.65], hspace=.38, wspace=.39,
                                left=.055, right=.97, top=.86, bottom=.085)
        rows = []; selections = {}
        for col, (reference, title) in enumerate(REFERENCES):
            upper = outer[0, col].subgridspec(2, 2, height_ratios=[4, 1], width_ratios=[1, .05], hspace=.05, wspace=.06)
            main = fig.add_subplot(upper[0, 0]); residual = fig.add_subplot(upper[1, 0], sharex=main)
            color = fig.add_subplot(upper[:, 1]); histogram = fig.add_subplot(outer[1, col])
            use = population & t[reference+'_valid']
            raw_x = 10**((t[reference+'_mu'][use]-10)/5); raw_y = 10**((t['mu'][use, 2]-10)/5)
            keep, audit = clip_pair(raw_x, raw_y); selections[reference] = audit
            x, y = raw_x[keep], raw_y[keep]; pct = 100*(y/x-1)
            main.set_title(title, fontsize=15); main.set_ylabel('Photometric distance (kpc)', fontsize=13)
            residual.set(xlabel='Reference distance (kpc)', ylabel='Residual (%)', ylim=(-100, 100), yticks=[-100, 0, 100])
            residual.xaxis.label.set_size(13); residual.yaxis.label.set_size(12)
            histogram.set(xlabel='Relative distance residual (%)', ylabel='Objects', xlim=(-100, 100))
            histogram.xaxis.label.set_size(12); histogram.yaxis.label.set_size(12)
            plt.setp(main.get_xticklabels(), visible=False)
            if not len(x):
                main.text(.5, .5, 'No qualified reference', ha='center', transform=main.transAxes)
                color.set_axis_off(); continue
            bound = max(x.max(), y.max())*1.02
            edges = np.linspace(0, bound, 46); redges = np.linspace(-100, 100, 41)
            h, _, _ = np.histogram2d(x, y, bins=(edges, edges))
            hr, _, _ = np.histogram2d(x, pct, bins=(edges, redges))
            norm = LogNorm(1, max(h.max(), hr.max(), 2.))
            im = main.pcolormesh(edges, edges, np.ma.masked_equal(h.T, 0), norm=norm, cmap='viridis', rasterized=True)
            residual.pcolormesh(edges, redges, np.ma.masked_equal(hr.T, 0), norm=norm, cmap='viridis', rasterized=True)
            fig.colorbar(im, cax=color, label='Objects per bin')
            main.plot([0, bound], [0, bound], '--', color='black', lw=1)
            main.set(xlim=(0, bound), ylim=(0, bound)); residual.axhline(0, color='black', lw=1, ls='--')
            main.text(.03, .97, f'N = {len(x):,} / {len(raw_x):,}', va='top', transform=main.transAxes, fontsize=11)
            cut = np.linspace(0, max(x.max(), 1e-6), 9)
            for lo, hi in zip(cut[:-1], cut[1:]):
                ix = (x >= lo) & (x < hi)
                if ix.sum() >= 10:
                    q = np.percentile(pct[ix], [16, 50, 84]); center = np.median(x[ix])
                    residual.errorbar(center, q[1], yerr=[[q[1]-q[0]], [q[2]-q[1]]], color=ORANGE, fmt='o', ms=3, lw=1, capsize=2)
            counts, bins, _ = histogram.hist(pct, bins=np.linspace(-100, 100, 41), histtype='stepfilled', alpha=.35, color=BLUE)
            histogram.axvline(0, color='black', lw=1, ls='--')
            histogram.text(.03, .94, f'Outside view: {(abs(pct) > 100).sum():,}', transform=histogram.transAxes, va='top', fontsize=10)
            for panel, matrix, xe, ye in [('main', h, edges, edges), ('residual', hr, edges, redges)]:
                for i, j in np.argwhere(matrix > 0):
                    rows.append(dict(reference=reference, panel=panel, xlo=xe[i], xhi=xe[i+1], ylo=ye[j], yhi=ye[j+1], count=int(matrix[i, j])))
            for i, count in enumerate(counts): rows.append(dict(reference=reference, panel='histogram', xlo=bins[i], xhi=bins[i+1], ylo=0., yhi=0., count=int(count)))
        fig.suptitle(f'{label} | {group_name}', fontsize=20, y=.97)
        fig.text(.5, .905, 'Display only: independent x/y 3-sigma cut per reference; attached residual and histogram share that selection', ha='center', fontsize=12)
        save(fig, f'distance_{model}_{group}', rows, dict(model=model, population=group, selections=selections,
            residual_definition='100*(d_phot/d_reference - 1)', display_cut='single simultaneous mean +/-3 population sigma in linear kpc, each population/model/reference independently',
            metrics='Scientific tables remain un-clipped; primary references follow availability hierarchy'))


def conditions(t):
    valid = t['primary_valid']; residual = 100*np.expm1(np.log(10)*(t['mu'][:, 2]-t['primary_mu'])/5)
    # Reference absolute magnitude is a diagnostic axis only, never model input.
    mr = t['r0']-t['primary_mu']
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True); rows = []
    panels = [('color_color', t['gr'], t['rz'], '$g_0-r_0$', '$r_0-z_0$', [-.4, 2.3, -.5, 2.]),
              ('reference_CMD', t['gr'], mr, '$g_0-r_0$', 'Reference-based $M_r$ (mag)', [-.4, 2.3, -4, 14]),
              ('metallicity', t['corrected_feh'], residual, 'Corrected reference [Fe/H]', 'Relative distance residual (%)', [-3, .6, -100, 150])]
    for ax, (name, x, y, xlabel, ylabel, bounds) in zip(axes, panels):
        use = valid & np.isfinite(x) & np.isfinite(y)
        if name == 'metallicity': use &= t['corrected_feh_valid']
        if name == 'metallicity':
            h = ax.hist2d(x[use], y[use], bins=55, range=[bounds[:2], bounds[2:]], norm=LogNorm(), cmap='viridis')
            fig.colorbar(h[3], ax=ax, label='Objects per bin'); ax.axhline(0, color='black', ls='--', lw=1)
            stat, xe, ye = h[:3]
        else:
            xe = np.linspace(*bounds[:2], 41); ye = np.linspace(*bounds[2:], 41)
            stat = binned_statistic_2d(x[use], y[use], residual[use], statistic='median', bins=(xe, ye))[0]
            counts = np.histogram2d(x[use], y[use], bins=(xe, ye))[0]; stat[counts < 5] = np.nan
            im = ax.pcolormesh(xe, ye, stat.T, cmap='coolwarm', norm=TwoSlopeNorm(0, -60, 60), rasterized=True)
            fig.colorbar(im, ax=ax, label='Median relative residual (%)')
            if name == 'reference_CMD': ax.invert_yaxis()
        ax.set(xlabel=xlabel, ylabel=ylabel, title=name.replace('_', ' '))
        for i, j in np.argwhere(np.isfinite(stat)):
            rows.append(dict(panel=name, xlo=xe[i], xhi=xe[i+1], ylo=ye[j], yhi=ye[j+1], value=stat[i, j]))
    save(fig, 'population_conditions_multiband', rows, dict(model='multiband', source='primary',
        caveat='Seen reference selection; CMD uses reference distance only after inference. FeH is an evaluation coordinate, not an application feature.'))


def main():
    require_compute(); plt.rcParams.update(STYLE); inputs = {}
    for name, label, _ in MODELS:
        folder = D/'evaluation_results'/f'development_{name}_0'; path = folder/'per_star.npz'
        complete = json.loads((folder/'complete.json').read_text()); assert sha(path) == complete['outputs'][path.name]
        inputs[name] = sha(path); t = load(path); assert (t['TYPE'] == 'PSF').all()
        distances(t, name, label)
        if name == 'multiband_fit':
            source_path = D/'cohorts/development.npz'; source = load(source_path)
            np.testing.assert_array_equal(source['identity'][t['rows']], t['identity'])
            t['r0'] = optical_magnitudes({k: v[t['rows']] for k, v in source.items()})[0][:, 1]
            inputs['photometric_coordinates'] = sha(source_path)
            conditions(t)
    save_json(TABLE/'complete.json', dict(inputs=inputs, figures=21, job=os.getenv('SLURM_JOB_ID')))


if __name__ == '__main__': main()
