#!/usr/bin/env python3
"""Rebin immutable v2.0 predictions and deliver the full six-band atlas."""
import argparse
import itertools
import json
import os
from pathlib import Path
import shutil
import concurrent.futures
import multiprocessing

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator, MaxNLocator

import v2_0_figures as f
from v2_0_common import ROOT, OUT, DEST, load, subset, sha, save_json, csv_rows
from photdis.stage1 import require_compute
from photdis_figure_style import TRANSPARENT_STYLE, BLUE, ORANGE, GRAY

REV = ROOT / 'data/v2_0_revision'
BANDS = ('g', 'r', 'i', 'z', 'w1', 'w2')
COLORS = tuple(itertools.combinations(BANDS, 2))
GROUPS = ('all', 'dwarf', 'giant', 'other', 'unknown')
GROUP_NAMES = dict(all='All chosen references', dwarf='Conservative dwarfs',
                   giant='Giant reference', other='Other qualified spectral stars',
                   unknown='No reliable spectral class')
Q = None


def configure():
    f.FIG = REV / 'figures'
    f.TABLES = REV / 'manifests'
    f.FIGURE_PREFIX = 'figures/v2_0_revision'
    f.TABLE_PREFIX = 'manifests/v2_0_revision'
    f.TABLES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(TRANSPARENT_STYLE)


def data():
    q = load(DEST / 'chosen_expanded_A.npz')
    q = subset(q, (q['split'] == 2) & np.isfinite(q['ref_chosen']))
    assert len(q['identity']) == 47756
    assert (q['class_label'] == 1).sum() == 1163
    # Band-specific qualification is applied before any arbitrary combination.
    for band in BANDS:
        key = band + '0'
        valid = np.isfinite(q[key])
        if band == 'i':
            valid &= q['i_usable']
        if band in ('w1', 'w2'):
            valid &= q[band + '_measurement_state'] == 4
        q[key] = np.where(valid, q[key], np.nan)
    return q


def group_mask(q, group):
    if group == 'all':
        return np.ones(len(q['identity']), bool)
    return q['class_label'] == dict(dwarf=0, giant=1, other=2, unknown=-1)[group]


def band_label(band):
    return band.upper() if band.startswith('w') else band


def color_label(color):
    return '$(' + band_label(color[0]) + '-' + band_label(color[1]) + ')_0$'


def color(q, pair):
    values = []
    for band in pair:
        value = q[band + '0']
        if band == 'i':
            value = np.where(q['i_usable'], value, np.nan)
        if band in ('w1', 'w2'):
            value = np.where(q[band+'_measurement_state'] == 4, value, np.nan)
        values.append(value)
    return values[0] - values[1]


def finish_error(fig, ax, x, y, error, xe, ye, absolute, label):
    im = f.map_error(ax, x, y, error, xe, ye, label, absolute)
    cb = fig.colorbar(im, ax=ax, extend='max' if absolute else 'both')
    cb.set_label('Median |relative distance error| [%]' if absolute else 'Median relative distance bias [%]')
    if absolute:
        cb.set_ticks([1, 10, 100], labels=['≤1', '10', '100'])
    else:
        cb.set_ticks([-100, -10, -1, 0, 1, 10, 100])
    ax.set_title('Median |relative error| [%]' if absolute else 'Median relative bias [%]')


def combination(task):
    kind, a, b = task
    q = Q
    configure()
    f.REGISTRY.clear()
    x = color(q, a)
    if kind == 'cmd':
        yr = q[b + '0'] - q['ref_chosen']
        yp = q[b + '0'] - q['mu'][:, 2]
        y_label = '$M_{' + band_label(b) + '}$ [mag]'
        y_all = np.r_[yr, yp]
        name = 'cmd_' + b + '_' + '_'.join(a)
    else:
        yr = color(q, b)
        yp = None
        y_label = color_label(b)
        y_all = yr
        name = 'color_' + '_'.join(a) + '__' + '_'.join(b)
    # All classes use exactly the same edges, including the full coordinate tails.
    xe = np.linspace(*f.limits(x), 39)
    ye = np.linspace(*f.limits(y_all), 39)
    shared_valid = np.isfinite(x) & np.isfinite(yr)
    density_max = max(2, np.histogram2d(x[shared_valid], yr[shared_valid], [xe, ye])[0].max())
    if yp is not None:
        good = np.isfinite(x) & np.isfinite(yp)
        density_max = max(density_max, np.histogram2d(x[good], yp[good], [xe, ye])[0].max())
    records = []
    for group in GROUPS:
        f.begin()
        mask = group_mask(q, group)
        xx, yy, err = x[mask], yr[mask], q['epsilon_pct_chosen'][mask]
        fig, axes = plt.subplots(2, 2, figsize=(11.2, 9.2), layout='constrained') if kind == 'cmd' else plt.subplots(1, 3, figsize=(15.6, 5.2), layout='constrained')
        axes = np.asarray(axes).ravel()
        ys = [yy, yp[mask]] if kind == 'cmd' else [yy]
        for ax, values, title in zip(axes, ys, ['Reference coordinates', 'Predicted coordinates']):
            im = f.density(ax, xx, values, xe, ye, title)
            im.set_norm(LogNorm(1, density_max))
            fig.colorbar(im, ax=ax, label='Objects/bin')
            ax.set_title(title)
        offset = len(ys)
        for i, absolute in enumerate((False, True)):
            finish_error(fig, axes[offset + i], xx, yy, err, xe, ye, absolute, 'absolute' if absolute else 'signed')
        for ax in axes:
            ax.set(xlabel=color_label(a), ylabel=y_label, xlim=(xe[0], xe[-1]), ylim=(ye[-1], ye[0]) if kind == 'cmd' else (ye[0], ye[-1]))
        fig.suptitle(GROUP_NAMES[group], fontsize=17)
        label = name + '_' + group
        n = int((mask & shared_valid).sum())
        f.save(fig, label,
               'Each panel uses only its required valid bands. Reference M=m0-mu_chosen, predicted M=m0-mu50. Residual statistics are in reference coordinates. Signed=median(epsilon), absolute=median(abs(epsilon)), both percent. N>=20 performance cells, sparse gray crosses, zero cells transparent. Shared edges and density scale across classes. Frozen historical TYPE exceptions retained for audit.',
               section='combinations')
        record = f.REGISTRY[label]
        record.update(group=group, combination_kind=kind, x_bands=a, y_band_or_color=b,
                      input_n=int(mask.sum()), valid_n=n, missing_n=int(mask.sum())-n,
                      status='no_valid_data' if n == 0 else 'rendered',
                      data_reason='Required band qualification or finite magnitude absent' if n == 0 else '',
                      origin='new_combination', common_edges_across_classes=True)
        save_json(f.TABLES / 'figures' / (label + '.json'), record)
        records.append(record)
    return records


def distance_clip(x, y):
    """One simultaneous mean +/- 3 population-sigma cut in linear kpc."""
    x, y = np.asarray(x), np.asarray(y)
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    stats = dict(input_n=len(x), valid_n=int(valid.sum()))
    keep = valid.copy()
    rejected = []
    for label, values in (('x', x), ('y', y)):
        mean = float(np.mean(values[valid])) if valid.any() else None
        sigma = float(np.std(values[valid], ddof=0)) if valid.any() else None
        low, high = (mean - 3*sigma, mean + 3*sigma) if valid.any() else (None, None)
        axis_keep = valid & (values >= low) & (values <= high) if valid.any() else valid.copy()
        rejected.append(valid & ~axis_keep)
        keep &= axis_keep
        stats.update({label+'_mean_kpc':mean, label+'_sigma_kpc':sigma,
                      label+'_low_kpc':low, label+'_high_kpc':high,
                      label+'_rejected':int((valid & ~axis_keep).sum())})
    stats.update(retained_n=int(keep.sum()), clipped_n=int((valid & ~keep).sum()),
                 invalid_n=int((~valid).sum()), both_rejected=int((rejected[0] & rejected[1]).sum()))
    return keep, stats


def distance(q, group='all', section='main'):
    f.begin()
    group = 'giant' if group == 'evolved' else group
    t = subset(q, group_mask(q, group))
    fig = plt.figure(figsize=(17, 10.5))
    outer = fig.add_gridspec(2, 3, height_ratios=[5, 1.8], hspace=.42, wspace=.48,
                            top=.87, bottom=.08, left=.065, right=.96)
    selections = {}
    for j, source in enumerate(('Gaia', 'specdis', 'spdist')):
        upper = outer[0, j].subgridspec(2, 2, height_ratios=[4, 1],
                                      width_ratios=[1, .045], hspace=.045, wspace=.055)
        main = fig.add_subplot(upper[0, 0])
        residual = fig.add_subplot(upper[1, 0], sharex=main)
        cax = fig.add_subplot(upper[:, 1])
        hist = fig.add_subplot(outer[1, j])
        raw_x = 10 ** ((t['ref_' + source] - 10) / 5)
        raw_y = t['distance_kpc'][:, 2]
        keep, selection = distance_clip(raw_x, raw_y)
        selections[source] = selection
        f.panel(source+'_3sigma_selection', len(raw_x), selection['valid_n'], selection['retained_n'], **selection)
        f.ROWS.append(dict(panel=source+'_3sigma_selection', **selection))
        x, y = raw_x[keep], raw_y[keep]
        error = 100 * (y / x - 1)
        if not len(x):
            for ax in (main, residual, hist, cax):
                ax.set_axis_off()
            main.text(.5, .5, 'No eligible reference', ha='center', transform=main.transAxes)
            main.set_title(source+' · N=0')
            continue
        edges = np.geomspace(*10 ** np.array(f.limits(np.log10(np.r_[x, y]))), 46)
        center = np.arange(-100., 102., 2.)
        im = f.density(main, x, y, edges, edges, source+'_distance')
        rim = f.density(residual, x, error, edges, center, source+'_Attached residual')
        norm = LogNorm(1, max(2, im.norm.vmax, rim.norm.vmax))
        im.set_norm(norm); rim.set_norm(norm)
        cb = fig.colorbar(im, cax=cax, label='Objects/bin')
        if norm.vmax < 10:
            cb.set_ticks(np.arange(1, int(norm.vmax)+1))
            cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, pos: f'{value:g}'))
            cb.ax.yaxis.set_minor_locator(NullLocator())
        main.plot(edges, edges, '--', c=ORANGE, lw=1)
        main.set(xscale='log', yscale='log', xlim=(edges[0], edges[-1]),
                 ylim=(edges[0], edges[-1]), ylabel='Predicted distance [kpc]',
                 title=f'{source} · N={len(x):,}\n3σ clipped: {selection["clipped_n"]:,}')
        main.tick_params(axis='x', which='both', labelbottom=False)
        main.yaxis.set_major_locator(FixedLocator(np.geomspace(edges[0],edges[-1],4)[1:]))
        main.yaxis.set_major_formatter(FuncFormatter(lambda value,pos: f'{value:.3g}'))
        main.yaxis.set_minor_locator(NullLocator())
        residual.set(ylim=(-100, 100), yticks=[-100, 0, 100],
                     xlabel='Reference distance [kpc]', ylabel='Residual [%]')
        residual.set_ylabel('Residual [%]', fontsize=12)
        residual.axhline(0, c=GRAY, lw=.8)
        residual.xaxis.set_major_locator(FixedLocator(np.geomspace(edges[0],edges[-1],4)))
        residual.xaxis.set_major_formatter(FuncFormatter(lambda value,pos: f'{value:.3g}'))
        residual.xaxis.set_minor_locator(NullLocator())
        counts = np.histogram(error, center)[0]
        outside = len(error) - int(counts.sum())
        hist.stairs(counts, center, color=BLUE)
        hist.set(xlim=(-100,100), xlabel='Relative distance residual [%]', ylabel='Objects/bin',
                 title=f'Central 2 percentage-point histogram\nOutside ±100%: {outside:,}/{len(error):,}')
        hist.axvline(0, c=GRAY, lw=.8)
        hist.yaxis.set_major_locator(MaxNLocator(nbins=4,integer=True))
        label = source+'_Central 2 percentage-point histogram'
        f.panel(label, len(error), len(error), int(counts.sum()))
        for lo, hi, n in zip(center[:-1], center[1:], counts):
            f.ROWS.append(dict(panel=label, lo=lo, hi=hi, n=int(n)))
        for lo, hi in zip(edges[:-1], edges[1:]):
            selected = (x >= lo) & (x < hi)
            row = dict(panel=source+'_spread', xlo=lo, xhi=hi, n=int(selected.sum()))
            if selected.sum() >= 20:
                p16, median, p84 = np.quantile(error[selected], [.16, .5, .84])
                residual.plot(np.sqrt(lo*hi), median, 'o', ms=2, c='black')
                residual.vlines(np.sqrt(lo*hi), p16, p84, color='black', lw=.6)
                row.update(p16=p16, median=median, p84=p84)
            f.ROWS.append(row)
    fig.suptitle(GROUP_NAMES[group], fontsize=17)
    name = '03_distance_' + ('evolved' if group == 'giant' else group)
    f.save(fig, name,
           'Frozen expanded_A. Display-only simultaneous 3-sigma selection on reference x and predicted y distances in linear kpc: arithmetic mean +/- 3 population standard deviations (ddof=0), one pass, independently for each population/reference pair. All three panels use the same retained objects. Narrow attached residual axis is linear from -100% to 100%; its x-axis is shared with the main distance panel. Central histogram uses 2 percentage-point bins. Out-of-window and sigma-exclusion counts are retained in CSV/JSON. Trend bars are P16-P84 object spread, not median confidence intervals. Original uncut scientific metrics and anomaly audit remain unchanged. Reference-source samples overlap.',
           section=section, reference='Gaia/specdis/spdist')
    record = f.REGISTRY[name]
    record.update(distance_layout='main + attached residual (height ratio 4:1) + central histogram',
                  clipping=dict(method='simultaneous single-pass mean +/- 3*std(ddof=0)',
                                space='linear distance in kpc', independent_per='population/reference pair',
                                sources=selections), residual_limits_pct=[-100,100])
    save_json(f.TABLES/'figures'/(name+'.json'), record)


def distance_only():
    """Replace only the five distance plates, preserving all other products."""
    source = DEST/'chosen_expanded_A.npz'
    before = sha(source)
    expected = json.loads((ROOT/'manifests/v2_0_revision/revision_validation.json').read_text())['source_sha256']
    assert before == expected
    f.REGISTRY.clear()
    q = load(source)
    q = subset(q, q['split'] == 2)
    for group in GROUPS:
        distance(q, group, section='main' if group == 'all' else 'appendix')
    for record in f.REGISTRY.values():
        for p in record['panels']:
            assert p['plotted']+p['outside']+p['missing']==p['eligible']
    assert sha(source) == before
    save_json(f.TABLES/'distance_replacement.json', dict(source_sha256=before,
        job_id=os.environ.get('SLURM_JOB_ID'), figures=f.REGISTRY))
    print('COMPLETE distance replacement: 5 plates', flush=True)


def physical(q):
    qualified = q['rvs_parameter_quality']
    summary = []
    for group in GROUPS:
        t = subset(q, qualified & group_mask(q, group))
        for window in ('full', 'body'):
            f.begin()
            x, y = t['rvs_teff'], t['rvs_logg']
            if window == 'full':
                xe = np.linspace(*f.limits(q['rvs_teff'][qualified]), 49)
                ye = np.linspace(*f.limits(q['rvs_logg'][qualified]), 41)
            else:
                xe, ye = np.linspace(3000, 8000, 49), np.linspace(0, 5.5, 41)
            fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.2), layout='constrained')
            im = f.density(axes[0], x, y, xe, ye, 'Kiel density')
            fig.colorbar(im, ax=axes[0], label='Objects/bin')
            for ax, absolute in zip(axes[1:], (False, True)):
                finish_error(fig, ax, x, y, t['epsilon_pct_chosen'], xe, ye, absolute, 'absolute' if absolute else 'signed')
                sparse = np.isfinite(x) & np.isfinite(y) & (y < 2)
                ax.scatter(x[sparse], y[sparse], s=8, facecolors='none', edgecolors=GRAY, linewidths=.6)
            for ax in axes:
                ax.set(xlabel='RVS temperature [K]', ylabel=r'RVS $\log g$', xlim=(xe[-1], xe[0]), ylim=(ye[-1], ye[0]))
            fig.suptitle(f'Kiel diagram · {GROUP_NAMES[group]} · {window} · N={len(x):,}', fontsize=16)
            f.save(fig, f'kiel_{window}_{group}', 'Reliable RVS only. Full range retains every finite parameter, body is an explicit zoom. Low-gravity points remain visible irrespective of performance-cell N. No post-hoc point rejection.', section='physical')
        f.begin()
        x = t['rvs_teff']
        yr = t['g0'] - t['ref_chosen']
        yp = t['g0'] - t['mu'][:, 2]
        xe = np.linspace(*f.limits(q['rvs_teff'][qualified]), 49)
        ye = np.linspace(*f.limits(np.r_[q['g0'][qualified]-q['ref_chosen'][qualified], q['g0'][qualified]-q['mu'][qualified, 2]]), 45)
        fig, axes = plt.subplots(2, 2, figsize=(11.2, 9.2), layout='constrained')
        for ax, y, label in ((axes[0, 0], yr, 'Reference absolute g'), (axes[0, 1], yp, 'Predicted absolute g')):
            im = f.density(ax, x, y, xe, ye, label)
            fig.colorbar(im, ax=ax, label='Objects/bin')
            ax.set_title(label)
        for ax, absolute in zip(axes[1], (False, True)):
            finish_error(fig, ax, x, yr, t['epsilon_pct_chosen'], xe, ye, absolute, 'absolute' if absolute else 'signed')
        for ax in axes.ravel():
            ax.set(xlabel='RVS temperature [K]', ylabel='Legacy $M_g$ [mag]', xlim=(xe[-1], xe[0]), ylim=(ye[-1], ye[0]))
        fig.suptitle(GROUP_NAMES[group], fontsize=16)
        f.save(fig, 'temperature_Mg_' + group, 'Temperature versus absolute Legacy g, not bolometric luminosity. Reference M=g0-mu_chosen; predicted M=g0-mu50. Reliable RVS parameter gate retained.', section='physical')
    # Report actual endpoints and sentinel values without deleting finite points.
    for key in ('rvs_teff', 'rvs_logg', 'rvs_feh'):
        values = q[key]
        for mask, label in ((np.ones(len(values), bool), 'all chosen'), (qualified, 'RVS qualified')):
            a = values[mask]
            finite = a[np.isfinite(a)]
            summary.append(dict(parameter=key, sample=label, n=len(a), finite=len(finite),
                                min=float(finite.min()), max=float(finite.max()),
                                at_min=int((finite == finite.min()).sum()), at_max=int((finite == finite.max()).sum()),
                                sentinel_le_minus9000=int((finite <= -9000).sum()), zero=int((finite == 0).sum())))
    csv_rows(f.TABLES / 'spectral_parameter_boundaries.csv', summary)


def legacy_figures():
    configure()
    # Read-only compact statistics needed by the original 55 figure families.
    for path in (DEST / 'manifests').glob('*'):
        if path.is_file() and path.name in ('snapshot.json','training_brightness_hist.csv'):
            shutil.copy2(path, f.TABLES / path.name)
    q = load(DEST / 'chosen_expanded_A.npz')
    q = subset(q, q['split'] == 2)
    f.PRIMARY = q
    f.distance = distance
    f.scope(); f.coverage(q); distance(q); f.cmd(q); f.color_map(q)
    f.physical(q); f.conditions(q); f.uncertainty(q); f.availability()
    f.joint_states(); f.experiments(); f.appendix(q); f.reused_appendices()
    # Read the already checked per-bin PDF cache; four extra forward calls
    # would add no information to this presentation-only refresh.
    import csv
    with open(DEST/'manifests/figures/A_fixed_pdf_examples.csv') as stream:
        pdf_rows = list(csv.DictReader(stream))
    f.begin()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout='constrained')
    for ax, label in zip(axes.ravel(), dict.fromkeys(row['panel'] for row in pdf_rows)):
        rows = [row for row in pdf_rows if row['panel'] == label]
        edges = np.r_[[float(row['lo']) for row in rows], float(rows[-1]['hi'])]
        ax.stairs([float(row['density']) for row in rows], edges, color=BLUE)
        ax.axvline(float(rows[0]['reference']), color=ORANGE, label='Reference')
        ax.set(title=label.replace('evolved', 'giant'), xlabel='Distance modulus [mag]', ylabel='Density [1/mag]')
        ax.legend(fontsize=10)
        f.ROWS.extend(rows)
    f.save(fig, 'A_fixed_pdf_examples', 'Existing four frozen PDF cases, original bin values and reference unchanged. User-visible giant terminology updated.', section='appendix')
    physical(data())
    separate_flows()
    save_json(f.TABLES / 'base_registry.json', f.REGISTRY)



def separate_flows():
    import csv
    extraction = json.loads((ROOT/'manifests/baseline_v1_4_extraction_complete.json').read_text())
    with open(ROOT/'manifests/v2_0/A_selection_flow.csv') as stream:
        flow = list(csv.DictReader(stream))
    with open(ROOT/'manifests/v2_0/C_application_fields.csv') as stream:
        fields = list(csv.DictReader(stream))
    f.begin()
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), layout='constrained')
    lines = [
        [('Reference retrieval', f"1,600 south sweeps\n{extraction['scanned_rows']:,} input rows"),
         ('Assembled identities', f"{int(flow[0]['input']):,} objects"),
         ('Sequential reference gates', '\n'.join(r['stage']+': '+format(int(r['pass_n']),',') for r in flow[:4])),
         ('Independent qualification', '\n'.join(r['stage']+': '+format(int(r['pass_n']),',') for r in flow[4:]))],
        [('Application footprint', '14 preselected fields'),
         ('Unique application objects', f"{sum(int(r['unique_rows']) for r in fields):,}"),
         ('DECam prediction', f"{sum(int(r['predicted']) for r in fields):,} · baseline_v1"),
         ('North explicit no prediction', f"{sum(int(r['unique_rows'])-int(r['predicted']) for r in fields):,}")]]
    for ax, sequence in zip(axes, lines):
        ax.axis('off')
        for i, (label, value) in enumerate(sequence):
            application = ax is axes[1]
            x, y = (.5, .88 - i*.235)
            if application and i >= 2:
                x, y = (.24 if i == 2 else .76), .30
                label = 'DECam prediction' if i == 2 else 'North\nNo prediction'
                value = value.replace(' · ', '\n')
            ax.text(x, y, label+'\n'+value, ha='center', va='center', transform=ax.transAxes, fontsize=10 if application and i >= 2 else 12, bbox=dict(boxstyle='round,pad=.5', fc='none', ec=BLUE))
            if application and i >= 2:
                ax.annotate('', xy=(x,.39), xytext=(.5,.59), xycoords='axes fraction', arrowprops=dict(arrowstyle='->', color=GRAY))
            elif i < len(sequence)-1 and not (application and i == 1):
                ax.annotate('', xy=(.5,y-.14), xytext=(.5,y-.08), xycoords='axes fraction', arrowprops=dict(arrowstyle='->', color=GRAY))
            f.ROWS.append(dict(panel='reference' if ax is axes[0] else 'application', stage=label, value=value))
    f.save(fig, 'separate_reference_application_flows', 'Two distinct pipelines. The DR11 scale ratios are not same-cut retention fractions. Class/fit/parallax qualifications are parallel, and their counts overlap. Full-DR11 same-cut denominator and effective masked area remain uncomputed.', section='main', model='reference assembly + baseline_v1 application', reference='distinct populations', split='all')

def finalize():
    configure()
    registry = {}
    for path in sorted((f.TABLES / 'figures').glob('*.json')):
        record = json.loads(path.read_text())
        if 'name' not in record:
            continue
        for panel in record['panels']:
            assert panel['plotted'] + panel['outside'] + panel['missing'] == panel['eligible'], (record['name'], panel)
        for extension in ('png', 'pdf'):
            path = REV / 'figures' / record['section'] / (record['name'] + '.' + extension)
            assert path.exists() and sha(path) == record[extension + '_sha256']
        registry[record['name']] = record
    original_names = set(json.loads((DEST/'manifests/figure_registry.json').read_text()))
    assert original_names <= set(registry), sorted(original_names-set(registry))
    for name, record in registry.items():
        record.setdefault('origin', 'redrawn_existing' if name in original_names else 'new_diagnostic')
        record.setdefault('status', 'rendered')
    expected = 5 * (6 * 15 + 105)
    assert sum(r.get('combination_kind') is not None for r in registry.values()) == expected
    assert len(registry) == 1050, len(registry)
    for name, digest in json.loads((OUT/'freeze.json').read_text())['hashes'].items():
        assert sha(ROOT/name) == digest, name
    save_json(f.TABLES / 'figure_registry.json', registry)
    csv_rows(f.TABLES/'complete_figure_inventory.csv', [dict(name=r['name'], section=r['section'], status=r['status'], origin=r['origin'], png=r['png'], pdf=r['pdf'], csv=r['csv']) for r in registry.values()])
    csv_rows(f.TABLES / 'combination_inventory.csv', [dict(name=r['name'], kind=r['combination_kind'], group=r['group'], valid_n=r['valid_n'], missing_n=r['missing_n'], status=r['status'], origin=r['origin']) for r in registry.values() if r.get('combination_kind')])
    save_json(f.TABLES / 'revision_validation.json', dict(figures=len(registry), exhaustive_combinations=expected, count_conservation=True, source_sha256=sha(DEST / 'chosen_expanded_A.npz')))
    print('COMPLETE revision', len(registry), 'figures', flush=True)


def main():
    require_compute()
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['base', 'combinations', 'finalize', 'distance'])
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    configure()
    if args.stage == 'base':
        legacy_figures()
    elif args.stage == 'finalize':
        finalize()
    elif args.stage == 'distance':
        distance_only()
    else:
        global Q
        Q = data()
        tasks = [('cmd', c, b) for c in COLORS for b in BANDS]
        tasks += [('color', a, b) for a, b in itertools.combinations(COLORS, 2)]
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('fork')) as pool:
            for records in pool.map(combination, tasks):
                print('combination complete', records[0]['name'], flush=True)


if __name__ == '__main__':
    main()
