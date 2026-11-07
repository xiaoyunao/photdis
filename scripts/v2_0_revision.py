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


def distance(q, group='all', section='main'):
    f.begin()
    group = 'giant' if group == 'evolved' else group
    t = subset(q, group_mask(q, group))
    fig, axes = plt.subplots(5, 3, figsize=(16.5, 22), layout='constrained')
    for j, source in enumerate(('Gaia', 'specdis', 'spdist')):
        valid = np.isfinite(t['ref_' + source])
        x = 10 ** ((t['ref_' + source][valid] - 10) / 5)
        y = t['distance_kpc'][valid, 2]
        error = t['epsilon_pct_' + source][valid]
        if not len(x):
            for ax in axes[:, j]:
                ax.text(.5, .5, 'No eligible reference', ha='center', transform=ax.transAxes)
                ax.set_axis_off()
            axes[0,j].set_title(source+' · N=0')
            f.panel(source, len(valid), 0)
            continue
        edges = np.geomspace(*10 ** np.array(f.limits(np.log10(np.r_[x, y]))), 46)
        im = f.density(axes[0, j], x, y, edges, edges, source + '_distance')
        fig.colorbar(im, ax=axes[0, j], label='Objects/bin')
        axes[0, j].plot(edges, edges, '--', c=ORANGE)
        axes[0, j].set(xscale='log', yscale='log', xlabel='Reference distance [kpc]', ylabel='Predicted distance [kpc]', title=source)
        for axis in (axes[0,j].xaxis, axes[0,j].yaxis):
            axis.set_major_locator(FixedLocator(np.geomspace(edges[0],edges[-1],4)))
            axis.set_major_formatter(FuncFormatter(lambda value,pos: f'{value:.3g}'))
            axis.set_minor_locator(NullLocator())
        full = np.linspace(*f.limits(np.r_[error, 0]), 102)
        center = np.arange(-100., 102., 2.)
        for k, bins, label in ((1, full, 'Full linear residual'), (2, center, 'Central linear residual')):
            im = f.density(axes[k, j], x, error, edges, bins, source + '_' + label)
            fig.colorbar(im, ax=axes[k, j], label='Objects/bin')
            outside = int(((error < bins[0]) | (error > bins[-1])).sum())
            axes[k, j].set(xscale='log', ylim=(bins[0], bins[-1]), xlabel='Reference distance [kpc]', ylabel='Relative distance residual [%]', title=f'{label} · outside {outside}/{len(error)}')
            axes[k, j].axhline(0, c=GRAY, lw=.8)
            axes[k,j].xaxis.set_major_locator(FixedLocator(np.geomspace(edges[0],edges[-1],4)))
            axes[k,j].xaxis.set_major_formatter(FuncFormatter(lambda value,pos: f'{value:.3g}'))
            axes[k,j].xaxis.set_minor_locator(NullLocator())
        for k, bins, label in ((3, full, 'Full equal-width histogram'), (4, center, 'Central 2 percentage-point histogram')):
            counts = np.histogram(error, bins)[0]
            axes[k, j].stairs(counts, bins, color=BLUE)
            axes[k, j].set(xlabel='Relative distance residual [%]', ylabel='Objects/bin', title=label)
            axes[k, j].axvline(0, c=GRAY, lw=.8)
            axes[k,j].yaxis.set_major_locator(MaxNLocator(nbins=5,integer=True))
            f.panel(source + '_' + label, len(error), len(error), int(counts.sum()))
            for lo, hi, n in zip(bins[:-1], bins[1:], counts):
                f.ROWS.append(dict(panel=source + '_' + label, lo=lo, hi=hi, n=int(n)))
        for lo, hi in zip(edges[:-1], edges[1:]):
            selected = (x >= lo) & (x < hi)
            row = dict(panel=source + '_spread', xlo=lo, xhi=hi, n=int(selected.sum()))
            if selected.sum() >= 20:
                p16, median, p84 = np.quantile(error[selected], [.16, .5, .84])
                axes[2, j].plot(np.sqrt(lo*hi), median, 'o', ms=2, c='black')
                axes[2, j].vlines(np.sqrt(lo*hi), p16, p84, color='black', lw=.6)
                row.update(p16=p16, median=median, p84=p84)
            f.ROWS.append(row)
    fig.suptitle(GROUP_NAMES[group], fontsize=17)
    original_group = 'evolved' if group == 'giant' else group
    f.save(fig, '03_distance_' + original_group,
           'Frozen expanded_A. Source-specific same-star overlaps are allowed; source N cannot be summed as independent stars. Every residual y-axis and histogram count axis is linear. Center window uses 2 percentage-point bins. Trend bars are P16–P84 object spread, not median confidence intervals. Complete sample retained.',
           section=section, reference='Gaia/specdis/spdist')


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
    parser.add_argument('stage', choices=['base', 'combinations', 'finalize'])
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    configure()
    if args.stage == 'base':
        legacy_figures()
    elif args.stage == 'finalize':
        finalize()
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
