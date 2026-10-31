#!/usr/bin/env python3
"""Small aggregate report/figures from completed frozen field evaluations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'.deps')]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from photdis.stage1 import require_compute, write_json


def number(value, digits=3):
    return '—' if value is None else f'{value:.{digits}f}'


def figures(result, folder):
    folder.mkdir(parents=True, exist_ok=True)
    fields, folds = result['inventory']['fields'], result['folds']
    systems = list(result['config']['release_system'].values())
    plt.rcParams.update({'font.size': 9, 'figure.dpi': 140})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, system in zip(axes, systems):
        for item in fields:
            if system not in item['systems']:
                continue
            h = item['systems'][system]['histograms']['r']
            n = np.asarray(h['optical']); bins = np.array(result['config']['r_bins'])
            x = (bins[:-1]+bins[1:])/2
            ratio = np.divide(h['w1w2_snr5'], n, out=np.full(len(n), np.nan), where=n>0)
            ax.plot(x, ratio, 'o-', alpha=.7, label=str(item['field_index']))
        ax.set(title=system, xlabel='Measured r (AB)', ylabel='W1,W2 S/N>=5 fraction of clean grz', ylim=(0, 1.02))
        if ax.lines:
            ax.legend(title='Field index', ncol=3, fontsize=7)
    fig.suptitle('Photometric availability; no spectroscopic label requirement')
    fig.tight_layout(); fig.savefig(folder/'01_wise_availability.png'); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey='row')
    colors = {'A': '#2765a5', 'B': '#c37b16', 'C': '#327f4f'}
    for col, system in enumerate(systems):
        for row, cls in enumerate(('giants', 'dwarfs')):
            ax = axes[row, col]
            for model in ('A', 'B', 'C'):
                for cell in folds:
                    if cell['system']!=system or cell['panel']!='common5' or model not in cell['models']:
                        continue
                    metric = cell['models'][model]['strata']['all'][cls]
                    if metric['rate'] is None:
                        continue
                    rate, lo, hi = metric['rate'], *metric['interval95']
                    x = cell['field_index'] + {'A': -.2, 'B': 0, 'C': .2}[model]
                    ax.errorbar(x, rate, yerr=[[max(0, rate-lo)], [max(0, hi-rate)]],
                                fmt='o', color=colors[model], capsize=2)
                ax.plot([], [], 'o', color=colors[model], label=model)
            ax.set(title=system, xlabel='Held-out field index', ylabel='Giant TPR' if row==0 else 'Dwarf FPR', ylim=(-.03, 1.03))
            ax.legend()
    fig.suptitle('Same-star WISE>=5 comparison; Wilson intervals conditional on one field')
    fig.tight_layout(); fig.savefig(folder/'02_class_transfer.png'); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    pi = result['config']['purity_giant_fractions']
    for ax, system in zip(axes, systems):
        for cell in folds:
            if cell['system']!=system or cell['panel']!='common5' or 'C' not in cell['models']:
                continue
            p = cell['models']['C']['strata']['all']['purity_if_two_classes']
            if all(v is not None for v in p):
                ax.plot(pi, p, 'o-', label=str(cell['field_index']))
        ax.set(xscale='log', title=system, xlabel='Assumed target giant fraction', ylabel='Two-class purity from point rates', ylim=(0, 1.02))
        if ax.lines:
            ax.legend(title='Field index', ncol=3, fontsize=7)
    fig.suptitle('Sensitivity to population prior; not calibrated full-catalog purity')
    fig.tight_layout(); fig.savefig(folder/'03_purity_prior.png'); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, system in zip(axes, systems):
        ax.axhline(0, color='gray', lw=1)
        for model in ('A', 'B', 'C'):
            cells = [c for c in folds if c['system']==system and c['panel']=='common5' and model in c['abundance']]
            ax.plot([c['field_index'] for c in cells],
                    [c['abundance'][model]['paired_log_gain'] for c in cells], 'o', color=colors[model], label=model)
        ax.set(title=system, xlabel='Held-out field index', ylabel='Mean log-density gain / star vs marginal')
        ax.legend()
    fig.suptitle('Hidden FeH; matched stars; spectral errors convolved once')
    fig.tight_layout(); fig.savefig(folder/'04_feh_information.png'); plt.close(fig)
    return [str(p) for p in sorted(folder.glob('*.png'))]


def report(result):
    inv, cfg = result['inventory'], result['config']
    lines = ['# Existing-field giant selection and abundance information v1', '',
             'Protocol: [registered design](giant_information_protocol_v1.md). '
             'This is leave-one-whole-field-out evaluation on existing labeled fields, '
             'not an untouched final sample or a field-giant distance release.', '',
             f"Stored clean records: {inv['n_clean_photometric_records']:,}; unique LS photometric "
             f"measurements: {inv['n_unique_clean_ls_measurements']:,}. Spectral records: "
             f"{inv['n_spectral_records']:,}; connected identity groups: {inv['n_spectral_identity_groups']:,}; "
             f"conflicting groups: {inv['n_conflicting_groups']}; cross-field groups: {inv['n_crossfield_groups']}.", '',
             'The LS measurement count is not an independent-star count. Uncertain spectral '
             'classes and raw unassessed Redrock types remain in the audit. Base/clean counts '
             'are conditional on the stored diagnostic parent; optical availability does not require WISE.', '',
             '| Field | Camera | Stored parent | Base | Clean records | W1>=5 | W1W2>=5 | W1W2>=20 |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for f in inv['fields']:
        for system, s in f['systems'].items():
            n = s['total']; c = f['cutflow']
            lines.append(f"| {f['field_index']}: {f['field']} | {system} | {c['stored_parent']} | {c['base']} | {n['optical']} | {n['w1_snr5']} | {n['w1w2_snr5']} | {n['w1w2_snr20']} |")
    lines += ['', 'Stored-parent/base columns are field totals (not camera subtotals for a mixed field). '
              'Clean and WISE columns are camera-specific.', '',
              '## Same-star WISE>=5 class comparison', '',
              'Threshold=0.5 is fixed for class-balanced scores. Values below are TPR / FPR. '
              'Wilson intervals, temperature/r/error strata, high-S/N panel, optical-only '
              'panel and color extrapolation flags are in the aggregate manifest.', '',
              '| Field | Camera | Test N | A | B | C |', '|---|---|---:|---|---|---|']
    for cell in result['folds']:
        if cell['panel']!='common5':
            continue
        values = []
        for m in ('A', 'B', 'C'):
            s = cell['models'].get(m, {}).get('strata', {}).get('all')
            values.append(number(s['giants']['rate'])+' / '+number(s['dwarfs']['rate']) if s else cell['status'])
        lines.append(f"| {cell['field_index']} | {cell['system']} | {cell['n_test']} | "+' | '.join(values)+' |')
    lines += ['', '## Hidden abundance comparison', '',
              'Mean log-density gains relative to the training-only Gaussian abundance '
              'marginal are evaluated on identical stars. Measurement-error convolution is '
              'included; widths and coverage refer to noisy spectral labels. The fitted '
              'residual width is not intrinsic population scatter. These fixed simple '
              'models omit coefficient uncertainty; finite-sample coverage is an outcome '
              'to inspect, not guaranteed calibration.', '',
              '| Field | Camera | Panel | FeH test N | A gain | B gain | C gain |',
              '|---|---|---|---:|---:|---:|---:|']
    for cell in result['folds']:
        if cell.get('abundance_status')!='scored':
            continue
        gains = [number(cell['abundance'].get(m, {}).get('paired_log_gain')) for m in ('A', 'B', 'C')]
        lines.append(f"| {cell['field_index']} | {cell['system']} | {cell['panel']} | {cell['abundance_test_n']} | "+' | '.join(gains)+' |')
    lines += ['', '## Finite transfer decision', '',
              'A stable abundance gain requires >=3 fields with >=10 test giants each, '
              'a positive whole-field bootstrap lower limit, and positive signs in >=75% '
              'of those fields. Overlapping training folds make this a descriptive transfer '
              'uncertainty, not a new survey-level significance measurement.', '']
    for system, panels in result['transfer'].items():
        for panel, items in panels.items():
            for model, value in items['abundance'].items():
                if panel=='optical' and model!='A':
                    continue
                status = 'passes registered limited-family gain criterion' if value['stable_gain'] else 'no stable gain established (weak or insufficient evidence)'
                lines.append(f"- {system}, {panel}, {model}: {value['n_fields']} informative fields; "
                             f"gain {number(value['mean'])}, field-bootstrap interval "
                             f"[{number(value['interval95'][0])}, {number(value['interval95'][1])}]; {status}.")
    failed = [f"{r['system']}/{r['field_index']}/{r['panel']}: {r['status']}"
              for r in result['folds'] if r['status']!='scored']
    lines += ['', 'Unscored class cells: '+('; '.join(failed) if failed else 'none')+'.', '',
              '## Population and distance boundary', '',
              'The evidence applies to DESI-qualified 4200–5000 K low-gravity versus '
              'high-gravity stars within these native-camera, clean, high-latitude low-dust '
              'samples. The three common-panel models have identical train/test stars. '
              'Changing the WISE S/N panel changes its population; its performance difference '
              'is not a causal noise-only gain. Pure optical availability remains separate.', '',
              'The purity-prior figure assumes only two transported classes. Unknown '
              'temperatures, intermediate gravities, nonstellar contamination, DESI targeting '
              'and different field populations prevent reading it as full-catalog purity. '
              'Selected labeled-giant temperature/logg/FeH distributions are provided per fold '
              'in the manifest; they characterize that labeled subset only.', '',
              'No lower-RGB phase identification has been established. A useful WISE score '
              'can define a finite-depth evolved-star candidate layer, but RGB/RC/HB separation '
              'and an external luminosity calibration for its actual phase/FeH range are '
              'still required before distances. Where FeH gains fail the criterion, retain '
              'explicit external population priors. Even successful q(FeH|colors) must enter '
              'a consistent joint distance model: reusing it as an independent prior with the '
              'same color likelihood would double-count information.', '',
              'The next calibration must target the observed K-giant candidate population '
              'and its independently established evolutionary phases, rather than attach '
              'the present M3/M53 lower-RGB curve to every low-gravity object. No new field, '
              'color-cut search, distance-VAC regressor or RGB-curve tuning was used.', '',
              '## Artifacts', '']
    lines += [f'- `{p}`' for p in result['figures']]
    return '\n'.join(lines)+'\n'


def main():
    require_compute()
    source = Path('data/giant_information_v1/results.json')
    destination = Path('manifests/giant_information_v1.json')
    if destination.exists():
        raise FileExistsError(destination)
    result = json.loads(source.read_text())
    result['figures'] = figures(result, Path('figures/giant_information_v1'))
    result['result_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    result['summary_code_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_json(destination, result)
    Path('docs/giant_information_result_v1.md').write_text(report(result))


if __name__ == '__main__':
    main()
