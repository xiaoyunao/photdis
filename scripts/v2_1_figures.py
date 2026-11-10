#!/usr/bin/env python3
"""Compact scientific plates rendered from audited CSVs, without catalog scans."""
import csv
import json
import hashlib
import re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
from photdis_figure_style import STYLE, BLUE, ORANGE, GREEN, GRAY

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT/'manifests/v2_1'
FIG = ROOT/'figures/v2_1'
COLORS = [BLUE, ORANGE, GREEN, GRAY]
CLASSES = ['dwarf', 'evolved', 'other', 'unknown']
LABELS = ['Dwarf', 'Giant reference', 'Other stellar', 'Unlabeled']
DIST = ['0–2', '2–5', '5–10', '10–20', '20–50', '50+']


def read(name):
    with open(TABLES/(name+'.csv')) as f:
        return list(csv.DictReader(f))


def select(rows, **conditions):
    return [r for r in rows if all(r.get(k) == str(v) for k, v in conditions.items())]


def number(row, key):
    return float(row.get(key) or 'nan')


def save(fig, name, rows, sources, caption):
    FIG.mkdir(parents=True, exist_ok=True)
    dest = TABLES/'figures'
    dest.mkdir(exist_ok=True)
    for artist in fig.findobj(match=Text):
        if re.search(r'[\u3400-\u9fff]', artist.get_text()):
            raise ValueError('Non-English figure text')
    fig.savefig(FIG/(name+'.png'), dpi=300, bbox_inches='tight')
    fig.savefig(FIG/(name+'.pdf'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(dest/(name+'.csv'), 'w') as f:
        writer = csv.DictWriter(f, fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    metadata = dict(name=name, caption=caption, selection='Verified DR11 PSF; development test; no display clipping',
                    rows=len(rows), model='expanded_A',
                    inputs={p: hashlib.sha256((TABLES/(p+'.csv')).read_bytes()).hexdigest() for p in sources})
    if name.startswith('03'):
        metadata.update(model='Fixed same-label class probes and named frozen partition controls',
                        selection='Matched or broad PSF development sets as labeled; template simulations use training identities, never independent validation')
    if name == '03b_optical_candidate':
        metadata['model'] = 'Broad grz probe; one same-identity regression versus class-partition distance candidate'
    if name.startswith('06'):
        metadata['model'] = 'expanded_A; isolated frozen weak-flux prototype and PSF nonstellar control samples in labeled panels'
    (dest/(name+'.json')).write_text(json.dumps(metadata, indent=2)+'\n')


def distance():
    rows = select(read('distance_class'), split='development_test', reference='chosen')
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    selected = []
    for i, (name, label, color) in enumerate(zip(CLASSES, LABELS, COLORS)):
        data = sorted(select(rows, group=name), key=lambda r: number(r, 'low_kpc'))
        selected += data
        n = np.array([number(r, 'n') for r in data])
        axes[0, 0].bar(np.arange(6)+(i-1.5)*.18, n, width=.18, color=color, label=label)
        for ax, key, scale in [(axes[0, 1], 'median_relative', 100), (axes[1, 0], 'tail_1', 100),
                               (axes[1, 1], 'coverage68', 100)]:
            value = np.array([number(r, key)*scale for r in data])
            ax.plot(np.arange(6), value, 'o-', color=color, label=label)
            small = n < 20
            ax.scatter(np.arange(6)[small], value[small], marker='x', s=80, color='black', zorder=4)
    axes[0, 0].set(yscale='log', ylabel='Reference count')
    axes[0, 0].legend(fontsize=10)
    axes[0, 1].set(ylabel='Median distance residual (%)')
    axes[0, 1].axhline(0, color='black', lw=1, ls='--')
    axes[1, 0].set(ylabel=r'Severe fraction: $|\Delta\mu|>1$ (%)', ylim=(-3, 103))
    axes[1, 1].set(ylabel='68% interval coverage (%)', ylim=(-3, 103))
    axes[1, 1].axhline(68, color='black', lw=1, ls='--')
    for ax in axes.ravel():
        ax.set_xticks(np.arange(6), DIST)
        ax.set_xlabel('Reference heliocentric distance (kpc)')
        ax.axvline(2.5, color=GRAY, ls=':', lw=1.5)
    fig.suptitle('Full PSF development sample · expanded A · chosen reference\nBlack crosses: fewer than20 objects', fontsize=16)
    save(fig, '01_distance_population', selected, ['distance_class'],
         'Reference-conditioned performance, not actual selected-near sample or independent accuracy. Sparse bins are descriptive.')


def movement():
    rows = select(read('migration'), split='development_test', reference='chosen')
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    selected = []
    for ax, name, label in zip(axes.ravel(), ['all']+CLASSES, ['All']+LABELS):
        r = select(rows, group=name)[0]
        selected.append(r)
        matrix = np.array([[number(r, 'near_selected'), number(r, 'near_rejected')],
                           [number(r, 'far_selected'), number(r, 'far_rejected')]])
        fraction = matrix/np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        ax.imshow(fraction, vmin=0, vmax=1, cmap='Blues', aspect='auto')
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, f'{int(value):,}\n{fraction[i,j]*100:.1f}%', ha='center', va='center',
                    color='white' if fraction[i,j] > .65 else 'black')
        ax.set_xticks([0, 1], ['Selected near', 'Rejected'])
        ax.set_yticks([0, 1], ['Reference near', 'Reference far'])
        ax.grid(False)
        ax.set_title(label+f"\nSelected far fraction {100*number(r, 'far_contamination'):.1f}%", fontsize=13)
    ax = axes.ravel()[-1]
    vals = [number(select(rows, group=c)[0], 'far_selected') for c in CLASSES]
    ax.bar(np.arange(4), vals, color=COLORS)
    ax.set_xticks(np.arange(4), LABELS, rotation=25, ha='right')
    ax.set_ylabel('Far references selected as near')
    ax.set_title('Composition of far contamination', fontsize=13)
    fig.suptitle(r'Actual median selection: $d_{50}<10$ kpc · PSF development test', fontsize=17)
    save(fig, '02_near_far_migration', selected, ['migration'],
         'Row percentages and counts. Far contamination denominator is all selected objects within each stated reference class. Reference point boundaries; uncertainty separately tabulated.')


def conditions():
    rows = select(read('distance_class'), split='development_test', reference='chosen')
    extra = all((TABLES/(name+'.csv')).exists() for name in ['systematic_sensitivity', 'weak_flux_probability', 'nonstellar_near'])
    fig, axes = plt.subplots(2 if extra else 1, 3, figsize=(16, 10 if extra else 5), constrained_layout=True)
    axes = axes.ravel()
    selected = []
    corrected = (TABLES/'metallicity_quality.csv').exists()
    metal = select(read('metallicity_quality'), split='development_test', abundance='corrected') if corrected else []
    for name, flo, label, color in zip(['dwarf_FeH_-inf_-1.5', 'dwarf_FeH_-1.5_-0.5', 'dwarf_FeH_-0.5_inf'],
                                  ['-inf', '-1.5', '-0.5'], ['[Fe/H] < -1.5', '-1.5 to -0.5', '[Fe/H] >= -0.5'], COLORS):
        data = sorted(select(metal, feh_low=flo) if corrected else select(rows, group=name), key=lambda r: number(r, 'low_kpc'))
        selected += data
        values = np.array([100*number(r, 'median_relative') for r in data])
        axes[0].plot(np.arange(6), values, 'o-', color=color, label=label)
        sparse = np.array([number(r, 'n') < 20 for r in data])
        axes[0].scatter(np.arange(6)[sparse], values[sparse], color='black', marker='x', s=65, zorder=4)
    axes[0].axhline(0, color='black', ls='--', lw=1)
    axes[0].set(ylabel='Dwarf median distance residual (%)', title='Corrected [Fe/H], valid domain' if corrected else 'Raw RVS metallicity strata')
    axes[0].legend(fontsize=9)
    for name, color in zip(['error_0_0.03', 'error_0.03_0.1', 'error_0.1_inf'], COLORS):
        data = sorted(select(rows, group=name), key=lambda r: number(r, 'low_kpc'))
        selected += data
        values = np.array([100*number(r, 'tail_1') for r in data])
        axes[1].plot(np.arange(6), values, 'o-', color=color, label=name.replace('error_', 'mag error ').replace('_', ' to '))
        sparse = np.array([number(r, 'n') < 20 for r in data])
        axes[1].scatter(np.arange(6)[sparse], values[sparse], color='black', marker='x', s=65, zorder=4)
    axes[1].set(ylabel='Severe distance error (%)', title='Optical quality')
    axes[1].legend(fontsize=9)
    for ax in axes[:2]:
        ax.set_xticks(np.arange(6), DIST)
        ax.set_xlabel('Reference distance (kpc)')
        ax.axvline(2.5, color=GRAY, ls=':')
    mig = select(read('migration'), split='development_test', reference='chosen')
    sky = [r for r in mig if r['group'].startswith('cell_')]
    selected += sky
    axes[2].scatter([number(r, 'n') for r in sky], [100*number(r, 'far_contamination') for r in sky], color=BLUE)
    for r in sky:
        axes[2].annotate(r['group'].replace('cell_', ''), (number(r, 'n'), 100*number(r, 'far_contamination')), fontsize=9)
    axes[2].set(xlabel='Chosen references per sky cell', ylabel='Selected far contamination (%)', title='Development sky variation', xscale='log')
    sources = ['distance_class', 'migration']+(['metallicity_quality'] if corrected else [])
    if extra:
        stress = read('systematic_sensitivity')
        nominal = select(stress, scenario='nominal')[0]
        altered = [r for r in stress if r['scenario'] != 'nominal']
        selected += stress
        axes[3].barh(np.arange(len(altered)), [number(r, 'gaia_observation_log_score')-number(nominal, 'gaia_observation_log_score') for r in altered], color=BLUE)
        axes[3].set_yticks(np.arange(len(altered)), [r['scenario'].replace('_', ' ')+' '+r['sign'] for r in altered], fontsize=9)
        axes[3].axvline(0, color=GRAY, lw=1)
        axes[3].set(xlabel='Change in observed-parallax log score', title='Prespecified systematic stress')
        weak = select(read('weak_flux_probability'), split='development_test', population='all', threshold='0.95')
        for group, label, color in [('W1_weak:r18_20', 'Weak W1 (N=69)', BLUE), ('W2_weak_given_W1:r18_20', 'Weak W2, W1 measured (N=3010)', ORANGE)]:
            rr = [select(weak, group=group, model=model)[0] for model in ['optical', 'signed_W1', 'signed_W1W2']]
            selected += rr
            axes[4].plot(np.arange(3), [number(r, 'brier') for r in rr], 'o-', color=color, label=label)
        axes[4].set_xticks(np.arange(3), ['Optical', '+ signed W1', '+ signed W2'], rotation=15)
        axes[4].set(ylabel='Near-event Brier score', title='18–20 mag: frozen weak-flux prototype')
        axes[4].legend(fontsize=9)
        control = select(read('nonstellar_near'), split='2')
        for kind, color in [('GALAXY', ORANGE), ('QSO', BLUE)]:
            rr = select(control, spectype=kind)
            selected += rr
            axes[5].plot([number(r, 'threshold') for r in rr], [100*number(r, 'false_near_fraction') for r in rr], 'o-', color=color, label=kind+f" (N={rr[0]['predicted_n']})")
        axes[5].set(xlabel='Near-probability threshold', ylabel='False near selection (%)', title='Quality-assessed PSF nonstellar controls')
        axes[5].legend(fontsize=9)
        sources += ['systematic_sensitivity', 'weak_flux_probability', 'nonstellar_near']
    fig.suptitle('Full PSF development strata · no display clipping · crosses: N<20', fontsize=17)
    save(fig, '06_distance_quality_sky', selected, sources,
         'Observed reference strata only. Corrected FeH uses author domain and positive formal errors; diagnostic, never application input. Sparse and selection-dependent evidence; no final-validation claim.')


def pdf_examples(by_source=False):
    rows = read('pdf_examples')
    ids = list(dict.fromkeys(r['identity'] for r in rows))
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for ax, identity in zip(axes.ravel(), ids):
        data = select(rows, identity=identity)
        x = np.array([(number(r, 'mu_low')+number(r, 'mu_high'))/2 for r in data])
        total = np.array([number(r, 'total_density') for r in data])
        channels = [('source_1', 'Gaia label contribution', BLUE), ('source_2', 'SpecDis label contribution', ORANGE)] if by_source else list(zip(CLASSES, LABELS, COLORS))
        for key, label, color in channels:
            field = 'unlabeled' if key == 'unknown' else key
            ax.plot(x, [number(r, field) for r in data], color=color, lw=3 if by_source else 1.2, label=label)
        ax.plot(x, total, color='black', lw=.8 if by_source else 1.4, label='Total with noise')
        ax.plot(x, [number(r, 'nominal_density') for r in data], color='black', lw=1, ls='--', label='Total, nominal colors')
        ref = number(data[0], 'reference_mu')
        ax.axvline(ref, color='#9467bd', ls=':', label='Chosen reference')
        ax.axvline(15, color=GRAY, ls='--', alpha=.5)
        significant = total > total.max()*.001
        lo = min(x[significant].min(), ref, 15)-.5
        hi = max(x[significant].max(), ref, 15)+.5
        ax.set(xlim=(lo, hi), xlabel='Distance modulus (mag)', ylabel='Reference mass density / mag',
               title=data[0]['category'].replace('_', ' ')+f"\nq(near)={number(data[0], 'qnear'):.3f}")
        ax.set_title(ax.get_title(), fontsize=12)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Frozen QRF joint reference contributions · illustrative PSF development cases\n'+('Source channels sum to the original PDF; training labels, not independent truth' if by_source else 'Class channels sum to the original PDF; unlabeled is not a physical population'), fontsize=15)
    save(fig, '04b_joint_source_pdf' if by_source else '04_joint_reference_pdf', rows, ['pdf_examples'],
         'Deterministic identity-hash examples within stated outcome strata. Full exported grid retained despite display limits. Solid class curves include shared-g Sobol128 noise. Reference contribution is not calibrated physical class-distance posterior.')


def near_probability():
    rel = select(read('reliability'), split='development_test', reference='chosen', boundary='point_reference')
    risk = select(read('risk_retention'), split='development_test', reference='chosen', boundary='point_reference')
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    selected = []
    for name, label, color in zip(['all', 'dwarf', 'evolved', 'unknown'], ['All', 'Dwarf', 'Giant reference', 'Unlabeled'], ['black', BLUE, ORANGE, GRAY]):
        data = select(rel, group=name)
        selected += data
        axes[0].plot([number(r, 'mean_q') for r in data], [number(r, 'observed_near') for r in data], 'o-', color=color, label=label)
        small = [r for r in data if 0 < number(r, 'n') < 20]
        axes[0].scatter([number(r, 'mean_q') for r in small], [number(r, 'observed_near') for r in small], marker='x', color='black')
        data = select(risk, group=name, evolved_weight_stress='1.0')
        selected += data
        axes[1].plot([number(r, 'completeness') for r in data], [number(r, 'far_contamination') for r in data], 'o-', color=color, label=label)
        if name == 'all':
            for r in data:
                axes[1].annotate('q>'+r['threshold'], (number(r, 'completeness'), number(r, 'far_contamination')), xytext=(-14, 20), textcoords='offset points', ha='right', fontsize=9, arrowprops=dict(arrowstyle='-', color='black', lw=.6))
    axes[0].plot([0, 1], [0, 1], ls='--', color=GRAY)
    axes[0].set(xlim=(0, 1), ylim=(0, 1), xlabel='Mean predicted near probability', ylabel='Observed reference-near fraction', title='Reference-event reliability')
    axes[0].legend(fontsize=9)
    axes[1].set(xlabel='Reference-near completeness', ylabel='Far fraction among selected', title='Risk versus retention', ylim=(0, 1))
    for factor, label, color in [('0.5', 'Giant mass x0.5', BLUE), ('1.0', 'Original mass', GRAY), ('2.0', 'Giant mass x2', ORANGE)]:
        data = select(risk, group='all', evolved_weight_stress=factor)
        selected += data
        axes[2].plot([number(r, 'threshold') for r in data], [number(r, 'far_contamination') for r in data], 'o-', color=color, label=label)
    axes[2].set(xlabel='Near-probability threshold', ylabel='Selected far contamination', title='Branch-weight stress, not calibrated priors')
    axes[2].legend(fontsize=9)
    fig.suptitle('expanded A · full PSF development reference sample · full histogram probabilities', fontsize=15)
    save(fig, '05_near_probability', selected, ['reliability', 'risk_retention'],
         'Thresholds fixed at0.5/0.8/0.95; cal selects rule, development test evaluates. Class strata are diagnostics, not an equal-per-class calibration requirement. Stress multipliers are not real-population prior estimates. Boundary-safe rows tabulated separately.')


def information():
    rows = select(read('information_scores'), split='development_test', group='all', raw_or_calibrated='calibrated')
    neighbors = read('color_neighbors')
    degraded = read('template_degradation')
    broad = (TABLES/'optical_probe.csv').exists() and (TABLES/'probe_spatial_bootstrap.csv').exists()
    fig, axes = plt.subplots(3 if broad else 2, 2, figsize=(13, 13 if broad else 9), constrained_layout=True)
    names = ['grz', 'griz', 'grzW1', 'grzW1W2']
    selected = list(rows)+degraded
    for method, label, color in [('frozen_partitions_same_labels', 'Frozen partitions, common labels', BLUE),
                                  ('fixed_forest', 'Fixed classification forest', ORANGE)]:
        data = [select(rows, bands=name, method=method)[0] for name in names]
        axes[0, 0].plot(np.arange(4), [number(r, 'log_gain_over_prior') for r in data], 'o-', color=color, label=label)
        axes[0, 1].plot(np.arange(4), [number(r, 'brier') for r in data], 'o-', color=color, label=label)
    axes[0, 0].axhline(0, color=GRAY, ls='--')
    axes[0, 0].set(ylabel='Log score gain over prior\n(nats per object)', title='Held-out development class information')
    axes[0, 0].legend(fontsize=10)
    axes[0, 1].set(ylabel='Multiclass Brier score', title='Calibrated probabilities; lower is better')
    for c, label, color in [(0, 'Dwarf', BLUE), (1, 'Giant reference', ORANGE), (2, 'Other stellar', GREEN)]:
        values = []
        for name in names:
            group = select(neighbors, bands=name, class_label=c)
            v = np.array([number(r, 'same_class_mass') for r in group])
            v = v[np.isfinite(v)]
            values.append(float(np.mean(v)) if len(v) else np.nan)
            selected.append(dict(panel='local_same_class', bands=name, class_label=c, n=len(v), mean=values[-1]))
        axes[1, 0].plot(np.arange(4), values, 'o-', color=color, label=label)
    axes[1, 0].set(ylabel='Local same-class reference mass', ylim=(0, 1), title='Full color covariance; bounded neighbor search')
    axes[1, 0].legend(fontsize=10)
    for condition, label, color in [('high_quality', 'Original bright templates', GREEN), ('target_noise', 'Same templates + target noise', GRAY)]:
        data = [select(degraded, bands=name, condition=condition)[0] for name in names]
        axes[1, 1].plot(np.arange(4), [number(r, 'log_gain_over_prior') for r in data], 'o-', color=color, label=label)
    axes[1, 1].set(ylabel='Template-conditional\nlog score gain', title='Shared band noise and common target routes')
    axes[1, 1].legend(fontsize=10)
    for ax in axes.ravel():
        ax.set_xticks(np.arange(4), names)
    sources = ['information_scores', 'color_neighbors', 'template_degradation']
    if broad:
        rr = select(read('optical_probe'), split='development_test')
        gg = ['all', 'no_W1', 'r18_20:no_W1']
        for method, label, color in [('frozen_partitions_same_labels', 'Frozen partitions, common labels', BLUE), ('fixed_forest', 'Fixed classification forest', ORANGE)]:
            data = [select(rr, method=method, group=g)[0] for g in gg]
            selected += data
            axes[2, 0].plot(np.arange(3), [number(r, 'log_gain_over_prior') for r in data], 'o-', color=color, label=label)
        axes[2, 0].set_xticks(np.arange(3), ['All grz', 'No W1', '18–20, no W1'])
        axes[2, 0].set(ylabel='Log score gain over prior', title='Broad optical probe, identical label pool')
        axes[2, 0].legend(fontsize=9)
        rr = read('probe_spatial_bootstrap')
        selected += rr
        mid = np.array([number(r, 'mean_log_gain') for r in rr])
        lo = np.array([number(r, 'spatial_bootstrap_low') for r in rr])
        hi = np.array([number(r, 'spatial_bootstrap_high') for r in rr])
        axes[2, 1].errorbar(np.arange(len(rr)), mid, yerr=[np.maximum(mid-lo, 0), np.maximum(hi-mid, 0)], fmt='o', color=BLUE, capsize=4)
        axes[2, 1].axhline(0, color=GRAY, ls='--')
        axes[2, 1].set_xticks(np.arange(len(rr)), [r['scope']+' '+r['bands'] for r in rr], rotation=20, ha='right', fontsize=10)
        axes[2, 1].set(ylabel='Classifier minus QRF log score', title='Paired development-sky resampling')
        sources += ['optical_probe', 'probe_spatial_bootstrap']
    fig.suptitle('Matched PSF band controls · identical labels and objects\nFixed forest calibration uses cal only; original QRF partition training differs', fontsize=15)
    save(fig, '03_color_information', selected, sources,
         'Single fixed classifier framework. Information gain is reference/model conditional, not a physical upper bound. Common i+W1+W2 sample has narrower coverage than optical parent. Degraded training templates are not independent validation.')


def optical_candidate():
    neighbors = read('color_neighbors_broad')
    degradation = read('template_degradation_broad')
    candidate = select(read('partition_candidate'), split='development_test')
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    selected = degradation+candidate
    for kind, label, color in [(0, 'Dwarf', BLUE), (1, 'Giant reference', ORANGE), (2, 'Other stellar', GREEN)]:
        means = []
        for name in ('all', 'no_W1', 'r18_20_no_W1'):
            rr = select(neighbors, class_label=kind)
            if name != 'all':
                rr = [r for r in rr if r['W1'] == 'False']
            if name == 'r18_20_no_W1':
                rr = [r for r in rr if 18 <= number(r, 'r') < 20]
            values = np.array([number(r, 'same_class_mass') for r in rr])
            values = values[np.isfinite(values)]
            means.append(values.mean() if len(values) else np.nan)
            selected.append(dict(panel='broad_neighbors', group=name, class_label=kind, n=len(values), mean=means[-1]))
        axes[0, 0].plot(np.arange(3), means, 'o-', label=label, color=color)
    groups = ['all', 'no_W1', 'r18_20_no_W1']
    for condition, label, color in [('original_templates', 'Original templates', GREEN), ('target_noise', 'Same templates + target noise', GRAY)]:
        rr = [select(degradation, condition=condition, group=g)[0] for g in groups]
        axes[0, 1].plot(np.arange(3), [number(r, 'log_gain_over_prior') for r in rr], 'o-', color=color, label=label)
    for ax in axes[0]:
        ax.set_xticks(np.arange(3), ['All grz', 'No W1', '18–20, no W1'])
        ax.legend(fontsize=9)
    axes[0, 0].set(ylabel='Local same-class reference mass', ylim=(0, 1), title='Broad optical covariance neighborhoods')
    axes[0, 1].set(ylabel='Template-conditional\nlog score gain', title='Actual target noise, optical-only routes')
    axes[0, 1].set_xticks(np.arange(3), ['All grz\nN=2000', 'No W1\nN=1351', '18–20, no W1\nN=23'])
    for model, label, color in [('control', 'Regression partitions', BLUE), ('candidate', 'Class partitions', ORANGE)]:
        rr = [select(candidate, model=model, group=g)[0] for g in ['all', 'dwarf', 'evolved', 'other', 'unknown']]
        axes[1, 0].plot(np.arange(5), [100*number(r, 'tail_1') for r in rr], 'o-', color=color, label=label)
        axes[1, 1].plot(np.arange(5), [number(r, 'gaia_log') for r in rr], 'o-', color=color, label=label)
    for ax in axes[1]:
        ax.set_xticks(np.arange(5), ['All', 'Dwarf', 'Giant ref.', 'Other', 'Unlabeled'])
        ax.legend(fontsize=9)
    axes[1, 0].set(ylabel='Severe distance error (%)', title='Single candidate: same training and test IDs')
    axes[1, 1].set(ylabel='Observed-parallax log score\n(symmetric-log scale)', title='Full histogram prediction; higher is better')
    axes[1, 1].set_yscale('symlog', linthresh=.5)
    axes[1, 1].set_yticks([-100, -10, -1, 0, 1], ['-100', '-10', '-1', '0', '1'])
    fig.suptitle('Broad optical evidence and one prespecified W1 partition candidate\nDevelopment evidence; no independent final-validation claim', fontsize=15)
    save(fig, '03b_optical_candidate', selected,
         ['color_neighbors_broad', 'template_degradation_broad', 'partition_candidate'],
         'Top: broad optical pool and conditional degradation of training templates. Bottom: single W1 candidate and fair refit control, identical68004 PSF members, fixed complexity, natural histogram mass. Cal guards decide; development scores are reported without tuning.')


if __name__ == '__main__':
    plt.rcParams.update(STYLE)
    distance()
    movement()
    conditions()
    if (TABLES/'information_scores.csv').exists():
        information()
    if (TABLES/'pdf_examples.csv').exists():
        pdf_examples()
        pdf_examples(by_source=True)
    if (TABLES/'reliability.csv').exists():
        near_probability()
    if (TABLES/'partition_candidate.csv').exists():
        optical_candidate()
