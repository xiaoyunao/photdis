#!/usr/bin/env python3
"""English scientific plates from compact v2.2 audit tables only."""
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
TABLES = ROOT/'manifests/v2_2'; FIG = ROOT/'figures/v2_2'
COLORS = [BLUE, ORANGE, GREEN, GRAY]


def read(name):
    with open(TABLES/(name+'.csv')) as f:
        return list(csv.DictReader(f))


def sel(rows, **kw):
    return [r for r in rows if all(r.get(k) == str(v) for k, v in kw.items())]


def val(r, key):
    return float(r.get(key) or 'nan')


def save(fig, name, sources, caption):
    FIG.mkdir(parents=True, exist_ok=True); dest = TABLES/'figures'; dest.mkdir(exist_ok=True)
    for a in fig.findobj(match=Text):
        assert not re.search(r'[\u3400-\u9fff]', a.get_text()), a.get_text()
    fig.savefig(FIG/(name+'.png'), dpi=300, bbox_inches='tight')
    fig.savefig(FIG/(name+'.pdf'), dpi=300, bbox_inches='tight'); plt.close(fig)
    rows = [dict(source=source, **r) for source in sources for r in read(source)]
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(dest/(name+'.csv'), 'w') as f:
        w = csv.DictWriter(f, fields, lineterminator='\n'); w.writeheader(); w.writerows(rows)
    (dest/(name+'.json')).write_text(json.dumps(dict(name=name, caption=caption,
        selection='Verified DR11 PSF; original scientific denominators; no display clipping; no independent final validation',
        sources={s: hashlib.sha256((TABLES/(s+'.csv')).read_bytes()).hexdigest() for s in sources}), indent=2)+'\n')


def reliability():
    r = sel(read('reliability'), group='all', reference='chosen', boundary='point_reference')
    risk = read('selection_risk')
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for split, label, color in [('cal', 'Calibration', BLUE), ('development_test', 'Development', ORANGE)]:
        rr = sel(r, split=split)
        ax[0].plot([val(x, 'mean_q') for x in rr], [val(x, 'observed_near') for x in rr], 'o-', label=label, color=color)
        ax[1].stairs([val(x, 'n') for x in rr], np.linspace(0, 1, 11), color=color, label=f'{label}: N={sum(int(x["n"]) for x in rr):,}')
        rr = sel(risk, split=split)
        x = np.arange(3)+(0 if split == 'cal' else .08)
        y = np.array([val(z, 'far_contamination')*100 for z in rr])
        low = np.array([val(z, 'wilson_low')*100 for z in rr]); high = np.array([val(z, 'wilson_high')*100 for z in rr])
        ax[2].errorbar(x, y, yerr=[y-low, high-y], fmt='o', color=color, label=label+' (Wilson 95%)')
    ax[0].plot([0, 1], [0, 1], '--', color=GRAY); ax[0].set(xlabel='Mean predicted near probability', ylabel='Observed near fraction', xlim=(0, 1), ylim=(0, 1)); ax[0].legend(fontsize=10)
    ax[1].set(yscale='log', xlabel='Predicted near probability', ylabel='Reference count'); ax[1].legend(fontsize=9)
    ax[2].set(xticks=np.arange(3), xticklabels=['0.5', '0.8', '0.95'], xlabel='Near-probability threshold', ylabel='Far contamination (%)'); ax[2].legend(fontsize=9)
    fig.suptitle('expanded A · endpoint-complete reliability and selection risk')
    save(fig, '01_reliability', ['reliability', 'selection_risk'], 'Tolerance-only endpoint repair restores308cal and379development chosen rows. Every stratum conserves its denominator. Threshold decisions unchanged; sky-bootstrap intervals are in the table.')


def selected():
    data = sel(read('selected_distance'), split='development_test', reference='chosen', low_kpc=0, high_kpc='inf')
    names = ['all', 'dwarf', 'evolved', 'other', 'unknown', 'dwarf_corrected_FeH_-inf_-1.5']
    labels = ['All', 'Dwarf', 'Giant ref.', 'Other', 'Unlabeled', 'Metal-poor\ndwarf']
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for threshold, color in zip([.5, .8, .95], COLORS):
        rr = [sel(data, group=g, threshold=threshold)[0] for g in names]
        for ax, key, ylabel in zip(axes.ravel(), ['median_relative', 'tail_1', 'coverage68', 'nmad_relative'], ['Median distance residual (%)', 'Severe fraction (%)', '68% interval coverage (%)', 'Relative-distance NMAD (%)']):
            ax.plot(np.arange(6), [100*val(r, key) for r in rr], 'o-', color=color, label=f'q > {threshold}')
            ax.set(xticks=np.arange(6), xticklabels=labels, ylabel=ylabel); ax.tick_params(axis='x', labelsize=10)
    axes[0, 0].axhline(0, color=GRAY, ls='--'); axes[1, 0].axhline(68, color=GRAY, ls='--')
    axes[0, 1].legend(fontsize=10); fig.suptitle('Distance quality AFTER near-probability selection · expanded A\nClasses and corrected [Fe/H] are diagnostics only')
    save(fig, '02_selected_distance', ['selected_distance'], 'All selected objects, including far contaminants. Severe=|delta mu|>1mag. Low contamination is not precise individual distance. Unknown/other retained. Metal-poor means correctedFeH<−1.5 within the established quality/domain cuts.')


def application():
    data = read('application_summary'); fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for model, color in [('expanded_A', BLUE), ('baseline_v1', ORANGE)]:
        rr = [sel(data, model=model, population='application', threshold=q, group='all')[0] for q in (.5, .8, .95)]
        axes[0, 0].plot([.5, .8, .95], [val(r, 'weighted_n')/1e6 for r in rr], 'o-', label=model, color=color)
        r = rr[-1]; x = 0 if model == 'expanded_A' else 1; bottom = 0
        for key, c, label in [('evidence_zero_weight', GRAY, 'No reference'), ('evidence_sparse_weight', ORANGE, '1–19 references'), ('evidence_20plus_weight', GREEN, '20+ references')]:
            v = val(r, key)/val(r, 'weighted_n')*100
            axes[0, 1].bar(x, v, bottom=bottom, color=c, label=label if x == 0 else None); bottom += v
    axes[0, 0].set(xlabel='Near-probability threshold', ylabel='Weighted selected count (million)'); axes[0, 0].legend(fontsize=10)
    axes[0, 1].set(xticks=[0, 1], xticklabels=['expanded A', 'baseline v1'], ylabel='High-q selected weight (%)'); axes[0, 1].legend(fontsize=9)
    for pop, color in [('application', BLUE), ('development', ORANGE)]:
        for ax, groups, labels in [(axes[1, 0], ['r_-inf_18', 'r_18_20', 'r_20_22', 'r_22_25'], ['<18', '18–20', '20–22', '22–25']), (axes[1, 1], ['error_0_0.03', 'error_0.03_0.1', 'error_0.1_0.2', 'error_0.2_inf'], ['<0.03', '0.03–0.1', '0.1–0.2', '0.2+'])]:
            rr = [sel(data, model='expanded_A', population=pop, threshold=.95, group=g)[0] for g in groups]
            total = val(sel(data, model='expanded_A', population=pop, threshold=.95, group='all')[0], 'weighted_n')
            ax.plot(np.arange(4), [val(r, 'weighted_n')/total*100 for r in rr], 'o-', color=color, label=pop)
            ax.set(xticks=np.arange(4), xticklabels=labels, ylabel='High-q population share (%)'); ax.legend(fontsize=10)
    axes[1, 0].set_xlabel('Observed r magnitude'); axes[1, 1].set_xlabel('Maximum optical magnitude error')
    fig.suptitle('Fixed 26,000-object weighted PSF application sample\nq > 0.95; cell occupancy measures evidence availability, not accuracy')
    save(fig, '03_application', ['application_summary'], 'expanded_A is shared by development/application. baseline_v1 reproduces actual catalog routes. Counts refer to13DECam fields, not wholeDR11. Evidence cells fix magnitude/colors/error/WISE; no spatial independence is implied.')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    groups = ['all', 'no_W1', 'W1_only', 'W1W2']
    for pop, color in [('application', BLUE), ('development', ORANGE)]:
        rr = [sel(data, model='expanded_A', population=pop, threshold=.95, group=g)[0] for g in groups]
        for ax, key, label in zip(axes, ['mean_width68', 'ambiguous_fraction', 'mean_unlabeled_contribution'], ['Mean 68% width (mag)', 'Multimodal fraction', 'Mean unlabeled contribution']):
            ax.plot(np.arange(4), [val(r, key) for r in rr], 'o-', color=color, label=pop); ax.set(xticks=np.arange(4), xticklabels=['All', 'No W1', 'W1 only', 'W1/W2'], ylabel=label); ax.tick_params(axis='x', labelsize=10)
    axes[0].legend(fontsize=10); fig.suptitle('High-q PDF structure · same expanded A model')
    save(fig, '03b_pdf_structure', ['application_summary'], 'Weighted application versus reference-selected development. Unlabeled contribution is missing class evidence, not a physical fourth population.')


def noise():
    data = read('noise_transfer'); fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for ax, key, label in zip(axes.ravel(), ['point_log_gain_over_prior', 'averaged_log_gain_over_prior', 'tail_1', 'coverage68'], ['Point-class log gain (nats)', 'Marginal-class log gain (nats)', 'Distance severe fraction (%)', 'Distance 68% coverage (%)']):
        scale = 100 if key in ('tail_1', 'coverage68') else 1
        for condition, color in [('shifted_clean', BLUE), ('target_noise', ORANGE)]:
            means, lows, highs = [], [], []
            for lo in (18, 20, 22):
                rr = sel(data, condition=condition, group='all', r_low=lo)
                vv = [val(r, key)*scale for r in rr]; means.append(np.mean(vv)); lows.append(np.min(vv)); highs.append(np.max(vv))
            ax.plot(np.arange(3), means, 'o-', color=color, label=condition.replace('_', ' ')); ax.fill_between(np.arange(3), lows, highs, color=color, alpha=.18)
        ax.set(xticks=np.arange(3), xticklabels=['18–20', '20–22', '22–25'], xlabel='Matched target r magnitude', ylabel=label)
        ax.axhline(68 if key == 'coverage68' else 0, color=GRAY, ls='--')
    axes[0, 0].legend(fontsize=10); fig.suptitle('Held-out templates · fixed full Sobol128 distance workflow\nFour target-noise repeats; bands show min–max, not independent-sample confidence')
    save(fig, '04_noise_transfer', ['noise_transfer', 'noise_cluster_bootstrap'], 'Templates disjoint from fitting; matched observed colors, WISE state and target errors. Same identity retains role. Shifted reference distances are conditional simulations, not new faint truth. Paired source-cluster intervals provided inCSV.')


def far():
    data = sel(read('far_shape'), split='development_test', group='evolved')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for kind, color in zip(['chosen', 'Gaia', 'specdis', 'spdist'], COLORS):
        rr = [sel(data, reference=kind, low_kpc=lo)[0] for lo in (10, 20)]
        for ax, key, label in zip(axes, ['mean_reference_window_mass', 'conditional_coverage68', 'conditional_median_relative'], ['Mean mass within reference ±1 mag', 'Oracle-class 68% coverage', 'Oracle-class median relative residual']):
            ax.plot([0, 1], [val(r, key) for r in rr], 'o-', color=color, label=kind+' (N='+str(int(val(rr[0], 'n')))+'/'+str(int(val(rr[1], 'n')))+')')
            ax.set(xticks=[0, 1], xticklabels=['10–20', '20–50'], xlabel='Reference distance (kpc)', ylabel=label)
    axes[0].legend(fontsize=10); axes[1].axhline(.68, color=GRAY, ls='--'); axes[2].axhline(0, color=GRAY, ls='--')
    fig.suptitle('Far giant-reference branches · fixed expanded A\nShared reference sources; oracle class is diagnostic only')
    save(fig, '05_far_shape', ['far_shape', 'observable_branch_frequency', 'color_selection'], 'Separate Gaia/SpecDis/SPDist denominators and uncertainty windows. No new independent distance-scale confirmation. Conditional class distributions and their relative weights remain distinct questions; no fitted reweighting.')


def support():
    tr = read('training_luminosity_support'); gain = read('W1_information_contribution'); inv = read('validation_inventory')
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for group, color in zip(['dwarf', 'evolved', 'other', 'unlabeled'], COLORS):
        rr = sel(tr, group=group); axes[0].plot(np.arange(8), [val(r, 'n') for r in rr], 'o-', color=color, label=group)
    axes[0].set(yscale='log', xticks=np.arange(8), xticklabels=['<−2', '−2–0', '0–2', '2–4', '4–6', '6–8', '8–10', '10+'], xlabel='Training absolute g magnitude', ylabel='Verified-PSF distance labels'); axes[0].tick_params(axis='x', rotation=45, labelsize=9); axes[0].legend(fontsize=9)
    axes[1].bar(np.arange(3), [val(r, 'total_contribution') for r in gain], color=COLORS[:3]); axes[1].set(xticks=np.arange(3), xticklabels=['Dwarf', 'Giant ref.', 'Other'], ylabel='Contribution to total W1 log gain'); axes[1].axhline(0, color=GRAY, lw=1)
    for wise, color in [('False', BLUE), ('True', ORANGE)]:
        rr = sel(inv, W1=wise)[:4]; axes[2].plot(np.arange(4), [val(r, 'n') for r in rr], 'o-', label='No W1' if wise == 'False' else 'W1', color=color)
    axes[2].set(yscale='log', xticks=np.arange(4), xticklabels=['<18', '18–20', '20–22', '22–25'], xlabel='Observed r magnitude', ylabel='Unused identities (existing sky)'); axes[2].legend(fontsize=9)
    fig.suptitle('Retain unlabeled luminosity evidence; distinguish validation availability')
    save(fig, '06_evidence_limits', ['training_luminosity_support', 'W1_information_contribution', 'validation_inventory'], 'Unused42076identity pool has no current qualified class/distance labels; weak-parallax eligibility is distinct. Buffered new-sky reservation remains5. These are acquisition candidates, not completed independent validation.')


if __name__ == '__main__':
    plt.rcParams.update(STYLE)
    for function in (reliability, selected, application, noise, far, support):
        function()
