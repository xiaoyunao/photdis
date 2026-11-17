#!/usr/bin/env python3
"""Quantify the paired distance comparison and observed stellar-branch migration."""
import json
import os

import healpy as hp
import numpy as np

from photdis.stage1 import require_compute
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_8_direct_figures import BLUE, GREEN, ORANGE, STYLE, plt, save
from final_catalogue_benchmark import OUT as BENCHMARK, load_parent, masks, published
from final_catalogue_population import OUT as POPULATION, phase
from release_readiness_compare import available, stats
from v3_final_scan import OUT

REFERENCES = ('SpecDis', 'SPDist')
POPULATIONS = ('dwarf_feh_ge_minus1p5', 'dwarf_feh_lt_minus1p5', 'intermediate_gravity',
               'giant', 'dwarf_feh_unknown', 'other_stellar')
NAMES = ('Ordinary dwarf', 'Metal-poor dwarf', 'Intermediate gravity', 'Giant')
REGIONS = ('all', 'north', 'south_low')
COLORS = ('all', 'blue', 'intermediate_color', 'redder')
METRICS = ('nmad_fraction', 'severe_fraction', 'median_relative', 'median_modulus_mag')
SHELLS = (0, 2, 5, 10, np.inf)
BLOCK_SIDES = (8, 4)
DRAWS = 2000
SEED = 43115


def measure(ratio):
    residual = ratio - 1
    center = np.median(residual, axis=0)
    delta_mu = 5*np.log10(ratio)
    return np.stack((1.4826*np.median(np.abs(residual-center), axis=0),
                     (np.abs(delta_mu)>1).mean(axis=0), center, np.median(delta_mu, axis=0)))


def paired_blocks(ratio, blocks, rng):
    if not len(ratio):
        return 0, None, np.full((4, 2), np.nan), np.full((DRAWS, 4), np.nan)
    _, inverse, counts = np.unique(blocks, return_inverse=True, return_counts=True)
    order = np.argsort(inverse, kind='stable')
    members = np.split(order, np.cumsum(counts)[:-1])
    differences = np.full((DRAWS, 4), np.nan)
    if len(members) >= 2:
        for j, draw in enumerate(rng.integers(len(members), size=(DRAWS, len(members)))):
            m = measure(ratio[np.concatenate([members[i] for i in draw])])
            differences[j] = m[:, 0]-m[:, 1]
    return len(members), float(counts.max()/len(ratio)), measure(ratio), differences


def benchmark(t, q, dest):
    rng = np.random.default_rng(SEED)
    p, bj = q['Photdis'], q['BJ_photogeometric']
    both = available(p) & available(bj)
    m = masks(t)
    rows, draws = [], {}
    sky = {n:hp.ang2pix(n, t['RA'], t['DEC'], lonlat=True, nest=True) for n in BLOCK_SIDES}
    for source in REFERENCES:
        d = t[source.lower()+'_kpc']
        parent = both & (d>=2) & (d<5)
        for group in ('weak_gaia_dwarfs', 'weak_gaia_ordinary', 'weak_gaia_poor'):
            for region in REGIONS:
                for color in COLORS:
                    use = parent & m[group] & m[region] & m[color]
                    for n, cells in sky.items():
                        if n != BLOCK_SIDES[0] and (region, color) != ('all', 'all'):
                            continue
                        ratios = np.column_stack((p[use, 1], bj[use, 1]))/d[use, None]
                        k, largest, point, distribution = paired_blocks(ratios, cells[use], rng)
                        key = f'{source}_{group}_{region}_{color}_{n}'
                        draws[key] = distribution
                        lo, hi = np.quantile(distribution, [.025, .975], axis=0)
                        for j, name in enumerate(METRICS):
                            row = dict(reference=source, cohort=group, region=region, color=color,
                                       block_nside=n, n=int(use.sum()), n_blocks=k, largest_block_fraction=largest,
                                       metric=name, photdis=point[j,0], bj=point[j,1],
                                       photdis_minus_bj=point[j,0]-point[j,1], lo95=lo[j], hi95=hi[j])
                            rows.append(row)
    csv_rows(dest/'paired_bootstrap.csv', rows)
    atomic_npz(dest/'paired_draws.npz', **draws)
    counts = [r for r in rows if r['cohort']=='weak_gaia_dwarfs' and r['region']=='all'
              and r['color']=='all' and r['block_nside']==8 and r['metric']==METRICS[0]]
    assert [(r['reference'], r['n']) for r in counts] == [('SpecDis',9355), ('SPDist',8927)]
    return rows


def populations(t, groups, dest):
    q = t['qrf_quantiles_kpc'][:, 1:4]
    support = (t['state']>=2) & available(q)
    selected = (t['state']==4) & support
    pred = q[:, 1]
    rows, migration = [], []
    for source in REFERENCES:
        d = t[source.lower()+'_kpc']
        for pop in (*POPULATIONS, 'all_labeled', 'class_conflict', 'qualified_nonstellar', 'unclassified'):
            cat = np.isin(groups, POPULATIONS) if pop=='all_labeled' else (groups==pop)
            for region in REGIONS:
                base = cat & np.isfinite(d) & ((t['region']==region) if region!='all' else True)
                for a, b in ((0, np.inf), *zip(SHELLS[:-1], SHELLS[1:])):
                    k = base & (d>=a) & (d<b)
                    k_support, k_chosen = k & support, k & selected
                    k_core = k_chosen & (pred>=2) & (pred<5)
                    for stage, admitted in (('supported', k_support), ('working', k_chosen), ('core_2_5', k_core)):
                        rows.append(dict(reference=source, region=region, population=pop,
                            ref_lo=a, ref_hi=b, stage=stage, parent_n=int(k.sum()), supported_n=int(k_support.sum()),
                            retention_from_parent=admitted.sum()/k.sum() if k.any() else None,
                            retention_from_supported=admitted.sum()/k_support.sum() if k_support.any() else None,
                            **stats(q[admitted], d[admitted])))
                    for a_pred, b_pred in zip(SHELLS[:-1], SHELLS[1:]):
                        in_pred = (pred>=a_pred) & (pred<b_pred)
                        migration.append(dict(reference=source, region=region, population=pop,
                            ref_lo=a, ref_hi=b, pred_lo=a_pred, pred_hi=b_pred,
                            supported=int((k_support & in_pred).sum()), working=int((k_chosen & in_pred).sum())))
    csv_rows(dest/'population.csv', rows)
    csv_rows(dest/'migration.csv', migration)
    return rows, migration


def physics_figure(rows, migration, dest):
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), layout='constrained')
    for i, ref in enumerate(REFERENCES):
        base = dict(reference=ref, region='all', ref_lo=2, ref_hi=5)
        a, b = ([next(r for r in rows if r['population']==p and r['stage']==s
                     and all(r[key]==val for key,val in base.items())) for p in POPULATIONS[:4]]
                for s in ('supported', 'core_2_5'))
        y = np.arange(len(NAMES))
        ax = axes[i, 0]
        x = [r['retention_from_supported'] or 0 for r in b]
        ax.barh(y, 100*np.array(x), color=BLUE, height=.55)
        for j, (aa, bb) in enumerate(zip(a,b)):
            ax.annotate(f"{bb['n']:,}/{aa['n']:,}", (100*x[j], j), xytext=(4,0),
                        textcoords='offset points', va='center', fontsize=9)
        ax.set(title=f'{ref} · core retention', xlabel='Selected / supported (%)',
               yticks=y, yticklabels=NAMES, ylim=(3.55,-.55), xlim=(0,126))
        for k, key, name in ((1,'bias','Median relative error (%)'), (2,'tail','|Modulus error| > 1 mag (%)')):
            ax = axes[i, k]
            for data, shift, color, label in ((a,-.12,BLUE,'Supported parent'),(b,.12,ORANGE,'Core 2–5 kpc')):
                values = [100*r[key] if r[key] is not None else np.nan for r in data]
                ax.scatter(values, y+shift, color=color, label=label, zorder=4, s=31)
            if k == 1:
                ax.axvline(0, color='0.5', linestyle='--', linewidth=1)
            ax.set(xlabel=name, yticks=y, yticklabels=[], ylim=(3.55,-.55))
            if i==0 and k==1:
                ax.legend(loc='best', fontsize=9)
        ax = axes[i, 3]
        hist = np.array([[sum(r['working'] for r in migration if r['population']==p and r['reference']==ref
                             and r['region']=='all' and r['pred_lo']==2 and r['ref_lo']==d0)
                          for d0 in SHELLS[:-1]] for p in POPULATIONS[:4]])
        left = np.zeros(4)
        for j, (color, label) in enumerate(zip((BLUE, GREEN, ORANGE, '#81649c'),
                ('<2 kpc', '2–5 kpc', '5–10 kpc', '≥10 kpc'))):
            x = np.divide(100*hist[:,j], hist.sum(axis=1), out=np.zeros(4), where=hist.sum(axis=1)>0)
            ax.barh(y, x, left=left, color=color, height=.58, label=label)
            left += x
        ax.set(title=f'{ref} · arrival into the core', xlabel='Reference distance composition (%)',
               yticks=y, yticklabels=[], ylim=(3.55,-.55), xlim=(0,100))
        if i==0:
            ax.legend(fontsize=8, ncol=2)
    fig.suptitle('Luminosity class changes which stars enter the 2–5 kpc sample')
    save(fig, dest, 'F20_population_physics',
         [r for r in rows if r['population'] in POPULATIONS[:4] and r['region']=='all'
          and r['ref_lo']==2 and r['ref_hi']==5])


def main():
    require_compute()
    dest = OUT/'science'
    dest.mkdir(parents=True, exist_ok=True)
    t, meta = load_parent()
    q, _, inputs = published(t, meta['outputs']['ids.npy'])
    inputs['benchmark_parent'] = sha(BENCHMARK/'parent.json')
    benchmark(t, q, dest)
    groups = phase(t, inputs)
    rows, migration = populations(t, groups, dest)
    mark = POPULATION/'complete.json'
    inputs['conditional_lf'] = sha(mark)
    for name, digest in json.loads(mark.read_text())['outputs'].items():
        verified(POPULATION/name, digest)
    atomic_npz(dest/'plot_reference.npz', **{k:t[k] for k in ('TYPE','ls_id','RA','DEC','region',
               'specdis_kpc','spdist_kpc')}, group=groups)
    with plt.rc_context(STYLE):
        physics_figure(rows, migration, dest)
    save_json(dest/'complete.json', dict(inputs=inputs, job=os.getenv('SLURM_JOB_ID'),
        seed=SEED, draws=DRAWS, block_nsides=BLOCK_SIDES,
        outputs={p.name:sha(p) for p in dest.iterdir() if p.name!='complete.json'}))


if __name__ == '__main__':
    main()
