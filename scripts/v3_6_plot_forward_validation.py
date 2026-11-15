#!/usr/bin/env python3
"""Render stratified signed-Gaia diagnostics and conditional LF contamination."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
from matplotlib.ticker import PercentFormatter
import numpy as np

from photdis_figure_style import BLUE, GRAY, GREEN, ORANGE, STYLE
from v3_6_plot_shell_retention import ROOT, sha


WEAK = ROOT/'data/v3_6/weak_gaia'
LF = ROOT/'data/v3_6/lf_sensitivity/report'
DISTANCE = ROOT/'data/v3_6/distance_evidence'
CATALOG = ROOT/'data/v3_6/catalog_observables'
OUT = ROOT/'data/v3_6/figures/forward_validation'
CUTS = ('primary','strict')
HISTORY = 'outside_recorded_history'
GROUPS = {'all':'All', 'r18_20':'r = 18–20', 'r18_20_blue':'Faint blue',
          'r18_20_pred_2_5':'Faint,\nassigned 2–5', 'r18_20_pred_5_10':'Faint,\nassigned 5–10'}
MODELS = {'QRF':('QRF',BLUE), 'nominal':('Nominal SED',ORANGE), 'uniform_mu':('Broad prior',GRAY)}
SCENARIOS = {'nominal':'Nominal', 'empirical_galactic':'Empirical Galactic',
             'rgb_disk_rich':'Disk-rich RGB', 'rgb_halo_rich':'Halo-rich RGB',
             'more_faint_other':'More faint other'}
SHELLS = ('0–2','2–5','5–10')
ALL_SHELLS = (*SHELLS,'10–20','20–50','>50')


def read(directory,names):
    marker = directory/'complete.json'
    receipt = json.loads(marker.read_text())
    inputs = {str(marker.relative_to(ROOT)):sha(marker)}
    result = {}
    for name in names:
        path = directory/f'{name}.csv'
        digest = sha(path)
        if digest!=receipt['outputs'][path.name]:
            raise ValueError(f'Input differs from its publication receipt: {path}')
        with path.open() as f:
            result[name] = list(csv.DictReader(f))
        inputs[str(path.relative_to(ROOT))] = digest
    return result,inputs


def deliver(fig,name,rows,inputs,caption):
    OUT.mkdir(parents=True,exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    tab = OUT/f'{name}.csv'
    with tab.open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=fields,lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    files = [tab]
    for ext in ('png','pdf'):
        path = OUT/f'{name}.{ext}'
        fig.savefig(path,dpi=300)
        files.append(path)
    plt.close(fig)
    marker = OUT/f'{name}.json'
    marker.write_text(json.dumps(dict(caption=caption,dpi=300,inputs=inputs,
        outputs={str(path.relative_to(ROOT)):sha(path) for path in files}),indent=2)+'\n')


def finite(values):
    return np.array([float(x) if x not in (None,'') else np.nan for x in values])


def weak():
    result,inputs = read(WEAK,('scores','pit','forward_grid'))
    for cut in CUTS:
        rows = {name:[r for r in data if r['selection']==cut and r['history']==HISTORY]
                for name,data in result.items()}
        scores = {(r['group'],r.get('model')):r for r in rows['scores'] if r.get('model') in MODELS}
        counts = {g:int(scores.get((g,'QRF'),{}).get('n',0)) for g in GROUPS}
        labels = [f'{title}\nN={counts[k]:,}' for k,title in GROUPS.items()]
        with plt.rc_context(STYLE):
            fig,grid = plt.subplots(2,2,figsize=(15.1,9.0),layout='constrained')
            a,b,c,d = grid.ravel()
            x = np.arange(len(GROUPS))
            for i,(model,(label,color)) in enumerate(MODELS.items()):
                dscore = finite([scores.get((g,model),{}).get('relative_to_uniform_mu') for g in GROUPS])
                negative = finite([scores.get((g,model),{}).get('predicted_negative_fraction') for g in GROUPS])
                dx = (i-1)*.22
                a.bar(x+dx,dscore,width=.21,color=color,label=label)
                b.scatter(x+dx,negative,s=36,c=color,label=label)
                data = sorted((r for r in rows['pit'] if r['group']=='r18_20' and r['model']==model),
                              key=lambda r:float(r['lo']))
                if data:
                    edges = [float(data[0]['lo']),*[float(r['hi']) for r in data]]
                    c.stairs(finite([r['weighted_fraction'] for r in data]),edges,color=color,lw=2)
            observed = finite([scores.get((g,'QRF'),{}).get('observed_negative_fraction') for g in GROUPS])
            b.scatter(x,observed,marker='x',color='black',label='Observed',s=57,zorder=4)
            a.axhline(0,color=GRAY,ls='--',lw=1)
            a.set(title='Conditional log score',ylabel='Gain over broad prior [nat per star]')
            a.legend(fontsize=10,frameon=False)
            b.set(title='Negative parallaxes',ylabel='Fraction',ylim=(0,1))
            b.yaxis.set_major_formatter(PercentFormatter(1))
            b.legend(fontsize=10,frameon=False)
            for ax in (a,b):
                ax.set_xticks(x,labels,fontsize=10)
                ax.xaxis.grid(False)
            c.axhline(.1,color='black',ls=':',lw=1)
            c.set(title='Predictive calibration, r = 18–20',xlabel='Conditional probability integral transform',
                  ylabel='Weighted bin fraction',xlim=(0,1))
            prior = [r for r in rows['forward_grid'] if r['group']=='r18_20_pred_2_5']
            for odds,color in zip((.25,1.,4.),(GREEN,BLUE,ORANGE)):
                variants = [r for r in prior if float(r['original_far_branch_odds_multiplier'])==odds]
                scales = sorted({float(r['parallax_scale']) for r in variants})
                edges = [[float(r['relative_to_frozen_qrf']) for r in variants if float(r['parallax_scale'])==k]
                         for k in scales]
                central = {float(r['parallax_scale']):float(r['relative_to_frozen_qrf'])
                           for r in variants if float(r['parallax_offset_mas'])==0.}
                if scales:
                    d.fill_between(scales,[min(v) for v in edges],[max(v) for v in edges],color=color,alpha=.15)
                    d.plot(scales,[central[k] for k in scales],'-o',ms=4,color=color,label=f'{odds:g}× far odds')
            d.axhline(0,color='black',ls=':',lw=1)
            d.set(title='Faint stars assigned to 2–5 kpc',xlabel='Parallax scale multiplier',
                  ylabel='Gain over frozen QRF [nat per star]')
            if prior:
                d.legend(fontsize=10,frameon=False)
            else:
                d.text(.5,.55,'No qualified comparison',transform=d.transAxes,ha='center')
            fig.suptitle(cut.title()+' selection · weak Gaia outside recorded history')
            tab = [dict(table=key,**r) for key,data in rows.items() for r in data]
            caption = ('Stratified weak-parallax evaluation outside recorded validation and training history. '
                       'Scores and PIT condition on observed corrected S/N < 5. Negative parallaxes remain '
                       'in every score. The stress grid varies QRF far-branch odds and the parallax scale.'
                       ' Shading spans the tested common zero-point shifts. N is the observed draw count; '
                       'the tables also give weighted and effective counts. Brightness and assigned-distance '
                       'cohorts can overlap.')
            deliver(fig,f'F08_conditional_weak_gaia_{cut}',tab,inputs,caption)


def lf():
    result,inputs = read(LF,('shell_counts','migration','spatial_counts'))
    estimands = {'far_giant_assigned_near':'Giant beyond 10 kpc', 'far_any_assigned_near':'Any star beyond 10 kpc'}
    key = {(r['selection'],r['scenario'],r['shell_kpc'],r['estimand']):r for r in result['shell_counts']
           if r['sky']=='all_sky' and r['estimand'] in estimands}
    with plt.rc_context(STYLE):
        fig,axes = plt.subplots(2,2,figsize=(13.9,8.7),layout='constrained',sharex=True,sharey='col')
        x = np.arange(len(SCENARIOS))
        for i,cut in enumerate(CUTS):
            for j,(estimand,label) in enumerate(estimands.items()):
                ax = axes[i,j]
                for k,(shell,color) in enumerate(zip(SHELLS,(BLUE,GREEN,ORANGE))):
                    data = [key[cut,model,shell,estimand] for model in SCENARIOS]
                    frac = finite([r['fraction'] for r in data])
                    lo,hi = [finite([r[f'fraction_sky_95_{side}'] for r in data]) for side in ('lo','hi')]
                    points = x+(k-1)*.20
                    ax.scatter(points,frac,c=color,s=27,label=shell+' kpc')
                    ax.vlines(points,lo,hi,color=color,alpha=.6,lw=2)
                ax.set(title=cut.title()+' · '+label,ylabel='Fraction of QRF assignment')
                ax.yaxis.set_major_formatter(PercentFormatter(1))
                ax.xaxis.grid(False)
                if i==1:
                    ax.set_xticks(x,SCENARIOS.values(),rotation=20,ha='right',fontsize=11)
        axes[0,0].legend(fontsize=10,frameon=False)
        caption = ('Expected contamination of three QRF-assigned shells under each population scenario, '
                   'weighted to each photometric cut. The two columns respectively require the latent '
                   'class to be giant or allow every modeled class. Vertical intervals are 95% paired '
                   'sky-block bootstrap intervals conditional on the stated scenario. The source table '
                   'also retains estimated design errors, predicted class and distance migration, and '
                   'the spatial subdivisions.')
        rows = [dict(table=kind,**r) for kind,data in result.items() for r in data]
        deliver(fig,'F09_conditional_lf_contamination',rows,inputs,caption)


def migration():
    result,inputs = read(DISTANCE,('migration',))
    data = result['migration']
    sources = dict.fromkeys(r['source'] for r in data)
    for source in sources:
        rows = [r for r in data if r['source']==source]
        field = {(r['selection'],r['reference_class'],r['reference_shell'],r['predicted_shell']):r for r in rows}
        with plt.rc_context(STYLE):
            fig,axes = plt.subplots(2,3,figsize=(14.6,8.9),layout='constrained',sharex=True,sharey=True)
            for i,label in enumerate(('dwarf','giant')):
                for j,cut in enumerate(('supported',*CUTS)):
                    ax = axes[i,j]
                    grid = [[field[cut,label,a,b] for b in ALL_SHELLS] for a in ALL_SHELLS]
                    hard = np.asarray([[int(r['hard_count']) for r in line] for line in grid])
                    weight = np.asarray([[float(r['qrf_weighted_count']) for r in line] for line in grid])
                    n = hard.sum(axis=1)
                    denominator = weight.sum(axis=1)
                    np.testing.assert_allclose(n,denominator,rtol=0,atol=1e-5)
                    frac = np.divide(weight,denominator[:,None],out=np.full_like(weight,np.nan),
                                     where=denominator[:,None]>0)
                    im = ax.imshow(frac,vmin=0,vmax=1,cmap='Blues',aspect='auto',interpolation='nearest')
                    for a,b in zip(*np.nonzero(hard)):
                        color = 'white' if frac[a,b]>.58 else 'black'
                        ax.text(b,a,f'{hard[a,b]:,}',color=color,ha='center',va='center',fontsize=9)
                    if n.sum()==0:
                        ax.text(.5,.53,'No qualified references',transform=ax.transAxes,ha='center')
                    ax.set_title(f'{cut.title()} · {label}s · N={n.sum():,}',fontsize=13)
                    ax.set_xticks(range(6),ALL_SHELLS,rotation=28,ha='right',fontsize=10)
                    ax.set_yticks(range(6),ALL_SHELLS,fontsize=10)
                    ax.grid(False)
                    if j==0:
                        ax.set_ylabel('Reference distance [kpc]',fontsize=13)
                    if i==1:
                        ax.set_xlabel('QRF assigned distance [kpc]',fontsize=13)
            fig.colorbar(im,ax=axes,shrink=.8,label='QRF posterior shell fraction')
            fig.suptitle(f'{source} · reference to assigned distance')
            caption = ('Each reference row belongs to one reference-distance shell. Color sums the '
                       'full QRF PDF over assigned shells and divides by the number of qualified '
                       'reference rows in that reference shell. Annotations count hard median '
                       'assignments. The table also includes unknown and other classes in '
                       'the same qualified cohort. Sources can share stars. The displayed '
                       'fractions use the reference-matched denominator for this survey.')
            deliver(fig,f'F09_direct_migration_{source}',rows,inputs,caption)


def cartesian():
    marker = CATALOG/'complete.json'
    receipt = json.loads(marker.read_text())
    path = CATALOG/'observables.npz'
    digest = sha(path)
    if digest!=receipt['outputs'][path.name]:
        raise ValueError('Cartesian summary differs from its receipt')
    with np.load(path,allow_pickle=False) as z:
        data,edges = z['bright_red_cartesian'],z['cartesian_edges']
    np.testing.assert_array_equal(data[:,0].sum(axis=(1,2)),receipt['bright_red_d50_0_2'])
    rows = []
    with plt.rc_context(STYLE):
        fig,axes = plt.subplots(2,2,figsize=(11.8,9.4),layout='constrained',sharex=True,sharey=True)
        scale = LogNorm(1,max(2,int(np.max(data[1:]))))
        for i,cut in enumerate(CUTS,1):
            for j,plane in enumerate(('XY','XZ')):
                ax = axes[i-1,j]
                h = data[i,j]
                loc = ax.pcolormesh(edges,edges,np.ma.masked_equal(h.T,0),cmap='viridis',norm=scale,rasterized=True)
                ax.add_patch(Circle((0,0),2.,fill=False,color=GRAY,ls='--',lw=.8))
                ax.set(xlim=(-2,2),ylim=(-2,2),aspect='equal',title=f'{cut.title()} · {plane} · N={h.sum():,}',
                       ylabel=f'{plane[1]} [kpc]')
                if i==2:
                    ax.set_xlabel('X [kpc]')
                for a,b in zip(*np.nonzero(h)):
                    rows.append(dict(selection=cut,plane=plane,x_lo=edges[a],x_hi=edges[a+1],
                                     other_lo=edges[b],other_hi=edges[b+1],stars=int(h[a,b])))
        fig.colorbar(loc,ax=axes,shrink=.84,label='Stars per 40 pc × 40 pc bin')
        fig.suptitle('Bright red analogs · QRF median distance below 2 kpc')
        inputs = {str(marker.relative_to(ROOT)):sha(marker),str(path.relative_to(ROOT)):digest}
        caption = ('Heliocentric projection for the frozen bright-red observable domain. The QRF '
                   'median is required to lie below 2 kpc. X points toward the Galactic center, Y '
                   'follows Galactic rotation, and Z points north. Counts use hard point assignments, '
                   'with the same count scale for both catalog cuts. The footprint and '
                   'photometric selection limit which lines of sight appear in the figure.')
        deliver(fig,'F10_near_cartesian_catalog',rows,inputs,caption)


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('weak','lf','migration','cartesian'))
    {'weak':weak,'lf':lf,'migration':migration,'cartesian':cartesian}[p.parse_args().mode]()
