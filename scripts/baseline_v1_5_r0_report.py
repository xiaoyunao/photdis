#!/usr/bin/env python3
"""Render the fixed contrast and v1.4 coverage correction on allocated compute."""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import PercentFormatter
import numpy as np

from baseline_v1_5_r0 import (
    BASELINE, DEST, OUT, REFINED, load, metal_layers, save_json, sha, truth,
)
from baseline_v1_4_coverage import application_coverage
from photdis.diagnostics_v1_4 import distance_kpc, residual_stats
from photdis.stage1 import require_compute
from photdis_figure_style import BLUE, GRAY, GREEN, ORANGE, STYLE

FIG = DEST/'figures'
MODELS = ['Two grz colors', 'Two colors + $r_0$']
COLORS = [BLUE, GREEN]


def save(fig, name):
    path = FIG/f'{name}.png'
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return sha(path)


def distance_figure(t, a, b):
    fig, axes = plt.subplots(2,3,figsize=(15.6,9.5),layout='constrained')
    for i, (key, title) in enumerate([('Gaia','Gaia anchors'),('specdis','SpecDis'),('spdist','SPDist')]):
        ref = truth(t,key)
        use = np.isfinite(ref)
        x = distance_kpc(ref[use])
        old, new = (distance_kpc(p[use,2]) for p in (a,b))
        limits = np.log10(np.r_[x,old,new])
        lo, hi = limits.min()-.05, limits.max()+.05
        artists = []
        for ax, y, label in zip(axes[:,i],[old,new],MODELS):
            artist = ax.hexbin(x,y,xscale='log',yscale='log',extent=(lo,hi,lo,hi),
                               gridsize=45,mincnt=1,cmap='Blues')
            artists.append(artist)
            ax.plot(10**np.array([lo,hi]),10**np.array([lo,hi]),'--',color=ORANGE,lw=1.8)
            ax.set(xlabel='Reference distance [kpc]',ylabel='Predicted distance [kpc]',
                   title=f'{title}\n{label} · N={len(x):,}',aspect='equal')
        norm = LogNorm(1,max(2,max(h.get_array().max() for h in artists)))
        for h in artists:
            h.set_norm(norm)
        fig.colorbar(artists[-1],ax=axes[:,i],label='Stars per hexagon',shrink=.7)
    fig.suptitle('Same stars, separated by distance reference',fontsize=20)
    return save(fig,'distance_by_reference')


def cmd_figure(t,a,b):
    fig,axes=plt.subplots(2,3,figsize=(15.6,9.5),layout='constrained',sharex=True,sharey=True)
    chosen = truth(t,'chosen')
    gr = t['g0']-t['r0']
    all_use = np.isfinite(chosen) & np.isin(t['class_label'],[0,1])
    absolute = [t['g0']-m for m in (chosen,a[:,2],b[:,2])]
    ys = np.concatenate([v[all_use] for v in absolute])
    extent = (gr[all_use].min()-.02,gr[all_use].max()+.02,ys.min()-.1,ys.max()+.1)
    for row,(code,name,cmap) in enumerate([(0,'Dwarfs','Blues'),(1,'Evolved','Oranges')]):
        use = all_use & (t['class_label']==code)
        artists = []
        for ax,label,y in zip(axes[row],['Chosen reference',*MODELS],absolute):
            artists.append(ax.hexbin(gr[use],y[use],extent=extent,gridsize=47,mincnt=1,cmap=cmap))
            ax.set(title=f'{name} · {label}\nN={use.sum():,}',xlabel=r'$(g-r)_0$',ylabel=r'Legacy $M_g$ [mag]',
                   ylim=(extent[3],extent[2]))
        norm = LogNorm(1,max(2,max(h.get_array().max() for h in artists)))
        for h in artists:
            h.set_norm(norm)
        fig.colorbar(artists[-1],ax=axes[row],label='Stars per hexagon',shrink=.78)
    fig.suptitle('Reference and predicted color–absolute-magnitude diagrams',fontsize=20)
    return save(fig,'cmd_comparison')


def paired_figure(scores):
    fig, axes = plt.subplots(1,2,figsize=(14.5,5.8),layout='constrained')
    names=['Dwarf','Evolved','Other','Unknown','Metal-poor dwarf','18–20 / no W1']
    keys=['class0','class1','class2','class-1','dwarf_corrected_feh_lt_m1p5','r18_20:no_W1']
    data=[scores[k]['chosen'] for k in keys]
    y=np.arange(len(keys))
    ax=axes[0]
    for arm,offset,color,label in zip(['old','new'],[-.18,.18],COLORS,MODELS):
        ax.barh(y+offset,[r[arm].get('tail_1',np.nan) for r in data],height=.34,color=color,label=label)
    ax.set(title='Severe distance errors',xlabel=r'Fraction with $|\Delta\mu|>1$ mag',xlim=(0,1),yticks=y,
           yticklabels=[f'{name}  (N={r["paired"]["n"]:,})' for name,r in zip(names,data)])
    ax.xaxis.set_major_formatter(PercentFormatter(1));ax.invert_yaxis();ax.legend(loc='lower right')
    ax=axes[1]
    for k,offset,color,label in [('rescued',-.18,GREEN,'Rescued'),('harmed',.18,ORANGE,'Harmed')]:
        values=[r['paired'].get(k,0) for r in data]
        ax.barh(y+offset,values,height=.34,color=color,label=label)
        for j,n in enumerate(values):
            ax.annotate(f'{n:,}',(n,j+offset),xytext=(4,0),textcoords='offset points',va='center')
    ax.margins(x=.18)
    ax.set(title='Same-star changes',xlabel='Number of stars',yticks=y,yticklabels=names)
    ax.invert_yaxis();ax.legend(loc='lower right')
    return save(fig,'paired_class_risk')


def gaia_figure(scores):
    fig,axes=plt.subplots(1,2,figsize=(13,5.2),layout='constrained',sharey=True)
    for ax,key,title in zip(axes,['all','class1'],['All Gaia parallaxes','Evolved-star parallaxes']):
        g=scores[key]['full_Gaia']
        ax.axhline(.1,color=GRAY,ls=':',lw=1.4)
        for arm,label,c in zip(['old','new'],MODELS,COLORS):
            ax.stairs(np.asarray(g[arm]['pit_histogram'])/g['n'],np.linspace(0,1,11),
                      color=c,lw=2.4,label=label)
        ax.set(xlabel='Observed-parallax PIT',ylabel='Fraction per bin',title=f'{title}\nN={g["n"]:,} · {g["negative"]:,} negative')
        ax.legend()
    return save(fig,'gaia_pit')


def main():
    require_compute()
    FIG.mkdir(parents=True,exist_ok=True)
    result=json.loads((DEST/'result.json').read_text())
    paths=[BASELINE/'evaluation.npz',REFINED/'evaluation.npz',OUT/'references.npz']
    assert all(sha(p)==result['inputs'][str(p.relative_to(OUT.parents[1]))] for p in paths)
    assert sha(DEST/'evaluation.npz')==result['evaluation_sha256']
    old, new=load(paths[0]),load(DEST/'evaluation.npz')
    assert np.array_equal(old['identity'],new['identity']) and np.array_equal(old['rows'],new['rows'])
    with np.load(paths[2]) as z:
        rows=old['rows'][z['split'][old['rows']]==2]
        t={k:z[k][rows] for k in ['identity','class_label','g0','r0','nonstellar_risk',
              'anchor_astrometry','p_corr','label_source','ref_distance','specdis_distance','spdist_distance']}
    keep=np.isin(old['rows'],rows)
    assert np.array_equal(t['identity'],old['identity'][keep])
    a,b=old['mu'][keep],new['mu'][keep]
    matched=load(OUT/'diagnostics/matched_griz.npz')
    metal={name:residual_stats(matched['mu'][mask],matched['ref_chosen'][mask])
           for name,mask in metal_layers(matched).items()}
    plt.rcParams.update(STYLE)
    fig,coverage=application_coverage()
    scores=result['scores']['development_test']
    images={'application_coverage':save(fig,'application_coverage'),
            'distance_by_reference':distance_figure(t,a,b),'cmd_comparison':cmd_figure(t,a,b),
            'paired_class_risk':paired_figure(scores),'gaia_pit':gaia_figure(scores)}
    save_json(DEST/'figures.json',dict(sha256=images,coverage=coverage,matched_griz_metallicity=metal,
              result_sha256=sha(DEST/'result.json'),render_job_id=os.getenv('SLURM_JOB_ID')))
    print('Verified and rendered',len(images),'figures',flush=True)


if __name__=='__main__':
    main()
