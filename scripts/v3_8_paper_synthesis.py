#!/usr/bin/env python3
"""Render fixed-selection comparison and signed parallax distributions."""
import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from photdis_figure_style import BLUE, GRAY, GREEN, ORANGE, STYLE

ROOT = Path(__file__).resolve().parents[1]
COUNTS = ROOT/'manifests/v3_7/selection'


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def save(fig,dest,tab,name,rows,inputs):
    dest.mkdir(parents=True,exist_ok=True)
    tab.mkdir(parents=True,exist_ok=True)
    path = tab/f'{name}.csv'
    with path.open('w') as f:
        w = csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)),lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    outputs = [path]
    for extension in ('pdf','png'):
        a = dest/f'{name}.{extension}'
        fig.savefig(a,dpi=300)
        outputs.append(a)
    plt.close(fig)
    def hashes(paths):
        return {p.name:sha256(p.read_bytes()).hexdigest() for p in paths}
    (tab/f'{name}.json').write_text(json.dumps(dict(inputs=hashes(inputs),outputs=hashes(outputs)),indent=2)+'\n')


def selection():
    p,c = COUNTS/'reference.csv',COUNTS/'predicted_composition.csv'
    raw,comp = read(p),read(c)
    baselines = ['primary','lower_dwarf']
    fig,axs = plt.subplots(2,2,figsize=(13,8),layout='constrained')
    rows = []
    for j,(cut,tag,color) in enumerate(zip(baselines,['Original primary','Fixed candidate'],[BLUE,ORANGE])):
        x = np.arange(2)+(j-.5)*.32
        selected = []
        for source in ('SpecDis','SPDist'):
            m = next(r for r in raw if r['source']==source and r['selection']==cut and
                     r['domain']=='all' and r['reference_class']=='dwarf' and r['history']=='all' and
                     r['reference_shell']=='5–10')
            contaminant = next(r for r in comp if r['source']==source and r['selection']==cut and
                               r['domain']=='all' and r['reference_class']=='giant' and r['history']=='all' and
                               r['predicted_shell']=='5–10')
            selected.append(dict(source=source,selection=cut,reference_dwarfs=int(m['n']),
                correctly_assigned_dwarfs=int(m['hard_2']),
                median_bias_pct=100*float(m['median_relative']),
                far_giants_assigned=int(contaminant['reference_gt10_hard_count'])))
        for ax,field,title in zip(axs.flat,
                    ('reference_dwarfs','correctly_assigned_dwarfs','median_bias_pct','far_giants_assigned'),
                    ('Selected reference dwarfs at 5–10 kpc','Same dwarfs assigned 5–10 kpc',
                     'Median distance bias of selected dwarfs (%)','Reference giants beyond 10 kpc assigned 5–10 kpc')):
            bars = ax.bar(x,[r[field] for r in selected],width=.30,color=color,label=tag)
            ax.bar_label(bars,fmt='%.1f' if field=='median_bias_pct' else '%d',fontsize=10,padding=2)
            ax.set_xticks(range(2),['SpecDis','SPDist'])
            ax.set_title(title,fontsize=14)
            ax.margins(y=.17)
        rows.extend(selected)
    axs[0,0].legend()
    fig.suptitle('What changes when the dwarf-score threshold is lowered?\nSame photometric distances, matched references')
    return fig,'F06_selection_outcomes',rows,[p,c]


def signed(data):
    p = data/'candidate/parallax_distribution.csv'
    rows = [r for r in read(p) if r['scope']=='outside_recorded_history']
    f,a = plt.subplots(1,2,figsize=(14,5.5),layout='constrained',sharex=True,sharey=True)
    for ax,cut in zip(a,('primary','candidate')):
        z = [r for r in rows if r['selection']==cut and np.isfinite(float(r['lo'])) and np.isfinite(float(r['hi']))]
        edge = np.array([float(r['lo']) for r in z]+[float(z[-1]['hi'])])
        starts = np.arange(0,len(z),3)
        plot_edges = edge[np.r_[starts,len(z)]]
        for field,color,label in (('observed',GRAY,'Gaia observed'),('qrf',BLUE,'QRF forward prediction'),
                                  ('broad',ORANGE,'Broad distance baseline')):
            m = np.array([float(r[field]) for r in z])
            ax.stairs(np.add.reduceat(m,starts)/np.diff(plot_edges),plot_edges,label=label,color=color)
        ax.set(xlabel='Corrected signed parallax (mas)',title='Original primary' if cut=='primary' else 'Fixed candidate')
        ax.axvline(0,ls=':',color=GRAY,lw=1)
    a[0].set_ylabel('Probability density (mas⁻¹)')
    a[0].legend(fontsize=10)
    f.suptitle('Weak Gaia parallaxes at photometric median distances of 5–10 kpc\n'
               'Weighted draw outside recorded history · predictions conditioned on the weak-parallax cut')
    return f,'F07_signed_parallax_distribution',rows,[p]


def recovered(data):
    path = data/'reliability/recovered_types.npz'
    output = []
    f,axes = plt.subplots(2,3,figsize=(16,9),layout='constrained',sharex='col',sharey='col')
    with np.load(path,allow_pickle=False) as z:
        if not (z['TYPE']=='PSF').all():
            raise ValueError('Recovered references require native PSF')
        planes = [('gr','reference_Mr','Dereddened g − r (mag)','Reference absolute r',
                   (.0,1.8),(-3.,8.)),
                  ('rvs_teff','rvs_logg','Effective temperature (K)','Surface gravity log g',
                   (3500.,8000.),(0.,5.5)),
                  ('rvs_feh','rvs_logg','Spectroscopic [Fe/H]','Surface gravity log g',
                   (-3.,.8),(0.,5.5))]
        for k,src in enumerate(('SpecDis','SPDist')):
            for j,(x,y,xlab,ylab,xlim,ylim) in enumerate(planes):
                ax = axes[k,j]
                for g,color,name in (('primary',BLUE,'Primary'),('new',ORANGE,'New candidates')):
                    m = (z['source']==src)&z[g]&np.isfinite(z[x])&np.isfinite(z[y])
                    ax.scatter(z[x][m],z[y][m],s=6,alpha=.16,color=color,linewidths=0,label=name,rasterized=True)
                    in_axes = m&(z[x]>=xlim[0])&(z[x]<=xlim[1])&(z[y]>=ylim[0])&(z[y]<=ylim[1])
                    output.append(dict(source=src,panel=f'{x}/{y}',selection=g,n=int(m.sum()),in_axes=int(in_axes.sum())))
                ax.set(xlim=xlim,ylim=ylim[::-1],xlabel=xlab,ylabel=ylab,title=src)
                if j==1:
                    ax.axhline(4.2,color=GRAY,ls=':')
        axes[0,0].legend()
    f.suptitle('Photometry and spectroscopy of stars recovered at reference distances of 5–10 kpc')
    return f,'A01_recovered_stars',output,[path]


def main():
    g = argparse.ArgumentParser(description=__doc__)
    g.add_argument('data',type=Path)
    g.add_argument('mode',choices=['selection','signed','recovered'])
    a = g.parse_args()
    with plt.rc_context(STYLE):
        plot = {'signed':signed,'recovered':recovered}
        fig,name,rows,inputs = selection() if a.mode=='selection' else plot[a.mode](a.data)
        save(fig,ROOT/'figures/v3_8',ROOT/'manifests/v3_8/figures',name,rows,inputs)


if __name__=='__main__':
    main()
