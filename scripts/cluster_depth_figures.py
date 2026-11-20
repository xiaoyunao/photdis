#!/usr/bin/env python3
"""Render the CMD depth results in the preserved Pal 5 style."""
import argparse
import json
from pathlib import Path

from matplotlib.colors import TwoSlopeNorm
import numpy as np

from environment_figures import BLUE, GRAY, ORANGE, PAPER, STYLE, check_templates, figure, plt, read, sha
from v2_1_near import ROOT, save_json
from v3_10_figures import regression

SYSTEMS = json.loads((ROOT/'configs/cluster_depth_20261120.json').read_text())['systems']


def columns(rows,key):
    return np.array([float(r.get(key) or np.nan) for r in rows])


def cmd(source,dest):
    data=read(source,'cmd')
    layout,axes=plt.subplots(3,2,figsize=(7.15,7.5),layout='constrained')
    norm=TwoSlopeNorm(vmin=-1.5,vcenter=0.,vmax=1.5)
    for j,(key,p) in enumerate(SYSTEMS.items()):
        for i,b in enumerate(('gr','rz')):
            rows=[r for r in data if r['system']==key and r['color']==b]
            ax=axes[j,i]
            n=columns(rows,'N')
            pic=ax.scatter(columns(rows,'coord'),columns(rows,'Mr'),
                          c=columns(rows,'bias'),s=5+2*np.sqrt(n),cmap='RdBu_r',norm=norm,
                          alpha=.85,linewidths=0,rasterized=True)
            ax.invert_yaxis()
            ax.set(title=p['label']+f' · {int(n.sum()):,} members',
                   xlabel=r'$(g-r)_0$ (mag)' if i==0 else r'$(r-z)_0$ (mag)')
            if i==0:
                ax.set_ylabel(r'$M_{r,mathrm{ref}}$ (mag)')
    layout.colorbar(pic,ax=axes.ravel(),label='Median QRF modulus residual (mag)',
                    shrink=.84,extend='both')
    figure(dest,'native_cmd_distance_residuals',layout,data)


def depth(source,dest):
    rows=read(source,'continuous')
    layout,axes=plt.subplots(3,3,figsize=(7.15,6.9),layout='constrained',sharey='col')
    data=[]
    for j,(key,p) in enumerate(SYSTEMS.items()):
        for i,axis,metric,xlabel in [
            (0,'r','bias','Observed $r$ (mag)'),
            (1,'Mr','bias','Reference $M_r$ (mag)'),
            (2,'Mr','c68','Reference $M_r$ (mag)')]:
            ax=axes[j,i]
            for group,color,label in [('lower',BLUE,'Lower sequence'),
                                       ('clean_lower',ORANGE,'Outer clean lower sequence')]:
                q=[r for r in rows if r['system']==key and r['axis']==axis and r['group']==group and int(r['N'])>=8]
                if not q:
                    continue
                x=(columns(q,'lo')+columns(q,'hi'))/2
                y=columns(q,metric)
                gaps=np.flatnonzero(np.diff(x)>.51)+1
                for k,segment in enumerate(np.split(np.arange(len(x)),gaps)):
                    ax.plot(x[segment],y[segment],color=color,marker='o',ms=2.7,lw=.85,
                            label=label if k==0 else None)
                if metric=='bias':
                    ax.vlines(x,columns(q,'bias_lo'),columns(q,'bias_hi'),color=color,lw=.55,alpha=.65)
                data.extend(q)
            if metric=='bias':
                ax.axhline(0,color=GRAY,ls=':',lw=.7)
                ax.axhline(.3,color=GRAY,ls='--',lw=.6)
            else:
                ax.axhline(.68,color=GRAY,ls=':',lw=.7)
                ax.set_ylim(0,1)
            if axis=='r':
                ax.axvline(20,color=GRAY,ls=':',lw=.65)
            if j==0:
                ax.set_title(('Apparent magnitude','Absolute magnitude','68% interval')[i])
            if j==2:
                ax.set_xlabel(xlabel)
            if i==0:
                ax.set_ylabel(p['label']+'\nModulus residual (mag)')
            if i==2:
                ax.set_ylabel('Reference included')
    axes[0,1].legend(frameon=False,fontsize=5.8,loc='best')
    figure(dest,'lower_sequence_depth_and_coverage',layout,data)


def compare(source,dest):
    pairs=read(source,'paired')
    scales=read(source,'scales')
    layout,axs=plt.subplots(1,2,figsize=(7.15,3.3),layout='constrained')
    data=[]
    for j,(cluster,p) in enumerate(SYSTEMS.items()):
        for i,(m,label,color) in enumerate([('mist','Matched MIST',BLUE),('ridge_M5','M5 ridge',ORANGE),
                                               ('ridge_M30','M30 ridge',GRAY)]):
            row=next(r for r in pairs if r['system']==cluster and r['subset']=='faint_lower'
                                      and r['model']==m)
            if int(row['support'])==0:
                continue
            data.append(row)
            y=j+(i-1)*.19
            x,lo,hi=[float(row.get(k) or np.nan) for k in ('mean_delta_crps','crps_lo','crps_hi')]
            axs[0].plot([lo,hi],[y,y],color=color,lw=.8)
            axs[0].plot(x,y,'o',color=color,ms=3.5,label=label if j==0 else None)
        use=[r for r in scales if r['system']==cluster and r['subset']=='faint_lower_clean']
        for i,(anchor,label,color) in enumerate([('rrl_hst','HST W1',BLUE),
                     ('baumgardt_vasiliev','Combined scale',ORANGE),
                     ('secondary_published','Other published',GRAY)]):
            match=[r for r in use if r['anchor']==anchor]
            if not match:
                continue
            row=match[0]
            data.append(row)
            x,lo,hi=[float(row[k]) for k in ('bias','lo','hi')]
            y=j+(i-1)*.19
            axs[1].plot([lo,hi],[y,y],color=color,lw=.8)
            axs[1].plot(x,y,'o',color=color,ms=3.5,label=label if j==0 else None)
    for i,ax in enumerate(axs):
        ax.set(yticks=np.arange(len(SYSTEMS)),yticklabels=[p['label'] for p in SYSTEMS.values()],
               ylim=(len(SYSTEMS)-.6,-.45))
        ax.axvline(0,color=GRAY,ls=':',lw=.7)
        ax.legend(frameon=False,fontsize=6.1,loc='best')
    axs[0].set(xlabel='Paired CRPS change (mag)',title='Independent CMD fits')
    axs[1].axvline(.30,color=GRAY,ls='--',lw=.7)
    axs[1].set(xlabel='Median QRF modulus residual (mag)',title='Outer clean faint stars')
    figure(dest,'cmd_comparison_and_reference_scales',layout,data)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path);p.add_argument('dest',type=Path)
    p.add_argument('--regression',action='store_true')
    a=p.parse_args();a.dest.mkdir(parents=True,exist_ok=True)
    locked=check_templates()
    with plt.rc_context(PAPER):
        cmd(a.source,a.dest);depth(a.source,a.dest);compare(a.source,a.dest)
    if a.regression:
        replay=a.dest.parent/'style_replay'
        replay.mkdir(exist_ok=True)
        with plt.rc_context(STYLE):
            save_json(a.dest/'regression.json',regression(replay))
    save_json(a.dest/'complete.json',dict(style=locked,
        input={p.name:sha(p) for p in sorted(a.source.glob('*.csv'))},
        outputs={p.name:sha(p) for p in sorted(a.dest.iterdir()) if p.name!='complete.json'}))


if __name__=='__main__':
    main()
