#!/usr/bin/env python3
"""Pal 5 style figures from compact environment-pilot tables."""
import argparse
import csv
from pathlib import Path

import numpy as np

from environment_pilot import C, OUT, finish
from halo_common import PAPER, plt
from photdis_figure_style import BLUE, GRAY, GREEN, ORANGE
from stream_common import figure
from v2_1_near import save_json, sha
from v3_10_figures import check_templates, regression, STYLE
from v3_8_direct_figures import density


def read(source, name):
    with (source/(name+'.csv')).open() as f:
        return list(csv.DictReader(f))


def numbers(rows,key):
    return np.array([float(row.get(key) or np.nan) for row in rows])


def render(source, dest):
    dest.mkdir(parents=True, exist_ok=True)
    check_templates()
    fields=[x[0] for x in C['fields']]
    bounds=np.geomspace(.03,20,55)
    score=read(source,'source_scores')
    gaia=[r for r in score if r['source']=='gaia']
    with plt.rc_context(PAPER):
        fig, axes=plt.subplots(2,3,figsize=(7.15,4.8),layout='constrained',sharex=True,sharey=True)
        counts=[]
        for i,(field,ax) in enumerate(zip(fields,axes.flat)):
            rows=[r for r in gaia if r['field']==field]
            d=numbers(rows,'reference_kpc')
            pred=d*10**(numbers(rows,'bias')/5)
            hist=density(fig,ax,d,pred,bounds,bounds,label='Stars per cell')
            ax.plot(bounds[[0,-1]],bounds[[0,-1]],'--',color=ORANGE,lw=.8)
            ax.set(xscale='log',yscale='log',title=f'{field} · {len(rows):,} stars')
            counts.append(dict(field=field,stars=len(rows),inside_axes=int(hist.sum())))
            if i>=3:
                ax.set_xlabel('Gaia geometric distance (kpc)')
            if i%3==0:
                ax.set_ylabel('QRF median distance (kpc)')
        figure(dest,'environment_distance_comparison',fig,counts)

    contrasts=[r for r in read(source,'environment_contrasts')
               if r['source']=='gaia' and r['cohort']=='all_nonfit']
    usable=[r for r in contrasts if r.get('delta_bias') not in ('',None)]
    labels={'high_vs_low_fields':'Low vs high, matched environment',
            'high_vs_mid_fields':'Mid vs high, matched environment',
            'higher_sfd_column':'High vs low dust column',
            'higher_fracflux_r':'High vs low crowding'}
    with plt.rc_context(PAPER):
        fig, ax=plt.subplots(figsize=(7.15,3.0),layout='constrained')
        ax.axvspan(-.1,.1,color='.93',lw=0)
        ax.axvline(0,color=GRAY,ls=':',lw=.7)
        for j,row in enumerate(contrasts):
            if row not in usable:
                continue
            x=float(row['delta_bias'])
            lower,upper=float(row['bias_lo']),float(row['bias_hi'])
            color=BLUE if row['precise_010']=='True' else GRAY
            if np.isfinite(lower+upper):
                ax.plot([lower,upper],[j,j],color=color,lw=.8)
            ax.plot(x,j,'o',color=color,ms=4)
        ax.set(yticks=np.arange(len(contrasts)),yticklabels=[
               labels.get(r['contrast'],r['contrast'].replace('_vs_',' vs ')) for r in contrasts],
               xlabel='Difference in median distance modulus residual (mag)')
        ax.invert_yaxis()
        figure(dest,'environment_matched_scales',fig,contrasts)

    dust=[r for r in read(source,'dust_pairs') if r['cohort']=='all'
          and r['field']!='ALL' and r['method']=='iteration' and int(r['n'])>0]
    with plt.rc_context(PAPER):
        fig,ax=plt.subplots(1,2,figsize=(7.15,2.75),layout='constrained')
        for j,key in enumerate(('bias','crps')):
            ax[j].axvline(0,color=GRAY,ls=':',lw=.7)
            if j==0:
                ax[j].axvspan(-.1,.1,color='.93',lw=0)
            for i,row in enumerate(dust):
                x=float(row['delta_'+key]); lo=float(row[key+'_lo']); hi=float(row[key+'_hi'])
                if np.isfinite(lo+hi):
                    ax[j].plot([lo,hi],[i,i],color=BLUE,lw=.8)
                ax[j].plot(x,i,'o',color=BLUE,ms=4)
            ax[j].set(yticks=np.arange(len(dust)),yticklabels=[r['field'] for r in dust],
                xlabel=['Change in median modulus residual (mag)','Change in mean CRPS (mag)'][j])
            ax[j].invert_yaxis()
        figure(dest,'finite_distance_dust_comparison',fig,dust)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=OUT/'results')
    parser.add_argument('--dest',type=Path,default=OUT/'figures')
    parser.add_argument('--baseline-regression',action='store_true')
    args=parser.parse_args()
    render(args.source,args.dest)
    if args.baseline_regression:
        target=OUT/'style_replay'
        target.mkdir(parents=True,exist_ok=True)
        with plt.rc_context(STYLE):
            result=regression(target)
        save_json(args.dest/'style_regression.json',result)
    finish(args.dest,{'results':sha(args.source/'complete.json'),'style':check_templates(),
                      'script':sha(Path(__file__))})


if __name__=='__main__':
    main()
