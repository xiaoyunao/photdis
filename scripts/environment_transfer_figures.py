#!/usr/bin/env python3
"""Render the three registered environment plates with the locked Pal 5 style."""
import argparse
from pathlib import Path

import numpy as np
from matplotlib.ticker import NullFormatter

from environment_figures import (BLUE, GRAY, GREEN, ORANGE, PAPER, STYLE,
                                 check_templates, figure, numbers, plt, read, regression)
from v2_1_near import save_json, sha


def scale_panel(source,dest):
    data = [r for r in read(source,'scores') if r['stratum']=='all']
    contrast = [r for r in read(source,'contrasts') if r['stratum']=='all']
    fields = list(dict.fromkeys(r['field'] for r in data))
    fig, axes = plt.subplots(2,2,figsize=(7.15,4.2),layout='constrained')
    metrics = ('bias','nmad','crps','log_score')
    labels = ('Median modulus residual (mag)','NMAD (mag)','Change in mean CRPS (mag)',
              'Change in mean log score (nat)')
    for ax,metric,label in zip(axes.flat,metrics,labels):
        ax.set(yticks=range(len(fields)),yticklabels=fields,xlabel=label)
        ax.invert_yaxis()
        if metric=='bias':
            ax.axvspan(-.1,.1,color='.93',lw=0)
        if metric!='nmad':
            ax.axvline(0,color=GRAY,ls=':',lw=.7)
        for j,field in enumerate(fields):
            if metric in ('crps','log_score'):
                values = [r for r in contrast if r.get('metric')==metric and r['field']==field]
                sets = [(values,'delta',j,BLUE,'Bayestar − SFD')]
            else:
                use = [r for r in data if r['field']==field and r['metric']==metric]
                sets = [([r for r in use if r['arm']==a],'value',j+offset,col,name)
                        for a,offset,col,name in [('A_full',-.18,GRAY,'SFD, full'),
                        ('B',0,ORANGE,'SFD, shared support'),('C',.18,BLUE,'Bayestar')]]
            for rows,key,y,col,name in sets:
                if rows:
                    x,lo,hi = (float(rows[0][k]) for k in (key,'lo','hi'))
                    if np.isfinite(lo+hi):
                        ax.plot([lo,hi],[y,y],color=col,lw=.9)
                    ax.plot(x,y,'o',color=col,ms=4,label=name if j==0 else None)
        if metric=='bias' and fields:
            ax.legend(frameon=False,fontsize=6,loc='best')
    figure(dest,'finite_distance_transfer_scores',fig,data+contrast)


def comparison(source,dest):
    data = read(source,'pairs')
    flow = read(source,'flow')
    stages = ('registered','map_valid','sfd_half_mass','converged','common')
    labels = ('Registered','Map valid','SFD ≥ 50%','Converged','Scored')
    ref = numbers(data,'reference_kpc')
    distance = np.concatenate([ref,*(ref*10**(numbers(data,a+'_delta_mu')/5) for a in ('B','C'))])
    limits = (min(.12,float(distance.min())/1.08),max(4.,float(distance.max())*1.08))
    fig, axes = plt.subplots(2,len(flow),figsize=(7.15,4.3),layout='constrained',squeeze=False)
    for j,r in enumerate(flow):
        name = r['field']
        rows = [a for a in data if a['field']==name]
        ax = axes[0,j]
        ref = numbers(rows,'reference_kpc')
        for arm,col,label in [('B',ORANGE,'SFD, shared support'),('C',BLUE,'Bayestar')]:
            dist = ref*10**(numbers(rows,arm+'_delta_mu')/5)
            ax.scatter(ref,dist,s=5,c=col,alpha=.48,rasterized=True,edgecolor='none',label=label)
        ax.plot(limits,limits,':',color=GRAY,lw=.7)
        ax.set(xscale='log',yscale='log',xlim=limits,ylim=limits,
               xlabel='Gaia distance (kpc)',title=f'{name} · {len(rows)} stars')
        ticks = [.2,.5,1,2,4]
        ax.set_xticks(ticks,labels=[f'{x:g}' for x in ticks])
        ax.set_yticks(ticks,labels=[f'{x:g}' for x in ticks])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        if j==0:
            ax.set_ylabel('Photometric distance (kpc)')
            ax.legend(frameon=False,fontsize=6)
        ax = axes[1,j]
        n = [int(r[k]) for k in stages]
        ax.barh(range(len(n)),n,color=[GRAY,GRAY,ORANGE,BLUE,BLUE],height=.62)
        for k,value in enumerate(n):
            ax.text(value+7,k,str(value),va='center',fontsize=7)
        ax.set(xlim=(0,570),yticks=range(len(n)),yticklabels=labels if j==0 else [],
               xlabel='Stars in dust cohort')
        ax.invert_yaxis()
    figure(dest,'same_star_transfer_and_attrition',fig,data+flow)


def l219_panel(source,dest):
    data = read(source,'l219_stars')
    profiles = read(source,'l219_profiles')
    fig, axes = plt.subplots(2,3,figsize=(7.15,4.7),layout='constrained')
    states = numbers(data,'state')
    severe = np.array([r['severe']=='True' for r in data])
    selections = ((states<=2,GRAY,'Initial mass / color'),
                  ((states==3)|(states==4),ORANGE,'Updated mass / steps'),
                  ((states==5)&~severe,BLUE,'Converged'),
                  (severe,GREEN,'Severe residual'))
    coords = (('obs_gr','obs_r','Observed $g-r$ (mag)','$r$ (mag)'),
              ('gaia_bp_rp','gaia_absolute_g_uncorrected','Gaia BP − RP (mag)','$M_G$ (uncorrected, mag)'),
              ('FRACFLUX_R','nearest_detection_arcsec','FRACFLUX $r$','Nearest detection (arcsec)'),
              ('sfd','bayestar_at_gaia','SFD $E(B-V)$','Bayestar $E(B-V)$ at Gaia distance'),
              ('bayestar_converged','bayestar_residual','Iterated Bayestar $E(B-V)$','Modulus residual (mag)'))
    for k,(x,y,xlabel,ylabel) in enumerate(coords):
        ax = axes.flat[k if k<3 else k+1]
        xx,yy = numbers(data,x),numbers(data,y)
        for mask,color,label in selections:
            ax.scatter(xx[mask],yy[mask],s=7,c=color,alpha=.62,edgecolors='none',
                       label=label,rasterized=True)
        ax.set(xlabel=xlabel,ylabel=ylabel)
        if k<2:
            ax.invert_yaxis()
        if k==2:
            ax.set_xscale('symlog',linthresh=.003)
            ax.set_xticks([-.003,0,.003,.03],labels=['−.003','0','.003','.03'])
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_yscale('log')
        if k==3:
            ax.plot([0,1.4],[0,1.4],':',color=GRAY,lw=.7)
        if k==4:
            ax.axhline(0,ls=':',color=GRAY,lw=.7)
    axes[0,0].legend(frameon=False,fontsize=5.6,loc='best')
    ax = axes[1,0]
    for state,col,ls,label in [(1,GRAY,'-','Low initial mass'),(2,GRAY,'--','Color support'),
                               (3,ORANGE,'-','Low updated mass'),(4,ORANGE,'--','Step limit'),
                               (5,BLUE,'-','Converged')]:
        p = [r for r in profiles if int(r['state'])==state and int(r['n'])>=10]
        x = 10**((numbers(p,'mu')-10)/5)
        ax.plot(x,numbers(p,'center'),ls=ls,color=col,label=label,lw=.9)
    ax.set(xscale='log',xlabel='Map distance (kpc)',ylabel='Median Bayestar $E(B-V)$')
    ax.set_xticks([.4,1,2,4],labels=['0.4','1','2','4'])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.legend(frameon=False,fontsize=5.6)
    figure(dest,'l219_extinction_and_failures',fig,data+profiles)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('dest',type=Path)
    parser.add_argument('--baseline-regression',action='store_true')
    args = parser.parse_args()
    args.dest.mkdir(parents=True,exist_ok=True)
    lock = check_templates()
    with plt.rc_context(PAPER):
        scale_panel(args.source,args.dest)
        comparison(args.source,args.dest)
        l219_panel(args.source,args.dest)
    if args.baseline_regression:
        output = args.dest.parent/'style_replay'
        output.mkdir(exist_ok=True)
        with plt.rc_context(STYLE):
            save_json(args.dest/'style_regression.json',regression(output))
    save_json(args.dest/'complete.json',dict(style=lock,
        inputs={p.name:sha(p) for p in sorted(args.source.glob('*.csv'))},
        outputs={p.name:sha(p) for p in sorted(args.dest.iterdir()) if p.name!='complete.json'}))


if __name__=='__main__':
    main()
