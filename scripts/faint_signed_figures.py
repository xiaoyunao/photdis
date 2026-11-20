#!/usr/bin/env python3
"""Render the three signed-parallax plates from compact results."""
import argparse
from pathlib import Path

import numpy as np

from environment_figures import BLUE, ORANGE, GRAY, PAPER, STYLE, check_templates, figure, plt, read, sha
from v3_10_figures import regression
from v2_1_near import save_json


def power_plate(source,dest):
    rows = read(source,'power')
    fields = list(dict.fromkeys(x['field'] for x in rows))
    fig,axes = plt.subplots(1,2,figsize=(7.15,2.75),layout='constrained',sharey=True)
    for ax,field in zip(axes,fields):
        for m,color in ((20,BLUE),(21,ORANGE)):
            for stress,style in ((1,'-'),(2,'--')):
                q=[x for x in rows if x['field']==field and int(x['r_lo'])==m
                   and float(x['systematics_multiplier'])==stress]
                x=[float(r['shift_mag']) for r in q]
                y=[float(r['directional_power'] or r['rejection']) for r in q]
                ax.plot(x,y,ls=style,marker='o',ms=4,color=color,lw=.9,
                        label=f'{m} ≤ r < {m+1}' if stress==1 else None)
        ax.axhline(.80,color=GRAY,ls=':',lw=.8)
        ax.axhspan(.02,.08,color='.92',lw=0)
        ax.set(title=field,xlabel='Injected modulus shift (mag)',ylim=(0,1.02))
    axes[0].set_ylabel('Rejection and directional power')
    axes[0].legend(frameon=False)
    figure(dest,'signed_parallax_scale_power',fig,rows)


def observed_plate(source,dest):
    rows=read(source,'signed_snr')
    histogram=read(source,'pit')
    use='full' if any(q['mode']=='full' for q in rows) else 'pilot'
    fields=list(dict.fromkeys((q['field'],q['r_lo']) for q in rows if q['mode']==use))
    if use=='full':
        fields=[x for x in fields if x[0] in ('L032','M249')][:4]
    fig,axes=plt.subplots(2,len(fields),figsize=(7.15,4.1),layout='constrained',squeeze=False)
    displayed=[]
    for j,(field,mag) in enumerate(fields):
        q=[r for r in rows if (r['field'],r['r_lo'])==(field,mag) and r['mode']==use]
        p=[r for r in histogram if (r['field'],r['r_lo'])==(field,mag) and r['mode']==use]
        shown=np.array([float(r['snr_hi'])-.5 for r in q])
        total=sum(int(r['observed']) for r in q)
        ax=axes[0,j]
        ax.bar(shown,[int(r['observed'])/total for r in q],width=.92,color=BLUE,alpha=.6,label='Gaia')
        ax.plot(shown,[float(r['predicted'])/total for r in q],'o-',color=ORANGE,ms=2,lw=.8,label='Photdis')
        ax.axvline(0,color=GRAY,ls=':',lw=.7)
        ax.set(title=f'{field} · r {mag}–{int(mag)+1}',xlabel='Signed parallax S/N',xticks=[-4,0,4])
        ax=axes[1,j]
        ax.bar([float(r['pit_lo'])+.05 for r in p],[int(r['count'])/total for r in p],width=.09,color=BLUE,alpha=.6)
        ax.axhline(.10,color=GRAY,ls=':',lw=.7)
        ax.set(xlabel='Conditional PIT',xlim=(0,1),xticks=[0,.5,1])
        displayed.extend(q+p)
    axes[0,0].set_ylabel('Fraction per S/N bin')
    axes[1,0].set_ylabel('Fraction per PIT bin')
    axes[0,0].legend(frameon=False,fontsize=6)
    figure(dest,'signed_parallax_observed',fig,displayed)


def population_plate(source,dest):
    rows=read(source,'population_stars')
    scores=read(source,'population_matches')
    fig,axes=plt.subplots(1,3,figsize=(7.15,3.0),layout='constrained')
    for ax,x,y,xlabel,ylabel in [(axes[0],'gaia_bp_rp','gaia_MG','Gaia BP − RP (mag)','$M_G$ (uncorrected, mag)'),
                                (axes[1],'gr','rz','SFD-corrected $g-r$ (mag)','SFD-corrected $r-z$ (mag)')]:
        for group,color,label in [('dwarf_proxy',BLUE,'Dwarf proxy'),('uncertain',ORANGE,'Uncertain')]:
            q=[s for s in rows if s['group']==group and float(s['match_weight'])>0]
            ax.scatter([float(s[x]) for s in q],[float(s[y]) for s in q],s=5,ec='none',
                       alpha=.55,c=color,label=label,rasterized=True)
        ax.set(xlabel=xlabel,ylabel=ylabel)
    axes[0].invert_yaxis()
    axes[0].legend(frameon=False,fontsize=6)
    ax=axes[2]
    q=[s for s in scores if s.get('metric')=='C_crps']
    for i,s in enumerate(q):
        ax.plot([float(s['lo']),float(s['hi'])],[i,i],color=BLUE,lw=.8)
        ax.plot(float(s['change_difference']),i,'o',color=BLUE,ms=4)
    ax.axvline(0,color=GRAY,ls=':',lw=.7)
    ax.set(yticks=range(len(q)),yticklabels=[s['field'] for s in q],
           xlabel='Difference in CRPS change (mag)')
    figure(dest,'dust_population_common_support',fig,rows+q)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('dest',type=Path)
    parser.add_argument('--regression',action='store_true')
    args=parser.parse_args()
    args.dest.mkdir(parents=True,exist_ok=True)
    lock=check_templates()
    with plt.rc_context(PAPER):
        power_plate(args.source,args.dest)
        observed_plate(args.source,args.dest)
        population_plate(args.source,args.dest)
    if args.regression:
        output=args.dest.parent/'style_replay'
        output.mkdir(exist_ok=True)
        with plt.rc_context(STYLE):
            save_json(args.dest/'regression.json',regression(output))
    save_json(args.dest/'complete.json',dict(style=lock,
              inputs={p.name:sha(p) for p in sorted(args.source.glob('*.csv'))},
              outputs={p.name:sha(p) for p in sorted(args.dest.iterdir()) if p.name!='complete.json'}))


if __name__=='__main__':
    main()
