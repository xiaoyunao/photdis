#!/usr/bin/env python3
"""Render Pal 5-style figures from the compact weak WISE score manifest."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from photdis_figure_style import STYLE, BLUE, GREEN, ORANGE, GRAY

TITLE = {'optical':'Optical', 'w1':'Optical + W1', 'w1w2':'Optical + W1/W2'}
COLORS = {'optical':BLUE, 'w1':GREEN, 'w1w2':ORANGE}
CONTRASTS = [
    ('w1_weak', 'w1_minus_optical', 'Add weak W1', ('optical','w1')),
    ('w2_weak_given_w1', 'w1w2_minus_w1', 'Add weak W2 with valid W1', ('w1','w1w2')),
]


def save(fig, directory, name):
    fig.savefig(directory/f'{name}.png',dpi=300,bbox_inches='tight')
    plt.close(fig)


def paired(scores, directory):
    groups = [
        ('', 'All'), ('class0:', 'Dwarf'), ('class1:', 'Evolved'),
        ('class2:', 'Other / transition'), ('class-1:', 'Unknown'),
        ('dwarf_corrected_feh_lt_m1p5:', 'Metal-poor dwarf'), ('r18_20:', '18–20 all'),
        ('r18_20:class1:', '18–20 evolved'),
    ]
    fig, axes = plt.subplots(1,2,figsize=(14.5,6.5),layout='constrained')
    for ax,(key,comparison,title,arms) in zip(axes,CONTRASTS):
        text = []
        for j,(prefix,label) in enumerate(groups):
            p = scores[prefix+key]['chosen'][comparison]
            text.append(f'{label} (N={p["n"]:,})')
            if not p['n']:
                continue
            d = 100*p['tail_difference']
            lo,hi = p['tail_ci95']
            color = COLORS[arms[-1]]
            if lo is not None:
                ax.hlines(j,100*lo,100*hi,color=color,lw=2)
            ax.plot(d,j,'o',color=color,ms=7)
        ax.axvline(0,color=GRAY,ls='--',lw=1.2)
        ax.set(title=title,xlabel='Change in severe-error rate [pp]',
               yticks=range(len(text)),yticklabels=text,ylim=(len(text)-.5,-.5))
    fig.suptitle('Real weak measurements · same-star comparison',fontsize=20)
    save(fig,directory,'weak_flux_paired')


def gaia(scores,directory):
    fig,axes=plt.subplots(1,2,figsize=(13,5.4),layout='constrained',sharey=True)
    for ax,(key,_,title,arms) in zip(axes,CONTRASTS):
        g=scores['r18_20:'+key]['full_Gaia']
        if g['n']:
            for arm in arms:
                ax.stairs(np.asarray(g[arm]['pit_histogram'])/g['n'],np.linspace(0,1,11),
                          color=COLORS[arm],lw=2.3,label=TITLE[arm])
            ax.legend()
        ax.axhline(.1,color=GRAY,ls=':',lw=1.5)
        ax.set(title=f'{title}\nN={g["n"]:,} · {g["negative"]:,} negative',
               xlabel='Observed-parallax PIT',ylabel='Fraction per bin')
    fig.suptitle('Signed Gaia parallax checks · 18–20 mag',fontsize=20)
    save(fig,directory,'weak_flux_gaia')


def parent_figure(coverage,directory):
    labels=['W1','W2','Both bands','Either band']
    valid = ['w1_valid','w2_valid','both_valid','either_valid']
    a = np.array([coverage[k]['fraction'] for k in ('w1_high','w2_valid','both_high','either_high')])
    a[1] -= coverage['w2_weak']['fraction']
    b = np.array([coverage[k]['fraction'] for k in valid])
    fig,ax=plt.subplots(figsize=(10.2,5.7),layout='constrained')
    x=np.arange(len(labels))
    ax.bar(x,a,color=BLUE,label='Quality pass and S/N ≥ 5')
    ax.bar(x,b-a,bottom=a,color=ORANGE,label='Additional with signed weak flux')
    for j,y in enumerate(b):
        ax.annotate(f'{y:.1%}',(j,y),xytext=(0,5),textcoords='offset points',ha='center')
    ax.set(title='WISE measurement coverage in the weighted DECam parent',
           ylabel='Weighted fraction',xticks=x,xticklabels=labels,ylim=(0,max(b)*1.3))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend()
    save(fig,directory,'weak_flux_coverage')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result',type=Path)
    parser.add_argument('figure_directory',type=Path)
    args = parser.parse_args()
    args.figure_directory.mkdir(parents=True,exist_ok=True)
    result = json.loads(args.result.read_text())
    scores = result['scores']['development_test']
    plt.rcParams.update(STYLE)
    paired(scores,args.figure_directory)
    gaia(scores,args.figure_directory)
    parent_figure(result['parent'],args.figure_directory)


if __name__=='__main__':
    main()
