#!/usr/bin/env python3
"""Render the F04 reference-distance diagnostics with denominators and both cuts."""
import argparse
import csv
import json
import os
from pathlib import Path
from textwrap import fill

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from photdis.stage1 import require_compute
from photdis_figure_style import BLUE, GREEN, ORANGE, STYLE
from v2_1_near import ROOT, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_6_condition_errors import OUT as CONDITIONS
from v3_6_plot_distance_atlas import SOURCE_LABELS


OUT = ROOT/'data/v3_6/figures/condition_errors'
SELECTION = (('supported','Before cuts',BLUE),('primary','Primary',ORANGE),('strict','Strict',GREEN))
PROPERTIES = {
    'reference_kpc':'Reference distance [kpc]',
    'observed_r':'Observed r [mag]',
    'dereddened_g_r':'Dereddened g − r [mag]',
    'dereddened_r_z':'Dereddened r − z [mag]',
    'maximum_grz_error_mag':'Maximum grz magnitude error [mag]',
    'EBV':'E(B − V) [mag]',
    'W1':'W1 measurement state',
    'W2':'W2 measurement state',
    'spectroscopic_feh':'Spectroscopic [Fe/H] [dex]',
    'spectroscopic_logg':'Spectroscopic log g [dex]',
    'spectroscopic_teff':'Spectroscopic effective temperature [K]',
}
PAGES = {'photometry':tuple(PROPERTIES)[:6], 'wise_spectroscopy':tuple(PROPERTIES)[6:]}
METRICS = (
    ('n','Reference stars','count'),
    ('median_relative','Median relative bias [%]','relative'),
    ('nmad_relative','Relative NMAD [%]','relative'),
    ('severe_fraction','Fraction beyond 1 mag [%]','relative'),
    ('median_width68','Interval width [mag]','interval_width'),
    ('coverage68','Reference-center agreement [%]','center'),
    ('error_integrated_agreement68','Error-integrated agreement [%]','integrated'),
)


def value(r,k):
    return float(r[k]) if r.get(k) else np.nan


def bin_label(values):
    label = values['bin']
    if label.startswith('['):
        low,high = label[1:-1].split(', ')
        fmt = lambda s: '∞' if s=='inf' else ('−∞' if s=='-inf' else f'{float(s):g}')
        return f'{fmt(low)} to {fmt(high)}'
    return {'missing_or_unusable':'Missing / unusable','legal_weak_or_signed':'Weak / signed'}.get(label,label.title())


def prepare(rows):
    index = {(r['property'],int(r['bin_index']),r['selection']):r for r in rows}
    if len(index)!=len(rows):
        raise ValueError('Duplicated F04 stratum')
    output = {}
    for prop in PROPERTIES:
        bins = sorted({i for key,i,cut in index if key==prop})
        cells = [{cut:index[prop,i,cut] for cut,_,_ in SELECTION} for i in bins]
        for cell in cells:
            supported,primary,strict = [value(cell[cut],'n') for cut,_,_ in SELECTION]
            if not 0<=strict<=primary<=supported:
                raise ValueError(f'Broken selection denominators for {prop}')
        output[prop] = [cell for cell in cells if value(cell['supported'],'n')>0]
    return output


def draw(ax, cells, key, metric):
    n = len(cells)
    points = np.arange(n)
    extra = {'interval_width':'median_width95','center':'coverage95',
             'integrated':'error_integrated_agreement95'}.get(metric)
    transformed = metric in ('relative','center','integrated')
    factor = 100 if transformed else 1
    for offset,(cut,_,color) in zip((-.2,0,.2),SELECTION):
        y = points+offset
        first = np.array([value(c[cut],key) for c in cells])*factor
        second = np.array([value(c[cut],extra) for c in cells])*factor if extra else None
        if second is not None:
            ax.hlines(y,first,second,color=color,alpha=.38,lw=.75)
            ax.plot(second,y,'o',mfc='white',mec=color,ms=3.6,ls='None')
        if metric=='count':
            ax.set_xscale('symlog',linthresh=1)
            for x,point in zip(first,y):
                ax.annotate(f'{int(x):,}', (x,point), xytext=(4,0),textcoords='offset points',
                            color=color,va='center',fontsize=7.2,clip_on=False)
        ax.plot(first,y,'o',color=color,ms=3.6,ls='None')
    if metric=='count':
        max_n = max(value(c['supported'],'n') for c in cells)
        ax.set_xlim(-.4,max(2.,max_n*7))
    elif metric in ('center','integrated','relative') and key not in ('median_relative','nmad_relative'):
        ax.set_xlim(-3,103)
    elif key=='median_relative':
        ax.axvline(0,color='#777777',lw=.75)
    ax.set_ylim(n-.55,-.48)
    ax.yaxis.grid(False)
    ax.margins(x=.08)


def render(data,source,cohort,dest):
    with plt.rc_context(STYLE):
        for name,variables in PAGES.items():
            variables = [v for v in variables if data[v]]
            if not variables:
                continue
            heights = [max(2,len(data[v])) for v in variables]
            fig = plt.figure(figsize=(21.,max(4,1.8+.46*sum(heights))),layout='constrained')
            grid = fig.add_gridspec(len(variables)+1,len(METRICS)+1, height_ratios=[1.3,*heights],
                                    width_ratios=[.7,*([1]*len(METRICS))])
            heading = fig.add_subplot(grid[0,:])
            heading.axis('off')
            for row,var in enumerate(variables):
                cells = data[var]
                row_title = fig.add_subplot(grid[row+1,0])
                row_title.axis('off')
                row_title.text(.0,.5,fill(PROPERTIES[var],18),transform=row_title.transAxes,
                               va='center',ha='left',fontsize=10)
                for col,(key,title,metric) in enumerate(METRICS):
                    ax = fig.add_subplot(grid[row+1,col+1])
                    draw(ax,cells,key,metric)
                    ax.set_yticks(np.arange(len(cells)))
                    if col==0:
                        ax.set_yticklabels([bin_label(c['supported']) for c in cells],fontsize=8.3)
                    else:
                        ax.set_yticklabels([])
                    if row==0:
                        ax.set_title(fill(title,24),fontsize=10)
                    ax.tick_params(axis='x',labelsize=8)
            legend = [Line2D([],[],marker='o',ls='None',ms=5,color=color,label=title)
                      for _,title,color in SELECTION]
            legend += [Line2D([],[],marker='o',ls='None',color='#444444',label='68% interval'),
                       Line2D([],[],marker='o',ls='None',mfc='white',color='#444444',label='95% interval')]
            heading.legend(handles=legend,loc='center',ncols=5,frameon=False,fontsize=10)
            fig.suptitle(f'{source} | {cohort}')
            for suffix in ('png','pdf'):
                fig.savefig(dest/f'{name}.{suffix}',dpi=300)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    source = parser.parse_args().source
    require_compute()
    path = CONDITIONS/'complete.json'
    receipt = json.loads(path.read_text())
    table = verified(CONDITIONS/'conditions.csv',receipt['outputs']['conditions.csv'])
    out = OUT/source
    out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():
        raise FileExistsError(out/'complete.json')
    with table.open() as stream:
        rows = [r for r in csv.DictReader(stream) if r['source']==source
                 and r['reference_quality']=='high_quality' and r['history']=='outside_training'
                 and r['reference_class']=='all']
    if not rows:
        raise ValueError(f'No F04 data for {source}')
    csv_rows(out/'plotted_strata.csv',rows)
    render(prepare(rows),SOURCE_LABELS.get(source,source),'High-quality references outside QRF training',out)
    save_json(out/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),source=source,dpi=300,
        inputs={str(p.relative_to(ROOT)):sha(p) for p in (path,table,Path(__file__))},
        caption=('Reference-dependent, unweighted estimates for each selection. High-quality '
                 'reference stars outside QRF training can include other historical uses. '
                 'Empty strata are retained in the source table. Agreement with an uncertain '
                 'reference is reported both at the reference center and with its available '
                 'error integrated over the QRF interval. Neither panel measures true-distance coverage.'),
        outputs={p.name:sha(p) for p in out.iterdir() if p.name!='complete.json'}))


if __name__=='__main__':
    main()
