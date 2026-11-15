#!/usr/bin/env python3
"""Plot the frozen weak-parallax evidence and direct targeted giant detections."""
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from photdis_figure_style import BLUE, GRAY, GREEN, ORANGE, STYLE
from v3_6_plot_shell_retention import FIGURE, ROOT, TABLE, sha


INPUT = ROOT/'manifests/v3_5/validation'
MARKER = INPUT/'complete.json'
CUTS = ('primary','strict')
COHORTS = {
    'all':'Full draw',
    'r18_20':'Faint',
    'r18_20_color05_08':'Faint blue',
    'v34_analog_outside_original_sky':'Bright red, new footprint',
}
CLASSES = {'dwarfs':'Dwarf','near_giant':'Giant inside 10 kpc',
           'far_giant':'Giant beyond 10 kpc','unresolved_giant':'Giant, distance unresolved',
           'no_label':'Class unresolved'}


def open_rows(filename):
    path = INPUT/filename
    if sha(path)!=json.loads(MARKER.read_text())['outputs'][filename]:
        raise ValueError(f'{path} differs from the frozen validation source hash')
    with path.open() as stream:
        return list(csv.DictReader(stream)),path


def deliver(fig,name,rows,path,caption):
    TABLE.mkdir(parents=True,exist_ok=True)
    FIGURE.mkdir(parents=True,exist_ok=True)
    table = TABLE/f'{name}.csv'
    with table.open('w',newline='') as stream:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        writer = csv.DictWriter(stream,fieldnames=fields,lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    pictures = [FIGURE/f'{name}.{ext}' for ext in ('png','pdf')]
    for file in pictures:
        fig.savefig(file,dpi=300)
    plt.close(fig)
    (TABLE/f'{name}.json').write_text(json.dumps(dict(caption=caption,dpi=300,
        inputs={str(p.relative_to(ROOT)):sha(p) for p in (MARKER,path)},
        outputs={str(p.relative_to(ROOT)):sha(p) for p in (table,*pictures)}),indent=2)+'\n')


def gaia():
    data,path = open_rows('gaia_joint.csv')
    by_key = {(r['selection'],r['group'],float(r['lower_kpc'])):r for r in data}
    rows = []
    x = np.arange(len(COHORTS))
    with plt.rc_context(STYLE):
        fig,(a,b) = plt.subplots(1,2,figsize=(13.6,5.2),layout='constrained')
        for j,cut in enumerate(CUTS):
            color = (BLUE,GREEN)[j]
            qrf_prior = []
            for group,label in COHORTS.items():
                r = by_key[cut,group,0.]
                qrf,prior = (float(r[k]) for k in ('weak_signed_score_qrf','weak_signed_score_prior'))
                n = int(r['weak_gaia_outside_history'])
                qrf_prior.append((qrf-prior,n))
                rows.append(dict(panel='weak_log_score',selection=cut,cohort=label,
                    weak_outside_history_n=n,log_score_qrf=qrf,log_score_prior=prior,delta_qrf_prior=qrf-prior))
            dx = (j-.5)*.37
            for i,(gain,n) in enumerate(qrf_prior):
                a.bar(i+dx,gain,width=.33,color=color,label=cut.title() if i==0 else None)
                a.text(i+dx,gain+.015,f'{n:,}',ha='center',va='bottom',fontsize=8,rotation=90)
            faint = [by_key[cut,'r18_20_color05_08',shell] for shell in (0.,2.,5.)]
            total = float(faint[0]['projected_n'])
            masses = [float(r['qrf_mass']) for r in faint]
            bounds = [(float(r['shell_fraction_lower']),float(r['shell_fraction_upper']),
                       int(r['shell_interval_inside'])) for r in faint]
            masses.append(total-sum(masses))
            bounds.append((float(faint[0]['far_fraction_lower']),float(faint[0]['far_fraction_upper']),
                           int(faint[0]['gaia_far_interval'])))
            for i,(label,mass,(lo,hi,count)) in enumerate(zip(('0–2','2–5','5–10','>10'),masses,bounds)):
                b.vlines(i+dx,lo,hi,color=color,lw=3,alpha=.65)
                b.plot([i+dx,i+dx],[lo,hi],'_',color=color,ms=9)
                b.scatter(i+dx,mass/total,marker='o',c=color,s=27,zorder=3)
                rows.append(dict(panel='Gaia_interval_bounds',selection=cut,cohort='Faint blue',
                                 distance_kpc=label,draw_n=int(faint[0]['draw']),interval_inside_n=count,
                                 interval_count_lower=lo,interval_count_upper=hi,predicted_qrf_fraction=mass/total))
        a.set(title='Frozen signed-parallax score',ylabel='QRF vs. broad prior [nat per star]',
              xticks=x,xticklabels=('Full draw','Faint','Faint blue','Bright red\nnew footprint'),ylim=(0,1.38))
        a.legend(frameon=False,loc='upper left')
        b.set(title='Faint-blue distance constraints',ylabel='Fraction of the selection',xticks=x,
              xticklabels=('0–2','2–5','5–10','>10'),xlabel='Distance [kpc]',ylim=(0,1))
        b.yaxis.set_major_formatter(PercentFormatter(1))
        b.text(.02,.97,'Dots: QRF mass   Bars: Gaia interval counts',transform=b.transAxes,va='top',fontsize=10)
        for ax in (a,b):
            ax.xaxis.grid(False)
        caption = ('Left: the original v3.5 weighted signed-parallax log-score contrast for corrected '
                   'S/N < 5 outside recorded history, with unweighted counts on bars. The comparator '
                   'is uniform in distance modulus over 0.1–20 kpc. These are the original predictive '
                   'scores evaluated on that subset, before the v3.6 score conditional on the weak '
                   'parallax selection. Right: weighted counts of Gaia parallax intervals disjoint '
                   'from or entirely within each distance shell. Dots are the QRF mass fractions '
                   'on the same probability draw. Bounds describe interval ambiguity per shell, '
                   'including sources without decisive reference information.')
        deliver(fig,'F08_frozen_weak_parallax',rows,path,caption)


def spectra():
    records,path = open_rows('targeted_spectra_joint.csv')
    totals = {}
    for r in records:
        if r['group']!='all':
            continue
        k = r['survey'],r['selection']
        values = tuple(int(r[label]) for label in CLASSES)
        n = int(r['matched'])
        if sum(values)!=n or (k in totals and totals[k]!=(n,values)):
            raise ValueError('Spectroscopic class totals do not reconcile across reference bins')
        totals[k] = n,values
    names = ('DR20_MINESweeper','DR20_APO','DR19_Astra')
    labels = ('DR20 LCO','DR20 APO','DR19 Astra')
    colors = (BLUE,GREEN,ORANGE,'#a478bb',GRAY)
    rows = []
    with plt.rc_context(STYLE):
        fig,axes = plt.subplots(1,2,figsize=(13.4,5.2),layout='constrained')
        limit = 1.12*max(n for n,_ in totals.values())
        for ax,cut in zip(axes,CUTS):
            y = np.arange(len(names))
            counts = np.asarray([totals[name,cut][1] for name in names])
            lower = np.zeros(len(names))
            for j,((key,label),color) in enumerate(zip(CLASSES.items(),colors)):
                ax.barh(y,counts[:,j],left=lower,alpha=.84,color=color,label=label)
                for i,name in enumerate(names):
                    n = counts[i,j]
                    rows.append(dict(survey=name,selection=cut,group=key,targeted_count=int(n),
                                     matched=totals[name,cut][0]))
                    if n>=12:
                        ax.text(lower[i]+n/2,i,str(n),ha='center',va='center',fontsize=10)
                lower += counts[:,j]
            for i,n in enumerate(lower):
                ax.text(n+limit*.012,i,str(int(n)),ha='left',va='center',fontsize=10)
            ax.invert_yaxis()
            ax.set(yticks=y,yticklabels=labels,title=cut.title()+' selection',
                   xlabel='Targeted spectroscopic matches',xlim=(0,limit))
            ax.yaxis.grid(False)
        handles,labels = axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,frameon=False,fontsize=10,loc='outside lower center',ncols=3)
        caption = ('Direct spectroscopic evidence in the frozen targeting. Distance categories use '
                   'the published reference qualification and require the interval to lie entirely '
                   'inside or outside 10 kpc for the giant near and far labels. Each bar represents one '
                   'survey and one catalog cut. Targeting and unknown distances limit interpretation '
                   'to these specific matches. The two cuts overlap.')
        deliver(fig,'F09_frozen_spectroscopic_leakage',rows,path,caption)


if __name__=='__main__':
    gaia()
    spectra()
