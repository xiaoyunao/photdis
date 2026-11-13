#!/usr/bin/env python3
"""Render compact, already computed calibration tables; no catalog analysis."""
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from v2_1_near import ROOT,sha,save_json,csv_rows
from photdis_figure_style import STYLE,BLUE,ORANGE,GRAY


def main():
    source=ROOT/'manifests/v3_1'; fit=source/'luminosity_fit_refined'; guard=source/'luminosity_guard'
    paths=[fit/'complete.json',fit/'parameters.csv',guard/'complete.json',guard/'comparisons.csv']
    for folder,name in [(fit,'parameters.csv'),(guard,'comparisons.csv')]:
        record=json.loads((folder/'complete.json').read_text()); assert sha(folder/name)==record['outputs'][name]
    done=json.loads(paths[2].read_text()); assert done['numerical_passed'] and done['guard_passed']
    with paths[1].open() as f: parameters=list(csv.DictReader(f))
    with paths[3].open() as f: scores=list(csv.DictReader(f))
    out=ROOT/'figures/v3_1/calibration'; out.mkdir(parents=True,exist_ok=True)
    meta=source/'calibration_figure';meta.mkdir(exist_ok=True)
    plt.rcParams.update(STYLE); fig,axes=plt.subplots(2,3,figsize=(18,10))
    fig.subplots_adjust(left=.10,right=.98,top=.82,bottom=.17,hspace=.65,wspace=.58)
    groups=[('0','Dwarf',BLUE),('1','Evolved',ORANGE),('2','Other',GRAY)]; plotted=[]
    for ax,key,title in [(axes[0,0],'delta_mu','A | Luminosity shift (mag)'),(axes[0,1],'sigma_mu','B | Extra luminosity dispersion (mag)')]:
        for y,(c,label,color) in enumerate(groups):
            r=next(r for r in parameters if r['reference_class']==c and r['parameter']==key)
            ax.plot([float(r['lower95']),float(r['upper95'])],[y,y],lw=3,color=color)
            ax.scatter([float(r['point'])],[y],s=50,color=color,zorder=3)
            plotted.append(dict(panel=title,source='fit',**r))
        ax.set_yticks(range(3),[g[1] for g in groups]);ax.set_title(title,loc='left',fontsize=12)
        ax.set_xlabel('Point fit and 95% shared-sky bootstrap range',fontsize=10)
        ax.axvline(0,color='black',ls='--',lw=.8)
    specs=[(axes[0,2],'fit_channel_logscore',1.,'C | Predictive log-score change','Change in mean log score (nat)'),
        (axes[1,0],'coverage68',100.,'D | 68% reference-point agreement','Change (percentage points)'),
        (axes[1,1],'coverage95',100.,'E | 95% reference-point agreement','Change (percentage points)'),
        (axes[1,2],'gaia_tail',100.,'F | Signed-Gaia PIT tails','Change (percentage points)')]
    for ax,key,scale,title,label in specs:
        labels=[]
        for y,(c,name,color) in enumerate(groups):
            r=next(r for r in scores if r['variant']=='bootstrap' and r['reference_class']==c and r['metric']==key)
            lo,mean,hi=[scale*float(r[k]) for k in ['lo95','mean','hi95']]
            ax.plot([lo,hi],[y,y],lw=3,color=color);ax.scatter([mean],[y],s=50,color=color,zorder=3)
            labels.append(f"{name}\nN={r['n']}, sky={r['cells']}");plotted.append(dict(panel=title,source='guard',**r))
        ax.set_yticks(range(3),labels,fontsize=9);ax.set_title(title,loc='left',fontsize=12);ax.set_xlabel(label,fontsize=10)
        ax.axvline(0,color='black',ls='--',lw=.8)
    for ax in axes.flat:
        ax.set_ylim(2.5,-.5);ax.grid(axis='x',color='.92');ax.set_axisbelow(True)
    fig.suptitle('Constrained luminosity correction: calibration fit and held-out sky guards',fontsize=18,y=.965)
    fig.text(.5,.91,'Native DR11 PSF | reference prior | response12 | 32 shared calibration bootstrap draws',ha='center',fontsize=12)
    fig.text(.5,.865,'Evolved correction: only 3 fitting sky blocks; held-out guard: 39 stars in 2 sky blocks',ha='center',fontsize=12,color='#9b4b13')
    fig.text(.5,.105,'C–E: positive is better. F: negative is better. Guard intervals use the available sky blocks.',ha='center',fontsize=11)
    fig.text(.5,.067,'Reference-point agreement is not true coverage. Sparse sky/source ancestry limits independent calibration claims.',ha='center',fontsize=11)
    fig.text(.5,.03,'All 15 bootstrap guards and 96 numerical checks passed; final target-prior and external validation remain pending.',ha='center',fontsize=11)
    texts=[x.get_text() for x in fig.findobj(matplotlib.text.Text)]
    assert not any(any('\u4e00'<=c<='\u9fff' for c in t) for t in texts)
    outputs={}
    for ext in ['png','pdf']:
        path=out/f'luminosity_correction.{ext}';fig.savefig(path,dpi=300);outputs[str(path.relative_to(ROOT))]=sha(path)
    plt.close(fig);csv_rows(meta/'plotted.csv',plotted);outputs[str((meta/'plotted.csv').relative_to(ROOT))]=sha(meta/'plotted.csv')
    save_json(meta/'complete.json',dict(inputs={str(p.relative_to(ROOT)):sha(p) for p in paths+[ROOT/'scripts/v3_1_calibration_figure.py',ROOT/'scripts/photdis_figure_style.py']},
        outputs=outputs,plotted_rows=len(plotted),english_text_verified=True,visual_review_pending=True,
        scope='Rendering of compact computed tables only. No catalog scan or new sample; calibration guards, not final validation.'))


if __name__=='__main__':main()
