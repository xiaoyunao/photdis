#!/usr/bin/env python3
"""Small-table scientific summaries in the established English Pal5 style."""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from v2_1_near import ROOT,sha,save_json,csv_rows
from photdis_figure_style import STYLE,BLUE,GREEN,ORANGE,GRAY
SOURCE=ROOT/'manifests/v3_0/diagnostics'
OUT=ROOT/'figures/v3_0/diagnostics';TABLE=ROOT/'manifests/v3_0/figures/diagnostics'


def read(name):return list(csv.DictReader((SOURCE/(name+'.csv')).open()))


def save(fig,name,rows,sources,caption):
    OUT.mkdir(parents=True,exist_ok=True);TABLE.mkdir(parents=True,exist_ok=True)
    texts=[t.get_text() for t in fig.findobj(matplotlib.text.Text)]
    assert not any(any('\u4e00'<=c<='\u9fff' for c in t) for t in texts)
    fig.savefig(OUT/(name+'.png'),dpi=300);fig.savefig(OUT/(name+'.pdf'));plt.close(fig)
    csv_rows(TABLE/(name+'.csv'),rows)
    save_json(TABLE/(name+'.json'),dict(caption=caption,sources={n:sha(SOURCE/(n+'.csv')) for n in sources},
        png_sha256=sha(OUT/(name+'.png')),pdf_sha256=sha(OUT/(name+'.pdf')),csv_sha256=sha(TABLE/(name+'.csv')),
        english_text_verified=True,png_dpi=300,scope='Seen development; reference-conditioned evidence, not independent truth.'))


def main():
    plt.rcParams.update(STYLE)
    rows=read('information');groups=[('all','All labeled objects'),('dwarf','Dwarfs'),('evolved','Evolved reference'),('other','Other stars')]
    methods=[('probe_no_color','No color',GRAY),('probe_grz','Probe grz',BLUE),('probe_multiband','Probe multiband',GREEN),('v3_grz','v3 grz',ORANGE),('v3_multiband','v3 multiband','#8C564B')]
    fig,axes=plt.subplots(1,4,figsize=(16,4.8));selected=[]
    for ax,(g,title) in zip(axes,groups):
        for j,(m,label,color) in enumerate(methods):
            r=next(x for x in rows if x['method']==m and x['group']==g);selected.append(r)
            ax.bar(j,float(r['log_score']),color=color,width=.72)
        ax.set(title=title,xticks=range(len(methods)),xticklabels=[x[1] for x in methods])
        ax.tick_params(axis='x',rotation=35,labelsize=10);plt.setp(ax.get_xticklabels(),ha='right');ax.axhline(0,color='black',lw=.7)
        ax.text(.98,.05,f"N = {int(r['n']):,}",ha='right',transform=ax.transAxes,fontsize=11,bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
    fig.supylabel('Mean true-class log probability',x=.012,fontsize=14)
    fig.suptitle('Class information on identical development identities',y=.98,fontsize=18)
    fig.subplots_adjust(left=.065,right=.99,top=.83,bottom=.31,wspace=.4)
    save(fig,'class_information',selected,['information'],'Same labels and empirical composition. Missing spectral classes are not a fourth physical category; log score is higher when better.')
    rows=read('information_gain');fig,axes=plt.subplots(1,3,figsize=(13.5,4.8));selected=[]
    for ax,(g,title) in zip(axes,[('all','All labeled objects'),('evolved','Evolved reference'),('poor_dwarf','Metal-poor dwarfs')]):
        for j,(m,label,color) in enumerate(methods[1:]):
            r=next(x for x in rows if x['method']==m and x['group']==g);selected.append(r)
            value=float(r['logscore_gain']);lo=float(r['lo95']);hi=float(r['hi95'])
            ax.plot([lo,hi],[j,j],color=color,lw=2);ax.plot(value,j,'o',color=color,label=label)
        ax.axvline(0,color='black',ls='--',lw=.8);ax.set(title=title,yticks=range(4),yticklabels=[x[1] for x in methods[1:]])
        ax.tick_params(axis='y',labelsize=11)
    fig.supxlabel('Log-score gain over no-color probe',y=.04,fontsize=14)
    fig.suptitle('Information gains depend strongly on stellar population',fontsize=18)
    fig.subplots_adjust(left=.16,right=.98,bottom=.16,top=.8,wspace=.82)
    save(fig,'class_information_gain',selected,['information_gain'],'Paired 512 sky-cell bootstrap intervals; resampled seen cells, not new sky or independent stars.')
    rows=read('near_selection');fig,axes=plt.subplots(2,3,figsize=(14,8));selected=[]
    for col,(g,title) in enumerate([('all','All reference objects'),('dwarf','Dwarfs'),('evolved','Evolved reference')]):
        for m,label,color in [('qrf','Matched QRF',GRAY),('grz','v3 grz',BLUE),('multiband','v3 multiband',ORANGE)]:
            r=[x for x in rows if x['model']==m and x['reference']=='primary' and x['group']==g];r.sort(key=lambda x:float(x['threshold']));selected.extend(r)
            x=[float(t['threshold']) for t in r]
            axes[0,col].plot(x,[100*float(t['far_fraction']) if t['far_fraction'] else np.nan for t in r],'o-',label=label,color=color)
            axes[1,col].plot(x,[100*float(t['severe_fraction']) if t.get('severe_fraction') else np.nan for t in r],'o-',label=label,color=color)
        axes[0,col].set(title=title,ylabel='Reference far fraction (%)')
        axes[1,col].set(xlabel='Near-probability threshold',ylabel='Severe distance error (%)',xticks=[.5,.8,.95])
        axes[0,col].legend(fontsize=11,frameon=False)
    fig.suptitle('Near membership and selected distance quality are different tests',fontsize=18)
    fig.subplots_adjust(left=.08,right=.98,bottom=.09,top=.86,hspace=.28,wspace=.3)
    save(fig,'near_selection_risk',selected,['near_selection'],'Unclipped primary reference, with all distant contaminants retained. Severe error: absolute distance-modulus residual >1 mag. Reference selection does not determine application population composition.')
    rows=read('score_tails');fig,axes=plt.subplots(1,2,figsize=(12,5));selected=[]
    references=[('primary','Primary'),('specdis','SpecDis'),('spdist','SPDist'),('gaia_precise','Precise Gaia')]
    for j,(reference,label) in enumerate(references):
        r=next(x for x in rows if x['model']=='multiband' and x['group']=='all' and x['reference']==reference);selected.append(r)
        for ax in axes:
            ax.plot(j-.15,float(r['gain_mean']),'o',color=ORANGE,label='Mean gain' if j==0 else None)
            ax.plot(j,float(r['gain_median']),'s',color=BLUE,label='Median gain' if j==0 else None)
            ax.plot(j+.15,float(r['remainder_gain_mean']),'^',color=GREEN,label='Mean outside largest 1%' if j==0 else None)
    for ax in axes:
        ax.set(xticks=range(4),xticklabels=[x[1] for x in references],ylabel='v3 multiband minus QRF log score')
        ax.axhline(0,color='black',ls='--',lw=.8);ax.tick_params(axis='x',labelsize=12)
    axes[0].set_title('Full score difference');axes[1].set(title='Central differences',ylim=(-.12,.12))
    axes[0].legend(fontsize=10,frameon=False);fig.suptitle('Large average score gains are dominated by rare conflicts',fontsize=17)
    fig.subplots_adjust(left=.08,right=.98,bottom=.13,top=.8,wspace=.3)
    save(fig,'score_tail_sensitivity',selected,['score_tails'],'Largest 1% is defined by absolute paired score difference, separately per source. This is an explanatory sensitivity, not removal from primary scientific metrics.')
    files=sorted(p for p in TABLE.glob('*.json') if p.name not in ('complete.json','validation.json'));save_json(TABLE/'complete.json',dict(figures=len(files),files={str(p.relative_to(ROOT)):sha(p) for p in files}))
    print('SUMMARY FIGURES',len(files))


if __name__=='__main__':main()
