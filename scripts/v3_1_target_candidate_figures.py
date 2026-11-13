#!/usr/bin/env python3
"""Render authenticated compact candidate tables, retaining failed numerical gates."""
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from v2_1_near import ROOT,sha,save_json,csv_rows
from photdis_figure_style import STYLE,BLUE,ORANGE,GRAY

VIEWS=[('zero','Zero correction',GRAY,'o'),('point','Point correction',ORANGE,'s'),
       ('bootstrap_evidence','Evidence-weighted bootstrap',BLUE,'D'),('bootstrap_cut','Equal-weight cut','#6f732f','^')]
VARIANTS=['base','entry_seed','entry_level','native_fine']
VARIANT_LABELS=['Base','Independent\nentry seed','Previous\nentry level','Native fine\n(36 objects)']


def main():
    source=ROOT/'manifests/v3_1/target_ensemble/candidate'; done=json.loads((source/'complete.json').read_text())
    assert done['states']==34 and done['all_PSF'] and all(v['inputs_passed'] and v['n']==3540 for v in done['views'])
    paths=[source/'complete.json']; tables={}
    for name in ['counts','weights','checks']:
        p=source/(name+'.csv'); assert sha(p)==done['outputs'][p.name]; paths.append(p)
        with p.open() as f:tables[name]=list(csv.DictReader(f))
    counts=tables['counts']; assert len(counts)==4*4*18
    assert len({(r['view'],r['variant'],r['class_index'],r['distance_bin']) for r in counts})==len(counts)
    assert all(int(r['n'])==3540 and float(r['mean'])>=0 and 0<=int(r['lower95'])<=int(r['median'])<=int(r['upper95'])<=3540 for r in counts)
    main_view=next(v for v in done['views'] if v['view']=='bootstrap_evidence')
    assert not main_view['accepted'],'Revise visible gate status if a different authenticated result is supplied'
    main_checks=[r for r in tables['checks'] if r['view']=='bootstrap_evidence']; assert len(main_checks)==4
    out=ROOT/'figures/v3_1/target_candidate'; out.mkdir(parents=True,exist_ok=True)
    meta=ROOT/'manifests/v3_1/target_candidate_figures'; meta.mkdir(exist_ok=True)
    plt.rcParams.update(STYLE); outputs={}; texts=[]; plotted=[]

    def finish(fig,name):
        fig.canvas.draw()
        labels=[t.get_text() for t in fig.findobj(matplotlib.text.Text)];texts.extend(labels)
        assert not any(any('\u4e00'<=c<='\u9fff' for c in t) for t in labels)
        for ext in ['png','pdf']:
            p=out/(name+'.'+ext);fig.savefig(p,dpi=300);outputs[str(p.relative_to(ROOT))]=sha(p)
        plt.close(fig)

    fig,axes=plt.subplots(3,1,figsize=(13,12),sharex=True,sharey=True)
    fig.subplots_adjust(left=.12,right=.97,top=.79,bottom=.17,hspace=.28)
    for c,(ax,label) in enumerate(zip(axes,['Dwarf','Evolved','Other'])):
        for vi,(view,_,color,marker) in enumerate(VIEWS):
            rows=sorted([r for r in counts if r['view']==view and r['variant']=='base' and int(r['class_index'])==c],key=lambda r:int(r['distance_bin']))
            assert len(rows)==6; x=np.arange(6)+(vi-1.5)*.16
            for pos,r in zip(x,rows):
                ax.plot([pos,pos],[float(r['lower95']),float(r['upper95'])],color=color,lw=1.8)
                ax.plot(pos,float(r['mean']),marker=marker,ms=5.5,color=color,ls='none')
                plotted.append(dict(figure='conditional_counts',panel=label,source='counts.csv',**r))
        ax.set_title(label,loc='left',fontsize=14);ax.set_yscale('symlog',linthresh=1,linscale=1)
        ax.set_ylim(-.06,2600);ax.set_yticks([0,1,10,100,1000],['0','1','10','100','1000']);ax.set_ylabel('Expected count',fontsize=12)
        ax.set_xlim(-.55,5.55);ax.grid(axis='x',visible=False);ax.tick_params(labelsize=11)
    axes[-1].set_xticks(np.arange(6),['0–2','2–5','5–10','10–20','20–50','>50'])
    axes[-1].set_xlabel('Distance bin (kpc)',fontsize=13)
    handles=[Line2D([],[],color=color,marker=marker,lw=1.8,label=label) for _,label,color,marker in VIEWS]
    fig.legend(handles=handles,ncol=2,loc='upper center',bbox_to_anchor=(.54,.89),frameon=False,fontsize=11)
    fig.suptitle('Conditional candidate class–distance counts',y=.975,fontsize=19)
    fig.text(.5,.935,'3540 observed native DR11 PSF objects | fixed response12 / calibration12 | shared global draws',ha='center',fontsize=11)
    fig.text(.5,.81,'Original absolute-evidence seed gate failed for the main candidate; results retain that flag.',ha='center',fontsize=11,color='#8b4e14')
    fig.text(.5,.115,'Points: conditional expectations. Bars: 95% realized-count intervals from shared Monte Carlo catalogs.',ha='center',fontsize=10)
    half=[float(next(r for r in counts if r['view']==v and r['variant']=='base')['count_CDF_half_delta']) for v,_,_,_ in VIEWS]
    fig.text(.5,.085,'CDF half-sample discrepancies (legend order): '+', '.join(f'{x:.4f}' for x in half)+'. Y scale is linear below 1, logarithmic above.',ha='center',fontsize=10)
    fig.text(.5,.05,'Conditional model counts, not measured populations or physical densities. External validation remains pending.',ha='center',fontsize=10)
    finish(fig,'conditional_counts')

    fig,axes=plt.subplots(2,2,figsize=(15,10))
    fig.subplots_adjust(left=.09,right=.97,top=.82,bottom=.17,hspace=.65,wspace=.3)
    ax=axes[0,0]; weights=sorted([r for r in tables['weights'] if r['view']=='bootstrap_evidence' and r['variant']=='base'],key=lambda r:int(r['correction']))
    assert len(weights)==32 and [int(r['correction']) for r in weights]==list(range(2,34))
    ax.bar([int(r['correction']) for r in weights],[float(r['weight'])*100 for r in weights],width=.8,color=BLUE)
    ax.axhline(100/32,color='.25',ls='--',lw=1,label='Uniform finite prior')
    ax.set(xlim=(1,34),ylim=(0,8),xlabel='Frozen correction state',ylabel='Posterior weight (%)')
    ax.set_xticks([2,6,10,14,18,22,26,30,33]);ax.set_title('A | Finite luminosity-bootstrap weights',loc='left',fontsize=13)
    ax.legend(frameon=False,fontsize=10,loc='upper right');ax.text(.02,.96,f"ESS = {main_view['state_ESS']:.2f} / 32\nMaximum = {100*main_view['maximum_state_weight']:.2f}%",transform=ax.transAxes,va='top',fontsize=10)
    plotted.extend(dict(figure='weights_numerics',panel='A',source='weights.csv',**r) for r in weights)
    specifications=[(axes[0,1],'log_evidence_delta',.05,'B | Evidence sensitivity','Change in log evidence (nat)',(-.075,.29)),
        (axes[1,0],'maximum_probability_delta',.003,'C | Joint probability sensitivity','Maximum absolute probability change',(-.00012,.0036)),
        (axes[1,1],'maximum_count_mean_delta',.5,'D | Count expectation sensitivity','Maximum absolute count change',(-.02,.6))]
    ordered=[next(r for r in main_checks if r['variant']==v) for v in VARIANTS]
    for ax,key,limit,title,ylabel,ylim in specifications:
        values=[float(r[key]) for r in ordered];ax.plot(range(4),values,color=BLUE,marker='D',ms=5,lw=1.5)
        ax.axhline(limit,color='.25',ls='--',lw=1,label=f'Gate: {limit:g}')
        if key=='log_evidence_delta':ax.axhline(-limit,color='.25',ls='--',lw=1)
        ax.axhline(0,color='.5',lw=.7);ax.set_ylim(*ylim);ax.set_xlim(-.35,3.35)
        ax.set_xticks(range(4),VARIANT_LABELS,fontsize=10);ax.set_ylabel(ylabel,fontsize=11)
        ax.set_title(title,loc='left',fontsize=13);ax.legend(frameon=False,fontsize=10,loc='upper right')
        for i,value in enumerate(values):ax.annotate(f'{value:.4g}',(i,value),xytext=(0,7),textcoords='offset points',ha='center',fontsize=9)
        plotted.extend(dict(figure='weights_numerics',panel=title,source='checks.csv',display_metric=key,**r) for r in ordered)
    fig.suptitle('Candidate calibration-state support and numerical sensitivity',y=.975,fontsize=19)
    fig.text(.5,.93,'Main identity: 3540-object candidate cohort, response12 / calibration12, evidence-weighted luminosity bootstraps',ha='center',fontsize=11)
    fig.text(.5,.88,'Independent entry seed: evidence gate failed; probability, count and state-weight sensitivity gates passed.',ha='center',fontsize=11,color='#8b4e14')
    fig.text(.5,.10,'Native fine transfer audits 36 / 3540 candidate objects. The original failure is retained; no acceptance threshold changed.',ha='center',fontsize=10)
    fig.text(.5,.055,'Finite luminosity-bootstrap support is not the independent 64-SED response experiment or an absolute calibration guarantee.',ha='center',fontsize=10)
    finish(fig,'weights_numerics')
    csv_rows(meta/'plotted.csv',plotted);outputs[str((meta/'plotted.csv').relative_to(ROOT))]=sha(meta/'plotted.csv')
    save_json(meta/'complete.json',dict(inputs={str(p.relative_to(ROOT)):sha(p) for p in paths+[ROOT/'scripts/v3_1_target_candidate_figures.py',ROOT/'scripts/photdis_figure_style.py']},
        outputs=outputs,plotted_rows=len(plotted),figures=2,panels=7,english_text_verified=True,visual_review_pending=True,
        main_candidate_accepted=False,native_fine_audit_n=36,parent_n=3540,
        scope='Rendering of authenticated compact computed tables only. Original seed evidence failure and Monte Carlo interval precision remain visible; no new catalog computation or external reference access.'))


if __name__=='__main__':main()
