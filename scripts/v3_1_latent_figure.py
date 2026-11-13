#!/usr/bin/env python3
"""Render allocated latent-support/failure summaries, without catalog analysis."""
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from v2_1_near import ROOT,sha,save_json,csv_rows
from photdis_figure_style import STYLE,BLUE,ORANGE,GRAY


def main():
    source=ROOT/'manifests/v3_1/diagnostics/diagnostic/latent_failure_report';done=json.loads((source/'complete.json').read_text())
    for name in ('summary.csv','objects.csv'):assert sha(source/name)==done['outputs'][name]
    with (source/'summary.csv').open() as f:summary=list(csv.DictReader(f))
    with (source/'objects.csv').open() as f:objects=[r for r in csv.DictReader(f) if r['control']=='response12']
    def get(label,group,metric,reference='none'):
        rr=[r for r in summary if r['control']=='response12' and r['reference_class']==label and r['negative_group']==group and r['metric']==metric and r['source']==reference]
        assert len(rr)==1;return rr[0]
    out=ROOT/'figures/v3_1/latent';out.mkdir(parents=True,exist_ok=True);meta=ROOT/'manifests/v3_1/latent_figure';meta.mkdir(exist_ok=True)
    plt.rcParams.update(STYLE);fig,axes=plt.subplots(2,2,figsize=(16,11))
    fig.subplots_adjust(left=.10,right=.97,bottom=.17,top=.83,wspace=.40,hspace=.62);plotted=[]
    groups=[('-1','Unknown','#7856a3'),('0','Dwarf',BLUE),('1','Evolved',ORANGE),('2','Other',GRAY)]
    ax=axes[0,0]
    for x,(label,name,color) in enumerate(groups):
        r=get(label,'all','negative_upper');lo,mean,hi=[100*float(r[k]) for k in ['lo95','mean','hi95']]
        ax.plot([x,x],[lo,hi],color=color,lw=3);ax.scatter([x],[mean],color=color,s=55,zorder=3)
        ax.text(x,hi*1.35,f"N={r['n']}\nsky={r['cells']}",ha='center',fontsize=9);plotted.append(dict(panel='A',**r))
    ax.set_yscale('log');ax.set_ylim(.0005,5);ax.set_xlim(-.5,3.5);ax.set_xticks(range(4),[g[1] for g in groups]);ax.set_ylabel('Weighted any-negative upper bound (%)',fontsize=11)
    ax.set_title('A | Rare physical-support failures',loc='left',fontsize=13);ax.grid(axis='y',alpha=.2)
    ax=axes[0,1]
    for label,name,color in groups:
        rows=[r for r in objects if r['reference_class']==label]
        ax.scatter([100*float(r['W1_negative']) for r in rows],[100*float(r['W2_negative']) for r in rows],s=12,alpha=.55,color=color,label=name,rasterized=True)
        plotted.extend(dict(panel='B',**r) for r in rows)
    ax.set_xlabel('Intrinsic W1 negative probability (%)',fontsize=11);ax.set_ylabel('Intrinsic W2 negative probability (%)',fontsize=11)
    ax.set_title('B | W1 and W2 latent-negative probabilities',loc='left',fontsize=13);ax.legend(fontsize=9,loc='upper right');ax.grid(alpha=.2)
    intervals=['below0.001','0.001to0.01','0.01to0.1','atleast0.1'];labels=['<0.1%','0.1–1%','1–10%','>=10%']
    ax=axes[1,0];ticks=[]
    for y,(group,label) in enumerate(zip(intervals,labels)):
        r=get('1',group,'severe_abs_mu_gt1','primary');ticks.append(f"{label}\nN={r['n']}, sky={r['cells']}");plotted.append(dict(panel='C',**r))
        if int(r['n'])==0:ax.text(50,y,'No reference objects',ha='center',va='center',fontsize=9,color=GRAY);continue
        if r['lo95']:ax.plot([100*float(r['lo95']),100*float(r['hi95'])],[y,y],lw=3,color=ORANGE)
        ax.scatter([100*float(r['mean'])],[y],color=ORANGE,s=55,zorder=3)
    ax.set_yticks(range(4),ticks,fontsize=9);ax.set_xlim(-2,102);ax.set_ylim(3.5,-.5);ax.set_xlabel('Weighted fraction with |mu error| > 1 mag (%)',fontsize=11)
    ax.set_ylabel('Any-negative upper-bound group',fontsize=10);ax.set_title('C | Evolved-star failure persists at low negativity',loc='left',fontsize=13);ax.grid(axis='x',alpha=.2)
    ax=axes[1,1]
    for x,(group,label) in enumerate(zip(intervals,labels)):
        total=get('-1',group,'negative_upper');reference=get('-1',group,'severe_abs_mu_gt1','primary')
        n=int(total['n']);nr=int(reference['n']);ax.bar(x,n,color='#d9cbea',width=.7);ax.bar(x,nr,color='#7856a3',width=.7)
        ax.text(x,n+4,f'{nr}/{n}',ha='center',fontsize=10);plotted.extend([dict(panel='D',**total),dict(panel='D',**reference)])
    handles=[Patch(color='#7856a3',label='With primary distance reference'),Patch(color='#d9cbea',label='No primary distance reference')]
    ax.set_ylim(0,215);ax.set_xticks(range(4),labels);ax.set_ylabel('Observed objects (unweighted)',fontsize=11);ax.set_xlabel('Any-negative upper-bound group',fontsize=11)
    ax.set_title('D | Unknown-class outliers lack distance references',loc='left',fontsize=13);ax.legend(handles=handles,fontsize=9,loc='upper right');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Latent-flux support and measurement failure: same-object development audit',fontsize=17,y=.96)
    fig.text(.5,.91,'1,542 native DR11 PSF objects | response12 | fixed reference prior | exploratory sky-block intervals',ha='center',fontsize=12)
    fig.text(.5,.87,'A and C retain design weights. B shows individual probabilities. D shows raw evidence denominators.',ha='center',fontsize=11)
    fig.text(.5,.105,'Negative observed flux is legal; these probabilities concern the latent intrinsic-flux model.',ha='center',fontsize=11)
    fig.text(.5,.065,'Associations do not establish cause. Low-negativity evolved stars still have severe reference disagreement.',ha='center',fontsize=11)
    fig.text(.5,.027,'No new validation outcomes or physical population contamination measurements are shown.',ha='center',fontsize=11)
    text=[t.get_text() for t in fig.findobj(matplotlib.text.Text)];assert not any(any('\u4e00'<=c<='\u9fff' for c in t) for t in text)
    outputs={}
    for ext in ('png','pdf'):
        path=out/f'latent_support_failure.{ext}';fig.savefig(path,dpi=300);outputs[str(path.relative_to(ROOT))]=sha(path)
    plt.close(fig);keys=sorted(set().union(*(r.keys() for r in plotted)));csv_rows(meta/'plotted.csv',[{k:r.get(k) for k in keys} for r in plotted]);outputs[str((meta/'plotted.csv').relative_to(ROOT))]=sha(meta/'plotted.csv')
    save_json(meta/'complete.json',dict(inputs={str(p.relative_to(ROOT)):sha(p) for p in [source/'complete.json',source/'summary.csv',source/'objects.csv',ROOT/'scripts/v3_1_latent_figure.py',ROOT/'scripts/photdis_figure_style.py']},
        outputs=outputs,plotted_rows=len(plotted),english_text_verified=True,visual_review_pending=True,
        scope='Rendering of compact allocated diagnostic tables only; same-star associations and missing-reference denominators, not causal inference.'))


if __name__=='__main__':main()
