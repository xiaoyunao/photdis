#!/usr/bin/env python3
"""All-scenario radial recovery plates from completed authenticated summaries."""
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from v2_1_near import ROOT,sha,save_json,csv_rows
from photdis_figure_style import STYLE,BLUE,ORANGE


def main():
    source=ROOT/'manifests/v3_1/radial_summary';marker=source/'complete.json';done=json.loads(marker.read_text())
    assert done['catalogs']==176 and done['arms']==352 and done['scenarios']==22
    assert done['all_PSF'] and done['all_outcomes_retained'] and not done['shared_library_uncertainty_measured']
    inputs={str(marker.relative_to(ROOT)):sha(marker)};outputs={};plotted=[]
    def read(name):
        path=source/name;assert sha(path)==done['outputs'][name];inputs[str(path.relative_to(ROOT))]=sha(path)
        with path.open() as stream:return list(csv.DictReader(stream))
    paired=read('local_minus_broad_fraction_L1.csv');coverage=read('coverage95_nonzero_truth.csv')
    inventory=read('source_inventory.csv');assert len(inventory)==176
    scenarios=sorted({r['scenario'] for r in inventory},key=lambda s:(s.startswith('narrow_') and '_w' in s,s))
    assert len(scenarios)==22
    for s in scenarios:assert len([r for r in inventory if r['scenario']==s])==8
    labels=[s.replace('_',' ').replace('WISE','WISE').replace('kpc',' kpc') for s in scenarios]
    estimands=[('realized_selected_count','Selected realized counts'),('pre_entry_population_fraction','Pre-entry population fractions')]
    out=ROOT/'figures/v3_1/radial';out.mkdir(parents=True,exist_ok=True)
    meta=ROOT/'manifests/v3_1/radial_figures';meta.mkdir(exist_ok=True)
    plt.rcParams.update(STYLE)
    def finish(fig,name):
        fig.canvas.draw();text=[t.get_text() for t in fig.findobj(matplotlib.text.Text)]
        assert not any(any('\u4e00'<=c<='\u9fff' for c in t) for t in text)
        for ext in ['png','pdf']:
            path=out/f'{name}.{ext}';fig.savefig(path,dpi=300);outputs[str(path.relative_to(ROOT))]=sha(path)
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(20,14),sharey=True)
    fig.subplots_adjust(left=.22,right=.88,bottom=.18,top=.80,wspace=.65)
    for ax,(estimand,title) in zip(axes,estimands):
        for i,scenario in enumerate(scenarios):
            counts=[]
            for admission,color,offset in [('all_outcomes',BLUE,-.13),('numerically_accepted',ORANGE,.13)]:
                rows=[r for r in paired if r['scenario']==scenario and r['estimand']==estimand and r['admission']==admission and r['blocking']=='field']
                assert len(rows)==1;r=rows[0];n=int(r['eligible_catalogs']);counts.append(n)
                assert int(r['n'])==n and int(r['all_catalogs'])==8
                plotted.append(dict(figure='paired_radial_error',panel=estimand,source='local_minus_broad_fraction_L1.csv',**r))
                if n:
                    mean=100*float(r['mean']);ax.plot(mean,i+offset,'o',color=color,ms=5)
                    if r['lower95']!='' and r['upper95']!='':ax.plot(100*np.array([float(r['lower95']),float(r['upper95'])]),[i+offset]*2,color=color,lw=1.3)
            joint=[r for r in paired if r['scenario']==scenario and r['estimand']==estimand and r['admission']=='all_outcomes' and r['blocking']=='source_or_field']
            assert len(joint)==1;assert counts[0]==8
            plotted.append(dict(figure='paired_radial_error',panel=estimand,source='local_minus_broad_fraction_L1.csv',**joint[0]))
            ax.text(1.03,i,f"pass {counts[1]}/8; blocks {joint[0]['blocks']}",transform=ax.get_yaxis_transform(),fontsize=8,va='center')
        ax.axvline(0,color='.35',ls='--',lw=1);ax.set_yticks(range(22),labels,fontsize=10);ax.set_ylim(21.65,-.65)
        ax.set_title(title,fontsize=15);ax.set_xlabel('Local minus broad L1 fraction error (percentage points)',fontsize=11)
        ax.grid(axis='y',visible=False)
    axes[0].plot([],[],'o-',color=BLUE,label='All eight catalogs, failures retained')
    axes[0].plot([],[],'o-',color=ORANGE,label='Both arms numerically accepted')
    handles,legend=axes[0].get_legend_handles_labels();fig.legend(handles,legend,loc='upper center',bbox_to_anchor=(.52,.865),ncol=2,frameon=False,fontsize=11)
    fig.suptitle('Radial flexibility: paired recovery on every registered scenario',y=.975,fontsize=21)
    fig.text(.5,.925,'176 matched catalogs | 352 arms | 14 original scenarios + 8 narrow position / width designs',ha='center',fontsize=12)
    fig.text(.5,.89,'Negative values favor the local-structure arm. Parameter and realized-count recovery are different estimands.',ha='center',fontsize=11)
    fig.text(.5,.125,'Intervals resample whole fields, conditional on the shared empirical library. Right labels give accepted pairs and source-or-field blocks.',ha='center',fontsize=10)
    fig.text(.5,.085,'One joint block has no source/context uncertainty interval. Disconnected used-template blocks do not establish independent teacher scales.',ha='center',fontsize=10)
    fig.text(.5,.045,'Original numerical failures and the entry-table local-dust approximation remain. This is not independent real-population calibration.',ha='center',fontsize=10)
    finish(fig,'paired_radial_error')
    fig,axes=plt.subplots(2,2,figsize=(24,20))
    fig.subplots_adjust(left=.20,right=.94,bottom=.13,top=.87,hspace=.38,wspace=.65)
    ticks=[f'{c}:{d}' for c in ['D','E','O'] for d in ['0-2','2-5','5-10','10-20','20-50','>50']]
    cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#d9d9d9')
    for row,(estimand,title) in enumerate(estimands):
        for col,arm in enumerate(['broad','local']):
            ax=axes[row,col];matrix=np.full((22,18),np.nan);number=np.zeros((22,18),int);yl=[]
            for i,scenario in enumerate(scenarios):
                numpass=sum(r[arm+'_passed']=='True' for r in inventory if r['scenario']==scenario)
                yl.append(labels[i]+f' [{numpass}/8 pass]')
                for cls in range(3):
                    for dbin in range(6):
                        rows=[r for r in coverage if r['scenario']==scenario and r['estimand']==estimand and r['arm']==arm and r['admission']=='all_outcomes' and r['blocking']=='field' and int(r['class_index'])==cls and int(r['distance_bin'])==dbin]
                        assert len(rows)==1;r=rows[0];n=int(r['eligible_catalogs']);number[i,cls*6+dbin]=n
                        assert 0<=n<=8 and int(r['n'])==n
                        if n:matrix[i,cls*6+dbin]=100*float(r['mean'])
                        plotted.append(dict(figure='radial_nonzero_coverage',panel=f'{estimand}/{arm}',source='coverage95_nonzero_truth.csv',**r))
            mesh=ax.imshow(matrix,vmin=0,vmax=100,cmap=cmap,aspect='auto',interpolation='nearest')
            for i in range(22):
                for j in range(18):
                    n=number[i,j]
                    if n<8:ax.text(j,i,'NA' if n==0 else f'n{n}',ha='center',va='center',fontsize=6,color='black' if n==0 or matrix[i,j]>55 else 'white')
            for x in [5.5,11.5]:ax.axvline(x,color='white',lw=1.2)
            ax.set_yticks(range(22),yl,fontsize=8);ax.set_xticks(range(18),ticks,rotation=60,ha='right',fontsize=8)
            ax.set_title(f'{title} | {arm.capitalize()} arm',fontsize=14);ax.grid(False)
    bar=fig.colorbar(mesh,ax=axes.ravel().tolist(),fraction=.016,pad=.025);bar.set_label('Empirical 95% interval coverage (%)',fontsize=12)
    fig.suptitle('Class-distance recovery: nonzero-truth coverage with all failures retained',y=.975,fontsize=21)
    fig.text(.5,.94,'D = dwarf; E = evolved; O = other | distance-bin labels in kpc | row brackets show numerical passes',ha='center',fontsize=12)
    fig.text(.5,.905,'Each cell uses only catalogs with positive truth in that cell. n1-n7 mark smaller denominators; gray NA means no positive truth.',ha='center',fontsize=11)
    fig.text(.5,.085,'All 176 catalogs remain in the design. Coverage including structural zeros and accepted-only results are separately preserved in source tables.',ha='center',fontsize=11)
    fig.text(.5,.05,'Eight conditional realizations share source libraries. These are descriptive simulation frequencies, not independent real calibration or 18 independent bin trials.',ha='center',fontsize=11)
    finish(fig,'radial_nonzero_coverage')
    csv_rows(meta/'plotted.csv',plotted);outputs[str((meta/'plotted.csv').relative_to(ROOT))]=sha(meta/'plotted.csv')
    for path in [ROOT/'scripts/v3_1_radial_figures.py',ROOT/'scripts/photdis_figure_style.py']:inputs[str(path.relative_to(ROOT))]=sha(path)
    save_json(meta/'complete.json',dict(inputs=inputs,outputs=outputs,figures=2,panels=6,plotted_rows=len(plotted),
        english_text_verified=True,visual_review_pending=True,scientific_acceptance=False,
        scope='Complete actual radial summary only. Both estimands, all22scenarios, failed arms and sparse nonzero-truth denominators retained. No synthetic fallback or independent calibration claim.'))


if __name__=='__main__':main()
