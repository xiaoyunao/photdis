#!/usr/bin/env python3
"""Plot complete shared-state structure products and descriptive seen-system comparisons."""
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from v2_1_near import ROOT,sha,save_json,csv_rows
from photdis_figure_style import STYLE,BLUE,ORANGE


def main():
    base=ROOT/'manifests/v3_1/structure';inputs={};tables={};markers={}
    for stage,names in [('posterior',['summary.csv','numerical_checks.csv','membership.csv']),
                        ('report',['counts.csv','membership.csv','membership_comparison.csv'])]:
        marker=base/stage/'complete.json';done=json.loads(marker.read_text());markers[stage]=done
        assert done['n']==512 and done['states']==32 and done['all_PSF']
        inputs[str(marker.relative_to(ROOT))]=sha(marker)
        for name in names:
            path=base/stage/name;assert sha(path)==done['outputs'][name]
            inputs[str(path.relative_to(ROOT))]=sha(path)
            with path.open() as stream:tables[stage+'/'+name]=list(csv.DictReader(stream))
    summary={r['branch']:r for r in tables['posterior/summary.csv']}
    assert set(summary)=={'full','gaussian'}
    counts=tables['report/counts.csv'];checks=tables['posterior/numerical_checks.csv']
    assert len(counts)==4 and len(checks)==10
    truth=lambda x:str(x).lower()=='true'
    membership={};original={}
    for branch in ['full','gaussian']:
        membership[branch]={r['identity']:r for r in tables['report/membership.csv'] if r['branch']==branch}
        original[branch]={r['identity']:r for r in tables['posterior/membership.csv'] if r['branch']==branch}
        assert len(membership[branch])==len(original[branch])==512
        assert set(membership[branch])==set(original[branch])
        assert all(r['TYPE']=='PSF' for r in membership[branch].values())
        for identity,row in membership[branch].items():
            assert float(row['model_member_probability'])==float(original[branch][identity]['member_probability'])
    assert set(membership['full'])==set(membership['gaussian'])
    ids=sorted(membership['full'],key=int)
    for identity in ids:
        for key in ['radius_deg','membership_available','published_member_probability','published_p_ge_0_9']:
            assert membership['full'][identity][key]==membership['gaussian'][identity][key]
    out=ROOT/'figures/v3_1/structure';out.mkdir(parents=True,exist_ok=True)
    meta=ROOT/'manifests/v3_1/structure_figures';meta.mkdir(exist_ok=True)
    plt.rcParams.update(STYLE);outputs={};plotted=[]
    branches=[('full','Full likelihood',BLUE),('gaussian','Gaussian summary',ORANGE)]
    def finish(fig,name):
        fig.canvas.draw();labels=[t.get_text() for t in fig.findobj(matplotlib.text.Text)]
        assert not any(any('\u4e00'<=c<='\u9fff' for c in t) for t in labels)
        for ext in ['png','pdf']:
            path=out/f'{name}.{ext}';fig.savefig(path,dpi=300);outputs[str(path.relative_to(ROOT))]=sha(path)
        plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(20,12))
    fig.subplots_adjust(left=.07,right=.97,bottom=.18,top=.78,wspace=.38,hspace=.68)
    for j,(branch,label,color) in enumerate(branches):
        row=summary[branch]
        for k,component in enumerate(['member','background']):
            r=next(r for r in counts if r['branch']==branch and r['component']==component)
            x=k+(j-.5)*.2;median=float(r['median'])
            axes[0,0].plot([x,x],[float(r['lower95']),float(r['upper95'])],c=color,lw=1.5)
            axes[0,0].plot([x,x],[float(r['lower68']),float(r['upper68'])],c=color,lw=5)
            axes[0,0].scatter(x,median,c=color,s=28,marker=('o' if branch=='full' else 's'),label=label if k==0 else None)
            axes[0,0].scatter(x,float(r['mean']),facecolors='white',edgecolors=color,marker='D',s=22,zorder=4)
            plotted.append(dict(figure='structure_posterior',panel='A',source='report/counts.csv',**r))
        lo,med,hi=[float(row['common_distance_'+k+'_kpc']) for k in ['lower95','median','upper95']]
        axes[0,1].plot([lo,hi],[j,j],c=color,lw=2);axes[0,1].scatter(med,j,c=color,s=40,marker=('o' if branch=='full' else 's'))
        conditional=float(row['conditional_member_variance']);shared=float(row['shared_member_variance'])
        axes[0,2].bar(j,conditional,color=color,alpha=.5)
        axes[0,2].bar(j,shared,bottom=conditional,color=color,hatch='//')
        axes[1,0].bar(j,float(row['response_ESS']),color=color)
        axes[1,0].text(j,float(row['response_ESS']),f"  top: {row['top_response']}",ha='center',va='bottom',fontsize=9)
        for panel in ['B','C','D']:plotted.append(dict(figure='structure_posterior',panel=panel,source='posterior/summary.csv',**row))
        numerical=[r for r in checks if r['branch']==branch and r['comparison']!='quadrature_first']
        assert len(numerical)==4
        for k,(column,gate) in enumerate([('maximum_member_probability_delta',.003),('maximum_mean_count_delta',.5),
                                         ('maximum_common_mu_quantile_delta',.03),('log_evidence_delta',.05)]):
            value=max(abs(float(r[column]))/gate for r in numerical)
            axes[1,1].scatter(k+(j-.5)*.16,value,c=color,s=35,marker=('o' if branch=='full' else 's'))
            plotted.extend(dict(figure='structure_posterior',panel='E',source='posterior/numerical_checks.csv',display_metric=column,**r) for r in numerical)
        r=next(r for r in counts if r['branch']==branch and r['component']=='member')
        axes[1,2].bar(j,float(r['CDF_half_delta'])/.01,color=color)
        axes[1,2].text(j,float(r['CDF_half_delta'])/.01,f"{int(r['draws']):,} catalogs",ha='center',va='bottom',fontsize=9)
        plotted.append(dict(figure='structure_posterior',panel='F',source='report/counts.csv',**r))
    axes[0,0].set(title='A | Joint realized counts',ylabel='Objects among the selected 512')
    axes[0,0].set_xticks([0,1],['Members','Background']);axes[0,0].legend(frameon=False,fontsize=9)
    axes[0,1].set(title='B | Shared distance posterior',xlabel='Common distance (kpc)')
    axes[0,1].set_yticks([0,1],[r[1] for r in branches])
    axes[0,2].set(title='C | Member-count variance',ylabel='Variance (objects squared)')
    axes[1,0].set(title='D | Response-state concentration',ylabel='Effective states / 32',ylim=(0,36))
    axes[1,0].axhline(32,c='.4',ls='--',lw=1)
    axes[1,1].set(title='E | Maximum numerical change',ylabel='Absolute change / registered gate',yscale='symlog')
    axes[1,1].set_yscale('symlog',linthresh=1)
    axes[1,1].set_yticks([0,.5,1,2,5,10],['0','0.5','1','2','5','10'])
    axes[1,1].axhline(1,c='.4',ls='--',lw=1)
    axes[1,1].set_xticks(range(4),['Probability','Count','Common mu','Log evidence'],rotation=20,ha='right',fontsize=9)
    axes[1,2].set(title='F | Realized-count sampling check',ylabel='Alternating-half CDF difference / 0.01')
    axes[1,2].axhline(1,c='.4',ls='--',lw=1)
    for ax in [axes[0,2],axes[1,0],axes[1,2]]:ax.set_xticks([0,1],[r[1] for r in branches],fontsize=9)
    for ax in axes.flat:
        ax.title.set_fontsize(13);ax.xaxis.label.set_fontsize(11);ax.yaxis.label.set_fontsize(11)
        ax.tick_params(axis='y',labelsize=10)
    statuses=' | '.join(f"{label}: numerical={row['numerically_passed']}, sampling={next(r['sampling_passed'] for r in counts if r['branch']==branch)}" for branch,label,_ in branches for row in [summary[branch]])
    fig.suptitle('NGC288: full likelihood versus fair Gaussian summary',y=.975,fontsize=20)
    fig.text(.5,.93,'Same 512 photometric-first native-PSF objects | all 32 original shared response / calibration states',ha='center',fontsize=12)
    fig.text(.5,.885,statuses,ha='center',fontsize=11)
    fig.text(.5,.84,'One supplied common-scale prior: 9.03 kpc, lower/upper scales 0.56/0.48 kpc. This is not independent scale recovery.',ha='center',fontsize=11)
    fig.text(.5,.12,'A: thin/thick bars = 95%/68% realized-count intervals; circles/squares = medians; open diamonds = means. B: median and 95% interval.',ha='center',fontsize=10)
    fig.text(.5,.08,'C: translucent = conditional variance; hatched = shared-state variance. E: final quadrature, native grid and entry alternatives; symlog scale.',ha='center',fontsize=10)
    fig.text(.5,.04,'F is one sampling diagnostic; all moment / variance / quantile gates remain in the report. Counts condition on this sample; no physical density.',ha='center',fontsize=10)
    finish(fig,'structure_posterior')
    fig,axes=plt.subplots(1,2,figsize=(16,8));fig.subplots_adjust(left=.08,right=.96,bottom=.22,top=.72,wspace=.3)
    for branch,label,color in branches:
        rows=[membership[branch][identity] for identity in ids]
        radius=np.array([float(r['radius_deg']) for r in rows]);prob=np.array([float(r['model_member_probability']) for r in rows])
        available=np.array([truth(r['membership_available']) for r in rows])
        assert np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all()
        axes[0].scatter(radius,prob,c=color,s=12,alpha=.45,marker=('o' if branch=='full' else 's'),label=label,rasterized=True)
        refs=np.array([float(r['published_member_probability']) for r in rows if truth(r['membership_available'])])
        assert len(refs)==markers['report']['membership_available']
        axes[1].scatter(refs,prob[available],c=color,s=12,alpha=.45,marker=('o' if branch=='full' else 's'),label=f'{label} (n={available.sum()})',rasterized=True)
        plotted.extend(dict(figure='structure_membership',panel='A',source='report/membership.csv',**r) for r in rows)
        plotted.extend(dict(figure='structure_membership',panel='B',source='report/membership.csv',**r) for r in rows if truth(r['membership_available']))
    axes[0].set(title='A | Photometric membership versus radius',xlabel='Angular radius (degrees)',ylabel='Model membership probability',ylim=(-.03,1.03))
    axes[1].plot([0,1],[0,1],c='.4',ls='--',lw=1)
    axes[1].set(title='B | Seen Gaia membership model comparison',xlabel='Published membership probability',ylabel='Model membership probability',xlim=(-.03,1.03),ylim=(-.03,1.03))
    for ax in axes:ax.legend(frameon=False,fontsize=10)
    fig.suptitle('Descriptive membership comparison after the frozen structure fit',y=.97,fontsize=18)
    fig.text(.5,.9,f"All 512 selected objects | published membership available: {markers['report']['membership_available']} | published p >= 0.9: {markers['report']['published_p_ge_0_9']}",ha='center',fontsize=12)
    fig.text(.5,.82,'Membership information does not select the sample, fit the likelihood or update the shared posterior.',ha='center',fontsize=11)
    fig.text(.5,.77,'Joint numerical acceptance: ' + ' | '.join(
        f"{label} = {summary[branch]['numerically_passed']}" for branch,label,_ in branches),ha='center',fontsize=10)
    fig.text(.5,.13,'Missing published membership stays missing. Published probabilities are model outputs, not independent hard truth.',ha='center',fontsize=11)
    fig.text(.5,.07,'Seen-system reference-ensemble demonstration; separate from the corrected target candidate and new NGC1261 evaluation.',ha='center',fontsize=10)
    finish(fig,'structure_membership')
    csv_rows(meta/'plotted.csv',plotted);outputs[str((meta/'plotted.csv').relative_to(ROOT))]=sha(meta/'plotted.csv')
    for path in [ROOT/'scripts/v3_1_structure_figures.py',ROOT/'scripts/photdis_figure_style.py',ROOT/'docs/v3_1_structure_protocol.md']:
        inputs[str(path.relative_to(ROOT))]=sha(path)
    save_json(meta/'complete.json',dict(inputs=inputs,outputs=outputs,figures=2,panels=8,n=512,states=32,
        plotted_rows=len(plotted),english_text_verified=True,visual_review_pending=True,
        scientific_acceptance=False,scope='Authenticated full structure reports only; all numerical flags retained. No synthetic data, independent scale accuracy or final-candidate validation claim.'))


if __name__=='__main__':main()
