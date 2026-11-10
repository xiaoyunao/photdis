#!/usr/bin/env python3
"""Compact English plates; cache corrections first, model plates when ready."""
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import v2_2_figures as f
from photdis_figure_style import STYLE,BLUE,ORANGE,GREEN,GRAY
f.TABLES=f.ROOT/'manifests/v2_3';f.FIG=f.ROOT/'figures/v2_3'
read,sel,val=f.read,f.sel,f.val


def save(fig,name,sources,caption):
    f.save(fig,name,sources,caption)
    path=f.TABLES/'figures'/(name+'.json');meta=json.loads(path.read_text())
    meta['selection']='Verified DR11 PSF; scientific denominators and scores untrimmed; posterior example display bounds report retained mass; no independent final validation'
    path.write_text(json.dumps(meta,indent=2)+'\n')


def corrections():
    data=read('noise_baselines');fig,axes=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
    for method,label,color in [('prior_class','Frozen reference prior',GRAY),('uniform','Uniform (no colors)',GREEN),('design_composition','Template-design prior', '#9467bd'),('point_class','Noisy point prediction',BLUE),('averaged_class','Noisy error-averaged prediction',ORANGE)]:
        for ax,metric in zip(axes,['log_score','macro_log_score']):
            vv=[]
            for lo in (18,20,22):
                r=sel(data,r_low=lo,method=method,group='all');vv.append(np.mean([val(x,metric) for x in r if int(x['replica'])>=0]))
            ax.plot([0,1,2],vv,'o-',color=color,label=label)
            ax.set(xticks=[0,1,2],xticklabels=['18–20','20–22','22–25'],xlabel='Matched target r magnitude',ylabel='Mean log score (nats)' if metric=='log_score' else 'Equal-class mean log score (nats)')
    axes[0].legend(fontsize=9);fig.suptitle('Cached noise experiment · matched no-color benchmarks\nHigher is better; no new simulation or classifier fit')
    save(fig,'01_noise_benchmarks',['noise_baselines'],'Positive gain over a mismatched prior is not recovered faint-color information. Design prior uses known template composition; macro scores assign equal class weight. Four cached noisy repeats; paired source-cluster changes remain in v2.2.')
    data=read('reference_inventory');fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    keys=[('fit_eligible','Distance training',GRAY),('distance_evaluation_eligible','Chosen distance evaluation',GREEN),('weak_parallax_quality','Signed parallax S/N < 5',BLUE),('parallax_snr_ge5','Signed parallax S/N >= 5',ORANGE)]
    for ax,wise in zip(axes,['False','True']):
        r=sel(data,W1=wise)[:4]
        for i,(key,label,color) in enumerate(keys):ax.bar(np.arange(4)+(i-1.5)*.18,[val(x,key) for x in r],width=.18,color=color,label=label)
        ax.set(xticks=np.arange(4),xticklabels=['<18','18–20','20–22','22–25'],ylabel='Reserved-identity count',xlabel='Observed r magnitude',yscale='symlog',title='No qualified W1' if wise=='False' else 'Qualified W1')
    axes[0].legend(fontsize=8,loc='upper left');fig.suptitle('New identities: fitting and evaluation qualifications differ')
    save(fig,'02_reference_inventory',['reference_inventory'],'42076 strict-PSF new identities in existing sky.173 chosen evaluation references,3596 signed Gaia observations. Source-valid VAC counts in CSV include exclusions and are not interchangeable with clean chosen labels. No new-scale or independent-sky claim.')


def model_results():
    data=read('development_distance');fig,axes=plt.subplots(2,2,figsize=(14,10),constrained_layout=True)
    groups=['all','dwarf','evolved','unknown','metal_poor_dwarf','low_luminosity'];labels=['All','Dwarf','Giant ref.','Unlabeled','Metal-poor\ndwarf','Low\nluminosity']
    for model,color in [('control',BLUE),('candidate',ORANGE)]:
        rr=[sel(data,role='development_test',model=model,group=g,selection='all',reference='chosen',low_kpc=0,high_kpc='inf')[0] for g in groups]
        for ax,key,label in zip(axes.ravel(),['median_relative','tail_1','coverage68','interval68'],['Median relative residual (%)','Severe error fraction (%)','68% interval coverage (%)','68% interval score (mag)']):
            ax.plot(np.arange(6),[val(r,key)*(1 if key=='interval68' else 100) for r in rr],'o-',color=color,label=model)
            ax.set(xticks=np.arange(6),xticklabels=labels,ylabel=label);ax.tick_params(axis='x',labelsize=10)
    axes[0,0].axhline(0,color=GRAY,ls='--');axes[1,0].axhline(68,color=GRAY,ls='--');axes[0,1].legend()
    fig.suptitle('Same PSF training identities · complete grz distance distributions')
    save(fig,'03_model_comparison',['development_distance','paired_spatial_uncertainty'],'One fixed XD32 candidate versus same-identity PSF QRF. Unknown-class distance labels retained; no clipping. Cal decision independent of plotted development outcomes.')
    data=read('reserved_pit');fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    for ax,group in zip(axes,['all','no_W1','r_20_22']):
        for model,color in [('control',BLUE),('candidate',ORANGE)]:
            rr=sel(data,model=model,group=group,selection='all');n=sum(int(x['n']) for x in rr)
            ax.stairs([val(x,'n')/max(n,1) for x in rr],np.linspace(0,1,11),color=color,label=f'{model}: N={n:,}')
        ax.axhline(.1,color=GRAY,ls='--');ax.set(xlabel='Full observed-parallax PIT',ylabel='Fraction',title=group.replace('_',' '));ax.legend(fontsize=9)
    fig.suptitle('Reserved identities opened after model freeze · negative parallaxes retained')
    save(fig,'04_reserved_parallax',['reserved_pit','reserved_gaia'],'Same-sky new identities; actual signed observations and errors. Does not certify independent distance scale, true population composition or new sky.')
    data=read('application_scope');fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    for j,model in enumerate(['control','candidate']):
        rr=sel(data,model=model,selection='near_scope',r_low='-inf',r_high=20,WISE='all')[0]
        total=val(rr,'weight');axes[0].bar(j,total/1e6,color=[BLUE,ORANGE][j])
        absent=val(rr,'no_new_parallax_weight')/total*100 if total else 0
        axes[1].bar(j,absent,color=[BLUE,ORANGE][j])
    axes[0].set(xticks=[0,1],xticklabels=['PSF QRF','XD32'],ylabel='Weighted selected count (million)')
    axes[1].set(xticks=[0,1],xticklabels=['PSF QRF','XD32'],ylabel='Weight without new parallax evidence (%)',ylim=(0,100))
    fig.suptitle('Prospective near scope: r < 20, q > 0.95, optical quality and common support')
    save(fig,'05_near_scope',['application_scope','application_coverage','reserved_distance','reserved_risk'],'Only13existing DECam fields. Weights are predicted selection estimates, not validated nearby-star counts. Evidence occupancy is not a quality flag.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--models',action='store_true');a=p.parse_args();plt.rcParams.update(STYLE)
    corrections()
    if a.models:model_results()


def additional_results():
    data=read('far_conditional');fig,axes=plt.subplots(1,3,figsize=(16,5),constrained_layout=True)
    for model,color in [('control',BLUE),('candidate',ORANGE)]:
        for reference,ls in [('specdis','-'),('spdist','--')]:
            rr=[sel(data,model=model,role='development_test',reference=reference,low_kpc=lo)[0] for lo in (10,20)]
            for ax,key,label in zip(axes,['median_relative','tail_1','coverage68'],['Conditional distance residual (%)','Conditional severe fraction (%)','Conditional 68% coverage (%)']):
                ax.plot([0,1],[100*val(x,key) for x in rr],marker='o',ls=ls,color=color,label=model+' / '+reference)
                ax.set(xticks=[0,1],xticklabels=['10–20','20–50'],xlabel='Reference distance (kpc)',ylabel=label)
    axes[0].axhline(0,color=GRAY,ls=':');axes[2].axhline(68,color=GRAY,ls=':');axes[1].legend(fontsize=9)
    fig.suptitle('Far branches: reference class is an oracle diagnostic only')
    save(fig,'06_far_branches',['far_conditional','far_reference_disagreement'],'Known evolved class only used after prediction. Source differences and shapes are distinct from branch weights; no independent distance-scale evidence.')
    data=read('metal_poor_matching');fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    for model,color in [('control',BLUE),('candidate',ORANGE)]:
        for group,ls in [('metal_poor','-'),('matched_ordinary','--')]:
            rr=[sel(data,model=model,group=group,selected=False,low_kpc=lo,high_kpc=hi)[0] for lo,hi in [(0,2),(2,5),(5,10)]]
            for ax,key,label in zip(axes,['median_relative','coverage68'],['Median distance residual (%)','68% interval coverage (%)']):
                ax.plot([0,1,2],[100*val(x,key) for x in rr],'o',ls=ls,color=color,label=model+' / '+group.replace('_',' '))
                ax.set(xticks=[0,1,2],xticklabels=['0–2','2–5','5–10'],xlabel='Reference distance (kpc)',ylabel=label)
    axes[0].axhline(0,color=GRAY,ls=':');axes[1].axhline(68,color=GRAY,ls=':');axes[1].legend(fontsize=8)
    fig.suptitle('Metal-poor dwarfs: matched colors, temperature, source and errors')
    save(fig,'07_metal_poor',['metal_poor_matching'],'Same poor-metal identities across models; matched controls with replacement, matching calipers fixed before outcomes. Descriptive evidence, not causal identification. No true metallicity application cut.')
    data=read('development_distance');fig,axes=plt.subplots(1,3,figsize=(16,5),constrained_layout=True)
    groups=['all','dwarf','evolved','unknown','metal_poor_dwarf'];labels=['All','Dwarf','Giant ref.','Unlabeled','Metal-poor\ndwarf']
    for model,color in [('control',BLUE),('candidate',ORANGE)]:
        rr=[sel(data,role='development_test',model=model,group=g,selection='near_scope',reference='chosen',low_kpc=0,high_kpc='inf')[0] for g in groups]
        axes[0].plot(np.arange(5),[100*val(r,'tail_1') for r in rr],'o-',color=color,label=model)
        axes[1].plot(np.arange(5),[100*val(r,'median_relative') for r in rr],'o-',color=color)
        axes[2].plot(np.arange(5),[100*val(r,'n')/val(r,'eligible') for r in rr],'o-',color=color)
    for ax,label in zip(axes,['Severe error fraction (%)','Median distance residual (%)','Retained reference fraction (%)']):ax.set(xticks=np.arange(5),xticklabels=labels,ylabel=label);ax.tick_params(axis='x',labelsize=10)
    axes[0].legend();axes[1].axhline(0,color=GRAY,ls=':');fig.suptitle('Distance quality AFTER prospective near selection')
    save(fig,'08_selected_quality',['development_distance','development_risk'],'Selection retention is explicit. Low far contamination is not precise individual distance. Different model-selected groups cannot be called matched-object improvement.')


if __name__=='__main__' and a.models:
    additional_results()


def posterior_examples():
    data=read('posterior_examples');groups=list(dict.fromkeys(r['group'] for r in data));fig,axes=plt.subplots(2,3,figsize=(16,9),constrained_layout=True)
    for ax,group in zip(axes.ravel(),groups):
        base=sel(data,group=group);ref=val(base[0],'ref_mu');sig=val(base[0],'ref_sigma')
        lo=min(ref-2,min(val(x,'mu025') for x in base))-1;hi=max(ref+2,max(val(x,'mu975') for x in base))+1
        ax.axvspan(ref-sig,ref+sig,color=GRAY,alpha=.2);ax.axvline(ref,color=GRAY,ls='--')
        for model,color in [('control',BLUE),('candidate',ORANGE)]:
            rr=sorted(sel(base,model=model),key=lambda x:val(x,'mu'));x=np.array([val(r,'mu') for r in rr]);y=np.array([val(r,'density') for r in rr]);visible=(x>=lo)&(x<=hi)
            mass=float(np.sum(y[visible]*np.array([val(r,'bin_width') for r in rr])[visible]))
            ax.plot(x,y,color=color,label=f'{model}: q={val(rr[0],"qnear"):.3f}; shown={mass:.3f}')
        ax.set(xlabel='Distance modulus (mag)',ylabel='Probability density',title=group.replace('_',' '),xlim=(lo,hi));ax.legend(fontsize=8);ax.title.set_fontsize(14)
    fig.suptitle('Fixed-identity posterior examples · no selection by model success')
    save(fig,'09_posterior_examples',['posterior_examples'],'Lowest identity per reference-defined development stratum. Display bounds include both95% intervals and reference; displayed mass explicit, scientific metrics untrimmed. Full numerical distributions and mixture parameters retained.')


if __name__=='__main__' and a.models:
    posterior_examples()
