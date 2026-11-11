#!/usr/bin/env python3
"""English, source-linked scientific-use figures from compact actual tables."""
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import v2_2_figures as f
from photdis_figure_style import STYLE,BLUE,ORANGE,GREEN,GRAY
f.TABLES=f.ROOT/'manifests/v2_4';f.FIG=f.ROOT/'figures/v2_4'
read,sel,val=f.read,f.sel,f.val


def save(fig,name,sources,caption):
    f.save(fig,name,sources,caption)
    path=f.TABLES/'figures'/(name+'.json')
    meta=json.loads(path.read_text())
    meta['selection']='Verified DR11 PSF; original scientific denominators; display limits documented in caption; no independent final-validation claim'
    path.write_text(json.dumps(meta,indent=2)+'\n')


def diagnostics():
    r=read('anomaly_objects');bad=np.array([x['low_pit']=='True' for x in r]);fig,axes=plt.subplots(1,3,figsize=(16,5),constrained_layout=True)
    for ax,x,y,xlabel,ylabel in [(axes[0],'gr','rz','(g-r)0 (mag)','(r-z)0 (mag)'),(axes[1],'ra','dec','RA (deg)','Dec (deg)'),(axes[2],'parallax_error','RUWE','Parallax error (mas)','RUWE')]:
        xx=np.array([val(a,x) for a in r]);yy=np.array([val(a,y) for a in r])
        ax.scatter(xx[~bad],yy[~bad],s=10,alpha=.4,color=GRAY,label='Other near Gaia: 564')
        ax.scatter(xx[bad],yy[bad],s=16,alpha=.8,color=ORANGE,label='Conditional PIT < 0.05: 147');ax.set(xlabel=xlabel,ylabel=ylabel)
    axes[0].legend(fontsize=9);fig.suptitle('Fixed QRF near sample · 711 identical Gaia objects')
    save(fig,'01_anomaly_conditions',['anomaly_objects','anomaly_strata'],'Full711objects;147conditional low-PIT flags are diagnostic, not a new selection. QRF and XD are compared on these exact identities. W1 is a population marker, not an input to either grz model.')
    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    for flag,color,label in [('False',BLUE,'Other overlap: 136'),('True',ORANGE,'Low-PIT overlap: 60')]:
        q=sel(r,low_pit=flag,SPDist_qualified=True);x=np.array([val(z,'Gaia_minus_SPDist_sigma') for z in q]);delta=np.array([val(z,'SPDist_mu')-val(z,'control_mu50') for z in q])
        axes[0].hist(x,bins=np.linspace(-8,8,41),histtype='step',lw=2,color=color,label=label)
        axes[1].scatter(delta,x,color=color,s=16,alpha=.6,label=label)
    axes[0].set(xlabel='Gaia minus SPDist prediction / combined error',ylabel='Object count');axes[1].set(xlabel='SPDist minus QRF mu (mag)',ylabel='Standardized parallax difference');axes[0].legend(fontsize=10)
    for ax in axes:ax.axhline(0,color=GRAY,lw=.7)
    fig.suptitle('Exact-identity reference overlap · agreement is not independence')
    save(fig,'02_reference_conflict',['anomaly_objects','anomaly_overlap','anomaly_reference_consistency'],'196of711have qualified SPDist;60are low-PIT. Combined errors use a first-order SPDist parallax propagation and do not model cross-source covariance. Histogram display±8; all objects retained in CSV and statistics.')
    r=read('anomaly_pdf_examples');examples=[]
    for group in ['low_SPDist','low_no_SPDist','not_low_SPDist']:
        examples += [(group,k) for k in sorted({a['identity'] for a in r if a['group']==group},key=int)[:2]]
    fig,axes=plt.subplots(2,3,figsize=(16,9),constrained_layout=True)
    for ax,(group,identity) in zip(axes.ravel(),examples):
        for model,color in [('control',BLUE),('candidate',ORANGE)]:
            q=sel(r,identity=identity,model=model);x=np.array([val(a,'mu') for a in q]);y=np.array([val(a,'density') for a in q]);ax.plot(x,y,color=color,label='QRF' if model=='control' else 'Corrected XD')
        if np.isfinite(val(q[0],'SPDist_mu')):ax.axvline(val(q[0],'SPDist_mu'),color=GREEN,ls='--',label='SPDist')
        ax.set(xlim=(5,22),xlabel='Distance modulus (mag)',ylabel='Density per mag',title=group.replace('_',' '));ax.text(.02,.95,identity,transform=ax.transAxes,va='top',fontsize=8)
    axes[0,0].legend(fontsize=9);fig.suptitle('Deterministic identity examples · full posterior structure')
    save(fig,'03_same_identity_pdfs',['anomaly_pdf_examples'],'First two identity-ranked examples per group, not hand-picked posterior shapes. Rebinning for display only; native full PDF used for scientific metrics. Display5<mu<22; complete compact bins−5to30 are in CSV.')
    r=read('metal_source_pairs');fig,axes=plt.subplots(1,3,figsize=(16,5),constrained_layout=True)
    for ax,reference in zip(axes,['Gaia','specdis','spdist']):
        for j,(model,color) in enumerate([('control',BLUE),('candidate',ORANGE)]):
            q=[sel(r,model=model,reference=reference,low_kpc=lo,high_kpc=hi)[0] for lo,hi in [(0,2),(2,5),(5,10)]]
            x=np.arange(3)+(j-.5)*.12;y=[val(a,'control_mean') for a in q]
            ax.plot(x,y,'o-',color=color,label='QRF' if model=='control' else 'XD')
            ax.vlines(x,[val(a,'control_low') for a in q],[val(a,'control_high') for a in q],color=color,lw=2)
            if j==0:
                for xx,a in zip(x,q):ax.text(xx,.98,'N='+a['n']+'\nU='+a['unique_controls'],transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=9)
        ax.axhline(0,color=GRAY,ls='--');ax.set(xticks=range(3),xticklabels=['0–2','2–5','5–10'],xlabel='Both reference distances (kpc)',ylabel='Mean paired residual difference (mag)',title=reference)
    axes[0].legend(fontsize=9,loc='lower right');fig.suptitle('Metal-poor minus ordinary · repeated-control cluster intervals')
    save(fig,'04_metal_source_scales',['metal_source_pairs'],'Original2055pairs; both sides must qualify for each reference and distance range. N=pairs,U=unique ordinary controls. Bars:1000control-ID-cluster bootstrap95%interval; separate sky-pair intervals retained in CSV. Very small Gaia strata cannot establish an independent scale.')
    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    refs=['Gaia','specdis','spdist']
    for model,color in [('control',BLUE),('candidate',ORANGE)]:
        q=[sel(r,model=model,reference=ref,low_kpc=0,high_kpc='inf')[0] for ref in refs]
        axes[0].plot(range(3),[100*val(a,'poor_median_relative') for a in q],'o-',color=color,label=model+' poor')
        axes[0].plot(range(3),[100*val(a,'ordinary_median_relative') for a in q],'s--',color=color,label=model+' ordinary')
        axes[1].plot(range(3),[val(a,'poor_mean_mass1') for a in q],'o-',color=color,label=model+' poor')
        axes[1].plot(range(3),[val(a,'ordinary_mean_mass1') for a in q],'s--',color=color,label=model+' ordinary')
    for ax in axes:ax.set(xticks=range(3),xticklabels=refs,xlabel='Qualified reference source')
    axes[0].set_ylabel('Median distance residual (%)');axes[1].set(ylabel='Mean probability within ±1 mag',ylim=(0,1));axes[0].legend(fontsize=9)
    fig.suptitle('Source-separated scale offset and full-PDF support')
    save(fig,'05_metal_pdf_support',['metal_source_pairs'],'Sources use different pair subsets. Cross-source persistence is descriptive evidence, not an independent causal metallicity effect. Gaia only10pairs; SpecDis/SPDist share inputs/teacher limitations.')
    r=read('information_population');pair=read('information_paired_bands');fig,axes=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
    bands=['grz','griz','grzW1','grzW1W2']
    for method,color in [('cal_global_prior',GRAY),('cal_r_only_prior','#9C6B9E'),('cal_r_sky_prior',GREEN),('frozen_partitions',BLUE),('fixed_color_probe',ORANGE)]:
        q=[sel(r,bands=b,group='all',population='natural_reference',method=method)[0] for b in bands]
        axes[0].plot(range(4),[val(a,'log_score') for a in q],'o-',color=color,label=method.replace('_',' '))
    for j,(method,color) in enumerate([('fixed_forest',ORANGE),('frozen_partitions_same_labels',BLUE)]):
        q=sel(pair,method=method);x=np.arange(3)+(j-.5)*.1;axes[1].plot(x,[val(a,'mean') for a in q],'o-',color=color,label='Color probe' if j==0 else 'Frozen partitions')
        axes[1].vlines(x,[val(a,'low') for a in q],[val(a,'high') for a in q],color=color)
    axes[0].set(xticks=range(4),xticklabels=bands,ylabel='Mean log score (nats)',xlabel='Matched band set');axes[1].set(xticks=range(3),xticklabels=['+ i','+ W1','+ W2'],ylabel='Same-object log-score gain (nats)',xlabel='Additional photometric information');axes[1].axhline(0,color=GRAY,ls='--')
    axes[0].legend(fontsize=8);axes[1].legend(fontsize=9);fig.suptitle('Matched reference population · cal-only no-color baselines')
    save(fig,'06_information_gains',['information_population','information_paired_bands','information_uncertainty'],'33251same development identities,9sky clusters. r/sky prior fitted only on cal labels; no model refit. Error treatment held at original cache settings. Bars are sky-cluster intervals, not a physical information bound; matched i+WISE coverage is restricted.')


def population():
    r=read('application_joint_counts');fig,axes=plt.subplots(1,2,figsize=(14,5),constrained_layout=True);channels=['dwarf_reference','evolved_reference','other_reference','unlabeled_reference'];colors=[BLUE,ORANGE,GREEN,GRAY]
    for ax,selection in zip(axes,['observable_parent','near_scope']):
        bottom=np.zeros(6)
        for c,color in zip(channels,colors):
            q=sel(r,selection=selection,group='all',channel=c);q=[x for x in q if not (val(x,'low_kpc')==0 and val(x,'high_kpc')==10)];y=np.array([val(x,'expected_count') for x in q])/1e3
            ax.bar(range(6),y,bottom=bottom,color=color,label=c.replace('_reference',''));bottom+=y
        ax.set(xticks=range(6),xticklabels=['0–2','2–5','5–10','10–20','20–50','50+'],xlabel='Integrated distance (kpc)',ylabel='Weighted contribution (thousands)',title=selection.replace('_',' '))
    axes[0].legend(fontsize=9);fig.suptitle('Joint class-distance reference contributions · not verified population counts')
    save(fig,'07_joint_population_counts',['application_joint_counts'],'Exact frozen QRF leaf-joint marginals, Sobol128 shared errors. Unlabeled denotes missing reference labels, not a physical class. Weighted13field diagnostic sample; conditional uncertainty is not population calibration.')
    coverage=read('application_target_coverage');fig,axes=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
    for j,(selection,color) in enumerate([('observable_parent',BLUE),('near_scope',ORANGE)]):
        q=sel(r,selection=selection,group='all',low_kpc=0,high_kpc=10)[:4]
        axes[0].bar(np.arange(4)+(j-.5)*.3,[val(a,'product_marginals_count')-val(a,'expected_count') for a in q],width=.3,color=color,label=selection.replace('_',' '))
        q=sel(coverage,selection=selection,group='all')[0]
        axes[1].bar(np.arange(4)+(j-.5)*.3,[100*val(q,key)/val(q,'weight') for key in ['direct_distance_weight','Gaia_observed_weight','sparse_weight','dense_weight']],width=.3,color=color)
    axes[0].set(xticks=range(4),xticklabels=['Dwarf','Evolved','Other','Unlabeled'],ylabel='Marginal product minus joint count');axes[1].set(xticks=range(4),xticklabels=['Distance\nreference','Gaia\nobservation','Cell N\n1–19','Cell N\n20+'],ylabel='Fraction of selected weight (%)');axes[0].legend(fontsize=9);fig.suptitle('Probability dependence and actual evidence coverage')
    save(fig,'08_count_dependence_coverage',['application_joint_counts','application_target_coverage'],'Direct reference flags and observed Gaia are distinct from observable-cell support. Cell counts are descriptive availability, not guarantees of reliability; cells omit sky. Marginal-product discrepancy demonstrates why joint contributions are required.')
    r=read('population_recovery');populations=['ordinary_dominated','giant_rich','metal_poor_rich','balanced'];fig,axes=plt.subplots(2,2,figsize=(14,10),constrained_layout=True)
    for ax,pop in zip(axes.ravel(),populations):
        q=sel(r,population=pop,condition='target_noise',selection='all',channel='all')
        for key,color,label,marker in [('expected_truth',GRAY,'Injected distribution','s'),('point_count',ORANGE,'Point distance','^'),('pdf_count',BLUE,'Integrated PDF','o')]:ax.plot(range(6),[val(a,key) for a in q],marker+'-',color=color,label=label)
        ax.set(xticks=range(6),xticklabels=['0–2','2–5','5–10','10–20','20–50','50+'],xlabel='Distance (kpc)',ylabel='Expected count per 1,000',title=pop.replace('_',' '))
    axes[0,0].legend(fontsize=10);fig.suptitle('Independent empirical-SED population recovery · target measurement errors')
    save(fig,'09_population_recovery',['population_recovery'],'Held-out template identities, no samples from the tested model PDF. Same empirical SEDs shifted to five known distances; injected class/metal mixtures are stress designs, not Galaxy priors. Teacher-scale errors limit absolute interpretation.')
    fig,axes=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
    for j,(condition,color) in enumerate([('clean',GREEN),('target_noise',BLUE)]):
        total=[];near=[]
        for pop in populations:
            q=sel(r,population=pop,condition=condition,selection='all',channel='all');total.append(sum(abs(val(a,'bias')) for a in q)/1000)
            q=sel(r,population=pop,condition=condition,selection='near_scope',channel='all');near.append(sum(val(a,'pdf_count')-val(a,'expected_truth') for a in q if val(a,'high_kpc')<=10))
        axes[0].bar(np.arange(4)+(j-.5)*.3,total,width=.3,color=color,label=condition.replace('_',' '));axes[1].bar(np.arange(4)+(j-.5)*.3,near,width=.3,color=color)
    for ax in axes:ax.set(xticks=range(4),xticklabels=['Ordinary','Giant-rich','Metal-poor','Balanced']);ax.tick_params(axis='x',labelsize=11)
    axes[0].set_ylabel('Distance-count L1 error / 1,000');axes[1].set_ylabel('Near-count bias after near selection');axes[0].legend(fontsize=10);fig.suptitle('Noise and population assumptions affect posterior count recovery')
    save(fig,'10_recovery_noise_selection',['population_recovery'],'Same empirical templates and design weights across clean/noisy cases. Near selection changes true composition; comparisons retain injected truth after the identical observational and predicted selection.')
    fig,axes=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
    for pop,color in zip(populations,[BLUE,ORANGE,GREEN,GRAY]):
        q=sel(r,population=pop,condition='target_noise',selection='all',channel='all');axes[0].plot(range(6),[100*val(a,'template_bootstrap_coverage') for a in q],'o-',color=color,label=pop.replace('_',' '))
        q=sel(r,population=pop,condition='target_noise',selection='near_scope');q=[a for a in q if a['channel']!='all' and val(a,'high_kpc')<=10]
        y=[sum(val(a,'pdf_count')-val(a,'expected_truth') for a in q if a['channel']==c) for c in channels];axes[1].plot(range(4),y,'o-',color=color)
    axes[0].axhline(95,color='black',ls='--',lw=1);axes[0].set(xticks=range(6),xticklabels=['0–2','2–5','5–10','10–20','20–50','50+'],xlabel='Distance (kpc)',ylabel='Conditional interval coverage (%)',ylim=(-2,102));axes[1].set(xticks=range(4),xticklabels=['Dwarf','Evolved','Other','Unlabeled'],ylabel='Near joint-contribution minus injected count');axes[0].legend(fontsize=9)
    fig.suptitle('Conditional count intervals and missing-label contributions')
    save(fig,'11_count_interval_stress',['population_recovery','population_recovery_uncertainty','population_teacher_sensitivity'],'Conditional Bernoulli count intervals atN=1000, evaluated over500identity-cluster Bayesian-bootstrap pseudo-populations. Not a real-sky coverage claim. Unlabeled-reference contribution has no injected physical-class counterpart; coherent teacher±1sigma sensitivity is retained.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--diagnostics-only',action='store_true');args=parser.parse_args();plt.rcParams.update(STYLE);diagnostics()
    if not args.diagnostics_only:population()
