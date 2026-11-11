#!/usr/bin/env python3
"""English scientific plates for observed-catalog inference and calibration."""
import numpy as np
import matplotlib.pyplot as plt
import v2_5_figures as prior
from photdis_figure_style import STYLE,BLUE,ORANGE,GREEN,GRAY
prior.f.TABLES=prior.f.ROOT/'manifests/v2_6';prior.f.FIG=prior.f.ROOT/'figures/v2_6'
read,sel,val,save=prior.read,prior.sel,prior.val,prior.save
DESIGNS=['reference_like','ordinary_dominated','giant_rich']
LABELS=['Reference-like','Ordinary-dominated','Giant-rich']


def main():
    plt.rcParams.update(STYLE)
    r=read('observed_summary');fig,ax=plt.subplots(1,2,figsize=(15,5),constrained_layout=True)
    for j,(method,label,color) in enumerate([('point','QRF point',GRAY),('pdf','QRF PDF',BLUE),('population','Population fraction x N',ORANGE),('fixed_response','Observed-catalog posterior',GREEN),('response_marginalized','Response-marginalized','#9467bd')]):
        q=[sel(r,design=d,regime=0,method=method)[0] for d in DESIGNS];x=np.arange(3)+(j-2)*.15
        ax[0].bar(x,[100*val(z,'realized_L1') for z in q],width=.14,color=color,label=label)
        if method in ('fixed_response','response_marginalized'):ax[1].plot(range(3),[100*val(z,'coverage_2_5') for z in q],'o-',color=color,label=label)
    ax[0].set(ylabel='Realized distance-count L1 / N (%)');ax[1].set(ylabel='2-5 kpc count interval coverage (%)',ylim=(-3,105));ax[1].axhline(95,color='black',ls='--')
    for a in ax:a.set(xticks=range(3),xticklabels=LABELS);a.tick_params(axis='x',labelsize=11)
    ax[0].legend(fontsize=9);ax[1].legend(fontsize=10);fig.suptitle('Same observed catalogs · population fraction is a different estimand')
    save(fig,'01_observed_counts',['observed_summary'],'All methods compared with the SAME realized128latent objects/catalog, including QRF beyond50kpc mass.12catalogs/design, entrance only; libraries shared across realizations. Count intervals draw a common population parameter and conditional latent labels, not new catalogs.')

    r=read('response_controls');fig,ax=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
    for j,(mode,label,color) in enumerate([('known_response','Known response',BLUE),('same_template','Same template distribution',GREEN),('independent_transfer','Independent templates',ORANGE)]):
        bias=[];coverage=[]
        for d in DESIGNS:
            q=sel(r,mode=mode,design=d,bin=1);bias.append(100*np.mean([val(t,'mean')-val(t,'truth') for t in q]));coverage.append(100*np.mean([t['covered']=='True' for t in q]))
        ax[0].plot(range(3),bias,'o-',color=color,label=label);ax[1].plot(range(3),coverage,'o-',color=color,label=label)
    ax[0].axhline(0,color='black',ls='--');ax[1].axhline(95,color='black',ls='--');ax[0].set(ylabel='2-5 kpc population bias (percentage points)');ax[1].set(ylabel='Population interval coverage (%)',ylim=(-3,105))
    for a in ax:a.set(xticks=range(3),xticklabels=LABELS);a.tick_params(axis='x',labelsize=11)
    ax[0].legend(fontsize=10);fig.suptitle('Restricted controls · same four-parameter inference family')
    save(fig,'02_response_controls',['response_controls','reachable_truth'],'Fixed-parameter Bayesian interval coverage across12catalogs is a diagnostic, not an exact95%requirement at every true parameter. Same-template truth uses independent response Monte Carlo. Reachability is numerical corner reduction, not an interval-arithmetic proof.')

    r=read('nested_response');old=read('uncertain_population');fig,ax=plt.subplots(1,2,figsize=(15,5),constrained_layout=True)
    for d,label,color in zip(DESIGNS,LABELS,[BLUE,ORANGE,GREEN]):
        q=sel(r,design=d,bin=1);scales=sorted({int(t['scale']) for t in q});x=[int(sel(q,scale=s)[0]['sources']) for s in scales]
        ax[0].plot(x,[100*np.mean([val(t,'mean') for t in sel(q,scale=s)]) for s in scales],'o-',color=color,label=label)
        ax[1].plot(x,[100*max(abs(val(t,'mean')-val(t,'mc_half_mean')) for t in sel(q,scale=s)) for s in scales],'o-',color=color,label=label)
    ax[0].set(xlabel='Independent calibration sources',ylabel='2-5 kpc population fraction (%)');ax[1].set(xlabel='Independent calibration sources',ylabel='Maximum 128 vs 256 noise-draw change (pp)')
    ax[0].legend(fontsize=10);fig.suptitle('Nested real-source support · separate from noise replication')
    save(fig,'03_template_convergence',['nested_response','uncertain_population'],'Identity-hash nested quotas fixed before results. Noise draws per source are not independent calibration stars. Lines average the same12existing catalogs; no template-scale tuning on test outcomes.')

    r=read('latent_assignment');fig,ax=plt.subplots(1,2,figsize=(13,5.5),constrained_layout=True)
    for a,method,title in zip(ax,['observed','uncertain_observed'],['Fixed response','Expanded response + uncertainty']):
        q=sel(r,method=method,design='reference_like',regime=0);matrix=np.zeros((4,4))
        for g in range(4):
            for h in range(4):
                z=sel(q,true_group=g,inferred_group=h);matrix[g,h]=np.average([val(t,'mean_probability') for t in z],weights=[val(t,'n') for t in z])
        im=a.imshow(matrix,vmin=0,vmax=1,cmap='viridis');a.grid(False)
        for g in range(4):
            for h in range(4):a.text(h,g,f'{100*matrix[g,h]:.1f}',ha='center',va='center',color='white' if matrix[g,h]<.55 else 'black')
        a.set(xticks=range(4),yticks=range(4),xticklabels=['Ordinary','Poor','Evolved','Other'],yticklabels=['Ordinary','Poor','Evolved','Other'],xlabel='Posterior group',ylabel='Actual simulated group',title=title)
    fig.colorbar(im,ax=ax,label='Mean per-star posterior probability',shrink=.8);fig.suptitle('Reference-like observed catalog · latent assignments, not hard classes')
    save(fig,'04_latent_assignments',['latent_assignment','template_shapes'],'True labels used ONLY after inference. Group matrix averages per-object joint marginals across same12reference-like catalogs; repeated template identities do not add independent real-source evidence.')

    r=read('interval_endpoints');u=read('response_integration');fig,ax=plt.subplots(1,3,figsize=(17,5),constrained_layout=True)
    for regime,color,label in [('0',BLUE,'Entrance'),('1',ORANGE,'Complement')]:
        q=sel(r,regime=regime,target='distance',bin=1);ax[0].hist([100*val(t,'max_delta') for t in q],bins=18,histtype='step',lw=2,color=color,label=label)
    ax[0].set(xlabel='2-5 kpc interval endpoint change (pp)',ylabel='Catalogs');ax[0].legend(fontsize=10)
    for d,label,color in zip(DESIGNS,LABELS,[BLUE,ORANGE,GREEN]):
        q=sel(u,design=d);ax[1].scatter([val(t,'response_ess') for t in q],[100*val(t,'response_half_max_delta') for t in q],label=label,color=color)
        ax[2].scatter([100*val(t,'theta_half_max_delta') for t in q],[100*val(t,'response_half_max_delta') for t in q],color=color)
    ax[1].set(xlabel='Response-posterior ESS (maximum 16)',ylabel='8 vs 16 response-draw change (pp)');ax[1].legend(fontsize=9)
    ax[2].set(xlabel='Theta half-grid mean change (pp)',ylabel='Response half-prior mean change (pp)');fig.suptitle('Integration diagnostics · finite response-prior approximation remains explicit')
    save(fig,'05_integration_limits',['interval_endpoints','response_integration'],'Original2^18versus2^17quantile endpoints. Response joint posterior2^16versus2^15theta mean and8versus16response priors. Evidence weights include target likelihood; low responseESS signals finite-prior sensitivity, not more independent data.')

    r=read('luminosity_parameters');q=read('luminosity_changes');fig,ax=plt.subplots(1,3,figsize=(17,5),constrained_layout=True)
    for j,(source,color) in enumerate([('specdis',BLUE),('spdist',ORANGE)]):
        rows=[sel(r,source=source,parameter=p)[0] for p in ['source_offset','poor_offset','poor_scatter']];x=np.arange(3)+(j-.5)*.13
        ax[0].plot(x,[val(t,'mean') for t in rows],'o',color=color,label=source);ax[0].vlines(x,[val(t,'low') for t in rows],[val(t,'high') for t in rows],color=color)
        for a,family in zip(ax[1:],['teacher','grz']):
            for side,marker in [('poor','o'),('ordinary','s')]:
                z=sel(q,source=source,family=family,side=side,metric='high_tail')[0];a.plot([0,1],[100*val(z,'before'),100*val(z,'after')],marker+'-',color=color,label=source+' / '+side)
            a.set(xticks=[0,1],xticklabels=['Before','After'],ylabel='Gaia high PIT tail (%)',title='VAC-conditioned' if family=='teacher' else 'Pure grz latent-branch transfer');a.axhline(5,color='black',ls='--')
    ax[0].set(xticks=range(3),xticklabels=['Source offset','Poor offset','Poor scatter'],ylabel='Magnitude (mag)',title='Cal-role posterior');ax[0].axhline(0,color='black',ls='--');ax[0].legend(fontsize=10);ax[2].legend(fontsize=9)
    fig.suptitle('One source-conditioned luminosity trial · unknown branch stays latent')
    save(fig,'06_luminosity_trial',['luminosity_parameters','luminosity_changes'],'Cal identities disjoint from complete reused-control evaluation groups. Full signed Gaia likelihood; no SNR cut. Positive Mg offset means shorter distance. VAC-dependent scale checks and pure-photometric kernel transfer differ; no QRF retraining or deployment.')

    r=read('luminosity_changes');fig,ax=plt.subplots(1,3,figsize=(17,5),constrained_layout=True)
    for a,metric in zip(ax,['log_score','coverage68','coverage95']):
        for j,(source,color) in enumerate([('specdis',BLUE),('spdist',ORANGE)]):
            q=[sel(r,source=source,family=f,side=s,metric=metric)[0] for f,s in [('teacher','poor'),('teacher','ordinary'),('grz','poor'),('grz','ordinary')]];x=np.arange(4)+(j-.5)*.12
            a.plot(x,[val(t,'mean') for t in q],'o',color=color,label=source);a.vlines(x,[val(t,'low') for t in q],[val(t,'high') for t in q],color=color)
        a.axhline(0,color='black',ls='--');a.set(xticks=range(4),xticklabels=['VAC poor','VAC ordinary','grz poor','grz ordinary'],ylabel='After minus before: '+metric.replace('_',' '));a.tick_params(axis='x',labelrotation=20,labelsize=10)
    ax[0].legend(fontsize=10);fig.suptitle('Same evaluation objects · reused-control cluster intervals')
    save(fig,'07_predictive_changes',['luminosity_changes'],'Intervals resample reused ordinary-control groups, not individual duplicated pair rows. Coverage is Gaia observation-predictive interval coverage, NOT true-distance coverage. Log-score units nats; coverage changes are fractions. Selection and shared-teacher dependence remain.')


if __name__=='__main__':main()
