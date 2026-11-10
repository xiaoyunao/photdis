#!/usr/bin/env python3
"""Corrected experiment plus explicit inference/selection evidence, English only."""
import json
import numpy as np
import matplotlib.pyplot as plt
import v2_3_figures as original
from photdis_figure_style import STYLE,BLUE,ORANGE,GREEN,GRAY
original.f.TABLES=original.f.ROOT/'manifests/v2_3_1'
original.f.FIG=original.f.ROOT/'figures/v2_3_1'
read,sel,val,save=original.read,original.sel,original.val,original.save


def corrections():
    data=read('inference_correction_by_error');fig,axes=plt.subplots(1,3,figsize=(16,5),constrained_layout=True)
    for role,color in [('development_test',BLUE),('reserved',ORANGE),('application',GREEN)]:
        rr=[sel(data,role=role,error_low=lo,error_high=hi,version='plus_g_corrected')[0] for lo,hi in [(0,.01),(.01,.03),(.03,.1),(.1,'inf')]]
        for ax,key,label in zip(axes,['median_abs_mu_change','mean_abs_q_change','changed_scope'],['Median absolute change in mu (mag)','Mean absolute change in near probability','Changed near-scope selection (%)']):
            y=[val(r,key)/(val(r,'n') or np.nan)*100 if key=='changed_scope' else val(r,key) for r in rr]
            ax.plot(range(4),y,'o-',color=color,label=role.replace('_',' '));ax.set(xticks=range(4),xticklabels=['<0.01','0.01–0.03','0.03–0.1','≥0.1'],xlabel='Maximum grz error (mag)',ylabel=label)
    axes[0].legend(fontsize=9);fig.suptitle('Shared-g inference correction · fitted model unchanged')
    save(fig,'10_inference_changes',['inference_correction_by_error'],'Same identities before/after correction. Old-minus-g predictions are archived and scientifically superseded. Empty strata remain missing; no clipping or refit.')
    data=read('reserved_selection_pit');fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    for ax,model in zip(axes,['control','candidate']):
        for score,color in [('raw',BLUE),('conditional',ORANGE)]:
            rr=sel(data,model=model,group='all',selection='near_scope',score=score);n=int(rr[0]['n'])
            ax.stairs([val(r,'count')/n for r in rr],np.linspace(0,1,11),color=color,label=f'{score}: N={n}')
        ax.axhline(.1,color=GRAY,ls='--');ax.set(title=model,xlabel='Observed-parallax PIT',ylabel='Fraction');ax.legend(fontsize=10)
    fig.suptitle('Previously inspected identities · near scope · selection conditioning')
    save(fig,'11_selected_parallax',['reserved_selection_pit','reserved_selection_scores','selection_qualification'],'Condition on replayed distance-admission rule with fixed source/error/class/spatial covariates. Does not model spectroscopic targeting, correlated reference selection or full application population. Original raw cal guards remain separate.')
    data=read('selection_forward_simulation');fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    for ax,model in zip(axes,['control','candidate']):
        r=sel(data,model=model,group='all',selection='near_scope')[0]
        for j,(score,color) in enumerate([('raw',BLUE),('conditional',ORANGE)]):
            ax.bar(np.arange(2)+(j-.5)*.25,[100*val(r,score+'_low'),100*val(r,score+'_high')],width=.25,color=color,label=score)
        ax.axhline(5,color=GRAY,ls='--');ax.set(title=model,xticks=[0,1],xticklabels=['PIT < 0.05','PIT > 0.95'],ylabel='Selected simulated fraction (%)');ax.legend()
    fig.suptitle('Forward-drawn observations after historical distance admission')
    save(fig,'12_selection_simulation',['selection_forward_simulation','selection_simulation_checks'],'Independent simulation halves estimate unselected/selected CDFs. Plot weights eligible simulated objects equally; Monte Carlo object counts and draw-weighted alternatives are in CSV. Reproduce selection effect under the model, not evidence of real population calibration. Fixed seed;2048 draws/object; real observed metrics separate.')
    data=read('far_full_pdf_mass');fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    labels=['SpecDis\n10–20','SpecDis\n20–50','SPDist\n10–20','SPDist\n20–50']
    for model,color,ls in [('control',BLUE,'-'),('candidate_old',GRAY,'--'),('candidate_corrected',ORANGE,'-')]:
        rr=[sel(data,model=model,reference=ref,low_kpc=lo)[0] for ref in ('specdis','spdist') for lo in (10,20)]
        axes[0].plot(range(4),[val(r,'mean_mass_within1') for r in rr],'o',ls=ls,color=color,label=model.replace('_',' '))
        axes[1].plot(range(4),[val(r,'reference_convolved_log') for r in rr],'o',ls=ls,color=color)
    for ax,label in zip(axes,['Mean full-PDF mass within 1 mag','Reference-convolved log score (nats)']):ax.set(xticks=range(4),xticklabels=labels,ylabel=label,xlabel='Reference / distance (kpc)')
    axes[0].legend(fontsize=9);fig.suptitle('Far giant references · full PDF without auxiliary class conditioning')
    save(fig,'13_far_full_pdf',['far_full_pdf_mass','far_conditional'],'Same reference identities per panel; full-PDF probability and convolved score do not use auxiliary class frequencies. QRF leaf-joint and XD fixed-component class oracles are structurally different.')
    data=read('reserved_distance');fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    for j,(model,color) in enumerate([('control',BLUE),('candidate',ORANGE)]):
        r=sel(data,model=model,group='all',selection='near_scope',reference='spdist',low_kpc=0,high_kpc='inf')[0]
        for ax,keys in zip(axes,[['median_relative','nmad_relative','tail_1'],['coverage68','coverage95']]):
            ax.bar(np.arange(len(keys))+(j-.5)*.28,[val(r,k)*100 for k in keys],width=.28,color=color,label=f'{model}: N={int(r["n"])}')
    axes[0].set(xticks=range(3),xticklabels=['Median residual','NMAD','Severe errors'],ylabel='Distance statistic (%)');axes[0].axhline(0,color=GRAY,lw=.8);axes[0].legend()
    axes[1].set(xticks=range(2),xticklabels=['68% interval','95% interval'],ylabel='Point-reference coverage (%)');axes[1].scatter([0,1],[68,95],marker='_',color=GRAY,s=400,label='Nominal');axes[1].legend(fontsize=9)
    fig.suptitle('Reserved SPDist references after near-scope selection')
    save(fig,'14_spdist_near',['reserved_distance','reserved_risk'],'SPDist is shared-teacher evidence, not independent truth. Selected identities may differ between models; no known reliable class labels. Point-reference interval coverage alone is not intrinsic calibration.')
    data=read('metal_poor_pair_retention');fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    for ax,model in zip(axes,['control','candidate']):
        rr=[sel(data,model=model,selected=True,low_kpc=lo,high_kpc=hi)[0] for lo,hi in [(0,2),(2,5),(5,10),(0,'inf')]];bottom=np.zeros(4)
        for key,label,color in [('both','Both',GREEN),('poor_only','Metal-poor side only',ORANGE),('ordinary_only','Ordinary side only',BLUE),('neither','Neither',GRAY)]:
            y=np.array([val(r,key) for r in rr]);ax.bar(range(4),y,bottom=bottom,color=color,label=label);bottom+=y
        ax.set(title=model,xticks=range(4),xticklabels=['0–2','2–5','5–10','All'],xlabel='Per-side reference distance (kpc)',ylabel='Original matched pairs')
    axes[0].legend(fontsize=9);fig.suptitle('Pair retention after distance stratum AND model near selection')
    save(fig,'15_pair_retention',['metal_poor_pair_retention','metal_poor_pair_conserving'],'Each column partitions the same2055pairs.354 unique ordinary controls reused; not independent pairs. Both-in-stratum and poor-side-defined paired estimates exported; former per-side statistics not pair-conserving.')
    data=read('application_coverage_definitions');fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    for j,(evidence,color) in enumerate([('development_only',BLUE),('cal_plus_development',ORANGE)]):
        rr=[sel(data,model=model,evidence=evidence,selection='near_scope')[0] for model in ('control','candidate')]
        for ax,key in zip(axes,['zero_weight','sparse1_19_weight']):ax.bar(np.arange(2)+(j-.5)*.3,[100*val(r,key)/val(r,'weight') for r in rr],width=.3,color=color,label=evidence.replace('_',' '))
    for ax,label in zip(axes,['Weight with zero references (%)','Weight with 1–19 references (%)']):ax.set(xticks=[0,1],xticklabels=['QRF control','Corrected XD32'],ylabel=label)
    axes[0].legend(fontsize=9);fig.suptitle('Same near selections · two explicit reference-count definitions')
    save(fig,'16_coverage_definitions',['application_coverage_definitions'],'Identical model selections and observable cells. v2.2 used development-only; v2.3 combined cal+development. Coverage differences do not measure accuracy or prove population reliability.')


if __name__=='__main__':
    plt.rcParams.update(STYLE)
    original.corrections();original.model_results('Previously inspected identities · corrected experiment · signed parallaxes');original.additional_results();original.posterior_examples();corrections()
    path=original.f.TABLES/'figures/07_metal_poor.json';meta=json.loads(path.read_text());meta['caption']+=' Distance/near masks here apply separately to each side, so stratified groups need not remain paired; see plate15 and pair-conserving CSV.';path.write_text(json.dumps(meta,indent=2)+'\n')
