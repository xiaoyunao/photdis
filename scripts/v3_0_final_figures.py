#!/usr/bin/env python3
"""Scientific count/recovery plots from completed, hashed v3 products."""
import argparse
import csv
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from v2_1_near import ROOT,load,sha,save_json,csv_rows
from photdis.stage1 import require_compute
from photdis_figure_style import STYLE,BLUE,GREEN,ORANGE,GRAY
D=ROOT/'data/v3_0'
OUT=ROOT/'figures/v3_0/final';TABLE=ROOT/'manifests/v3_0/figures/final'
BINS=['0–2','2–5','5–10','10–20','20–50','>50']


def read(path):return list(csv.DictReader(path.open()))


def save(fig,name,rows,inputs,caption):
    OUT.mkdir(parents=True,exist_ok=True);TABLE.mkdir(parents=True,exist_ok=True)
    assert not any(any('\u4e00'<=c<='\u9fff' for c in t.get_text()) for t in fig.findobj(matplotlib.text.Text))
    fig.savefig(OUT/(name+'.png'),dpi=300,bbox_inches='tight',pad_inches=.12)
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight',pad_inches=.12);plt.close(fig)
    csv_rows(TABLE/(name+'.csv'),rows)
    save_json(TABLE/(name+'.json'),dict(caption=caption,inputs={str(x.relative_to(ROOT)):sha(x) for x in inputs},
        png_sha256=sha(OUT/(name+'.png')),pdf_sha256=sha(OUT/(name+'.pdf')),csv_sha256=sha(TABLE/(name+'.csv')),
        english_text_verified=True,png_dpi=300,job=os.getenv('SLURM_JOB_ID')))


def recovery(name='all32_native_precision1'):
    base=D/'shared_recovery_summary'/name;record=json.loads((base/'complete.json').read_text())
    count=read(base/'counts.csv');cov=read(base/'coverage.csv');numeric=read(base/'numerical.csv')
    nominal=ROOT/'manifests/v3_0/recovery_final/summary.csv';old=read(nominal)
    scenarios=sorted({r['scenario'] for r in count});labels=[s.replace('_',' ').capitalize().replace('wise','WISE').replace('4kpc','4 kpc') for s in scenarios]
    rows=[]
    for scenario in scenarios:
        for axis in ['distance','class_distance']:
            values=[]
            for catalog in sorted({int(r['catalog']) for r in count if r['scenario']==scenario}):
                selected=[r for r in count if r['scenario']==scenario and r['axis']==axis and int(r['catalog'])==catalog]
                values.append(sum(abs(float(r['mean'])-int(r['truth'])) for r in selected)/128)
            rows.append(dict(scenario=scenario,axis=axis,method='shared_target',L1=float(np.mean(values)),catalogs=len(values)))
        for method in ['qrf_pdf','v3_reference','v3_target']:
            r=next(x for x in old if x['view']=='multiband' and x['scenario']==scenario and x['method']==method)
            rows.append(dict(scenario=scenario,axis='distance',method=method,L1=float(r['mean_distance_count_L1']),catalogs=int(r['catalogs'])))
    fig,ax=plt.subplots(figsize=(11,8))
    for j,(method,label,color) in enumerate([('qrf_pdf','Matched QRF PDF',GRAY),('v3_reference','Nominal reference prior',GREEN),('v3_target','Nominal target prior',BLUE),('shared_target','Shared-response target',ORANGE)]):
        value=[next(r['L1'] for r in rows if r['scenario']==s and r['method']==method and r['axis']=='distance') for s in scenarios]
        ax.plot(100*np.array(value),np.arange(len(value))+(j-1.5)*.14,'o',color=color,label=label,ms=5)
    ax.set(yticks=range(len(labels)),yticklabels=labels,xlabel='Distance-count L1 error (%)',title='Population recovery with shared uncertainty')
    ax.invert_yaxis();ax.tick_params(axis='y',labelsize=11);ax.legend(frameon=False,fontsize=10,loc='lower right');ax.grid(axis='x',alpha=.15)
    fig.subplots_adjust(left=.28,right=.98,top=.91,bottom=.1)
    save(fig,'shared_recovery_L1',rows,[base/'complete.json',base/'counts.csv',nominal],f"Eight 128-object continuous realizations per scenario; shared templates, not newly independent stars. {record['numerically_passed']}/112 final numerical passes. L1 is count error, not per-star precision; every case retained.")
    fig,axes=plt.subplots(1,2,figsize=(13,8));table=[]
    for ax,axis,title in zip(axes,['distance','class_distance'],['Distance counts','Class × distance counts']):
        bins=6 if axis=='distance' else 18
        matrix=np.array([[float(next(r for r in cov if r['scenario']==s and r['axis']==axis and int(r['bin'])==b)['coverage95']) for b in range(bins)] for s in scenarios])
        im=ax.imshow(matrix*100,cmap='viridis',vmin=0,vmax=100,aspect='auto')
        ax.set(title=title,yticks=range(len(labels)),yticklabels=labels,xticks=range(bins),xticklabels=BINS if bins==6 else BINS*3)
        ax.tick_params(axis='both',labelsize=9);ax.tick_params(axis='x',rotation=90 if bins==18 else 0)
        if bins==18:
            ax.axvline(5.5,color='white',lw=1);ax.axvline(11.5,color='white',lw=1)
            ax.set_xlabel('Dwarf / evolved / other: distance bins (kpc)',fontsize=11)
        else:ax.set_xlabel('Distance bin (kpc)')
        table += [r for r in cov if r['axis']==axis]
    axes[0].set_position([.17,.12,.25,.76]);axes[1].set_position([.60,.12,.27,.76])
    cax=fig.add_axes([.90,.12,.015,.76]);fig.colorbar(im,cax=cax,label='Empirical 95% interval coverage (%)')
    fig.suptitle('Realized-count interval coverage: eight catalogs per cell',fontsize=17)
    save(fig,'shared_recovery_coverage',table,[base/'coverage.csv'], 'Coverage advances in steps of12.5%. Shared population and finite-response variation plus latent membership draws; zero-truth-count cells and numerically flagged cases remain.')
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    x=np.arange(len(numeric));delta=np.array([float(r['final_marginal_half_delta']) for r in numeric]);ess=np.array([float(r['response_ess']) for r in numeric])
    axes[0].semilogy(x,delta,'o',ms=3,color=BLUE);axes[0].axhline(.005,color=ORANGE,ls='--',label='Frozen tolerance');axes[0].legend(frameon=False,fontsize=11)
    axes[0].set(ylabel='Joint half-chain error',xlabel='Simulated catalog',title='Marginal Monte Carlo precision')
    axes[1].plot(x,ess,'o',ms=3,color=GREEN);axes[1].set(ylabel='Response ESS',xlabel='Simulated catalog',title='Reference-support concentration')
    fig.subplots_adjust(left=.09,right=.98,bottom=.17,top=.86,wspace=.32)
    save(fig,'shared_recovery_numerics',numeric,[base/'numerical.csv'],'Numerical precision and finite-response support are distinct. Small response ESS is a remaining modeling/integration limitation, not strong scientific confidence.')


def population():
    base=D/'ensemble_products/all32_native_refined';record=json.loads((base/'complete.json').read_text());rows=read(base/'count_summary.csv')
    fig,axes=plt.subplots(1,3,figsize=(14,4.8))
    for c,(ax,title,color) in enumerate(zip(axes,['Dwarf','Evolved','Other'],[BLUE,ORANGE,GREEN])):
        r=[next(x for x in rows if int(x['class_index'])==c and int(x['distance_bin'])==b) for b in range(6)]
        mean=np.array([float(x['mean']) for x in r]);lo=np.array([float(x['lo95']) for x in r]);hi=np.array([float(x['hi95']) for x in r])
        ax.vlines(range(6),lo,hi,color=color,lw=2);ax.plot(range(6),mean,'o',color=color)
        ax.set(title=title,xticks=range(6),xticklabels=BINS,xlabel='Distance bin (kpc)',ylabel='Observed sample count');ax.tick_params(axis='x',labelsize=10)
    fig.suptitle('Joint counts for the 26,000 sampled targets',fontsize=18);fig.subplots_adjust(left=.07,right=.98,bottom=.15,top=.81,wspace=.35)
    save(fig,'sample_joint_counts',rows,[base/'complete.json',base/'count_summary.csv'],'Posterior means and95%latent-count intervals including shared parameters. These are sample counts, not survey-expanded totals or intrinsic density. Finite response support remains limited.')
    weights=read(base/'response_weights.csv');z=np.array([float(r['log_evidence']) for r in weights]);order=np.argsort(z)[::-1]
    fig,ax=plt.subplots(figsize=(10,8));ax.plot(z[order]-z.max(),range(len(z)),'o',color=BLUE)
    ax.set(yticks=range(len(z)),yticklabels=[weights[i]['global_response_draw'] for i in order],xlabel='Log evidence relative to highest-weight response',ylabel='Global reference-response draw',title=f"Finite response support: effective sample size = {record['response_effective_sample_size']:.2f}")
    ax.tick_params(axis='y',labelsize=9);ax.invert_yaxis();fig.subplots_adjust(left=.12,right=.98,bottom=.1,top=.92)
    save(fig,'response_concentration',weights,[base/'response_weights.csv'],'All32fullrefit source/calibration realizations. Concentration within this discrete prior does not prove convergence over continuous reference uncertainty.')
    counts=load(base/'counts.npz');cov=counts['covariance'];sd=np.sqrt(np.maximum(np.diag(cov),0));corr=np.divide(cov,sd[:,None]*sd[None,:],out=np.zeros_like(cov),where=(sd[:,None]*sd[None,:])>0)
    fig,ax=plt.subplots(figsize=(9,8));im=ax.imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r');fig.colorbar(im,ax=ax,label='Posterior count correlation',fraction=.045,pad=.04)
    ticks=np.arange(18);labels=[f'{c}: {b}' for c in ['D','E','O'] for b in BINS]
    ax.set(xticks=ticks,xticklabels=labels,yticks=ticks,yticklabels=labels,title='Correlated class–distance counts')
    ax.tick_params(axis='both',labelsize=9);plt.setp(ax.get_xticklabels(),rotation=90)
    for v in [5.5,11.5]:ax.axhline(v,color='black',lw=.5);ax.axvline(v,color='black',lw=.5)
    fig.subplots_adjust(left=.14,right=.94,bottom=.18,top=.92)
    save(fig,'sample_count_correlation',[dict(i=i,j=j,covariance=float(cov[i,j]),correlation=float(corr[i,j])) for i in range(18) for j in range(18)], [base/'complete.json'],'D/E/O mean dwarf/evolved/other; each group contains six distance bins. Counts have fixed total26000; covariance includes negative normalization correlations.')


def catalog():
    base=D/'catalog/summary';record=json.loads((base/'complete.json').read_text());rows=read(base/'joint_counts.csv');fields=read(base/'fields.csv');prior=read(base/'prior_comparison.csv')
    fig,axes=plt.subplots(1,3,figsize=(14,4.8))
    names=['0_2','2_5','5_10','10_20','20_50','above_50']
    for c,(ax,title,color) in enumerate(zip(axes,['Dwarf','Evolved','Other'],[BLUE,ORANGE,GREEN])):
        selected=[next(r for r in rows if r['scope']=='all_qualified_fields' and int(r['class_index'])==c and r['distance_group']==b) for b in names]
        mean=[float(r['mean']) for r in selected];lo=[float(r['lower95']) for r in selected];hi=[float(r['upper95']) for r in selected]
        ax.vlines(range(6),lo,hi,color=color,lw=2);ax.plot(range(6),mean,'o',color=color)
        ax.set(title=title,xticks=range(6),xticklabels=BINS,xlabel='Distance bin (kpc)',ylabel='Qualified-catalog count');ax.tick_params(axis='x',labelsize=10);ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    fig.suptitle(f"Conditional candidate-catalog counts: {record['qualified']:,} objects",fontsize=18)
    fig.subplots_adjust(left=.07,right=.98,bottom=.15,top=.80,wspace=.38)
    save(fig,'catalog_joint_counts',rows,[base/'complete.json',base/'joint_counts.csv'],'Unit-weight actual qualified catalog. Means and95%shared latent-count intervals under modular transfer from the26000sample; not full-catalog refit, independently calibrated class census, or intrinsic density.')
    fig,axes=plt.subplots(1,2,figsize=(13,5));valid=[r for r in fields if int(r['candidate'])>0];x=np.arange(len(valid))
    axes[0].bar(x,[int(r['candidate']) for r in valid],color=GRAY,label='Qualified input')
    axes[0].bar(x,[int(r['near95_selected']) for r in valid],color=BLUE,label='Near probability >0.95')
    axes[0].set(xticks=x,xticklabels=[r['field'] for r in valid],xlabel='Existing DECam field index',ylabel='Objects',title='Candidate coverage and near selection');axes[0].legend(frameon=False,fontsize=11)
    for label,key,color in [('Nominal reference','nominal_reference_mean',GRAY),('Matched response reference','matched_response_reference_mean',GREEN),('Target population','target_mean',ORANGE)]:
        values=[sum(float(r[key]) for r in prior if int(r['distance_bin'])==b) for b in range(6)]
        axes[1].plot(range(6),values,'o-',label=label,color=color)
    axes[1].set(xticks=range(6),xticklabels=BINS,xlabel='Distance bin (kpc)',ylabel='Expected observed count',title='Reference and target-prior counts');axes[1].legend(frameon=False,fontsize=10)
    fig.subplots_adjust(left=.08,right=.98,bottom=.15,top=.86,wspace=.3)
    save(fig,'catalog_scope_prior',prior,[base/'fields.csv',base/'prior_comparison.csv'],'Qualifiedparent1,183,990 ofhistorical9,390,518 rows; north remainsunmodeled. Nearselectionmodel leakage is not measured real contamination. Allposteriorfar tails retained.')


def scale():
    plan=json.loads((D/'catalog/transfer_plan.json').read_text());assert len(plan['responses'])==1;r=plan['responses'][0]
    name=f'refit_{r["draw"]:02}_s{r["draws"]}';base=D/'scale_example'
    reference=read(base/'reference/distance_distribution.csv');target=read(base/name/'distance_distribution.csv')
    config=json.loads((ROOT/'configs/v3_0_independent_scale.json').read_text());mu=10+5*np.log10(config['distance_kpc'])
    fig,axes=plt.subplots(1,3,figsize=(14,4.8));rows=[]
    for ax,group,title in zip(axes,['all','members','nonmembers'],['All photometric objects','Post-associated members','Remaining photometric objects']):
        for data,label,color in [(reference,'Nominal reference prior',GRAY),(target,'Conditional target transfer',ORANGE)]:
            selected=[r for r in data if r['group']==group];ax.plot([float(r['mu']) for r in selected],[float(r['expected_count']) for r in selected],color=color,label=label)
            rows.extend([dict(r,view=label) for r in selected])
        ax.axvline(mu,color=BLUE,ls='--',label='Independent dynamical scale');ax.set(title=title,xlim=(8,22),xlabel='Distance modulus (mag)',ylabel='Expected objects per 0.05 mag');ax.legend(frameon=False,fontsize=9)
    fig.suptitle('NGC 288 field: distance marginalization and foreground mixture',fontsize=17)
    fig.subplots_adjust(left=.06,right=.99,bottom=.15,top=.8,wspace=.32)
    save(fig,'ngc288_prior_comparison',rows,[base/'reference/complete.json',base/name/'complete.json'],'One shared9.03kpc dynamicalscale with-0.56/+0.48kpc errors; membershipGaia-conditioned afterpurephotometricpredictions. Curves are posterior expectations, not independent-star Poisson uncertainty or intrinsic density. Nominal-reference versus target is not an isolated prior-only contrast; matched-response comparisons are reported separately.')


def main():
    require_compute();p=argparse.ArgumentParser();p.add_argument('stage',choices=['recovery','population','catalog','scale']);p.add_argument('--recovery-name',default='all32_native_precision1');a=p.parse_args()
    plt.rcParams.update(STYLE)
    if a.stage=='recovery': recovery(a.recovery_name)
    else: dict(population=population,catalog=catalog,scale=scale)[a.stage]()
    print('FINAL FIGURES',a.stage,flush=True)


if __name__=='__main__':main()
