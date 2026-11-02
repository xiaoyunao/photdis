#!/usr/bin/env python3
"""Unified scorecard and figures from fixed baseline artifacts, never retunes."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'.deps')]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from photdis.stage1 import require_compute
from photdis.baseline_v0 import metrics
from baseline_v0_fit import truth_mu,strata,load,OUT,CFG
from kgiant_distance_inventory import write_json,sha
FIG=Path('figures/baseline_v0');FIG.mkdir(exist_ok=True,parents=True)
def save(fig,name):fig.savefig(FIG/name,dpi=150);plt.close(fig)
def density(ax,x,y,title,xlabel,ylabel,limits=None):
    good=np.isfinite(x)&np.isfinite(y);x=x[good];y=y[good]
    if len(x):
        h=ax.hexbin(x,y,gridsize=65,mincnt=1,bins='log',cmap='viridis')
        ax.figure.colorbar(h,ax=ax,fraction=.045,pad=.02,label='Sources / hexagon')
    ax.set(title=f'{title}: N={len(x):,}',xlabel=xlabel,ylabel=ylabel)
    if limits:ax.set(xlim=limits,ylim=limits);ax.plot(limits,limits,'r--',lw=.8)

def main():
    require_compute();t=load();fit=json.load(open(OUT/'fit_result.json'));prep=json.load(open(OUT/'prepared.json'));n=len(t['g0'])
    ix=np.flatnonzero(t['split']==2);mu=np.full((len(ix),5),np.nan);muc=mu.copy();support=np.zeros(len(ix),bool);amb=support.copy();route=np.zeros(len(ix),np.int8)
    lookup=np.full(n,-1,int);lookup[ix]=np.arange(len(ix));evaluation={}
    for code,k in enumerate(('A','B','C')):
        with np.load(OUT/(k+'_evaluation.npz')) as p:
            test=t['split'][p['rows']]==2;dest=lookup[p['rows'][test]];sp=p['supported'][test]
            take=np.ones(len(dest),bool) if k=='A' else sp&(fit['route'][k]['adopt']|~support[dest])
            dd=dest[take];mu[dd]=p['mu'][test][take];muc[dd]=p['mu_cal'][test][take];support[dd]=sp[take];amb[dd]=p['ambiguous'][test][take];route[dd]=code
            evaluation[k]={x:p[x][test] for x in ['mu','rows','supported','class_weights']}
    assert np.isfinite(mu).all()
    layer=strata(t,ix);layer.update(supported=support,outside=~support,ambiguous=amb)
    for lo,hi in [(-5,.4),(.4,.8),(.8,1.2),(1.2,1.6),(1.6,5)]:
        color=t['g0'][ix]-t['r0'][ix];layer[f'gr_{lo}_{hi}']=(color>=lo)&(color<hi)
    scores={ref:{k:metrics(mu[m],truth_mu(t,ref)[ix][m]) for k,m in layer.items()} for ref in ['chosen','Gaia','specdis','spdist']}
    result=dict(preparation=prep,fit=fit,recommended_test=scores,calibrated_test={k:metrics(muc,truth_mu(t,k)[ix]) for k in ['chosen','Gaia','specdis','spdist']},n_test_references=len(ix),supported_fraction=float(support.mean()),ambiguous_fraction=float(amb.mean()),routes={str(k):int((route==k).sum()) for k in range(3)})
    fig,ax=plt.subplots(2,3,figsize=(15,9),layout='constrained')
    for j,ref in enumerate(['Gaia','specdis','spdist']):
        truth=truth_mu(t,ref)[ix];good=np.isfinite(truth);x=(truth[good]-10)/5;y=(mu[good,2]-10)/5
        density(ax[0,j],x,y,ref,'log10(reference d/kpc)','log10(phot d/kpc)',[-2,3])
        residual=mu[good,2]-truth[good];ax[1,j].hist(residual,bins=np.linspace(-5,5,101),histtype='step');ax[1,j].set(xlabel='phot - reference distance modulus [mag]',ylabel='N',title=f'Outside display: {int((abs(residual)>5).sum())}');ax[1,j].axvline(0,color='k',lw=.7)
    fig.suptitle('Complete development-test references; no true type supplied; all support states')
    save(fig,'03_distance_references.png')
    truth=truth_mu(t,'chosen')[ix];good=np.isfinite(truth);gr=t['g0'][ix]-t['r0'][ix];mgref=t['g0'][ix]-truth;mgphot=t['g0'][ix]-mu[:,2]
    fig,ax=plt.subplots(2,2,figsize=(12,10),layout='constrained')
    train=(t['split']==0)&t['fit_eligible']
    for a,source,title in zip(ax[0],[1,2],['Training: Gaia anchors','Training: supplementary SPECDIS']):
        mask=train&(t['label_source']==source)
        density(a,(t['g0']-t['r0'])[mask],t['Mg'][mask],title,'(g-r)0','Mg');a.set(xlim=(-.5,2),ylim=(17,-6))
    for a,v,title in zip(ax[1],[mgref,mgphot],['Same test sources: reference','Same test sources: photometric median']):
        density(a,gr[good],v[good],title,'(g-r)0','Mg');a.set(xlim=(-.5,2),ylim=(17,-6))
    save(fig,'04_same_source_cmd.png')
    result['training_population']={}
    for name,entry in fit['models'].items():
        tier=name[-1] if '_on_' in name else name
        mask=train&(np.ones(n,bool) if tier=='A' else t[tier])
        result['training_population'][name]=dict(n=int(mask.sum()),classes={str(k):int((mask&(t['class_label']==k)).sum()) for k in [-1,0,1,2]},Mg_percentiles=np.percentile(t['Mg'][mask],[0,1,50,99,100]).tolist(),sigma_mu_median=float(np.median(t['sigma_mu'][mask])))
    fig,ax=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    stages=['Optical records','Unique sources','Reference candidates','Training core A']
    values=[prep['n_parent_records'],prep['n_unique_parent'],prep['n_reference'],fit['models']['A']['n_train']]
    ax[0].barh(stages,values);ax[0].set(xscale='log',xlabel='Source / record count',title='Existing 14-field sample flow');ax[0].invert_yaxis()
    for y,v in enumerate(values):ax[0].text(v,y,f' {v:,}',va='center',fontsize=9)
    for j,source in enumerate([1,2]):
        counts=[int(((t['split']==split)&(t['label_source']==source)).sum()) for split in range(4)]
        ax[1].bar(np.arange(4)+j*.35,counts,width=.35,label=['Gaia','SPECDIS supplement'][j])
    ax[1].set(xticks=np.arange(4)+.175,xticklabels=['train area','cal area','test area','north'],ylabel='References (before precision cut)',yscale='log');ax[1].legend()
    save(fig,'00_sample_flow.png')
    fig,ax=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for j,(tag,names) in enumerate([('W1',['A_on_B','B']),('W1W2',['A_on_C','B_on_C','C'])]):
        for ref,marker in [('Gaia','o'),('specdis','s')]:
            vals=[fit['same_star'][tag][name][ref].get('nmad_mag',np.nan) for name in names];ax[j].plot(names,vals,marker=marker,label=ref)
        ax[j].set(title=tag+' identical train and test identities',ylabel='Distance modulus NMAD [mag]');ax[j].legend()
    save(fig,'02_same_star_band_gain.png')
    fig,ax=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for a,k in zip(ax,('A','B','C')):
        matrix=np.array(fit['models'][k]['class_confusion']);rates=matrix/np.maximum(matrix.sum(axis=1)[:,None],1);a.imshow(rates,vmin=0,vmax=1,cmap='Blues')
        for (i,j),v in np.ndenumerate(matrix):a.text(j,i,str(v),ha='center',va='center',fontsize=9)
        a.set(title=k+' available test references',xticks=range(3),yticks=range(3),xticklabels=['dwarf','evolved','transition'],yticklabels=['dwarf','evolved','transition'],xlabel='Empirical type score maximum',ylabel='DESI evidence')
    save(fig,'06_coarse_type_confusion.png')
    # Weak/negative observations retained in parallax space; descriptive only.
    weak=t['astro_quality'][ix]&t['zpt_valid'][ix]&(t['p_corr'][ix]/t['gaia_parallax_error'][ix]<10)
    par=t['p_corr'][ix];phot=10**((10-mu[:,2])/5)
    fig,ax=plt.subplots(1,2,figsize=(12,5),layout='constrained');density(ax[0],par[weak],phot[weak],'Weak/negative observed parallax','Corrected observed parallax [mas]','Photometric median parallax [mas]')
    z=(par[weak]-phot[weak])/t['gaia_parallax_error'][ix][weak];ax[1].hist(z,bins=np.linspace(-10,10,81),histtype='step');ax[1].set(xlabel='(observed - photometric median)/Gaia error',title='Selected descriptive residual; not a zero-point test')
    save(fig,'07_weak_parallax.png');result['weak_parallax']=dict(n=int(weak.sum()),negative=int((weak&(par<0)).sum()),note='SNR-selected descriptive comparison, no inverse weak-parallax labels, no unconditioned PIT/zero-point inference')
    # Representative empirical luminosity distributions, showing resolved ambiguity.
    with np.load(OUT/'A_evaluation.npz') as p:
        test=t['split'][p['rows']]==2;selected=np.flatnonzero(test&p['ambiguous'])[:4]
        if len(selected)<4:selected=np.unique(np.r_[selected,np.flatnonzero(test)[::max(1,int(test.sum()/4))][:4]])[:4]
        fig,axs=plt.subplots(2,2,figsize=(10,7),layout='constrained')
        for a,row in zip(axs.flat,selected):
            a.plot(p['centers'],p['pdf'][row]);a.set(title=f"Identity {t['identity'][p['rows'][row]]}",xlabel='Effective Mg (shared g noise included)',ylabel='Bin probability')
        save(fig,'08_distribution_examples.png')
    catalog_paths=[OUT/f'catalog_{i:02}.json' for i in range(14)]
    if all(p.exists() for p in catalog_paths):
        cats=[json.load(open(p)) for p in catalog_paths];result['application']=cats
        assert all(c['n_unique']==f['unique_parent'] for c,f in zip(cats,prep['fields']))
        south=[c for c in cats if c['field'] not in CFG['north_fields']]
        fig,ax=plt.subplots(2,2,figsize=(12,9),layout='constrained')
        for a,key in zip(ax.flat,['r','gr','error']):
            edges=np.array(south[0]['histograms'][key]['edges']);base=sum(np.array(c['histograms'][key]['counts']['A']) for c in south)
            for k in ['B','C']:
                y=sum(np.array(c['histograms'][key]['counts'][k]) for c in south);a.plot((edges[:-1]+edges[1:])/2,y/np.maximum(base,1),'o-',label=k)
            a.set(xlabel=key,ylabel='Fraction of unique clean optical parent',ylim=(0,1));a.legend()
        for k in ['B','C']:ax[1,1].plot([c['field'] for c in south],[c['availability'][k]/max(c['n_unique'],1) for c in south],'o-',label=k)
        ax[1,1].set(xlabel='Field index',ylabel='WISE coverage',ylim=(0,1));ax[1,1].legend();save(fig,'01_wise_coverage.png')
        # Quality-performance loss measured on test; optical coverage on full parent.
        limits=[.03,.06,.12,.25];curve=[]
        for limit in limits:
            mask=t['optical_error'][ix]<limit
            curve.append(dict(error_limit=limit,test=metrics(mu[mask],truth[mask]),n_test_retained=int(mask.sum()),n_parent_retained=sum(c['error_cumulative'][str(limit)] for c in south),references={ref:metrics(mu[mask],truth_mu(t,ref)[ix][mask]) for ref in ['Gaia','specdis']}))
        result['precision_coverage']=curve
        fig,ax=plt.subplots(1,2,figsize=(12,5),layout='constrained')
        fractions=[c['n_parent_retained']/sum(p['n_unique'] for p in south) for c in curve]
        for ref in ['Gaia','specdis']:
            values=[c['references'][ref].get('nmad_mag',np.nan) for c in curve]
            ax[0].plot(fractions,values,'o-',label=ref)
            for c,x,y in zip(curve,fractions,values):ax[0].annotate(str(c['error_limit']), (x,y),fontsize=8)
        ax[0].set(xlabel='Full optical parent retained fraction',ylabel='Test reference NMAD [mag]',title='Maximum optical-error cuts [mag]');ax[0].legend()
        for k in ['A','B','C']:ax[1].plot([c['field'] for c in south],[c['availability'][k] for c in south],'o-',label=k)
        ax[1].set(yscale='log',xlabel='Field',ylabel='Available unique sources');ax[1].legend();save(fig,'05_precision_coverage.png')
        result['complete_application']=True
    else:result['complete_application']=False;result['pending_catalogs']=[str(p) for p in catalog_paths if not p.exists()]
    result['artifact_checks']=dict(unique_reference_ids=len(np.unique(t['identity']))==len(t['identity']),test_ids_disjoint_train=not bool(np.intersect1d(t['identity'][t['split']==0],t['identity'][ix]).size),all_test_have_finite_prediction=bool(np.isfinite(mu).all()),same_training_controls=all(fit['models'][a]['train_identity_sha256']==fit['models'][b]['train_identity_sha256'] for a,b in [('A_on_B','B'),('A_on_C','C'),('B_on_C','C')]),all_available_parent_rows_accounted_for=result['complete_application'])
    write_json(OUT/'report.json',result);np.savez_compressed(OUT/'test_recommended.npz',rows=ix,mu=mu,mu_cal=muc,supported=support,ambiguous=amb,route=route)
    print('Report complete; all application finished:',result['complete_application'],flush=True)
if __name__=='__main__':main()
