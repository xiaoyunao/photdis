#!/usr/bin/env python3
"""One calibrated VAC luminosity-scale trial with a latent grz transfer check."""
import json
import os
import numpy as np
from scipy.special import logsumexp,ndtr
from numpy.polynomial.hermite import hermgauss
from v2_1_near import ROOT,load,subset,sha,save_json,csv_rows,atomic_npz
from v2_3_diagnostics import references
from baseline_v1_4_experiments import arrays
from photdis.audit import rvs_metallicity
from photdis.reference_v1_4 import unique_join
from photdis.stage1 import require_compute
from photdis.density_v2_3 import ErrorDeconvolvedMixture
from photdis.science_v2_4 import cluster_mean_interval
from photdis.population_v2_5 import weighted_quantile
D=ROOT/'data/v2_6';T=D/'tables'


def signed_loglike(mu,sigma,p,error,offset=0,scatter=0,order=32):
    x,w=hermgauss(order);w=w/np.sqrt(np.pi)
    m=np.asarray(mu)[...,None]-np.asarray(offset)[...,None]+np.sqrt(2)*np.hypot(sigma,scatter)[...,None]*x
    par=10**((10-m)/5)
    z=(np.asarray(p)[...,None]-par)/np.asarray(error)[...,None]
    result=logsumexp(-.5*z*z-np.log(np.asarray(error)[...,None]*np.sqrt(2*np.pi))+np.log(w),axis=-1)
    # When astrometry is narrower, integrate over its Gaussian rather than
    # trying to resolve it with a broad lognormal's fixed quadrature nodes.
    ln_sigma=np.log(10)/5*np.hypot(sigma,scatter)
    narrow=(np.asarray(p)>5*np.asarray(error))&(np.asarray(p)*ln_sigma>np.asarray(error))
    if np.any(narrow):
        pp=np.asarray(p)[narrow,None]+np.sqrt(2)*np.asarray(error)[narrow,None]*x
        safe=np.maximum(pp,1e-300);ls=ln_sigma[narrow,None]
        center=(10-np.asarray(mu)[narrow]+np.broadcast_to(offset,np.shape(mu))[narrow])*np.log(10)/5
        logdensity=-.5*((np.log(safe)-center[:,None])/ls)**2-np.log(safe*ls*np.sqrt(2*np.pi))
        logdensity[pp<=0]=-np.inf
        result[narrow]=logsumexp(logdensity+np.log(w),axis=-1)
    return result


def score_draws(mu,p,error,rng):
    par=10**((10-mu)/5);z=(p[:,None]-par)/error[:,None]
    log=logsumexp(-.5*z*z-np.log(error[:,None]*np.sqrt(2*np.pi)),axis=1)-np.log(mu.shape[1])
    pit=ndtr(z).mean(axis=1);predictive=par+rng.normal(size=par.shape)*error[:,None]
    interval=np.quantile(predictive,[.025,.16,.84,.975],axis=1).T
    return log,pit,interval


def calibrate(t,rows,poor,ref,sigma,source):
    grid=np.linspace(-1,1,41);scatter=np.linspace(0,.6,13);total=np.linspace(-2,2,81)
    ordinary=rows[~poor[rows]];metal=rows[poor[rows]]
    ol=np.array([signed_loglike(ref[ordinary],sigma[ordinary],t['p_corr'][ordinary],t['gaia_parallax_error'][ordinary],offset=z).sum() for z in grid])
    pl=np.empty((81,13))
    for j,z in enumerate(total):
        for k,s in enumerate(scatter):pl[j,k]=signed_loglike(ref[metal],sigma[metal],t['p_corr'][metal],t['gaia_parallax_error'][metal],offset=z,scatter=s).sum()
    a,b,s=np.meshgrid(np.arange(41),np.arange(41),np.arange(13),indexing='ij')
    ll=ol[a]+pl[a+b,s];w=np.exp(ll-logsumexp(ll));parameters=np.column_stack((grid[a.ravel()],grid[b.ravel()],scatter[s.ravel()]))
    atomic_npz(D/f'luminosity_posterior_{source}.npz',parameters=parameters,weight=w.ravel(),identity=t['identity'][rows],TYPE=t['TYPE'][rows])
    out=[]
    for k,name in enumerate(('source_offset','poor_offset','poor_scatter')):
        lo,med,hi=weighted_quantile(parameters[:,k],w.ravel());out.append(dict(source=source,parameter=name,mean=float(w.ravel()@parameters[:,k]),low=lo,median=med,high=hi,
            cal_poor=len(metal),cal_ordinary=len(ordinary),negative=int(np.sum(t['p_corr'][rows]<0)),weak=int(np.sum(t['p_corr'][rows]/t['gaia_parallax_error'][rows]<5)),boundary=float(w.ravel()@((abs(parameters[:,0])>.95)|(abs(parameters[:,1])>.95)|(parameters[:,2]>.55)))))
    return parameters,w.ravel(),out


def photometric_draws(t,rows,c,calref,calsigma,base,parameters,w,source):
    # No spectroscopic label or FeH from evaluation objects enters this function.
    m,e,ids=arrays(t,rows);absolute=c['absolute'].copy();supported=np.isfinite(calref)&np.isfinite(calsigma)&(calsigma>0)
    # Caller supplies source-specific absolute photometry, already teacher-swapped.
    teacher_sigma=np.where(supported,calsigma,c['sigma'])
    member=c['member'];prior=member/member.sum(axis=0)[None,:]*base[None,:];prior=prior.sum(axis=1);prior/=prior.sum()
    model=ErrorDeconvolvedMixture(components=len(prior));model.weights=prior
    model.means=np.column_stack((absolute[:,0]-absolute[:,1],absolute[:,1]-absolute[:,2],absolute[:,0]))
    model.covariances=np.zeros((len(prior),3,3));model.covariances[:,0,0]=.08**2;model.covariances[:,1,1]=.08**2;model.covariances[:,2,2]=.2**2+teacher_sigma**2
    cw,mean,sd=model.conditional(m,e);rng=np.random.default_rng(261210);param=parameters[rng.choice(len(w),4096,p=w)]
    probability_group=member/member.sum(axis=0)[None,:]*base[None,:];probability_group/=probability_group.sum(axis=1)[:,None]
    before=np.empty((len(rows),4096));after=before.copy();joint_before=np.zeros((len(rows),4,6));joint_after=joint_before.copy()
    edges=np.array([-np.inf,10+5*np.log10(2),10+5*np.log10(5),15,10+5*np.log10(20),10+5*np.log10(50),np.inf])
    for k in range(len(rows)):
        ix=rng.choice(len(prior),4096,p=cw[k]);g=(rng.random(4096)[:,None]>np.cumsum(probability_group[ix],axis=1)).sum(axis=1);g=np.minimum(g,3)
        before[k]=m[k,0]-mean[k,ix]+sd[k,ix]*rng.normal(size=4096)
        apply=supported[ix]&(g<2);poor=g==1
        after[k]=before[k]-apply*(param[:,0]+poor*param[:,1])+apply*poor*param[:,2]*rng.normal(size=4096)
        for out,mu in [(joint_before,before[k]),(joint_after,after[k])]:
            dist=np.clip(np.searchsorted(edges,mu,side='right')-1,0,5);out[k]=np.bincount(6*g+dist,minlength=24).reshape(4,6)/4096
    atomic_npz(D/f'photometric_luminosity_{source}.npz',identity=ids,TYPE=t['TYPE'][rows],rows=rows,mu_before=before,mu_after=after,joint_before=joint_before,joint_after=joint_after)
    return before,after


def main():
    require_compute();T.mkdir(parents=True,exist_ok=True)
    files=[ROOT/'scripts/v2_6_luminosity.py',ROOT/'docs/v2_6_luminosity_design.md',ROOT/'docs/v2_6_protocol.md',ROOT/'data/v2_1/development.npz',ROOT/'data/v2_3/metal_poor_pairs.npz',ROOT/'data/v2_4/prospective_identity_reservation.npz']
    freeze={str(p.relative_to(ROOT)):sha(p) for p in files};marker=D/'luminosity_freeze.json'
    if marker.exists():assert json.loads(marker.read_text())==freeze
    else:save_json(marker,freeze)
    t=load(ROOT/'data/v2_1/development.npz');pairs=load(ROOT/'data/v2_3/metal_poor_pairs.npz');fit=load(ROOT/'data/v2_3/training.npz')['identity']
    a=unique_join(pairs['poor_identity'],t['identity']);b=unique_join(pairs['control_identity'],t['identity']);assert min(a.min(),b.min())>=0
    evalids=np.unique(np.r_[t['identity'][a],t['identity'][b]])
    feh,valid=rvs_metallicity(t['rvs_feh'],t['rvs_teff'],t['rvs_logg']);quality=valid&t['rvs_parameter_quality']&np.isfinite(t['rvs_feh_err'])&(t['rvs_feh_err']>0)
    poor=(t['class_label']==0)&quality&(feh< -1.5);ordinary=(t['class_label']==0)&quality&(feh>=-.5)
    eligible=(t['TYPE']=='PSF')&~t['nonstellar_risk']&~t['identity_conflict']&t['parallax_eligible']&np.isfinite(t['p_corr'])&np.isfinite(t['gaia_parallax_error'])&(t['gaia_parallax_error']>0)&~np.isin(t['identity'],fit)
    cal=(t['split']==1)&eligible&~np.isin(t['identity'],evalids)
    c0=load(ROOT/'data/v2_5/cal_templates.npz');ci=unique_join(c0['identity'],t['identity']);assert min(ci)>=0
    assert not np.intersect1d(c0['identity'],evalids).size
    base=np.array(json.loads((ROOT/'data/v2_5/templates.json').read_text())['base'])
    summaries=[];objects=[];changes=[];audit=[];rng=np.random.default_rng(261220)
    for source in ('specdis','spdist'):
        ref,sigma=references(t,source);sourceok=np.isfinite(ref)&np.isfinite(sigma)&(sigma>0)
        rr=[]
        for group in (ordinary,poor):
            rows=np.flatnonzero(cal&sourceok&group)
            with np.errstate(over='ignore'):rank=t['identity'][rows].astype('u8')*np.uint64(11400714819323198485)
            rr.extend(rows[np.argsort(rank,kind='stable')[:512]])
        rr=np.array(rr);assert len(rr)>40 and not np.intersect1d(t['identity'][rr],evalids).size
        parameters,w,summary=calibrate(t,rr,poor,ref,sigma,source);summaries+=summary
        use=eligible[a]&eligible[b]&sourceok[a]&sourceok[b];aa=a[use];bb=b[use];union=np.unique(np.r_[aa,bb]);index={r:k for k,r in enumerate(union)}
        cr={k:v.copy() for k,v in c0.items()};known=sourceok[ci];cr['absolute'][known]+= (t['ref_chosen'][ci[known]]-ref[ci[known]])[:,None]
        before,after=photometric_draws(t,union,cr,ref[ci],sigma[ci],base,parameters,w,source)
        param=parameters[rng.choice(len(w),4096,p=w)]
        teacher=ref[union,None]+sigma[union,None]*rng.normal(size=before.shape)
        teacher_after=teacher-param[:,0][None,:]-poor[union,None]*param[:,1][None,:]+poor[union,None]*param[:,2][None,:]*rng.normal(size=before.shape)
        scores={}
        for method,mu in [('teacher_before',teacher),('teacher_after',teacher_after),('grz_before',before),('grz_after',after)]:
            log,pit,iv=score_draws(mu,t['p_corr'][union],t['gaia_parallax_error'][union],rng);scores[method]=(log,pit,iv)
            for side,rs in [('poor',aa),('ordinary',bb)]:
                for r in rs:
                    k=index[r];objects.append(dict(source=source,method=method,side=side,identity=int(t['identity'][r]),TYPE='PSF',
                        control_identity=int(t['identity'][bb[np.flatnonzero(aa==r)[0]]]) if side=='poor' else int(t['identity'][r]),
                        parallax=t['p_corr'][r],error=t['gaia_parallax_error'][r],log_score=log[k],pit=pit[k],
                        p025=iv[k,0],p16=iv[k,1],p84=iv[k,2],p975=iv[k,3],mu16=float(np.quantile(mu[k],.16)),mu50=float(np.median(mu[k])),mu84=float(np.quantile(mu[k],.84)),teacher_mu=ref[r]))
        ia=np.array([index[r] for r in aa]);ib=np.array([index[r] for r in bb])
        for family in ('teacher','grz'):
            for side,ii in [('poor',ia),('ordinary',ib)]:
                for metric in ('log_score','low_tail','high_tail','coverage68','coverage95'):
                    values=[]
                    for suffix in ('before','after'):
                        log,pit,iv=scores[family+'_'+suffix];p=t['p_corr'][union]
                        v={'log_score':log,'low_tail':(pit<.05).astype(float),'high_tail':(pit>.95).astype(float),
                           'coverage68':((p>=iv[:,1])&(p<=iv[:,2])).astype(float),'coverage95':((p>=iv[:,0])&(p<=iv[:,3])).astype(float)}[metric];values.append(v[ii])
                    changes.append(dict(source=source,family=family,side=side,metric=metric,n=len(ii),before=float(values[0].mean()),after=float(values[1].mean()),**cluster_mean_interval(values[1]-values[0],t['identity'][bb])))
        audit.append(dict(source=source,cal=len(rr),evaluation_pairs=len(aa),unique_controls=len(np.unique(bb)),identity_overlap=0,group_overlap=0,
            nonPSF=int((t['TYPE']!='PSF').sum()),missing_TYPE=int((t['TYPE']=='').sum()),cal_evaluation_excluded=int(((t['split']==1)&np.isin(t['identity'],evalids)).sum()),
            cal_negative=int(np.sum(t['p_corr'][rr]<0)),evaluation_negative=int(np.sum(t['p_corr'][union]<0)),unknown_metallicity_latent=True,
            limits='VAC-conditioned selection; source/Gaia dependence not modeled; no universal correction or deployment; source nuisance calibrated with ordinary stars.'))
        print('LUMINOSITY SOURCE COMPLETE',source,len(rr),len(aa),flush=True)
    csv_rows(T/'luminosity_parameters.csv',summaries);csv_rows(T/'luminosity_objects.csv',objects);csv_rows(T/'luminosity_changes.csv',changes)
    save_json(D/'luminosity_admission.json',audit);assert all(sha(ROOT/k)==v for k,v in freeze.items())
    save_json(D/'luminosity_complete.json',dict(job=os.getenv('SLURM_JOB_ID'),complete=True,prospective_unopened=True))


if __name__=='__main__':main()
