#!/usr/bin/env python3
"""Correct random power cohorts and preselection branch-mixture conditioning."""
from scipy.special import ndtr, ndtri
from scipy.stats import kstest
from photdis.stage1 import require_compute
from sv18_common import cached, index_join
from sv18_predictive import gaussian_mixture, row_stats
from paper_v1_benchmark import read_scan
from final_catalogue_benchmark import load_parent
from halo_common import *


def wilson(k,n):
    if not n:return np.nan,np.nan
    z=1.959963984540054
    p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return center-half,center+half


def selected_components(mu,base_mass,error,kind,window,fraction,scale):
    shifted=mu+5*np.log10(1+scale)
    both=np.concatenate((shifted,shifted+2),axis=1)
    mass=np.concatenate(((1-fraction)*base_mass,fraction*base_mass),axis=1)
    center=10**((10-both)/5) if kind=='gaia' else both
    lower=np.zeros_like(center) if window is None else ndtr(window[:,0,None]-center/error[:,None])
    upper=np.ones_like(center) if window is None else ndtr(window[:,1,None]-center/error[:,None])
    weights=mass*(upper-lower)
    total=weights.sum(axis=1)
    normalized=np.divide(weights,total[:,None],out=np.zeros_like(weights),where=total[:,None]>1e-14)
    branch=normalized[:,mu.shape[1]:].sum(axis=1)
    return center,lower,upper,normalized,total,branch


def simulate(prepared,error,rng):
    center,lo,hi,weights,total,_=prepared
    cdf=np.cumsum(weights,axis=1)
    index=(cdf>=rng.random(len(error))[:,None]).argmax(axis=1)
    rows=np.arange(len(error))
    u=lo[rows,index]+rng.random(len(error))*(hi[rows,index]-lo[rows,index])
    y=center[rows,index]+error*ndtri(np.clip(u,1e-12,1-1e-12))
    return y,total>1e-14


def main():
    require_compute()
    d=folder('power');inputs={}
    _,detail,_=read_scan(inputs)
    t,_=load_parent()
    j=index_join(t['ls_id'],detail['ls_id']);good=j>=0
    t={k:v[good] for k,v in t.items()}
    detail={k:v[j[good]] for k,v in detail.items()}
    reference=cached(inputs,('TYPE','ls_id','p_corr','p_err','specdis_sig','spdist_sig'))
    j=index_join(t['ls_id'],reference['ls_id']);assert (j>=0).all()
    for k in ('p_corr','p_err','specdis_sig','spdist_sig'):t[k]=reference[k][j]
    del reference
    old=ROOT/'data/science_validation_20261118/predictive'
    with np.load(verified(old/'sampled_ids.npz',json.loads((old/'complete.json').read_text())['outputs']['sampled_ids.npz'])) as a:
        frozen={k:a[k] for k in a.files}
    inputs['original_predictive']=sha(old/'complete.json')
    results=[];check=[];cohorts=[];private={};diagnostic=[]
    for ref in ('gaia','specdis','spdist'):
        rows=index_join(frozen[ref+'_all_ls_id'],t['ls_id']);assert (rows>=0).all()
        y=t['p_corr'][rows] if ref=='gaia' else 10+5*np.log10(t[ref+'_kpc'][rows])
        error=t['p_err'][rows] if ref=='gaia' else t[ref+'_sig'][rows]
        window=None if ref!='gaia' else np.column_stack((np.where(t['gaia_state'][rows]==4,-np.inf,5),
                                                       np.where(t['gaia_state'][rows]==4,5,10)))
        edges=detail['qrf_mu_edges'][rows];mass=detail['qrf_mu_mass'][rows]
        u,density=gaussian_mixture(edges,mass,y,error,ref,window)
        check.append(dict(reference=ref,**row_stats(u,density)))
        valid=np.isfinite(u)&np.isfinite(error)&(error>0)
        positions=np.flatnonzero(valid)
        chosen=RNG.choice(positions,size=min(C['power']['N'],len(positions)),replace=False)
        old_positions=positions[:C['power']['N']]
        for label,ii in (('previous_first_rows',old_positions),('uniform_valid_rows',chosen)):
            cohort=t['region'][rows[ii]]
            for side in np.unique(cohort):
                mask=cohort==side
                cohorts.append(dict(reference=ref,sampling=label,native=side,n=int(mask.sum()),
                    median_r=np.median(t['r'][rows[ii[mask]]]),mean_parent_index=np.mean(rows[ii[mask]])))
        private[ref+'_ls_id']=t['ls_id'][rows[chosen]]
        private[ref+'_TYPE']=t['TYPE'][rows[chosen]]
        private[ref+'_RA']=t['RA'][rows[chosen]];private[ref+'_DEC']=t['DEC'][rows[chosen]]
        private[ref+'_old_ls_id']=t['ls_id'][rows[old_positions]]
        e=edges[chosen];w=mass[chosen];ee=error[chosen];ww=None if window is None else window[chosen]
        mid=(e[:,:-1]+e[:,1:])/2;half=np.diff(e,axis=1)/2
        nodes,quadrature=np.polynomial.legendre.leggauss(3)
        mu=(mid[:,:,None]+half[:,:,None]*nodes).reshape(len(e),-1)
        wm=(w[:,:,None]*(quadrature/2)).reshape(len(e),-1)
        for scale in C['power']['scale_fraction']:
            for branch in C['power']['branch_fraction']:
                prepared=selected_components(mu,wm,ee,ref,ww,branch,scale)
                k=0;negative=[];sizes=[]
                for draw in range(C['power']['draws']):
                    simulated,ok=simulate(prepared,ee,RNG)
                    pit,_=gaussian_mixture(e[ok],w[ok],simulated[ok],ee[ok],ref,None if ww is None else ww[ok])
                    v=pit[np.isfinite(pit)]
                    if len(v):
                        k+=kstest(v,'uniform').pvalue<.05
                        sizes.append(len(v));negative.append(int((simulated[ok]<0).sum()))
                lo,hi=wilson(k,len(sizes))
                results.append(dict(reference=ref,scale_fraction=scale,wrong_branch_mass=branch,n=len(e),
                    replications=len(sizes),rejections=k,rejection_fraction=k/len(sizes) if sizes else None,
                    ci_lo=lo,ci_hi=hi,mean_valid=np.mean(sizes),mean_negative=np.mean(negative),
                    mean_selected_branch_fraction=np.mean(prepared[-1])))
                print(ref,scale,branch,k,len(sizes),flush=True)
    # One near/far analytic case makes the prior/selected branch distinction explicit.
    mu=np.full((1,1),10.);wm=np.ones_like(mu);err=np.array([.2]);window=np.array([[-np.inf,5.]])
    prepared=selected_components(mu,wm,err,'gaia',window,.3,0)
    z0=ndtr(5-1/.2);z1=ndtr(5-10**(-.4)/.2)
    expected=.3*z1/(.7*z0+.3*z1)
    np.testing.assert_allclose(prepared[-1][0],expected,atol=1e-14)
    diagnostic.append(dict(prior_branch=.3,base_acceptance=z0,far_acceptance=z1,
        analytic_selected_fraction=expected,joint_component_selected_fraction=prepared[-1][0]))
    csv_rows(d/'power.csv',results);csv_rows(d/'primary_replay.csv',check)
    csv_rows(d/'cohort_distribution.csv',cohorts);csv_rows(d/'branch_conditioning.csv',diagnostic)
    atomic_npz(d/'power_ids.npz',**private)
    with plt.rc_context(PAPER):
        f,axes=plt.subplots(1,3,figsize=(7.1,2.9),layout='constrained',sharey=True)
        for ax,ref in zip(axes,('gaia','specdis','spdist')):
            for branch,color in zip((0,.1,.3),(BLUE,ORANGE,GREEN)):
                a=[r for r in results if r['reference']==ref and r['wrong_branch_mass']==branch]
                x=[r['scale_fraction'] for r in a];y=[r['rejection_fraction'] for r in a]
                ax.plot(x,y,'o-',color=color,ms=3,label=f'{branch:g}')
                ax.fill_between(x,[r['ci_lo'] for r in a],[r['ci_hi'] for r in a],color=color,alpha=.13)
            ax.set(title=ref.capitalize(),xlabel='Injected distance-scale change',ylim=(0,1))
        axes[0].set_ylabel('Detection fraction')
        axes[0].legend(title='Prior far-branch mass',frameon=False)
        figure(d,'corrected_scale_branch_power',f,results)
    complete(d,inputs,branch_scope='mixture fraction before SNR selection; joint component acceptance normalization',
             cohort_scope='uniform positions from final valid fixed 2048 source cohort, no sorting-prefix selection',
             interval_scope='Wilson 95% Monte Carlo interval on the fixed-model detection probability')


if __name__=='__main__':main()
