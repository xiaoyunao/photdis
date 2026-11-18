#!/usr/bin/env python3
"""Observation-space calibration and scale/branch sensitivity on archived PDFs."""
from scipy.special import ndtr, ndtri
from scipy.stats import kstest
from photdis.stage1 import require_compute
from sv18_common import *
from paper_v1_benchmark import read_scan
from final_catalogue_benchmark import load_parent
from paper_v1_scan import locate


def gaussian_mixture(edges,mass,y,error,kind,window=None,shift=0,return_selection=False):
    mid=(edges[:,:-1]+edges[:,1:])/2+shift
    half=np.diff(edges,axis=1)/2
    nodes,weights=np.polynomial.legendre.leggauss(3)
    numerator=np.zeros(len(y));selection=np.zeros(len(y));density=np.zeros(len(y))
    for node,weight in zip(nodes,weights/2):
        mu=mid+half*node
        center=10**((10-mu)/5) if kind=='gaia' else mu
        z=(y[:,None]-center)/error[:,None]
        a=0 if window is None else ndtr(window[:,0,None]-center/error[:,None])
        b=1 if window is None else ndtr(window[:,1,None]-center/error[:,None])
        numerator+=weight*np.sum(mass*(ndtr(z)-a),axis=1)
        selection+=weight*np.sum(mass*(b-a),axis=1)
        density+=weight*np.sum(mass*norm.pdf(z)/error[:,None],axis=1)
    result=(np.divide(numerator,selection,out=np.full(len(y),np.nan),where=selection>1e-12),
            np.divide(density,selection,out=np.full(len(y),np.nan),where=selection>1e-12))
    return (*result,selection) if return_selection else result


def row_stats(u,density):
    good=np.isfinite(u)&(u>=0)&(u<=1)&np.isfinite(density)&(density>0)
    v=u[good]
    return dict(n=len(u),valid=int(good.sum()),mean_log_score=np.log(density[good]).mean() if good.any() else None,
       median_pit=np.median(v) if len(v) else None,
       **{f'coverage_{int(100*c)}':((v>=(1-c)/2)&(v<=(1+c)/2)).mean() if len(v) else None
          for c in P['predictive']['nominal_coverage']})


def main():
    require_compute()
    inputs={};d=out('predictive')
    _,detail,_=read_scan(inputs)
    t,marker=load_parent()
    ii=index_join(t['ls_id'],detail['ls_id']);use=ii>=0
    t={k:v[use] for k,v in t.items()};detail={k:v[ii[use]] for k,v in detail.items()}
    np.testing.assert_array_equal(t['ls_id'],detail['ls_id'])
    r=cached(inputs,('TYPE','ls_id','p_corr','p_err','pop','specdis_sig','spdist_sig'))
    ix=index_join(t['ls_id'],r['ls_id']);assert (ix>=0).all()
    for k in ('p_corr','p_err','pop','specdis_sig','spdist_sig'):t[k]=r[k][ix]
    del r
    edges=detail['qrf_mu_edges'];mass=detail['qrf_mu_mass']
    allrows=[];hist=[];power=[];sensitivity=[];sampling=[];sample_ids={}
    for ref in ('gaia','specdis','spdist'):
        qual=np.isin(t['gaia_state'],(4,5)) if ref=='gaia' else np.isfinite(t[ref+'_kpc'])&(t[ref+'_kpc']>0)
        sets={'all':qual,'north':qual&(t['region']=='north'),'south':qual&(t['region']!='north'),
              'ordinary':qual&(t['pop']==1),'metal_poor':qual&(t['pop']==2),
              'intermediate_gravity':qual&(t['pop']==4)}
        for label,eligible in sets.items():
            rows=np.flatnonzero(eligible)
            parent_n=len(rows)
            if len(rows)>P['predictive']['max_rows_per_group']:
                rows=np.sort(RNG.choice(rows,P['predictive']['max_rows_per_group'],replace=False))
            if not len(rows):continue
            y=t['p_corr'][rows] if ref=='gaia' else 10+5*np.log10(t[ref+'_kpc'][rows])
            error=t['p_err'][rows] if ref=='gaia' else t[ref+'_sig'][rows]
            good=np.isfinite(error)&(error>0)&np.isfinite(y)
            rows,y,error=rows[good],y[good],error[good]
            if not len(rows):continue
            sampling.append(dict(reference=ref,group=label,eligible_parent=parent_n,used=len(rows),
                negative=int((y<0).sum()) if ref=='gaia' else 0))
            sample_ids[ref+'_'+label+'_ls_id']=t['ls_id'][rows]
            sample_ids[ref+'_'+label+'_TYPE']=t['TYPE'][rows]
            window=None
            if ref=='gaia':
                window=np.column_stack((np.where(t['gaia_state'][rows]==4,-np.inf,5),
                                        np.where(t['gaia_state'][rows]==4,5,10)))
            for mul in P['predictive']['error_multipliers']:
                adjusted_window=None if window is None else window/mul
                u,dens=gaussian_mixture(edges[rows],mass[rows],y,error*mul,ref,adjusted_window)
                allrows.append(dict(reference=ref,group=label,error_multiplier=mul,
                    scale_fraction=0,negative=int((y<0).sum()) if ref=='gaia' else 0,
                    median_pdf_width_mu=np.median(5*np.log10(t['qrf_quantiles_kpc'][rows,3]/t['qrf_quantiles_kpc'][rows,1])),
                    **row_stats(u,dens)))
                if mul==1:
                    h=np.histogram(np.clip(u[np.isfinite(u)],0,1),np.linspace(0,1,11))[0]
                    hist.extend(dict(reference=ref,group=label,bin_lo=a,bin_hi=b,n=int(n))
                        for a,b,n in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:],h))
            for eps in P['predictive']['distance_scale_fraction']:
                u,dens=gaussian_mixture(edges[rows],mass[rows],y,error,ref,window,5*np.log10(1+eps))
                allrows.append(dict(reference=ref,group=label,error_multiplier=1,
                     scale_fraction=eps,negative=int((y<0).sum()) if ref=='gaia' else 0,
                     **row_stats(u,dens)))
            center=10+5*np.log10(t['qrf_quantiles_kpc'][rows,2])
            for width in P['predictive']['error_multipliers']:
                scaled=center[:,None]+width*(edges[rows]-center[:,None])
                u,dens=gaussian_mixture(scaled,mass[rows],y,error,ref,window)
                allrows.append(dict(reference=ref,group=label,error_multiplier=1,scale_fraction=0,
                   pdf_width_multiplier=width,median_pdf_width_mu=width*np.median(
                       5*np.log10(t['qrf_quantiles_kpc'][rows,3]/t['qrf_quantiles_kpc'][rows,1])),
                   **row_stats(u,dens)))
            u0,f0,z0=gaussian_mixture(edges[rows],mass[rows],y,error,ref,window,return_selection=True)
            u1,f1,z1=gaussian_mixture(edges[rows],mass[rows],y,error,ref,window,2.,return_selection=True)
            for fraction in P['predictive']['wrong_branch_mass']:
                normalizer=(1-fraction)*z0+fraction*z1
                u=((1-fraction)*z0*np.nan_to_num(u0)+fraction*z1*np.nan_to_num(u1))/normalizer
                dens=((1-fraction)*z0*np.nan_to_num(f0)+fraction*z1*np.nan_to_num(f1))/normalizer
                allrows.append(dict(reference=ref,group=label,error_multiplier=1,scale_fraction=0,
                   added_far_branch_mass=fraction,**row_stats(u,dens)))
            if ref!='gaia':
                sigpred=2.5*np.log10(t['qrf_quantiles_kpc'][rows,3]/t['qrf_quantiles_kpc'][rows,1])
                dm=5*np.log10(t['qrf_quantiles_kpc'][rows,2]/t[ref+'_kpc'][rows])
                for rho in P['predictive']['reference_cross_error_correlation']:
                    variance=np.maximum(sigpred**2+error**2-2*rho*sigpred*error,1e-8)
                    sensitivity.append(dict(reference=ref,group=label,rho=rho,n=len(rows),
                        gaussian_joint_approx_log_score=np.mean(norm.logpdf(dm,scale=np.sqrt(variance))),
                        variance_mean=np.mean(variance)))
            # Conditional forward simulations evaluate the finite-sample power
            # against a fixed, known predictive distribution.
            if label=='all':
                rr=rows[:256];ee=error[:256]; ww=None if window is None else window[:256]
                mids=(edges[rr,:-1]+edges[rr,1:])/2
                half=np.diff(edges[rr],axis=1)/2
                nodes,quadrature=np.polynomial.legendre.leggauss(3)
                components=(mids[:,:,None]+half[:,:,None]*nodes).reshape(len(rr),-1)
                component_mass=(mass[rr,:,None]*(quadrature/2)).reshape(len(rr),-1)
                for eps in P['predictive']['distance_scale_fraction']:
                    for branch in (0.,*P['predictive']['wrong_branch_mass']):
                        rejections=[];negative=[]
                        for draw in range(P['predictive']['power_replications']):
                            shifted=components+5*np.log10(1+eps)
                            wrong=RNG.random(len(rr))<branch
                            shifted=shifted+wrong[:,None]*2.0
                            center=10**((10-shifted)/5) if ref=='gaia' else shifted
                            lower=np.zeros_like(center) if ww is None else ndtr(ww[:,0,None]-center/ee[:,None])
                            upper=np.ones_like(center) if ww is None else ndtr(ww[:,1,None]-center/ee[:,None])
                            weights=component_mass*(upper-lower)
                            total=weights.sum(axis=1)
                            valid=total>1e-12
                            if not valid.any():continue
                            weights=np.divide(weights,total[:,None],out=np.zeros_like(weights),where=total[:,None]>0)
                            c=np.cumsum(weights,axis=1)
                            jj=(c>=RNG.random(len(rr))[:,None]).argmax(axis=1)
                            kk=np.arange(len(rr))
                            prob=lower[kk,jj]+RNG.random(len(rr))*(upper[kk,jj]-lower[kk,jj])
                            simulated=center[kk,jj]+ee*ndtri(np.clip(prob,1e-12,1-1e-12))
                            u,_=gaussian_mixture(edges[rr][valid],mass[rr][valid],simulated[valid],ee[valid],
                                     ref,None if ww is None else ww[valid])
                            v=u[np.isfinite(u)]
                            if len(v):
                                rejections.append(kstest(v,'uniform').pvalue<.05)
                                negative.append(int((simulated[valid]<0).sum()))
                        power.append(dict(reference=ref,scale_fraction=eps,wrong_branch_mass=branch,
                            n=len(rr),replications=len(rejections),rejection_fraction=np.mean(rejections),
                            mean_negative=np.mean(negative)))
    csv_rows(d/'observation_space_metrics.csv',allrows)
    csv_rows(d/'observation_space_pit.csv',hist)
    csv_rows(d/'scale_branch_power.csv',power)
    csv_rows(d/'shared_error_sensitivity.csv',sensitivity)
    csv_rows(d/'sampling.csv',sampling)
    atomic_npz(d/'sampled_ids.npz',**sample_ids)
    with plt.rc_context(PAPER):
        f,axes=plt.subplots(1,3,figsize=(7.15,2.85),layout='constrained',sharey=True)
        for ax,ref in zip(axes,('gaia','specdis','spdist')):
            for group,col in (('ordinary',BLUE),('metal_poor',ORANGE),('intermediate_gravity',GREEN)):
                a=next((r for r in allrows if r['reference']==ref and r['group']==group and r['scale_fraction']==0 and r['error_multiplier']==1),None)
                if a is None:continue
                nominal=P['predictive']['nominal_coverage']
                ax.plot(nominal,[a[f'coverage_{int(100*c)}'] for c in nominal],'o-',color=col,ms=3,label=group.replace('_',' '))
            ax.plot([0,1],[0,1],ls=':',color='.5',lw=.7)
            ax.set(title=REFERENCE_NAMES[ref],xlabel='Nominal probability',xlim=(0,1),ylim=(0,1))
        axes[0].set_ylabel('Observation-space central fraction')
        axes[0].legend(fontsize=5.5,frameon=False,loc='upper left')
        figure(d,'measurement_space_calibration',f,allrows)
    complete(d,inputs,spectral_likelihood='Gaussian VAC-summary approximation, zero reference-prior removal; intrinsic SPDist included once',
             power_scope='conditional Gaussian forward simulation, random subset up to 256; finite-bin approximation tested')


if __name__=='__main__':
    main()
