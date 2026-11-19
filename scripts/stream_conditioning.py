#!/usr/bin/env python3
"""One fixed population and statistic: vary the distance conditioning alone."""
from concurrent.futures import ThreadPoolExecutor
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
import healpy as hp
from photdis.stage1 import require_compute
from stream_common import *
from halo_common import BLUE,ORANGE,GREEN
from halo_balance import moments,contrast
from halo_balance_compare import median
from sv18_common import cached,index_join,interval


def propensity(x,y,mult):
    use=mult>0
    if len(np.unique(y[use]))<2:return np.full(len(y),np.nan)
    mean=np.average(x[use],axis=0,weights=mult[use]);sd=np.sqrt(np.average((x[use]-mean)**2,axis=0,weights=mult[use]))
    z=np.clip((x-mean)/np.maximum(sd,1e-5),-8,8);a=np.column_stack((z,z*z,z*z*z,z[:,1]*z[:,2]))
    center=np.average(a[use],axis=0,weights=mult[use]);scale=np.sqrt(np.average((a[use]-center)**2,axis=0,weights=mult[use]));a=(a-center)/np.maximum(scale,1e-5)
    fit=LogisticRegression(C=10,max_iter=300,tol=1e-8,solver='newton-cholesky')
    fit.fit(a[use],y[use],sample_weight=mult[use]);assert fit.n_iter_[0]<300
    return expit(fit.decision_function(a))


def ess(w):return w.sum()**2/(w@w) if np.any(w) else 0.


def main():
    require_compute();d=folder('conditioning');inputs={}
    t=cached(inputs,('TYPE','ls_id','source_id','heldout','state','pop','physical','region','m','err','observed_r','EBV','q','RA','DEC',
        'gaia_stage','gaia_state','gaia_d','gaia_lo','gaia_hi','p_corr','p_err','specdis_stage','specdis_d','specdis_sig','spdist_stage','spdist_d','spdist_sig','dep'))
    roles=read(OLD/'roles',('reference_roles.npz','TYPE','ls_id','direct_fit_or_calibration'),inputs)
    np.testing.assert_array_equal(t['ls_id'],roles['ls_id']);p=t['physical']
    base=t['heldout']&(t['state']==3)&np.isin(t['pop'],(1,2))&~roles['direct_fit_or_calibration']
    raw_n=int(base.sum());base&=(p[:,2]>=4.1)&(p[:,2]<=4.8)&(p[:,4]>=4000)&(p[:,4]<=6500)
    gr=t['m'][:,0]-t['m'][:,1];rz=t['m'][:,1]-t['m'][:,2]
    base&=(gr>=.3)&(gr<=.8)&(rz>=.05)&(rz<=.65)&(t['observed_r']>=16)&(t['observed_r']<20)
    t={k:v[base] for k,v in t.items()};del roles
    assert (t['TYPE']=='PSF').all();rows=[];audits=[];support=[];private=[];drawtable=[]
    block=hp.ang2pix(16,t['RA'],t['DEC'],nest=True,lonlat=True)
    for ref in ('specdis','spdist','gaia'):
        for lo,hi in ((2,5),(5,10)):
            distance=t[ref+'_d'];sigma=(t['gaia_hi']-t['gaia_lo'])/(2*distance) if ref=='gaia' else t[ref+'_sig']*np.log(10)/5
            mr=t['m'][:,1]-10-5*np.log10(np.maximum(distance,1e-20))
            for domain in ('reference_range','common_luminosity_domain','magnitude_complete_rectangle'):
                use=(t[ref+'_stage']==3)&(distance>=lo)&(distance<hi)
                if domain!='reference_range':use&=(t['observed_r']<18)&(mr>=3)&(mr<=7)&(t['err'].max(axis=1)<=.05)&(sigma<=.15)
                if domain=='magnitude_complete_rectangle':
                    # A 1-mag luminosity slice within a 2-mag observed-r window:
                    # all Mr in [4.5,5.5], mu in [11.5,12.435] satisfy r16..18
                    # for 0<=Ar<=2.165*.03. This does not establish reference
                    # sampling completeness; its additional selection is explicit.
                    mu=10+5*np.log10(np.maximum(distance,1e-20))
                    use&=(mr>=4.5)&(mr<=5.5)&(mu>=11.5)&(mu<=12.5-2.165*.03)&(t['EBV']<.03)&(t['EBV']>=0)
                    depth=22.5-2.5*np.log10(2.5/np.log(10)/.05/np.sqrt(np.maximum(t['dep'],1e-20)))
                    use&=(depth[:,:2]>=19.5).all(axis=1)&(depth[:,2]>=18.1)
                    if ref=='gaia':use&=t['p_err']<=.03
                ii=np.flatnonzero(use)
                if not len(ii):support.append(dict(reference=ref,lo=lo,hi=hi,domain=domain,parent=0,status='no rows'));continue
                y=(t['pop'][ii]==2).astype(int)
                if min(np.bincount(y,minlength=2))<5:
                    support.append(dict(reference=ref,lo=lo,hi=hi,domain=domain,parent=len(ii),ordinary=int((y==0).sum()),metal_poor=int(y.sum()),status='fewer than5 in one group'));continue
                x=np.column_stack((t['observed_r'][ii],t['m'][ii,0]-t['m'][ii,1],t['m'][ii,1]-t['m'][ii,2],t['err'][ii].max(axis=1),t['EBV'][ii],distance[ii],t['region'][ii]==3))
                delta=5*np.log10(t['q'][ii,2]/distance[ii]);mg=t['m'][ii,0]-10-5*np.log10(distance[ii])
                finite=np.isfinite(x).all(axis=1)&np.isfinite(delta);ii=ii[finite];y=y[finite];x=x[finite];delta=delta[finite];mg=mg[finite]
                designs=[x[:,[0,1,2,3,4,6]],x];probs=[propensity(a,y,np.ones(len(y))) for a in designs]
                individual=[(p>=.05)&(p<=.95) for p in probs];common=individual[0]&individual[1]
                support.append(dict(reference=ref,lo=lo,hi=hi,domain=domain,parent=len(y),without=int(individual[0].sum()),with_distance=int(individual[1].sum()),intersection=int(common.sum()),ordinary=int((common&(y==0)).sum()),metal_poor=int((common&(y==1)).sum()),status='fixed intersection'))
                if min(np.bincount(y[common],minlength=2))<5:continue
                keys,inv=np.unique(block[ii],return_inverse=True)
                def bootstrap(draw):
                    rng=np.random.default_rng(C['seed']+draw*1009+lo);mult=np.bincount(rng.integers(len(keys),size=len(keys)),minlength=len(keys))[inv]
                    result=[]
                    for a in designs:
                        p=propensity(a,y,mult);w=mult*np.where(y==1,1-p,p)*common
                        result.append([contrast(delta,y,w),contrast(mg,y,w),median(delta[y==1],w[y==1])-median(delta[y==0],w[y==0])])
                    return result
                with ThreadPoolExecutor(8) as pool:draws=np.array(list(pool.map(bootstrap,range(C['conditioning']['sky_bootstrap_draws']))))
                for k,(label,p) in enumerate(zip(('without_distance','with_distance'),probs)):
                    w=np.where(y==1,1-p,p)*common
                    for value,outcome in ((delta,'modulus_residual'),(mg,'reference_Mg')):
                        for statistic in ('mean','median') if outcome=='modulus_residual' else ('mean',):
                            val=contrast(value,y,w) if statistic=='mean' else median(value[y==1],w[y==1])-median(value[y==0],w[y==0])
                            column=0 if outcome=='modulus_residual' and statistic=='mean' else 2 if statistic=='median' else 1
                            rows.append(dict(reference=ref,lo=lo,hi=hi,domain=domain,weight_design=label,outcome=outcome,statistic=statistic,
                                contrast=float(val),n=int(common.sum()),ordinary_ESS=ess(w[y==0]),metal_poor_ESS=ess(w[y==1]),**interval(draws[:,k,column])))
                    audits.extend(dict(reference=ref,lo=lo,domain=domain,weight_design=label,**r) for r in moments(x,y,w))
                    private.append(dict(TYPE=t['TYPE'][ii],ls_id=t['ls_id'][ii],reference=np.full(len(ii),ref),lo=np.full(len(ii),lo),domain=np.full(len(ii),domain),weight_design=np.full(len(ii),label),common=common,weight=w))
                for draw,a in enumerate(draws):drawtable.append(dict(reference=ref,lo=lo,domain=domain,draw=draw,without_delta=a[0,0],with_delta=a[1,0],paired_change=a[1,0]-a[0,0]))
                print(ref,lo,domain,len(y),int(common.sum()),flush=True)
    csv_rows(d/'estimands.csv',rows);csv_rows(d/'balance.csv',audits);csv_rows(d/'support.csv',support);csv_rows(d/'sky_draws.csv',drawtable)
    npz(d/'weighted_ids.npz',**{k:np.concatenate([p[k] for p in private]) for k in private[0]})
    signed_anchor(t,d,inputs)
    plot_results(d,rows)
    complete(d,inputs,initial_role_excluded_rows=raw_n,independent_phase_color_magnitude_rows=len(t['ls_id']),
        estimand='metal-poor minus ordinary weighted marginal contrast on a single fixed propensity-overlap intersection; identical IDs in both designs and every sky resample',
        selection='common luminosity domain remains reference-selected and Mr-truncated; stricter rectangle is complete only with respect to stated r window and formal photometric depth, not spectral targeting, Gaia selection, or metallicity completeness',
        interpretation='descriptive sensitivity; no causal physical FeH coefficient. Spectroscopic FeH and spectroscopic distances share teachers. Gaia anchor normalizes its signed-parallax window; cluster absolute anchor is in sequence stage.')


def plot_results(d,rows):
    with plt.rc_context(PAPER):
        f,axs=plt.subplots(1,3,figsize=(7.15,3.0),layout='constrained')
        for ax,(domain,outcome) in zip(axs,(('reference_range','modulus_residual'),('common_luminosity_domain','reference_Mg'),('magnitude_complete_rectangle','reference_Mg'))):
            labels=[]
            for i,(ref,lo) in enumerate((('specdis',2),('spdist',2),('gaia',2),('specdis',5),('spdist',5),('gaia',5))):
                labels.append(ref.upper() if ref=='gaia' else ref.capitalize())
                for k,(design,col) in enumerate(zip(('without_distance','with_distance'),(BLUE,ORANGE))):
                    a=[r for r in rows if r['reference']==ref and r['lo']==lo and r['domain']==domain and r['outcome']==outcome and r['weight_design']==design and r['statistic']=='mean']
                    if a:
                        r=a[0];ax.errorbar(i+(k-.5)*.2,r['contrast'],yerr=[[max(0,r['contrast']-r['ci_lo'])],[max(0,r['ci_hi']-r['contrast'])]],fmt='o',ms=3,color=col,lw=.8,label=design.replace('_',' ') if i==0 else None)
            ax.axhline(0,color='.4',ls=':',lw=.6);ax.set(xticks=range(6),xticklabels=labels,ylabel='Metal-poor minus ordinary (mag)',title={'reference_range':'Same population · residual','common_luminosity_domain':'Common luminosity domain','magnitude_complete_rectangle':'Magnitude-complete rectangle'}[domain]);ax.tick_params(axis='x',rotation=55,labelsize=6)
        axs[0].legend(frameon=False,fontsize=6)
        figure(d,'metallicity_conditioning',f,rows)


def signed_anchor(t,d,inputs):
    from paper_v1_benchmark import read_scan
    from sv18_predictive import gaussian_mixture,row_stats
    _,a,_=read_scan(inputs);j=index_join(a['ls_id'],t['ls_id']);use=j>=0
    a={k:v[use] for k,v in a.items()};z={k:v[j[use]] for k,v in t.items()}
    rows=[]
    for pop,name in ((1,'ordinary'),(2,'metal_poor')):
        good=np.isin(z['gaia_state'],(4,5,6))&(z['pop']==pop)&np.isfinite(z['p_corr'])&(z['p_err']>0)
        if not good.any():continue
        state=z['gaia_state'][good];window=np.column_stack((np.where(state==4,-np.inf,np.where(state==5,5,10)),np.where(state==4,5,np.where(state==5,10,np.inf))))
        for shift in (-.2,0,.2):
            u,density=gaussian_mixture(a['qrf_mu_edges'][good],a['qrf_mu_mass'][good],z['p_corr'][good],z['p_err'][good],'gaia',window,shift)
            rows.append(dict(population=name,shift_mu=shift,negative=int((z['p_corr'][good]<0).sum()),**row_stats(u,density)))
    csv_rows(d/'signed_gaia_anchor.csv',rows)


if __name__=='__main__':main()
