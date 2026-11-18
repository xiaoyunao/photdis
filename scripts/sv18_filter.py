#!/usr/bin/env python3
"""Selection-matched native CMD scans and empirical injections in real GD-1 fields."""
from astropy.io import fits
from scipy.special import logsumexp
from scipy.spatial import cKDTree
from photdis.stage1 import require_compute
from sv18_common import *
import paper_v2_filter as old
from v3_11_layers import read_receipt
from v3_17_sample import DEST as GD, NEW, INPUT
from v3_16_gd1 import NSTEP, EDGES
from v3_17_hess import MODULUS_CURVE

F=P['filter']
C=(old.COLOR[1:]+old.COLOR[:-1])/2
A=(old.APP[1:]+old.APP[:-1])/2


def observable(extinction,rlim):
    """Integrate observed-r admission over actual attenuation and CMD bin width."""
    shifts=np.quantile(extinction,(np.arange(101)+.5)/101)
    frac=np.clip((rlim-old.APP[:-1,None]-shifts[None,:])/np.diff(old.APP)[:,None],0,1).mean(axis=1)
    # The archived target and control sample is already PSF, mask, optical-error,
    # local grz-depth and FRACFLUX qualified. Its raw r0/color limits still apply.
    frac*=np.clip((old.APP[1:]-14)/np.diff(old.APP),0,1)
    color=np.clip((old.COLOR[1:]+.12)/np.diff(old.COLOR),0,1)
    return (color[:,None]*frac[None,:]).ravel()


def scan(signal,on,off,alpha):
    contrast=on-alpha*off
    variance=np.maximum(on+alpha*alpha*off,4.)
    numerator=signal@(contrast/variance)
    denominator=(signal*signal)@(1/variance)
    z=np.divide(numerator,np.sqrt(denominator),out=np.full(len(signal),np.nan),where=denominator>0)
    # Gaussian-contrast amplitude profile with nonnegative population amplitude.
    profile=.5*np.maximum(z,0)**2
    weight=np.exp(profile-logsumexp(profile))
    return z,numerator,denominator,weight


def recover(z,w):
    idx=int(np.nanargmax(z));cdf=np.cumsum(w)
    step=np.median(np.diff(old.MU))
    edges=np.r_[old.MU-step/2,old.MU[-1]+step/2]
    low,high=np.interp([.16,.84],np.r_[0,cdf],edges)
    return idx,low,high


def hess(color,mag):
    return np.histogram2d(color,mag,(old.COLOR,old.APP))[0].ravel()


def main():
    require_compute()
    inputs={};d=out('filter')
    t=read_receipt(GD/'sample','complete.json','desi.npz',
       ('TYPE','ls_id','r','r0','color','block','ribbon','tier','qrf','phi1'),inputs)
    area=read_receipt(GD/'sample','complete.json','area.npz',('desi','desi_good'),inputs)
    assert (t['TYPE']=='PSF').all()
    response=read(DEST/'observable',('selection.npz','response','area'),inputs)
    h,template_meta=old.template('M92',inputs)
    curve=np.sort(fits.getdata(verified(INPUT/'v4_Figure2_track_jarvis.fits',MODULUS_CURVE),1),order='phi1')
    inputs['distance_curve']=MODULUS_CURVE
    # Empirical confirmed M13 members provide a native independent sequence.
    seq=read_receipt(old.AUDIT/'M13','complete.json','native_join.npz',
      ('TYPE','g0','r0','z0','r','member','low_crowding','training_or_calibration','error_grz'),inputs)
    good=seq['member']&seq['low_crowding']&~seq['training_or_calibration']&(seq['r']<22)
    absolute=np.column_stack([seq[k][good] for k in ('g0','r0','z0')])-10-5*np.log10(old.SYSTEMS['M13'][1])
    seqerr=seq['error_grz'][good]
    assert len(absolute)>5 and (seq['TYPE'][good]=='PSF').all()
    cache=cached(inputs,('TYPE','ls_id','err','m','dep','state'))
    j=index_join(t['ls_id'],cache['ls_id'])
    real_errors=cache['err'][j[j>=0]]
    real_mags=cache['m'][j[j>=0]]
    del cache
    good=np.isfinite(real_errors).all(axis=1)&np.isfinite(real_mags).all(axis=1)&(real_errors>0).all(axis=1)
    real_errors,real_mags=real_errors[good],real_mags[good]
    features=np.column_stack((real_mags[:,1]/.5,(real_mags[:,0]-real_mags[:,1])/.15,
                              (real_mags[:,1]-real_mags[:,2])/.15))
    error_neighbors=cKDTree(features)
    rows=[];recoveries=[];selfchecks=[];arrays={};area_rows=[]
    segment=t['block']//NSTEP
    controls=np.flatnonzero(np.isin(np.round(NEW,2),(-5.8,-4.8,4.8,5.8)))
    blanks=np.flatnonzero(np.isin(np.round(NEW,2),(-7.8,-6.8,6.8,7.8)))
    tests=(0,*blanks)
    for response_index,si in enumerate(F['segment_indices']):
        ref_mu=float(np.interp((EDGES[si]+EDGES[si+1])/2,curve['phi1'],curve['r-Mr']))
        refdist=10**((ref_mu-10)/5)
        for k,rlim in enumerate(F['r_limits']):
            exp=area['desi'][k].reshape(len(EDGES)-1,NSTEP,len(NEW))[si].sum(axis=0)
            np.testing.assert_allclose(response['area'][k,response_index],exp,rtol=0,atol=1e-8)
            valid=area['desi_good'][k,si]
            use=(segment==si)&(t['r']<rlim)&valid[t['ribbon']]
            geo=[i for i in controls if valid[i] and exp[i]>0]
            if not valid[0] or len(geo)<2:
                area_rows.append(dict(segment=si,r_limit=rlim,usable=False,area_deg2=exp[0]))
                continue
            ar=t['r'][use]-t['r0'][use]
            response_by_field=response['response'][k,response_index].reshape(len(NEW),-1)
            background_response=np.average(response_by_field[geo],axis=0,weights=exp[geo])
            selected=response_by_field[0]
            common=(selected>0)&(background_response>0)
            selected=selected*common
            background=hess(t['color'][use&np.isin(t['ribbon'],geo)],t['r0'][use&np.isin(t['ribbon'],geo)])
            on=hess(t['color'][use&(t['ribbon']==0)],t['r0'][use&(t['ribbon']==0)])
            raw_on,raw_background=on.copy(),background.copy()
            ratio=np.divide(selected,background_response,out=np.zeros_like(selected),where=common)
            alpha=(exp[0]/exp[geo].sum())*ratio
            on*=common
            background*=common
            area_rows.append(dict(segment=si,r_limit=rlim,usable=True,area_deg2=exp[0],
                background_area=exp[geo].sum(),ar_min=ar.min(),ar_max=ar.max(),
                raw_target=int(on.sum()),raw_background=int(background.sum())))
            for width in F['widths']:
                raw=old.projected(h,width)
                vis=raw*selected
                fractions=vis.sum(axis=1)
                sig=np.divide(vis,fractions[:,None],out=np.zeros_like(vis),where=fractions[:,None]>0)
                key=f's{si}_r{rlim}_w{width}'
                arrays[key+'_raw']=raw
                arrays[key+'_selected']=sig
                arrays[key+'_observability']=selected
                for name,signal in (('legacy',raw),('selected',sig)):
                    z,num,den,w=(scan(signal,raw_on,raw_background,exp[0]/exp[geo].sum())
                                 if name=='legacy' else scan(signal,on,background,alpha))
                    peak,clo,chi=recover(z,w)
                    arrays[key+'_'+name+'_z']=z
                    for m in range(len(old.MU)):
                        rows.append(dict(segment=si,r_limit=rlim,width=width,method=name,
                            mu=old.MU[m],reference_mu=ref_mu,template_mass=raw[m].sum(),
                            selected_mass=fractions[m],numerator=num[m],denominator=den[m],
                            invisible_denominator=np.sum(raw[m]**2*(1-selected)/np.maximum(on+alpha*alpha*background,4)),
                            z=z[m],peak_mu=old.MU[peak],profile_lo=clo,profile_hi=chi,
                            selected_normalization='conditional selected count'))
                # Test response mismatch alone on the expected admitted template.
                true=int(np.argmin(abs(old.MU-ref_mu)))
                for method,signal in (('legacy',raw),('selected',sig)):
                    z,_,_,w=scan(signal,1000*sig[true],np.zeros_like(background),0)
                    peak,clo,chi=recover(z,w)
                    selfchecks.append(dict(segment=si,r_limit=rlim,width=width,method=method,
                        true_mu=old.MU[true],recovered_mu=old.MU[peak],
                        passed=abs(old.MU[true]-old.MU[peak])<=F['self_recovery_mu_tolerance']))
                # Fixed neighboring real sky ribbons. Injections retain the observed
                # field counts and use fresh native band-noise realizations.
                for blank in blanks:
                    if not valid[blank] or exp[blank]<=0:
                        continue
                    bgon=hess(t['color'][use&(t['ribbon']==blank)],t['r0'][use&(t['ribbon']==blank)])
                    blank_response=response_by_field[blank]
                    common_blank=(blank_response>0)&(background_response>0)
                    al=(exp[blank]/exp[geo].sum())*np.divide(blank_response,background_response,
                          out=np.zeros_like(blank_response),where=common_blank)
                    bgon*=common_blank
                    off=raw_background*common_blank
                    blank_sig=raw*blank_response*common_blank
                    blank_sig=np.divide(blank_sig,blank_sig.sum(axis=1)[:,None],out=np.zeros_like(blank_sig),
                                        where=blank_sig.sum(axis=1)[:,None]>0)
                    blank_rows=use&(t['ribbon']==blank)
                    blank_ar=t['r'][blank_rows]-t['r0'][blank_rows]
                    r_window=observable(blank_ar,rlim)
                    depth_given_r=np.divide(blank_response,r_window,out=np.zeros_like(blank_response),where=r_window>0)
                    depth_given_r=np.clip(depth_given_r,0,1).reshape(len(C),len(A))
                    for count in (0,*F['injection_counts']):
                        for draw in range(F['injection_trials']):
                            added=np.zeros_like(bgon)
                            admitted=0
                            if count:
                                ii=RNG.integers(len(absolute),size=count)
                                mu=ref_mu
                                mag=absolute[ii]+mu
                                # Error triplets are sampled together from the same
                                # native sky, matched to apparent r at this distance.
                                features=np.column_stack((mag[:,1]/.5,(mag[:,0]-mag[:,1])/.15,
                                                           (mag[:,1]-mag[:,2])/.15))
                                _,neighbors=error_neighbors.query(features,k=np.arange(1,17))
                                pick=neighbors[np.arange(count),RNG.integers(16,size=count)]
                                err=real_errors[pick]
                                additional=np.sqrt(np.maximum(err**2-seqerr[ii]**2,0))
                                mag+=RNG.normal(size=(count,3))*additional
                                ex=RNG.choice(blank_ar,size=count)
                                take=(mag[:,1]+ex<rlim)&(np.maximum(err,seqerr[ii]).max(axis=1)<.1)
                                ci=np.searchsorted(old.COLOR,mag[:,0]-mag[:,1],side='right')-1
                                ri=np.searchsorted(old.APP,mag[:,1],side='right')-1
                                inside=(ci>=0)&(ci<len(C))&(ri>=0)&(ri<len(A))
                                probability=np.zeros(count)
                                probability[inside]=depth_given_r[ci[inside],ri[inside]]
                                take&=RNG.random(count)<probability
                                added=hess((mag[:,0]-mag[:,1])[take],mag[take,1])
                                added*=common_blank
                                admitted=int(added.sum())
                            realized_on=RNG.poisson(bgon)+added
                            realized_off=RNG.poisson(off)
                            z,_,_,w=scan(blank_sig,realized_on,realized_off,al)
                            peak,clo,chi=recover(z,w)
                            trials=len(old.MU)*len(F['widths'])*len(F['r_limits'])*len(F['segment_indices'])
                            recoveries.append(dict(segment=si,r_limit=rlim,width=width,blank=int(blank),
                                generated=count,admitted=admitted,draw=draw,reference_mu=ref_mu,
                                recovered_mu=old.MU[peak],profile_lo=clo,profile_hi=chi,
                                profile_c68=clo<=ref_mu<=chi,
                                fractional_error=10**((old.MU[peak]-ref_mu)/5)-1,
                                within_tolerance=abs(10**((old.MU[peak]-ref_mu)/5)-1)<=.1,
                                z_max=z[peak],family_p=min(1.,norm.sf(z[peak])*trials)))
    csv_rows(d/'distance_scans.csv',rows)
    csv_rows(d/'template_self_recovery.csv',selfchecks)
    csv_rows(d/'empirical_injections.csv',recoveries)
    csv_rows(d/'effective_areas.csv',area_rows)
    atomic_npz(d/'template_support.npz',mu=old.MU,color=old.COLOR,apparent=old.APP,**arrays)
    assert all(r['passed'] for r in selfchecks if r['method']=='selected')
    summary=[]
    for n in (0,*F['injection_counts']):
        a=[r for r in recoveries if r['generated']==n]
        if a:
            summary.append(dict(generated=n,trials=len(a),admitted_median=np.median([r['admitted'] for r in a]),
                 bias=np.median([r['fractional_error'] for r in a]),c68=np.mean([r['profile_c68'] for r in a]),
                 within_tolerance=np.mean([r['within_tolerance'] for r in a]),
                 false_detection_fraction=np.mean([r['family_p']<norm.sf(3) for r in a])))
    csv_rows(d/'injection_summary.csv',summary)
    with plt.rc_context(PAPER):
        f,axes=plt.subplots(1,3,figsize=(7.15,2.85),layout='constrained')
        key=next(k for k in arrays if k.endswith('_selected'))
        k=key.removesuffix('_selected')
        a=arrays[k+'_raw']; b=arrays[key]; near=np.argmin(abs(old.MU-14.65))
        im=axes[0].pcolormesh(old.COLOR,old.APP,b[near].reshape(len(C),len(A)).T,
                            cmap='viridis',rasterized=True)
        axes[0].set(xlabel=r'$(g-r)_0$',ylabel=r'$r_0$ (mag)',ylim=(23,12))
        f.colorbar(im,ax=axes[0],fraction=.035,label='Selected template mass / bin')
        for name,color in (('legacy',BLUE),('selected',ORANGE)):
            axes[1].plot(old.D,arrays[k+'_'+name+'_z'],color=color,label=name.capitalize())
        axes[1].set(xlabel='Trial distance (kpc)',ylabel='CMD contrast Z',xscale='log')
        axes[1].legend(fontsize=6,frameon=False)
        for n,color in zip(F['injection_counts'],(BLUE,ORANGE,GREEN)):
            v=[r['fractional_error'] for r in recoveries if r['generated']==n]
            axes[2].hist(v,bins=np.linspace(-1,1.2,32),histtype='step',density=True,color=color,label=str(n))
        axes[2].axvline(0,color='.6',lw=.7,ls=':')
        axes[2].set(xlabel='Fractional distance residual',ylabel='Density')
        axes[2].legend(fontsize=6,frameon=False,title='Injected')
        figure(d,'filter_selection_recovery',f,summary)
    pass_injections=bool(summary) and all(abs(r['bias'])<=.1 and r['c68']>=.55 and r['within_tolerance']>=.68
               for r in summary if r['generated']>=100)
    null=[r for r in summary if r['generated']==0]
    pass_null=bool(null) and null[0]['false_detection_fraction']<=.05
    peaks=[r for r in rows if r['method']=='selected' and r['mu']==r['peak_mu']]
    pass_structure=bool(peaks) and all(abs(10**((r['peak_mu']-r['reference_mu'])/5)-1)<=.1 for r in peaks)
    passed=pass_injections and pass_null and pass_structure
    complete(d,inputs,template=template_meta,injection_sequence='native M13 members',
             pass_injections=bool(pass_injections),pass_null=bool(pass_null),pass_structure=bool(pass_structure),
             all_sky_search_authorized_by_evidence=bool(passed),
             interval_scope='nonnegative Gaussian contrast amplitude profile; empirical coverage measured')


if __name__=='__main__':
    main()
