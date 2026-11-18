#!/usr/bin/env python3
"""Locate foreground-peak Hess contributions and calibrate the available blank scans."""
from astropy.io import fits
from matplotlib.colors import SymLogNorm
from photdis.stage1 import require_compute
from halo_common import *
from sv18_common import DEST,index_join
from sv18_filter import hess,scan,recover
import paper_v2_filter as old
from v3_17_sample import DEST as GD,NEW,INPUT
from v3_16_gd1 import NSTEP,EDGES
from v3_17_hess import MODULUS_CURVE
from v3_11_layers import read_receipt


def main():
    require_compute();d=folder('hess');inputs={};arrays={};contributions=[];scans=[];injections=[]
    t=read_receipt(GD/'sample','complete.json','desi.npz',('TYPE','ls_id','r','r0','color','block','ribbon','phi1'),inputs)
    assert (t['TYPE']=='PSF').all()
    a=read_receipt(GD/'sample','complete.json','area.npz',('desi','desi_good'),inputs)
    response=read(DEST/'observable',('selection.npz','response','area'),inputs)
    templates=read(DEST/'filter',('template_support.npz',*['s%d_r%d_w%s_%s'%(si,r,w,k) for si in (1,2,5) for r in (20,21) for w in (.06,.09) for k in ('raw','selected','observability')]),inputs)
    curve=np.sort(fits.getdata(verified(INPUT/'v4_Figure2_track_jarvis.fits',MODULUS_CURVE),1),order='phi1')
    member=read(ROOT/'data/v3_18/members',('posteriors.npz','TYPE','ls_id','science'),inputs)
    member_ids=member['ls_id'][member['science']]
    segments=t['block']//NSTEP
    controls=np.flatnonzero(np.isin(np.round(NEW,2),(-5.8,-4.8,4.8,5.8)))
    blanks=np.flatnonzero(np.isin(np.round(NEW,2),(-7.8,-6.8,6.8,7.8)))
    shape=(len(old.COLOR)-1,len(old.APP)-1)
    for response_index,si in enumerate((1,2,5)):
        refmu=float(np.interp((EDGES[si]+EDGES[si+1])/2,curve['phi1'],curve['r-Mr']))
        for k,rlim in enumerate((20,21)):
            exp=a['desi'][k].reshape(len(EDGES)-1,NSTEP,len(NEW))[si].sum(axis=0)
            valid=a['desi_good'][k,si]
            geo=[i for i in controls if valid[i] and exp[i]>0]
            assert valid[0] and len(geo)>=2
            use=(segments==si)&(t['r']<rlim)&valid[t['ribbon']]
            resp=response['response'][k,response_index].reshape(len(NEW),-1)
            bgresponse=np.average(resp[geo],axis=0,weights=exp[geo])
            offraw=hess(t['color'][use&np.isin(t['ribbon'],geo)],t['r0'][use&np.isin(t['ribbon'],geo)])
            for width in (.06,.09):
                key=f's{si}_r{rlim}_w{width}'
                raw=templates[key+'_raw']
                for blank in (0,*blanks):
                    if not valid[blank] or exp[blank]<=0:continue
                    common=(resp[blank]>0)&(bgresponse>0)
                    alpha=exp[blank]/exp[geo].sum()*np.divide(resp[blank],bgresponse,out=np.zeros_like(bgresponse),where=common)
                    visible=raw*resp[blank]*common
                    fraction=visible.sum(axis=1)
                    signal=np.divide(visible,fraction[:,None],out=np.zeros_like(visible),where=fraction[:,None]>0)
                    onraw=hess(t['color'][use&(t['ribbon']==blank)],t['r0'][use&(t['ribbon']==blank)])
                    on=onraw*common;off=offraw*common
                    z,num,den,w=scan(signal,on,off,alpha);peak,low,high=recover(z,w)
                    scans.append(dict(segment=si,r_limit=rlim,width=width,blank=int(blank),mu_peak=old.MU[peak],
                        reference_mu=refmu,max_statistic=z[peak],profile_lo=low,profile_hi=high,on_count=on.sum(),
                        on_area=exp[blank],off_area=exp[geo].sum(),off_ribbons=' '.join(map(str,geo))))
                    if blank==0:
                        variance=np.maximum(on+alpha**2*off,4.)
                        mh=hess(t['color'][use&(t['ribbon']==0)&np.isin(t['ls_id'],member_ids)],
                                t['r0'][use&(t['ribbon']==0)&np.isin(t['ls_id'],member_ids)])*common
                        for label,node in (('peak',peak),('reference',int(np.argmin(abs(old.MU-refmu))))):
                            numerator=signal[node]*(on-alpha*off)/variance
                            denominator=signal[node]**2/variance
                            tag=key+'_'+label
                            arrays[tag+'_on']=on.reshape(shape);arrays[tag+'_off']=off.reshape(shape)
                            arrays[tag+'_alpha']=alpha.reshape(shape);arrays[tag+'_template']=signal[node].reshape(shape)
                            arrays[tag+'_visible']=resp[0].reshape(shape)
                            arrays[tag+'_numerator']=numerator.reshape(shape);arrays[tag+'_denominator']=denominator.reshape(shape)
                            order=np.argsort(-abs(numerator));totalabs=abs(numerator).sum()
                            for rank,j in enumerate(order[:30]):
                                ci,ri=np.unravel_index(j,shape)
                                contributions.append(dict(segment=si,r_limit=rlim,width=width,node=label,mu=old.MU[node],rank=rank+1,
                                   color_lo=old.COLOR[ci],r0_lo=old.APP[ri],on=on[j],off=off[j],alpha=alpha[j],
                                   template=signal[node,j],numerator=numerator[j],denominator=denominator[j],
                                   absolute_numerator_fraction=abs(numerator[j])/totalabs if totalabs else 0,
                                   cumulative_absolute_fraction=abs(numerator[order[:rank+1]]).sum()/totalabs if totalabs else 0))
                            mz,_,_,_=scan(signal,mh,np.zeros_like(mh),0.)
                            scans.append(dict(segment=si,r_limit=rlim,width=width,blank=-1,mu_peak=old.MU[int(np.nanargmax(mz))],
                                reference_mu=refmu,max_statistic=float(np.nanmax(mz)),profile_lo=None,profile_hi=None,
                                on_count=mh.sum(),on_area=exp[0],off_area=0,off_ribbons='confirmed members only'))
                        # Scalar template renormalization must leave every finite Z unchanged.
                        zz,_,_,_=scan(signal*7.3,on,off,alpha)
                        np.testing.assert_allclose(zz,z,atol=1e-12,equal_nan=True)
                    else:
                        true=int(np.argmin(abs(old.MU-refmu)))
                        prob=visible[true];prob=np.r_[prob,max(0.,1-prob.sum())];prob/=prob.sum()
                        for count in (100,300):
                            added=RNG.multinomial(count,prob,size=128)[:,:-1]
                            yy=RNG.poisson(on,size=(128,len(on)))+added
                            oo=RNG.poisson(off,size=(128,len(off)))
                            variance=np.maximum(yy+alpha**2*oo,4.)
                            num=signal@((yy-alpha*oo)/variance).T
                            den=signal**2@(1/variance).T
                            allz=np.divide(num,np.sqrt(den),out=np.full_like(num,np.nan),where=den>0)
                            for draw in range(128):
                                z1=allz[:,draw];at=int(np.nanargmax(z1));err=10**((old.MU[at]-refmu)/5)-1
                                injections.append(dict(segment=si,r_limit=rlim,width=width,blank=int(blank),generated=count,
                                    draw=draw,admitted=int(added[draw].sum()),recovered_mu=old.MU[at],reference_mu=refmu,
                                    fractional_error=err,within_tolerance=abs(err)<=.1,z_max=z1[at],
                                    generation='same projected template within archived apparent-CMD window'))
    csv_rows(d/'local_scans.csv',scans);csv_rows(d/'dominant_hess_cells.csv',contributions)
    csv_rows(d/'same_template_injections.csv',injections)
    # Every physical blank target is counted once after the exact complete local scan.
    maxima=[]
    for si in (1,2,5):
        for b in (0,*blanks):
            vals=[r['max_statistic'] for r in scans if r['segment']==si and r['blank']==b]
            if vals:maxima.append(dict(segment=si,blank=int(b),max_statistic=max(vals),search='89 mu x 2 widths x 2 nested depths'))
    calibration=[]
    for si in (1,2,5):
        target=next(r['max_statistic'] for r in maxima if r['segment']==si and r['blank']==0)
        null=[r['max_statistic'] for r in maxima if r['segment']==si and r['blank']>0]
        calibration.append(dict(segment=si,target_max=target,independent_target_regions=len(null),
            exceedances=int(np.sum(np.array(null)>=target)),finite_sample_tail=(1+np.sum(np.array(null)>=target))/(1+len(null)),
            minimum_resolvable_tail=1/(1+len(null)),scope='disjoint available target ribbons; same local off field reused'))
    csv_rows(d/'blank_maxima.csv',maxima);csv_rows(d/'blank_tail_calibration.csv',calibration)
    summary=[]
    for n in (100,300):
        a=[r for r in injections if r['generated']==n]
        summary.append(dict(generated=n,trials=len(a),admitted_median=np.median([r['admitted'] for r in a]),
            within_ten_percent=np.mean([r['within_tolerance'] for r in a]),median_error=np.median([r['fractional_error'] for r in a])))
    csv_rows(d/'same_template_summary.csv',summary)
    atomic_npz(d/'hess_contributions.npz',color=old.COLOR,apparent=old.APP,**arrays)
    with plt.rc_context(PAPER):
        for si in (1,2,5):
            f,axes=plt.subplots(4,2,figsize=(7.15,9.8),layout='constrained')
            key=f's{si}_r21_w0.06'
            stats=[]
            for row,stat in enumerate(('on','template','numerator','denominator')):
                pair=[arrays[key+'_'+label+'_'+stat] for label in ('peak','reference')]
                cap=max(float(np.max(abs(np.array(pair)))),1e-8)
                for col,(label,arr) in enumerate(zip(('peak','reference'),pair)):
                    ax=axes[row,col];signed=stat=='numerator'
                    z=ax.pcolormesh(old.COLOR,old.APP,arr.T,cmap='RdBu_r' if signed else 'viridis',
                        norm=SymLogNorm(max(cap/100,1e-10),vmin=-cap if signed else 0,vmax=cap),rasterized=True)
                    ax.set(xlabel=r'$(g-r)_0$',ylabel=r'$r_0$ (mag)',ylim=(23,12),title=label.capitalize()+' · '+stat)
                    f.colorbar(z,ax=ax,fraction=.035,extend='both' if signed else 'max',label='Counts / bin' if stat=='on' else 'Contribution / bin')
                    stats.append(dict(segment=si,node=label,statistic=stat,total=arr.sum(),maximum=cap))
            figure(d,f'foreground_hess_segment_{si}',f,stats)
    complete(d,inputs,template_scaling_test='Z invariant to factor7.3 at every finite node',
        trials='fixed mu per-selected-field contrast; local peak scan89*2*2; blank targets4 per segment; no full-sky search calibration',
        blank_dependence='target ribbons disjoint, shared local off fields and nested r limits; no independent6144-field claim',
        intervals='nonnegative fixed-variance Gaussian amplitude profile interval, not calibrated distance credible interval',
        injection_comparison='new same-template conditional-count control; old M13 empirical injections retained in science_validation_20261118/filter with their different source distribution/admission')


if __name__=='__main__':main()
