#!/usr/bin/env python3
"""Same-star photometry/uncertainty counterfactuals with both forests frozen."""
from photdis.stage1 import require_compute
from photdis.v3.data import stable_rank
from photdis.v3_3 import frozen_qrf
from halo_common import *
from halo_noise import models


def main():
    require_compute();d=folder('seam');inputs={}
    t={side:read(OUT/'native',(side+'_pairs.npz','TYPE','ls_id','m','err','r','q','state'),inputs) for side in ('south','north')}
    s,n=t['south'],t['north'];gr=s['m'][:,0]-s['m'][:,1]
    good=np.isfinite(s['m']).all(axis=1)&np.isfinite(n['m']).all(axis=1)
    good&=(s['err']>0).all(axis=1)&(n['err']>0).all(axis=1)&(s['err']<.1).all(axis=1)&(n['err']<.1).all(axis=1)
    rank=stable_rank(s['ls_id'],C['seed']+72);chosen=[]
    for lo,hi in ((14,18),(18,20),(20,21),(21,22),(22,23)):
        for a,b in ((-.2,.3),(.3,.6),(.6,1),(1,1.6)):
            ii=np.flatnonzero(good&(s['r']>=lo)&(s['r']<hi)&(gr>=a)&(gr<b))
            chosen.extend(ii[np.argsort(rank[ii])[:64]])
    ii=np.sort(np.asarray(chosen,int));assert len(np.unique(ii))==len(ii)
    predictions={};support={};offsets={}
    common_error=np.maximum(s['err'][ii],n['err'][ii])
    for modelside,code in (('south',0),('north',1)):
        forest,offset=models(bool(code),inputs);offsets[modelside]=offset
        for photoside in ('south','north'):
            for errorcase in ('native','common'):
                error=t[photoside]['err'][ii] if errorcase=='native' else common_error
                p=frozen_qrf(forest,t[photoside]['m'][ii],error,s['ls_id'][ii])
                key=modelside+'_'+photoside+'_'+errorcase
                predictions[key]=p['distance']*10**(offset/5);support[key]=p['supported']
    common=np.logical_and.reduce(list(support.values()))
    rows=[];decomp=[]
    for lo,hi in ((14,18),(18,20),(20,21),(21,22),(22,23)):
        for a,b in ((-.2,.3),(.3,.6),(.6,1),(1,1.6)):
            use=(s['r'][ii]>=lo)&(s['r'][ii]<hi)&(gr[ii]>=a)&(gr[ii]<b)
            jj=use&common
            for key,q in predictions.items():
                delta=5*np.log10(q[jj,2]/predictions['south_south_native'][jj,2])
                rows.append(dict(r_lo=lo,r_hi=hi,gr_lo=a,gr_hi=b,view=key,parent=int(use.sum()),common=int(jj.sum()),
                    support=int((use&support[key]).sum()),median_delta_mu=np.median(delta) if len(delta) else None,
                    applied_mu_offset=offsets[key.split('_')[0]]))
    atomic_npz(d/'counterfactuals.npz',TYPE=s['TYPE'][ii],south_ls_id=s['ls_id'][ii],north_ls_id=n['ls_id'][ii],
        r=s['r'][ii],gr=gr[ii],common=common,**predictions,**{'support_'+k:v for k,v in support.items()})
    csv_rows(d/'frozen_seam_decomposition.csv',rows)
    with plt.rc_context(PAPER):
        f,axes=plt.subplots(1,2,figsize=(7.1,3.15),layout='constrained')
        bins=np.linspace(-1.5,1.5,80)
        for errorcase,ls in (('native','-'),('common',':')):
            for key,color,label in (('south_north',BLUE,'Photometry change'),('north_south',ORANGE,'Forest + calibration change'),('north_north',GREEN,'Both changes')):
                q=predictions[key+'_'+errorcase]
                delta=5*np.log10(q[common,2]/predictions['south_south_'+errorcase][common,2])
                axes[0].hist(delta,bins=bins,histtype='step',density=True,color=color,ls=ls,label=label if errorcase=='native' else None)
        both=(s['state']==3)&(n['state']==3)
        h,xe,ye=np.histogram2d(gr[both],5*np.log10(n['q'][both,2]/s['q'][both,2]),(np.linspace(-.2,1.6,70),np.linspace(-1.5,1.5,80)))
        from matplotlib.colors import LogNorm
        z=axes[1].pcolormesh(xe,ye,np.ma.masked_equal(h.T,0),cmap='viridis',norm=LogNorm(1,max(2,h.max())),rasterized=True)
        f.colorbar(z,ax=axes[1],fraction=.035,label='Same-star pairs / bin')
        axes[0].set(xlabel='Counterfactual modulus change (mag)',ylabel='Density');axes[0].legend(frameon=False,fontsize=6)
        axes[1].set(xlabel=r'$(g-r)_{0,S}$',ylabel=r'$\mu_N-\mu_S$ (mag)')
        figure(d,'native_seam_frozen_decomposition',f,rows)
    complete(d,inputs,selected=len(ii),common=int(common.sum()),sampling='up to64 deterministic hash rows per 5r*4gr cell with both strict photometry',
        identity='same native-pair key and identical Sobol noise across counterfactuals',
        decomposition='2 frozen forests x 2 photometry systems x native/common max-error; same all-eight-supported subset',
        scope='consistency only; these input swaps are counterfactuals, not calibrated photometric-system interchange; no absolute-scale inference')


if __name__=='__main__':main()
