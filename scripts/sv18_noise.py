#!/usr/bin/env python3
"""Fixed-population native band-noise experiments using actual faint error triplets."""
import joblib
from photdis.stage1 import require_compute
from photdis.v3_3 import frozen_qrf
from sv18_common import *
from v3_9_catalog import from_north
from v3_3_catalog import FREEZE, read_frozen


def main():
    require_compute()
    inputs={};d=out('noise')
    keys=('TYPE','ls_id','source_id','region','observed_r','m','err','pop','heldout','quality','state',
          'gaia_stage','gaia_d','specdis_stage','specdis_d','q')
    t=cached(inputs,keys)
    rows=[];detail=[];error_summary=[]
    for side,north in (('south',False),('north',True)):
        native=(t['region']==3)==north
        base=native&t['heldout']&(t['quality']==7)&(t['state']==3)&(t['observed_r']>=16)&(t['observed_r']<18)
        base &= (t['pop']==1)&((t['gaia_stage']==3)|(t['specdis_stage']==3))
        ref=np.where(t['gaia_stage']==3,t['gaia_d'],t['specdis_d'])
        base&=(ref>0)&(ref<5)
        idx=np.flatnonzero(base)
        if len(idx)>P['noise']['bright_per_native']:
            idx=np.sort(RNG.choice(idx,P['noise']['bright_per_native'],replace=False))
        if not len(idx):continue
        if north:
            model,hh=from_north();forest=model['qrf'];offset=model['mu_offset'];inputs['north_model']=hh
        else:
            record=read_frozen()
            entry=record['models']['qrf']
            path=verified(ROOT/entry['path'],entry['sha256'])
            forest=joblib.load(path);offset=0;inputs['south_model']=sha(path)
        mags=t['m'][idx];err0=t['err'][idx]
        gr=mags[:,0]-mags[:,1];rz=mags[:,1]-mags[:,2]
        for low,high in P['noise']['r_bins']:
            pool=native&(t['observed_r']>=low)&(t['observed_r']<high)&np.isfinite(t['err']).all(axis=1)
            pp=np.flatnonzero(pool)
            if not len(pp):continue
            # Joint triplets keep the real dependence among band errors.
            target_error=[]
            for a,b in zip(gr,rz):
                close=pp[(abs(t['m'][pp,0]-t['m'][pp,1]-a)<.15)&
                         (abs(t['m'][pp,1]-t['m'][pp,2]-b)<.15)]
                if len(close)<20:
                    close=pp
                target_error.append(t['err'][RNG.choice(close,size=P['noise']['draws'],replace=True)])
            target_error=np.asarray(target_error)
            for draw in range(P['noise']['draws']):
                error=target_error[:,draw]
                extra=np.sqrt(np.maximum(error**2-err0**2,0))
                perturbed=mags+RNG.normal(size=mags.shape)*extra
                pred=frozen_qrf(forest,perturbed,error,t['source_id'][idx])
                q=pred['distance']*10**(offset/5)
                good=pred['supported']
                rows.append(dict(native=side,r_lo=low,r_hi=high,draw=draw,parent=len(idx),
                    supported=int(good.sum()),support_retention=good.mean(),experiment='noise_only_fixed_scale',
                    **metric(q[good],ref[idx][good])))
                for i in range(len(idx)):
                    detail.append(dict(native=side,r_lo=low,draw=draw,ls_id=int(t['ls_id'][idx[i]]),
                      supported=bool(good[i]),distance=q[i,2],reference=ref[idx[i]],
                      sigma_g=error[i,0],sigma_r=error[i,1],sigma_z=error[i,2]))
            error_summary.append(dict(native=side,r_lo=low,r_hi=high,donor_rows=len(pp),bright_rows=len(idx),
                median_sigma_g=np.median(target_error[:,:,0]),median_sigma_r=np.median(target_error[:,:,1]),
                median_sigma_z=np.median(target_error[:,:,2]),
                baseline_bias=np.median(t['q'][idx,2]/ref[idx]-1),
                perturbation='quadrature incremental native magnitude noise, shared g/r used coherently'))
    csv_rows(d/'noise_realizations.csv',rows)
    csv_rows(d/'error_donors.csv',error_summary)
    atomic_npz(d/'individual_noise.npz',
        **{k:np.array([r[k] for r in detail]) for k in detail[0]} if detail else {})
    summary=[]
    for native in ('south','north'):
        for lo,hi in P['noise']['r_bins']:
            a=[r for r in rows if r['native']==native and r['r_lo']==lo and r['n']]
            if not a:continue
            bias=np.array([r['bias'] for r in a])
            summary.append(dict(native=native,r_lo=lo,r_hi=hi,realizations=len(a),
                median_bias=np.median(bias),noise_draw_lo=np.quantile(bias,.025),noise_draw_hi=np.quantile(bias,.975),
                median_nmad=np.median([r['nmad'] for r in a]),median_c68=np.median([r['c68'] for r in a]),
                median_retention=np.median([r['support_retention'] for r in a]),
                scale_pass=abs(np.median(bias))<=P['noise']['success_bias_absolute']))
    csv_rows(d/'noise_summary.csv',summary)
    with plt.rc_context(PAPER):
        f,axes=plt.subplots(1,2,figsize=(7.1,3.05),layout='constrained')
        for native,col in (('south',BLUE),('north',ORANGE)):
            a=[r for r in summary if r['native']==native]
            x=[(r['r_lo']+r['r_hi'])/2 for r in a]
            axes[0].plot(x,[r['median_nmad'] for r in a],'o-',ms=3,color=col,label=native.capitalize())
            axes[1].plot(x,[r['median_retention'] for r in a],'o-',ms=3,color=col)
        axes[0].set_ylabel('Relative-distance NMAD')
        axes[1].set_ylabel('Retained PDF support')
        for ax in axes:ax.set_xlabel('Error-donor r (mag)')
        axes[0].legend(fontsize=7,frameon=False)
        figure(d,'fixed_population_noise',f,summary)
    complete(d,inputs,scope='conditional fixed bright-population noise experiment; no new true faint-distance validation')


if __name__=='__main__':
    main()
