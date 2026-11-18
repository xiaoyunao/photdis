#!/usr/bin/env python3
"""Replay the old band realizations with errors consistent with injected variance."""
import joblib
from photdis.stage1 import require_compute
from photdis.v3_3 import frozen_qrf
from sv18_common import cached, metric
from v3_9_catalog import from_north
from v3_3_catalog import read_frozen
from halo_common import *


def effective_errors(original,donor):
    extra=np.sqrt(np.maximum(donor**2-original**2,0))
    return extra,np.maximum(original,donor)


def models(north,inputs):
    if north:
        model,h=from_north();inputs['north_model']=h
        return model['qrf'],float(model['mu_offset'])
    entry=read_frozen()['models']['qrf'];p=verified(ROOT/entry['path'],entry['sha256'])
    inputs['south_model']=sha(p)
    return joblib.load(p),0.


def point_stats(center,reference):
    if not len(center):return dict(n=0,bias=np.nan,nmad=np.nan,severe=np.nan)
    q=np.repeat(center[:,None],5,axis=1)
    a=metric(q,reference)
    return {k:a[k] for k in ('n','bias','nmad','severe')}


def main():
    require_compute()
    d=folder('noise_fixed');inputs={}
    t=cached(inputs,('TYPE','ls_id','source_id','region','observed_r','m','err','pop','heldout','quality','state',
        'gaia_stage','gaia_d','specdis_stage','specdis_d','q','RA','DEC','dep','EBV'))
    old=ROOT/'data/science_validation_20261118/noise'
    original=read(old,('individual_noise.npz','TYPE','native','r_lo','draw','ls_id','supported','distance',
        'reference','sigma_g','sigma_r','sigma_z'),inputs)
    with (old/'noise_realizations.csv').open() as f:
        old_rows=list(csv.DictReader(f))
    rng=np.random.default_rng(20261118)
    rows=[];details=[];receivers=[];color_rows=[]
    for side,north in (('south',False),('north',True)):
        native=(t['region']==3)==north
        base=native&t['heldout']&(t['quality']==7)&(t['state']==3)&(t['observed_r']>=16)&(t['observed_r']<18)
        base&=(t['pop']==1)&((t['gaia_stage']==3)|(t['specdis_stage']==3))
        reference=np.where(t['gaia_stage']==3,t['gaia_d'],t['specdis_d'])
        base&=(reference>0)&(reference<5)
        idx=np.flatnonzero(base)
        if len(idx)>256:idx=np.sort(rng.choice(idx,256,replace=False))
        forest,offset=models(north,inputs)
        mags=t['m'][idx];err0=t['err'][idx]
        gr=mags[:,0]-mags[:,1];rz=mags[:,1]-mags[:,2]
        rec={k:t[k][idx] for k in ('TYPE','ls_id','source_id','RA','DEC','m','err','dep','q','EBV')}
        rec.update(r=t['observed_r'][idx],reference=reference[idx],native=np.full(len(idx),north,'u1'))
        receivers.append(rec)
        for low,high in ((20,21),(21,22),(22,23)):
            pool=np.flatnonzero(native&(t['observed_r']>=low)&(t['observed_r']<high)&np.isfinite(t['err']).all(axis=1))
            choices=[];fallback=[]
            for a,b in zip(gr,rz):
                close=pool[(abs(t['m'][pool,0]-t['m'][pool,1]-a)<.15)&(abs(t['m'][pool,1]-t['m'][pool,2]-b)<.15)]
                fallback.append(len(close)<20)
                if len(close)<20:close=pool
                choices.append(rng.choice(close,size=16,replace=True))
            choices=np.asarray(choices)
            for draw in range(16):
                donor=choices[:,draw];error=t['err'][donor]
                extra,eff=effective_errors(err0,error)
                normal=rng.normal(size=mags.shape)
                perturbed=mags+normal*extra
                mask=(original['native']==side)&(original['r_lo']==low)&(original['draw']==draw)
                np.testing.assert_array_equal(original['ls_id'][mask],t['ls_id'][idx])
                old_error=np.column_stack([original['sigma_'+b][mask] for b in 'grz'])
                np.testing.assert_array_equal(error,old_error)
                old_center=original['distance'][mask].copy();old_good=original['supported'][mask]
                clipped=(error<err0).any(axis=1)
                changed=np.flatnonzero(clipped)
                oldq=np.empty((0,5));newq=oldq.copy();newgood=old_good.copy();newcenter=old_center.copy()
                if len(changed):
                    prior=frozen_qrf(forest,perturbed[changed],error[changed],t['source_id'][idx[changed]])
                    corrected=frozen_qrf(forest,perturbed[changed],eff[changed],t['source_id'][idx[changed]])
                    oldq=prior['distance']*10**(offset/5);newq=corrected['distance']*10**(offset/5)
                    np.testing.assert_allclose(oldq[:,2],old_center[changed],rtol=2e-6,atol=1e-8,equal_nan=True)
                    np.testing.assert_array_equal(prior['supported'],old_good[changed])
                    newcenter[changed]=newq[:,2];newgood[changed]=corrected['supported']
                if draw==0:
                    proof=frozen_qrf(forest,perturbed,error,t['source_id'][idx])
                    np.testing.assert_allclose(proof['distance'][:,2]*10**(offset/5),old_center,rtol=2e-6,atol=1e-8,equal_nan=True)
                    np.testing.assert_array_equal(proof['supported'],old_good)
                legacy=next(r for r in old_rows if r['native']==side and float(r['r_lo'])==low and int(r['draw'])==draw)
                nold=int(legacy['n']);ref=reference[idx]
                assert nold==old_good.sum()
                coverage={}
                for level,a,b in (('c68',1,3),('c95',0,4)):
                    total=round(float(legacy[level])*nold)
                    if len(changed):
                        prev=old_good[changed]&(oldq[:,a]<=ref[changed])&(ref[changed]<=oldq[:,b])
                        current=newgood[changed]&(newq[:,a]<=ref[changed])&(ref[changed]<=newq[:,b])
                        total+=int(current.sum())-int(prev.sum())
                    coverage[level]=total/newgood.sum() if newgood.any() else None
                stats=point_stats(newcenter[newgood],ref[newgood])
                baseline=metric(t['q'][idx][newgood],ref[newgood])
                rows.append(dict(native=side,r_lo=low,r_hi=high,draw=draw,parent=len(idx),
                    old_supported=int(old_good.sum()),supported=int(newgood.sum()),clipped_rows=int(clipped.sum()),
                    clipped_bands=int((error<err0).sum()),fallback_rows=int(np.sum(fallback)),
                    **stats,**coverage,baseline_bias=baseline['bias'],baseline_nmad=baseline['nmad'],
                    old_bias=float(legacy['bias']),old_nmad=float(legacy['nmad']),
                    maximum_paired_distance_change=np.nanmax(abs(newcenter-old_center))))
                for a,b in ((-.2,.3),(.3,.6),(.6,1),(1,1.6)):
                    use=(gr>=a)&(gr<b);common=use&old_good&newgood
                    color_rows.append(dict(native=side,r_lo=low,draw=draw,gr_lo=a,gr_hi=b,parent=int(use.sum()),
                        clipped_rows=int((use&clipped).sum()),common=int(common.sum()),
                        mean_paired_relative_change=np.mean(newcenter[common]/old_center[common]-1) if common.any() else None))
                details.append(dict(TYPE=t['TYPE'][idx],ls_id=t['ls_id'][idx],donor_ls_id=t['ls_id'][donor],
                    native=np.full(len(idx),north,'u1'),r_lo=np.full(len(idx),low,'u1'),draw=np.full(len(idx),draw,'u1'),
                    donor_error=error,original_error=err0,effective_error=eff,standard_normal=normal,
                    old_supported=old_good,new_supported=newgood,old_distance=old_center,new_distance=newcenter,
                    fallback=np.asarray(fallback),donor_color_distance=np.hypot(t['m'][donor,0]-t['m'][donor,1]-gr,
                                                        t['m'][donor,1]-t['m'][donor,2]-rz)))
    def combine(parts):return {k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
    atomic_npz(d/'paired_realizations.npz',**combine(details))
    atomic_npz(d/'receivers.npz',**combine(receivers))
    csv_rows(d/'paired_noise_metrics.csv',rows);csv_rows(d/'clip_by_color.csv',color_rows)
    summary=[]
    for side in ('south','north'):
        for low in (20,21,22):
            a=[r for r in rows if r['native']==side and r['r_lo']==low]
            summary.append(dict(native=side,r_lo=low,draws=len(a),
                clipped_fraction=sum(r['clipped_rows'] for r in a)/sum(r['parent'] for r in a),
                median_bias=np.median([r['bias'] for r in a]),median_nmad=np.median([r['nmad'] for r in a]),
                median_old_nmad=np.median([r['old_nmad'] for r in a]),
                maximum_paired_distance_change=max(r['maximum_paired_distance_change'] for r in a)))
    csv_rows(d/'noise_correction.csv',summary)
    complete(d,inputs,scope='exact old source/donor/draw replay; only donor-error clips change model input',
        replay_checks='every donor triplet equals old archive; every clipped prediction and one full draw per bin reproduce original distances/support',
        efficiency='unchanged input rows reuse archived distances/support; full reference-center coverage updated by the exact changed-row contribution')


if __name__=='__main__':main()
