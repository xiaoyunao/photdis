#!/usr/bin/env python3
"""Test LF sensitivity by transporting the measured native selection response."""
import json
import os

import numpy as np

from photdis.reference_v1_4 import unique_join
from photdis.stage1 import galactic, require_compute
from v2_1_near import ROOT, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_6_vac_distances import read_columns
from release_readiness_population import classes
from v3_10_redrock import DEST as REDROCK
from final_catalogue_benchmark import parent
from final_catalogue_raw import DEST

OUT=DEST/'population'
SCENARIOS=('empirical','disk_rich','halo_rich')
LABELS=('dwarf_feh_ge_minus1p5','dwarf_feh_lt_minus1p5','dwarf_feh_unknown',
        'intermediate_gravity','giant','other_stellar')
MAG=((18.,19.),(19.,20.))
COLOR=((-np.inf,.5),(.5,.8),(.8,np.inf))
DRAWS=512
SEED=531709


def phase(t,inputs):
    raw=np.where(t['redrock_quality_assessed'],t['redrock_SPECTYPE'].astype('U8'),'')
    mark=REDROCK/'complete.json'
    p=verified(REDROCK/'qualified.npz',json.loads(mark.read_text())['outputs']['qualified.npz'])
    tab=read_columns(p,('ls_id','TYPE','qualified'))
    j=unique_join(t['ls_id'],tab['ls_id'])
    found=np.flatnonzero(j>=0)
    np.testing.assert_array_equal(t['TYPE'][found],tab['TYPE'][j[found]])
    k=found[tab['qualified'][j[found]]!='']
    conflict=k[(raw[k]!='')&(raw[k]!=tab['qualified'][j[k]])]
    raw[k]=tab['qualified'][j[k]]
    a=classes(t,raw)
    a[conflict]='class_conflict'
    inputs[str(mark.relative_to(ROOT))]=sha(mark)
    return a


def tilt(t,d,category,scenario):
    """Apply LF, metallicity and spatial alternatives on occupied reference support."""
    if scenario=='empirical':
        return np.ones(len(d))
    l,b=np.deg2rad(galactic(t['RA'],t['DEC']))
    dr=d*np.cos(b)
    z=d*np.sin(b)+.025
    R=np.hypot(8.-dr*np.cos(l),dr*np.sin(l))
    disk=np.exp(-(R-8.)/2.6-abs(z)/.3)+.12*np.exp(-(R-8.)/3.6-abs(z)/.9)
    halo=(np.hypot(R,z/.64)/8.)**(-2.8)*np.exp(-(np.hypot(R,z/.64)/100.)**2)
    # Relative to a fixed equal disk/halo reference tilt, common d^3 cancels.
    frac=.15 if scenario=='disk_rich' else .70
    spatial=((1-frac)*disk+frac*halo)/(.5*(disk+halo))
    mr=t['r0']-10-5*np.log10(d)
    giant=category==LABELS.index('giant')
    slope,mean,sigma=(.27,-.72,.22) if scenario=='disk_rich' else (.32,-1.46,.30)
    luminosity=np.where(giant,10**(slope*np.clip(mr,-4,3)),1.)
    f=t['rvs_feh']
    ef=t['rvs_feh_err']
    has=np.isfinite(f)&(ef>0)&(ef<=.3)
    log_ratio=.5*((f+1.0)/.7)**2-.5*((f-mean)/sigma)**2
    metallicity=np.where(has,np.exp(np.clip(log_ratio,-3.5,2.5)),1.)
    return np.clip(spatial*luminosity*metallicity,.02,40.)


def target_counts(inputs):
    p=ROOT/'data/final_science/evidence/complete.json'
    r=json.loads(p.read_text())
    t=read_columns(verified(p.with_name('selected.npz'),r['outputs']['selected.npz']),
                   ('TYPE','science','region','r','color'))
    assert (t['TYPE']=='PSF').all()
    inputs[str(p.relative_to(ROOT))]=sha(p)
    x=[]
    for side,i in (('north',3),('south_low',2)):
        base=(t['science']==2)&(t['region']==i)
        for r0,r1 in MAG:
            for c0,c1 in COLOR:
                sel=base&(t['r']>=r0)&(t['r']<r1)&(t['color']>=c0)&(t['color']<c1)
                x.append(dict(region=side,r_min=r0,r_max=r1,color_min=c0,color_max=c1,
                              core_sources=int(sel.sum())))
    return x


def forecast(weights,group,sky,n_test,rng):
    ix,k=np.unique(sky,return_inverse=True)
    cells=len(LABELS)*3
    sums=np.zeros((len(ix),cells))
    np.add.at(sums,(k,group),weights)
    sq=np.bincount(k,weights=weights**2,minlength=len(ix))
    counts=np.zeros((DRAWS,cells),'i8')
    for j in range(DRAWS):
        draw=rng.integers(len(ix),size=len(ix))
        summed=sums[draw].sum(axis=0)
        n_eff=summed.sum()**2/sq[draw].sum()
        p_old=summed/summed.sum()
        p=rng.dirichlet(p_old*n_eff+.25)
        counts[j]=rng.multinomial(n_test,p)
    return counts.reshape(DRAWS,len(LABELS),3)


def check_side(side,inputs,rows,controls,summaries,rng):
    t=parent(side,inputs)
    region=t['region'][0]
    group=phase(t,inputs)
    cat=np.full(len(group),-1,'i4')
    for i,key in enumerate(LABELS):
        cat[group==key]=i
    valid=(t['row_state']==7)&(t['state']>=2)
    train=valid&(t['sky_role']<=3)
    test=valid&(t['sky_role']==4)&t['evaluation_row']
    q=t['qrf_quantiles_kpc'][:,2]
    chosen=(t['state']==4)&(q>=2)&(q<5)
    r=t['r']; c=t['g0']-t['r0']
    tile=(t['RA']//10).astype(int)*18+((t['DEC']+90)//10).astype(int)
    for a,b in MAG:
        for c0,c1 in COLOR:
            obs=(r>=a)&(r<b)&(c>=c0)&(c<c1)
            base=dict(region=region,r_min=a,r_max=b,color_min=c0,color_max=c1)
            controls.append(dict(**base,**{f'parent_{k}':int((obs&test&(group==k)).sum())
                  for k in ('qualified_nonstellar','class_conflict','unclassified')},
                **{f'core_{k}':int((obs&test&chosen&(group==k)).sum())
                  for k in ('qualified_nonstellar','class_conflict','unclassified')}))
            for source in ('specdis','spdist'):
                d=t[source+'_kpc']
                good=t[source+'_quality']&np.isfinite(d)&(d>=.1)&(d<30)&(cat>=0)&obs
                tr=np.flatnonzero(good&train)
                te=np.flatnonzero(good&test)
                if len(tr)<50 or len(te)<50:
                    summaries.append(dict(**base,reference=source,scenario='unassessed',
                      train_n=len(tr),test_n=len(te),covered=None,tests=0))
                    continue
                dm=np.full(len(d),np.nan)
                usable=good&valid
                dm[usable]=abs(5*np.log10(q[usable]/d[usable]))
                outcome=cat*3+np.where(chosen,np.where(dm>1,2,1),0)
                obs_count=np.bincount(outcome[te],minlength=len(LABELS)*3).reshape(-1,3)
                draw_t={key:v[tr] for key,v in t.items() if key in ('RA','DEC','r0','rvs_feh','rvs_feh_err')}
                for scenario in SCENARIOS:
                    w=tilt(draw_t,d[tr],cat[tr],scenario)
                    pred=forecast(w,outcome[tr],tile[tr],len(te),rng)
                    cases=[(k,pred[:,i,1:].sum(axis=1),int(obs_count[i,1:].sum()))
                           for i,k in enumerate(LABELS)]
                    cases+=[('all_core',pred[:,:,1:].sum(axis=(1,2)),int(obs_count[:,1:].sum())),
                            ('severe_core',pred[:,:,2].sum(axis=1),int(obs_count[:,2].sum()))]
                    cover=0
                    for event,x,m in cases:
                        lo,mid,hi=np.quantile(x,[.025,.5,.975])
                        accepted=bool(lo<=m<=hi)
                        cover+=accepted
                        rows.append(dict(**base,reference=source,scenario=scenario,event=event,
                           train_n=len(tr),test_n=len(te),observed=m,expected=mid,lo95=lo,hi95=hi,covered=accepted))
                    summaries.append(dict(**base,reference=source,scenario=scenario,train_n=len(tr),
                                         test_n=len(te),covered=cover,tests=len(cases)))


def main():
    require_compute()
    OUT.mkdir(exist_ok=True,parents=True)
    inputs={}
    rows,controls,summary=[],[],[]
    rng=np.random.default_rng(SEED)
    for side in ('north','south'):
        check_side(side,inputs,rows,controls,summary,rng)
    for name,group in (('predictive_counts',rows),('nonstellar_controls',controls),
                       ('validation_by_cell',summary),('core_observable_counts',target_counts(inputs))):
        csv_rows(OUT/f'{name}.csv',group)
    save_json(OUT/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,seed=SEED,
           draws=DRAWS,scope='conditional on qualifying spectrum and QRF support',
           scripts=sha(ROOT/'scripts/final_catalogue_population.py'),
           outputs={p.name:sha(p) for p in OUT.glob('*.csv')}))


if __name__=='__main__':
    main()
