#!/usr/bin/env python3
"""Reduce the full native reference census and reproduce the frozen benchmark."""
import csv
import json
import os

import numpy as np

from photdis.stage1 import require_compute
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_9_references import join, subset
from final_science_catalog import REGIONS
from final_catalogue_benchmark import load_parent, masks, published
from release_readiness_compare import available
from paper_v1_scan import locate
from paper_v2_scan import AXES, CLASSES, NP, OUT, R, GR, RZ, REFS, SHARDS, SHELLS, bin_axis

BIN_AXES=('region','low_lat','observed_r','g_r','r_z','population','reference_distance')
TRAIN=('south_fit','north_gaia_fit','north_specdis_fit','north_calibration','north_test','north_class_fit')
POP_MIGRATION=(1,2,4,5)
PRED_EDGES=(0.,1.,2.,5.,10.,20.,np.inf)


def read_all(inputs):
    counts=np.zeros((3,np.prod(AXES[:-2]),5),'i8')
    sky=np.zeros((4,2,3,NP))
    c=np.zeros(4,'i8')
    blocks=[]
    for i in range(SHARDS):
        p=OUT/'scan'/f'{i:02}.json'
        r=json.loads(p.read_text())
        inputs[f'scan_{i:02}']=sha(p)
        c+=np.array([r[k] for k in ('parts','parent','pdf','matched')])
        key=f'{i:02}_summary.npz'
        with np.load(verified(p.with_name(key),r['outputs'][key])) as z:
            counts+=z['counts']; sky+=z['sky']
        key=f'{i:02}_matched.npz'
        with np.load(verified(p.with_name(key),r['outputs'][key])) as z:
            blocks.append({k:z[k] for k in z.files})
    np.testing.assert_array_equal(c[:3],(28917,72289386,43347612))
    t=join(blocks)
    assert (t['TYPE']=='PSF').all() and len(t['ls_id'])==c[3]
    assert len(np.unique(t['ls_id']))==len(t['ls_id'])
    positive=t['source_id']>0
    keys,n=np.unique(t['source_id'][positive],return_counts=True)
    # Ambiguous Gaia aliases retain record and qualified-reference counts. Exclude both
    # aliases from the independent frame and retain a diagnostic of their effect.
    t['alias']=positive&np.isin(t['source_id'],keys[n>1])
    t['heldout'] &= ~t['alias']
    path=ROOT/'manifests/v3_10/branch_cases/wd/wd_evidence.csv'
    with path.open() as f:
        wd={int(x['ls_id']) for x in csv.DictReader(f) if x['spectroscopic_wd']=='True' or
            (x['published_pwd'] and float(x['published_pwd'])>.75)}
    t['pop'][np.isin(t['ls_id'],list(wd))&~np.isin(t['pop'],(8,9))]=10
    inputs['wd_candidates']=sha(path)
    for kind in (0,1):
        np.testing.assert_allclose(sky[:,kind].sum(),43347612,rtol=0,atol=3e-6)
    np.testing.assert_array_equal(sky[:,0].sum(axis=(0,2)),[6237172,25229239,11881201])
    return counts,sky,t,c


def bins(t,d=None):
    g=t['m'][:,0]-t['m'][:,1]
    z=t['m'][:,1]-t['m'][:,2]
    rad=np.zeros(len(g),'i1') if d is None else bin_axis(d,SHELLS)
    return np.column_stack((t['region'],t['low_lat'],bin_axis(t['observed_r'],R),
                           bin_axis(g,GR),bin_axis(z,RZ),t['pop'],rad)).astype('i2')


def metric(q,d,modes=None):
    q=np.asarray(q); d=np.asarray(d)
    if len(d)==0:
        return dict(n=0,bias=None,nmad=None,severe=None,c68=None,c95=None,multiple_peak_fraction=None,mu_median=None)
    e=q[:,2]/d-1
    med=np.median(e)
    dm=5*np.log10(q[:,2]/d)
    return dict(n=len(d),bias=med,nmad=1.4826*np.median(np.abs(e-med)),severe=(np.abs(dm)>1).mean(),
                c68=((d>=q[:,1])&(d<=q[:,3])).mean(),c95=((d>=q[:,0])&(d<=q[:,4])).mean(),
                multiple_peak_fraction=(modes>=2).mean() if modes is not None else None,
                mu_median=np.median(dm))


def reference_uncertainty_mu(t, source, rows):
    """Return the one-sigma scale and optional 95% geometric halfwidth."""
    if source!='gaia':
        return t[source+'_sig'][rows],None
    sigma=(5/np.log(10))*t['p_err'][rows]/t['p_corr'][rows]
    halfwidth95=2.5*np.log10(t['gaia_hi'][rows]/t['gaia_lo'][rows])
    return sigma,halfwidth95


def summarize(t,counts,dest):
    x=bins(t)
    sh=AXES[:-2]
    group=np.ravel_multi_index(tuple(np.column_stack((x[:,:3],t['state'])).T),sh)
    rows=[]; roles=[]; met=[]; signed=[]; both=[]
    for i,src in enumerate(REFS):
        st=t[f'{src}_stage']
        d=t[f'{src}_d']
        good=st==3
        pdf=t['state']==3
        rgrid=bins(t,d)
        level=range(counts.shape[1])
        for j in level:
            r,l,m,s=np.unravel_index(j,sh)
            c=counts[i,j]
            if not c.sum(): continue
            take=group==j
            rows.append(dict(reference=src,region=REGIONS[r],low_lat=l,r_bin=m,state=s,
                 parent=int(c.sum()),unrecorded=int(c[0]),recorded=int(c[1:].sum()),
                 valid=int(c[2:].sum()),qualified=int(c[3:].sum()),
                 heldout=int((take&good&t['heldout']).sum()),gaia_alias=int((take&good&t['alias']).sum()),
                 gaia_tile_missing=int((take&t['gmissing']).sum())))
            for k,label in enumerate(TRAIN):
                roles.append(dict(reference=src,region=REGIONS[r],low_lat=l,r_bin=m,state=s,role=label,
                     matched=int((take&(st>=1)&((t['training']&(1<<k))>0)).sum()),
                     qualified=int((take&good&((t['training']&(1<<k))>0)).sum())))
        for frame,base in (('all',good&pdf),('heldout',good&pdf&t['heldout'])):
            j=np.flatnonzero(base)
            for label,axes in (('domain_magnitude_distance_pop',(0,1,2,5,6)),
                              ('domain_color_gr',(0,3,6)),('domain_color_rz',(0,4,6)),
                              ('domain_magnitude',(0,2)),('magnitude',(2,)),('total',())):
                k,ix=np.unique(rgrid[j][:,axes],axis=0,return_inverse=True)
                order=np.argsort(ix,kind='stable'); boundaries=np.r_[0,np.flatnonzero(np.diff(ix[order]))+1,len(j)]
                for b,(start,stop) in enumerate(zip(boundaries[:-1],boundaries[1:])):
                    if stop==start: continue
                    z=j[order[start:stop]]
                    sig,halfwidth95=reference_uncertainty_mu(t,src,z)
                    row=dict(reference=src,frame=frame,grouping=label)
                    row.update({x:None for x in BIN_AXES})
                    row.update({BIN_AXES[a]:int(v) for a,v in zip(axes,k[b])})
                    row.update(**metric(t['q'][z],d[z],t['peaks'][z]),median_reference_sigma_mu=np.median(sig),
                        median_reference_halfwidth95_mu=None if halfwidth95 is None else np.median(halfwidth95))
                    met.append(row)
    for a,b in (('specdis','spdist'),('gaia','specdis'),('gaia','spdist')):
        base=(t[a+'_stage']==3)&(t[b+'_stage']==3)&(t['state']==3)
        for k in range(len(R)):
            for region in range(4):
                for frame,keep in (('all',base),('heldout',base&t['heldout'])):
                    j=keep&(x[:,2]==k)&(x[:,0]==region)
                    d1,d2=t[a+'_d'][j],t[b+'_d'][j]
                    dv=5*np.log10(d1/d2)
                    both.append(dict(first=a,second=b,frame=frame,region=REGIONS[region],r_bin=k,n=int(j.sum()),
                         joint_median_mu=np.median(dv) if len(dv) else None,
                         joint_nmad_mu=1.4826*np.median(abs(dv-np.median(dv))) if len(dv) else None,
                         overlap_state='one row per common native LS_ID'))
    for region in range(4):
        for b in range(len(R)):
            for state in (4,5):
                for frame,extra in (('all',np.ones(len(x),bool)),('heldout',t['heldout'])):
                    j=(x[:,0]==region)&(x[:,2]==b)&extra&(t['state']==3)&(t['gaia_state']==state)
                    p=t['pit'][j]
                    v=np.isfinite(p)&(p>=-1e-6)&(p<=1+1e-6)
                    h=np.histogram(np.clip(p[v],0,1),np.linspace(0,1,11))[0]
                    signed.append(dict(region=REGIONS[region],r_bin=b,frame=frame,gaia_state=state,n=int(j.sum()),
                          negative=int((t['p_corr'][j]<0).sum()),pit_valid=int(v.sum()),
                          central68=((p[v]>=.16)&(p[v]<=.84)).mean() if v.any() else None,
                          **{f'bin_{k}':int(n) for k,n in enumerate(h)}))
    for filename,table in (('coverage',rows),('train_test_overlap',roles),('distance_metrics',met),
                           ('signed_gaia',signed),('common_references',both)):
        csv_rows(dest/f'{filename}.csv',table)


def replay_benchmark(t,dest,inputs):
    v,h=load_parent()
    p,_,_=published(v,h['outputs']['ids.npy'])
    order=np.argsort(t['ls_id'])
    a,b=locate(t['ls_id'][order],v['ls_id'])
    j=order[b]
    cross=np.full(len(v['ls_id']),-1,'i8'); cross[a]=j
    rows=[]
    for ref in ('SpecDis','SPDist'):
        r=ref.lower(); d=v[r+'_kpc']
        use=masks(v)['weak_gaia_dwarfs']&available(p['Photdis'])&available(p['BJ_photogeometric'])&(d>=2)&(d<5)
        k=cross[use]
        assert np.all(k>=0) and np.all(t['state'][k]==3)
        np.testing.assert_allclose(t[r+'_d'][k],d[use],rtol=1e-6)
        np.testing.assert_allclose(t['q'][k,2],p['Photdis'][use,1],rtol=1e-6)
        rows.append(dict(reference=ref,frame='frozen_F15',heldout_in_new_frame=int(t['heldout'][k].sum()),
                         **metric(t['q'][k],d[use],t['peaks'][k])))
        take=(t[r+'_stage']==3)&(t['q'][:,2]>0)&(t['observed_r']>=18)&(t['observed_r']<20)&(
             t[r+'_d']>=2)&(t[r+'_d']<5)&np.isin(t['gaia_state'],(3,4,5))&np.isin(t['pop'],(1,2,3))
        for label,j in (('full_join_nominal_window',take),('full_join_window_heldout',take&t['heldout'])):
            rows.append(dict(reference=ref,frame=label,heldout_in_new_frame=int(t['heldout'][j].sum()),
                             **metric(t['q'][j],t[r+'_d'][j],t['peaks'][j])))
    csv_rows(dest/'benchmark_replay.csv',rows)
    assert [r['n'] for r in rows[::3]]==[9355,8927]
    path=ROOT/'data/paper_v1_evidence/benchmark/complete.json'
    meta=json.loads(path.read_text())
    c=verified(path.with_name('observables.npz'),meta['outputs']['observables.npz'])
    with np.load(c) as a:
        z,d=a['counts'],a['reference_counts']
        union=(d>=20).any(axis=0)
        observed=z[:,union].sum(axis=1)
        np.testing.assert_array_equal(observed,(806611,0,376857))
    csv_rows(dest/'f15_observable_replay.csv',[dict(cohort=x,n=int(n),common_cells=int(m)) for x,n,m in
        zip(('full_pdf','near_core','mid_core'),z.sum(axis=1),observed)])
    inputs['v1_observables']=sha(path)


def main():
    require_compute()
    d=OUT/'reference'
    d.mkdir(exist_ok=True,parents=True)
    inputs={}
    c,s,t,n=read_all(inputs)
    summarize(t,c,d)
    replay_benchmark(t,d,inputs)
    atomic_npz(d/'sky.npz',sky=s)
    atomic_npz(d/'reference_rows.npz',**t)
    save_json(d/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,
      partitions=int(n[0]),parent=int(n[1]),pdf=int(n[2]),any_reference=int(n[3]),
      outputs={p.name:sha(p) for p in d.iterdir() if p.name!='complete.json'}))


if __name__=='__main__':
    main()
