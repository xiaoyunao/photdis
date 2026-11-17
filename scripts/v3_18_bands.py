#!/usr/bin/env python3
"""Measure native passband differences and replay fixed GD-1 CMD sensitivities."""
import json
import os

import numpy as np
from astropy.io import fits
from scipy.ndimage import shift

from photdis.reference_v1_4 import unique_join
from photdis.stage1 import require_compute
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_6_reference_photometry import OUT as SOUTH
from v3_8_direct_figures import BLUE, GRAY, STYLE, plt, save
from v3_10_figures import check_templates
from v3_11_layers import read_receipt
from v3_13_delivery import figure_check
from v3_9_references import subset
from v3_16_fit import ABS, arrays, cluster_template, magnitude_response, project
from v3_16_gd1 import EDGES, NSEG, NSTEP, P
from v3_16_sample import COLOR, D
from v3_17_distance import fitted
from v3_17_hess import MODULUS_CURVE, by_cells, detect
from v3_17_members import MEMBERS, table
from v3_17_sample import DEST as PRIOR, INPUT, NEW
from v3_18_members import CONFIG, DEST, C, read

OVERLAP = ROOT/'data/v3_9/expansion/validation/overlap'
DEPTHS = {'r < 20':1,'r < 21':2,'r < 22':3}
LABELS = ('g-r','r-z','r')


def pairs(inputs):
    names = ('TYPE','source_id','north_ls_id','south_ls_id','north_mag','south_mag',
             'reference_kpc','class_label','DEC','shared_support')
    t = read_receipt(OVERLAP,'complete.json','paired.npz',names,inputs)
    south = read_receipt(SOUTH,'complete.json','photometry.npz',
                         ('TYPE','ls_id','magnitude','magnitude_error'),inputs)
    ix = unique_join(t['south_ls_id'],south['ls_id'])
    assert (ix>=0).all() and (t['TYPE']=='PSF').all() and (south['TYPE'][ix]=='PSF').all()
    assert len(np.unique(t['source_id']))==len(t['source_id'])
    np.testing.assert_array_equal(t['south_mag'],south['magnitude'][ix][:,[0,1,3]])
    # The v3.9 pairing archive has already required errors below 0.10 mag on both sides.
    assert np.all(south['magnitude_error'][ix][:,[0,1,3]]<.1)
    n,s = t['north_mag'],t['south_mag']
    x = s[:,0]-s[:,1]
    d = n-s
    diff = np.c_[d[:,0]-d[:,1],d[:,1]-d[:,2],d[:,1]]
    ranges = C['color_edges']
    valid = np.isfinite(diff).all(axis=1)
    use = valid&(x>=ranges[0])&(x<ranges[-1])
    ref=t['reference_kpc']
    native_r=s[:,1]
    cube = [('all','all',valid),('GD-1 color','all',use),('GD-1 color, 5–10 kpc','all',
               use&(ref>=5)&(ref<10)),('GD-1 color, 2–5 kpc','all',use&(ref>=2)&(ref<5))]
    for a,b in zip(ranges,ranges[1:]):
        cube.append(('GD-1 color',f'{a:g} to {b:g}',use&(x>=a)&(x<b)))
        cube.append(('GD-1 color, 5–10 kpc',f'{a:g} to {b:g}',use&(x>=a)&(x<b)&(ref>=5)&(ref<10)))
    for a,b in zip(C['depth_edges'],C['depth_edges'][1:]):
        cube.append(('GD-1 color magnitude',f'{a:g} to {b:g}',use&(native_r>=a)&(native_r<b)))
    rng=np.random.default_rng(C['bootstrap_seed'])
    rows=[]
    for cohort,bin,can in cube:
        values=diff[can]
        for i,key in enumerate(LABELS):
            r=dict(cohort=cohort,bin=bin,quantity=key,n=len(values))
            if len(values):
                y=values[:,i]
                pc = np.quantile(y,(.16,.5,.84))
                r.update(median=pc[1],nmad=1.4826*np.median(abs(y-pc[1])),p16=pc[0],p84=pc[2])
                if len(y)>1:
                    interval = np.median(y[rng.integers(len(y),size=(C['bootstrap_draws'],len(y)))],axis=1)
                    r.update(median_p025=np.quantile(interval,.025),median_p975=np.quantile(interval,.975))
            rows.append(r)
    med = np.median(diff[use],axis=0)
    return rows,med,t,x,diff


def member_photometry(inputs):
    # Jarvis Table 7 photometry lacks per-source PHOTSYS in the public release.
    # Report its agreement separately without treating it as independent southern photometry.
    marker=MEMBERS/'complete.json'
    z=json.loads(marker.read_text())
    t=read(verified(MEMBERS/'members.csv',z['outputs']['members.csv']))
    inputs[str(marker.relative_to(ROOT))]=sha(marker)
    rows=[]
    for g in ('thin','cocoon'):
        q=[r for r in t if (r['group'],r['clean'])==(g,'True') and r['r0'] and r['color']]
        for p,f in [('r0','external_r0'),('color','external_color')]:
            d=np.array([float(r[p])-float(r[f]) for r in q])
            rows.append(dict(group=g,quantity=p,n=len(d),median=np.median(d),within_0p002=int((abs(d)<.002).sum())))
    return rows


def shifted_template(h,delta):
    result=shift(h,(delta[0]/np.diff(COLOR)[0],delta[2]/np.diff(ABS)[0]),
                 order=1,mode='constant',cval=0.,prefilter=False)
    return result/result.sum()


def synthetic(data,offset,inputs):
    fields=('TYPE','ls_id','block','ribbon','tier','r','r0','color','phi1','qrf')
    t=read_receipt(PRIOR/'sample','complete.json','desi.npz',fields,inputs)
    assert (t['TYPE']=='PSF').all()
    good=data['desi_good'].transpose(1,2,0)
    area=data['desi'].reshape(3,NSEG,NSTEP,len(NEW)).transpose(1,2,3,0)
    area=np.where(good[:,None,:,:],area,0.)
    g=table('v4_Figure3_empirical_isochrone.fits')
    seq=np.zeros(len(g),dtype=[('Mr','f8'),('g-r','f8')])
    curve=np.sort(fits.getdata(verified(INPUT/'v4_Figure2_track_jarvis.fits',MODULUS_CURVE),1),order='phi1')
    inputs['reference_track']=MODULUS_CURVE
    ref_cases=(('Published ridge',np.zeros(3)),('North minus South sensitivity',offset),
               ('South minus North sensitivity',-offset))
    holds=(1,2,5)
    ms=[]
    for label,a in ref_cases:
        seq['Mr'],seq['g-r']=g['Mr']+a[2],g['g-r']+a[0]
        values=by_cells(t,NEW,curve,seq)
        for group,(_,n,_) in zip(('all_native','common_qrf'),values):
            for j in holds:
                exp=area[j].sum(axis=0).T
                info=detect(n[j],exp,NEW,[1,2,3,4],[5,6,7,8],(0,1,2))
                if info:
                    stat,_,_=info
                    ms.append(dict(variant=label,group=group,segment=j,observed=int(n[j,:,0].sum()),**stat))
    segment=np.isin(t['block']//NSTEP,holds)
    response=magnitude_response((t['r']-t['r0'])[segment])
    q=segment&t['qrf']
    t=subset(t,q)
    t['qrf']=np.zeros((len(t['ls_id']),len(D)+2),'f4')
    h,q,v,n=arrays(t)
    templates=[cluster_template(name,inputs)[0] for name in ('M13','M92')]
    rows=[]
    draw=np.ones((1,NSTEP))
    for label,a in ref_cases:
        u=np.array([project(shifted_template(h0,a)) for h0 in templates])
        projected=u[None]*response[:,None,None,:]
        for j in holds:
            for depth,k in DEPTHS.items():
                d,z=fitted(h[j,:,:,:k],q[j,:,:,:k],v[j,:,:,:k],area[j,:,:,:k],projected[:k],draw,0)
                blanks=[fitted(h[j,:,:,:k],q[j,:,:,:k],v[j,:,:,:k],area[j,:,:,:k],
                               projected[:k],draw,b) for b in range(5,9)]
                control=np.max(np.abs(np.asarray([b[1][1:,0] for b in blanks if b is not None])),axis=0)
                for ii,name in enumerate(('M13','M92'),1):
                    rows.append(dict(variant=label,method=name,segment=j,depth=depth,psf=int(n[j,:,0,:k].sum()),
                                     distance_kpc=d[ii,0],z=z[ii,0],max_abs_blank_z=control[ii-1],
                                     signal=bool(z[ii,0]>=CLOSE and z[ii,0]>control[ii-1])))
    return ms,rows


CLOSE=3.


def run():
    inputs={'config':sha(CONFIG),'style':check_templates()}
    rows,med,t,x,diff=pairs(inputs)
    data=read_receipt(PRIOR/'sample','complete.json','area.npz',('desi','desi_good'),inputs)
    ms,d=synthetic(data,med,inputs)
    out=DEST/'bands'
    out.mkdir(parents=True,exist_ok=True)
    csv_rows(out/'photometry.csv',rows)
    csv_rows(out/'member_catalog_agreement.csv',member_photometry(inputs))
    csv_rows(out/'ridge_window.csv',ms)
    csv_rows(out/'cmd_sensitivity.csv',d)
    with plt.rc_context(STYLE):
        fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained',sharex=True,sharey=True)
        for k,(a,label) in enumerate(zip(axes,LABELS)):
            a.scatter(x,diff[:,k],s=6,color=BLUE,alpha=.20,rasterized=True)
            a.axhline(0,color=GRAY,ls='--')
            for z in rows:
                if z['cohort']=='GD-1 color' and z['bin']!='all' and z['quantity']==label and z['n']:
                    lo,hi=map(float,z['bin'].split(' to '))
                    a.hlines(z['median'],lo,hi,color=BLUE,lw=2)
            a.set(title=label,xlabel='DECam dereddened g − r (mag)',xlim=(.05,1.2),ylim=(-.15,.15))
        axes[0].set_ylabel('BASS/MzLS minus DECam (mag)')
        save(fig,out,'native_band_offsets',rows)
    figure_check(out/'native_band_offsets.png')
    atomic_npz(out/'paired_offsets.npz',TYPE=t['TYPE'],source_id=t['source_id'],
               south_color=x,difference=diff,reference_kpc=t['reference_kpc'],applied_median=med)
    save_json(out/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,applied_north_minus_south=med.tolist(),
                                     outputs={p.name:sha(p) for p in out.iterdir() if p.name!='complete.json'}))


if __name__=='__main__':
    require_compute()
    run()
