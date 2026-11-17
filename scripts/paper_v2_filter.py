#!/usr/bin/env python3
"""Evaluate an empirical native CMD matched filter on the two frozen control fields."""
from pathlib import Path
import csv
import json
import os

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import norm

from photdis.stage1 import galactic, require_compute
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_11_layers import read_receipt
from v3_13_clusters import SYSTEMS, native_config, zones
from v3_13_cmd import M5_CENTER,M5_D
from v3_14_audit import DEST as AUDIT, MASK
from v3_17_sample import DEST as GD, NEW
from v3_16_gd1 import EDGES, NSTEP
from v3_5_dr11_area import BANDS, random_memmap
from photdis.photometry import CLEAN_REJECT_MASKBITS
from paper_v2_scan import OUT

CONFIG=Path(__file__).with_name('paper_v2_filter.json')
if not CONFIG.exists():
    CONFIG=ROOT/'configs/paper_v2_filter.json'
P=json.loads(CONFIG.read_text())
def grid(spec):
    return np.arange(spec[0],spec[1]+spec[2]*.01,spec[2])
MU,COLOR,APP,ABS=[grid(P[k]) for k in ('mu_grid','color_limits','apparent_limits','absolute_limits')]
D=10**((MU-10)/5)


def template(name,inp):
    t=read_receipt(AUDIT/name,'complete.json','native_join.npz',
        ('TYPE','g0','r0','r','target','member','low_crowding','cluster_mask','training_or_calibration'),inp)
    assert np.all(t['TYPE']=='PSF')
    clean=t['low_crowding']&~t['cluster_mask']&~t['training_or_calibration']&(t['r']<22)
    c=t['g0']-t['r0']; m=t['r0']-10-5*np.log10(P['template_distance_kpc'][name])
    def hist(mask):
        return np.histogram2d(c[mask],m[mask],(COLOR,ABS))[0]
    if name=='M30':
        ok=clean&t['target']&t['member']
        h=hist(ok)
        meta=dict(method='members',target=int(ok.sum()))
    else:
        # This field's member-only strict subset truncates the lower main sequence.
        # Reuse the independently measured on/off CMD construction of v3.16.
        folder=AUDIT/name
        record=json.loads((folder/'complete.json').read_text())
        file=verified(folder/'area.csv',record['outputs']['area.csv'])
        with file.open() as f:
            area=[x for x in csv.DictReader(f) if x['limiting_r']=='22' and x['masked']=='False']
        a,b=(sum(float(x['area_deg2']) for x in area if x['zone']==z) for z in ('target','background'))
        assert a>0 and b>0
        on,off=clean&t['target'],clean&~t['target']
        h=np.maximum(hist(on)-(a/b)*hist(off),0)
        meta=dict(method='target_minus_background',target=int(on.sum()),control=int(off.sum()),
                  target_area=a,control_area=b,strict_members=int((on&t['member']).sum()))
    if h.sum()<20:
        raise ValueError(f'{name} has inadequate native template support')
    return h,meta


def projected(h,width):
    # Color measurement includes g-r, whose errors combine in quadrature.
    w=np.hypot(width,.05)
    hs=gaussian_filter(h,(np.sqrt(2)*w/np.diff(COLOR)[0],w/np.diff(ABS)[0]))
    a=(APP[1:]+APP[:-1])*.5
    b=(ABS[1:]+ABS[:-1])*.5
    signal=np.array([[np.interp(a-m,b,c,left=0,right=0) for c in hs] for m in MU]).reshape(len(MU),-1)
    return np.divide(signal,signal.sum(axis=1)[:,None],out=np.zeros_like(signal),
                     where=signal.sum(axis=1)[:,None]>0)


def az(ra,dec):
    x=(ra-M5_CENTER[0])*np.cos(np.deg2rad(M5_CENTER[1]))
    y=dec-M5_CENTER[1]
    return ((np.arctan2(y,x)+np.pi)*2/np.pi).astype(int)%4


def m5_area(inp):
    SYSTEMS['M5']=M5_CENTER,M5_D
    _,cfg,_=native_config('M5')
    p=Path(cfg['sweep_root']).parent.parent/'randoms/randoms-south-1-0.fits'
    raw,density,keys=random_memmap(p)
    a=np.zeros((2,5),'i8')
    start=p.stat()
    reject=MASK|sum(1<<b for b in CLEAN_REJECT_MASKBITS)
    for i in range(0,len(raw),300000):
        z=raw[i:i+300000]
        ra,dec=z[keys['RA']],z[keys['DEC']]
        j=(abs(ra-M5_CENTER[0])<2)&(abs(dec-M5_CENTER[1])<1.25)
        if not j.any(): continue
        z,ra,dec=z[j],ra[j],dec[j]
        t,b=zones('M5',ra,dec)
        k=np.where(t,0,1+az(ra,dec))
        ok=(t|b)&(z[keys['EBV']]>=0)&(z[keys['EBV']]<P['ebv_limit'])
        ok&=(z[keys['RELEASE']]==cfg['release'])&(np.char.strip(z[keys['PHOTSYS']])==b'S')
        ok&=(abs(galactic(ra,dec)[1])>30)&((z[keys['MASKBITS']].astype('i8')&reject)==0)
        ok&=np.logical_and.reduce([z[keys[f'NOBS_{v}']]>0 for v in BANDS])
        d=np.minimum.reduce([z[keys[f'PSFDEPTH_{v}']] for v in BANDS])
        for l,r in enumerate(P['r_limits']):
            floor=((2.5/np.log(10))/.1/10**((22.5-r)/2.5))**2
            a[l]+=np.bincount(k[ok&(d>=floor)],minlength=5)
    stop=p.stat()
    assert (start.st_size,start.st_mtime_ns)==(stop.st_size,stop.st_mtime_ns)
    inp['M5_random']=dict(bytes=start.st_size,mtime_ns=start.st_mtime_ns)
    return a/density


def m5(inp):
    p=read_receipt(AUDIT/'M5','complete.json','native_join.npz',
      ('TYPE','RA','DEC','EBV','r','g0','r0','target','low_crowding','cluster_mask',
       'training_or_calibration','qrf_mass','error_grz'),inp)
    assert np.all(p['TYPE']=='PSF')
    a=m5_area(inp)
    color=p['g0']-p['r0']
    zone=np.where(p['target'],0,1+az(p['RA'],p['DEC']))
    keep=(p['EBV']>=0)&(p['EBV']<P['ebv_limit'])&(abs(galactic(p['RA'],p['DEC'])[1])>30)
    keep&=~p['cluster_mask']&p['low_crowding']&~p['training_or_calibration']
    keep &= p['error_grz'].max(axis=1)<P['maximum_measurement_error_mag']
    result=[]
    for k,r in enumerate(P['r_limits']):
        n=[]; q=[];h=[]
        for i in range(5):
            j=keep&(p['r']<r)&(zone==i)
            n.append(int(j.sum()))
            q.append(int((p['qrf_mass'][j].sum(axis=1)>0).sum()))
            h.append(np.histogram2d(color[j],p['r0'][j],(COLOR,APP))[0].ravel())
        result.append((np.array(h),np.array(n),np.array(q),a[k],(1,3),(2,4)))
    return result


def gd1(inp):
    p=read_receipt(GD/'sample','complete.json','desi.npz',('TYPE','r','r0','color','tier','block','ribbon','qrf'),inp)
    a=read_receipt(GD/'sample','complete.json','area.npz',('desi','desi_good'),inp)
    assert np.all(p['TYPE']=='PSF')
    dims=(len(EDGES)-1,len(NEW))
    result=[]
    for k,r in enumerate(P['r_limits']):
        areas_by_depth=a['desi'][:k+1].reshape(k+1,*dims[:1],NSTEP,dims[1]).sum(axis=2)
        valid=a['desi_good'][:k+1].all(axis=0)
        # Restrict to the same frozen segments for every included ribbon, tier and method.
        ar=np.flatnonzero(np.isclose(NEW,0.)|np.isin(np.round(NEW,2),(-5.8,-4.8,4.8,5.8,-7.8,-6.8,6.8,7.8)))
        controls=np.flatnonzero(np.isin(np.round(NEW[ar],2),(-5.8,-4.8,4.8,5.8)))
        blanks=np.flatnonzero(np.isin(np.round(NEW[ar],2),(-7.8,-6.8,6.8,7.8)))
        # Archived depth masks are nested. Keep only segments whose effective
        # area agrees at every included depth, then count each sky area once.
        common_area=(areas_by_depth==areas_by_depth[-1:]).all(axis=0)
        keep_seg=(valid[:,ar]&common_area[:,ar]).all(axis=1)
        m=(p['r']<r)&keep_seg[p['block']//NSTEP]
        h=[];n=[];q=[]
        for i in ar:
            j=m&(p['ribbon']==i)
            n.append(int(j.sum()));q.append(int((j&p['qrf']).sum()))
            h.append(np.histogram2d(p['color'][j],p['r0'][j],(COLOR,APP))[0].ravel())
        area=areas_by_depth[-1,keep_seg][:,ar].sum(axis=0)
        result.append((np.array(h),np.array(n),np.array(q),area,controls,blanks))
    return result


def evaluate(fields,templates):
    rows=[];build={}
    trials=len(fields)*len(MU)*len(P['template_widths_mag'])*len(P['r_limits'])
    for name,runs in fields.items():
        for depth,(h,n,q,area,control,blanks) in zip(P['r_limits'],runs):
            for i,wt in enumerate(P['template_widths_mag']):
                sig=projected(templates[name],wt)
                for k in (0,*blanks):
                    bg=np.asarray(control)
                    valid=bool(area[k]>0 and len(bg)>=2 and (area[bg]>0).all() and h[k].sum()>0)
                    if valid:
                        scale=area[k]/area[bg].sum()
                        def count_z(v):
                            variance=v[k]+scale*scale*v[bg].sum()
                            return (v[k]-scale*v[bg].sum())/np.sqrt(variance) if variance>0 else np.nan
                        zraw,zqrf=count_z(n),count_z(q)
                        back=h[bg].sum(axis=0)*scale
                        diff=h[k]-back
                        var=np.maximum(h[k]+scale*back,4.)
                        d=((sig*sig)/var).sum(axis=1)
                        z=np.divide((sig*(diff/var)).sum(axis=1),np.sqrt(d),
                                   out=np.full(len(d),np.nan),where=d>0)
                        finite=np.isfinite(z)
                        j=np.argmax(np.where(finite,z,-np.inf))
                        zmax=z[j]
                        corrected=min(1.,norm.sf(zmax)*trials)
                    else:
                        scale=zmax=corrected=zraw=zqrf=np.nan
                        z=np.full(len(MU),np.nan)
                        j=0
                    label=f'{name}_{depth:g}_{i}_{k}'
                    rows.append(dict(system=name,r_limit=depth,width=wt,field=('target' if k==0 else 'blank'),
                       field_index=int(k),raw_n=int(n[k]),qrf_supported_n=int(q[k]),area_deg2=area[k],
                       bg_area_deg2=area[bg].sum(),area_scale=scale,valid=valid,raw_count_z=zraw,qrf_count_z=zqrf,z_max=zmax,
                       best_distance_kpc=D[j] if valid else None,family_p=corrected,key=label))
                    build[label]=z
    return rows,build,trials


def decide(rows,trials):
    decisions=[]
    for system in P['validation']:
        s=[r for r in rows if r['system']==system]
        target=[r for r in s if r['field']=='target']
        blanks=[r for r in s if r['field']=='blank' and r['valid']]
        # An archival fit can indicate positive detections. A formal tail claim also
        # requires independent blank areas and spatial replication for each variant.
        pcrit=norm.sf(P['significance'])
        levels={(r['r_limit'],r['width']):r for r in target}
        complete=len(levels)==4 and all(r['valid'] for r in levels.values())
        repeats=complete and all(r['family_p']<pcrit for r in levels.values())
        v=[r['best_distance_kpc'] for r in target if r['valid']]
        stable=bool(v) and max(v)/min(v)<1.15
        control_ok=[]
        for r in target:
            z=[abs(x['z_max']) for x in blanks
               if x['r_limit']==r['r_limit'] and x['width']==r['width']]
            control_ok.append(bool(z) and r['z_max']>max(z))
        above_blanks=complete and all(control_ok)
        decisions.append(dict(system=system,observed_variants=len(v),blank_variants=len(blanks),
              repeat_trial_adjusted=bool(repeats),stable_support=bool(stable),
              above_blanks=bool(above_blanks),passed=bool(repeats and above_blanks and stable)))
    return decisions


def main():
    require_compute()
    out=OUT/'filter'
    out.mkdir(exist_ok=True,parents=True)
    inputs={'preregistered':sha(CONFIG),'implementation':sha(Path(__file__))}
    a,m=template('M30',inputs)
    b,n=template('M92',inputs)
    fields={'M5':m5(inputs),'GD-1':gd1(inputs)}
    rows,curves,trials=evaluate(fields,{'M5':a,'GD-1':b})
    result=decide(rows,trials)
    csv_rows(out/'control_scores.csv',rows)
    csv_rows(out/'control_decisions.csv',result)
    atomic_npz(out/'filter_grid.npz',mu=MU,color=COLOR,apparent=APP,absolute=ABS,
               M30=a,M92=b,**curves)
    save_json(out/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,trials=trials,
             templates=dict(M30=m,M92=n),passed=any(x['passed'] for x in result),
             outputs={p.name:sha(p) for p in out.iterdir() if p.name!='complete.json'}))


if __name__=='__main__':
    main()
