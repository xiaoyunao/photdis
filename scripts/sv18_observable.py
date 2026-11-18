#!/usr/bin/env python3
"""Integrate native grz depth, extinction and mask selection over real random points."""
from photdis.stage1 import require_compute, galactic
from sv18_common import *
import paper_v2_filter as old
from v3_5_dr11_area import random_memmap
from v3_9_catalog import settings, OUT as V39
from v3_16_gd1 import MASK, required_depth, P as GEOMETRY, NSTEP
from v3_17_sample import read_track, newer, bounds, NEW
from v3_11_layers import read_receipt


def selected_fraction(limits,edges):
    values=np.sort(limits)
    if not len(values):
        return np.zeros(len(edges)-1)
    left=np.searchsorted(values,edges[:-1],side='right')
    right=np.searchsorted(values,edges[1:],side='right')
    sums=np.r_[0,np.cumsum(values)]
    partial=(sums[right]-sums[left]-edges[:-1]*(right-left))/np.diff(edges)
    return (len(values)-right+partial)/len(values)


def main():
    require_compute()
    d=out('observable');inputs={}
    cfg=settings()['north']
    coefficients=None
    for marker in sorted((V39/'north/reference').glob('[0-9][0-9].json')):
        r=json.loads(marker.read_text())
        for name,h in r['outputs'].items():
            if not name.endswith('.npz'):continue
            t=read_columns(verified(marker.with_name(name),h),('TYPE','EBV','transmission'))
            assert (t['TYPE']=='PSF').all()
            keep=(t['EBV']>.001)&np.isfinite(t['transmission'][:,[0,1,3]]).all(axis=1)
            coefficients=np.median(-2.5*np.log10(t['transmission'][keep][:,[0,1,3]])/t['EBV'][keep,None],axis=0)
            inputs['native_attenuation_reference']=sha(marker)
            break
        if coefficients is not None:break
    assert coefficients is not None and np.isfinite(coefficients).all()
    base=read_receipt(old.AUDIT/'M92','complete.json','native_join.npz',('TYPE','g0','r0','z0','target','low_crowding'),inputs)
    gr=base['g0']-base['r0'];rz=base['r0']-base['z0']
    color=(old.COLOR[1:]+old.COLOR[:-1])/2
    zcurve=np.array([np.median(rz[base['target']&base['low_crowding']&(abs(gr-c)<.10)])
                    if (base['target']&base['low_crowding']&(abs(gr-c)<.10)).any() else np.nan for c in color])
    valid=np.isfinite(zcurve);assert valid.sum()>3
    zcurve=np.interp(color,color[valid],zcurve[valid])
    path=Path(cfg['sweep_root']).parent.parent/'randoms/randoms-north-1-0.fits'
    raw,density,names=random_memmap(path)
    stat=path.stat();nodes=read_track();ra0,dec0,ra1,dec1=bounds(nodes)
    limits={};counts=np.zeros((2,len(P['filter']['segment_indices']),len(NEW)),'i8')
    for start in range(0,len(raw),350000):
        a=raw[start:start+350000]
        ra,dec=a[names['RA']],a[names['DEC']]
        keep=(ra>=ra0)&(ra<ra1)&(dec>=max(dec0,GEOMETRY['north_declination_min']))&(dec<dec1)
        if not keep.any():continue
        a,ra,dec=a[keep],ra[keep],dec[keep]
        _,_,block,ribbon,ok=newer(ra,dec,nodes)
        ebv=a[names['EBV']]
        ok&=(galactic(ra,dec)[1]>30)&(a[names['RELEASE']]==cfg['release'])
        ok&=(ebv>=0)&(ebv<.03)&((a[names['MASKBITS']].astype('i8')&MASK)==0)
        ok&=(np.char.strip(a[names['PHOTSYS']])==b'N')
        depth=np.column_stack([a[names['PSFDEPTH_'+b]] for b in 'GRZ'])
        ok&=np.logical_and.reduce([a[names['NOBS_'+b]]>0 for b in 'GRZ'])
        limiting=22.5-2.5*np.log10((2.5/np.log(10))/.1/np.sqrt(np.maximum(depth,1e-30)))
        extinction=ebv[:,None]*coefficients
        for k,rlim in enumerate(P['filter']['r_limits']):
            depth_ok=(depth>=required_depth()[k]).all(axis=1)
            for si,segment in enumerate(P['filter']['segment_indices']):
                for rib in range(len(NEW)):
                    use=ok&depth_ok&(block//NSTEP==segment)&(ribbon==rib)
                    n=int(use.sum())
                    if not n:continue
                    counts[k,si,rib]+=n
                    light=limiting[use]-extinction[use]
                    for ci,c in enumerate(color):
                        bound=np.minimum.reduce([rlim-extinction[use,1],light[:,0]-c,
                                                   light[:,1],light[:,2]+zcurve[ci]])
                        limits.setdefault((k,si,rib,ci),[]).append(bound)
    response=np.zeros((*counts.shape,len(color),len(old.APP)-1))
    for key,values in limits.items():
        response[key]=selected_fraction(np.concatenate(values),old.APP)
    # Match archived input color and r0 admission exactly.
    cfrac=np.clip((old.COLOR[1:]+.12)/np.diff(old.COLOR),0,1)
    rfrac=np.clip((old.APP[1:]-14)/np.diff(old.APP),0,1)
    response*=cfrac[None,None,None,:,None]*rfrac[None,None,None,None,:]
    atomic_npz(d/'selection.npz',response=response,area=counts/density,color=old.COLOR,
               apparent=old.APP,rz_curve=zcurve,extinction_coefficients=coefficients)
    csv_rows(d/'random_coverage.csv',[dict(r_limit=rlim,segment=seg,ribbon=int(rib),
          randoms=int(counts[k,si,rib]),area_deg2=counts[k,si,rib]/density)
          for k,rlim in enumerate(P['filter']['r_limits'])
          for si,seg in enumerate(P['filter']['segment_indices']) for rib in range(len(NEW))])
    m5=read_receipt(old.AUDIT/'M5','complete.json','native_join.npz',
          ('TYPE','target','member','EBV','r','low_crowding','cluster_mask'),inputs)
    m5counts=[]
    for ebv in (.03,.06,.10):
        eligible=m5['target']&m5['low_crowding']&~m5['cluster_mask']&(m5['r']<20)&(m5['EBV']<ebv)
        m5counts.append(dict(ebv_limit=ebv,native_target=int(eligible.sum()),members=int((eligible&m5['member']).sum()),
              minimum_target_ebv=float(m5['EBV'][m5['target']].min()),
              selection='existing PSF low-crowding unmasked target; broadening EBV changes experimental domain'))
    csv_rows(d/'M5_control_reachability.csv',m5counts)
    inputs['native_randoms']=dict(path=str(path),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)
    assert (path.stat().st_size,path.stat().st_mtime_ns)==(stat.st_size,stat.st_mtime_ns)
    complete(d,inputs,depth_model='PSFDEPTH Gaussian 0.10-mag flux-error threshold, all native random points',
       color_model='M92 median rz at fixed gr; missing templates interpolate only within empirical sequence',
       selection_scope='conditional on frozen PSF morphology, strict deblending and crowding sample')


if __name__=='__main__':
    main()
