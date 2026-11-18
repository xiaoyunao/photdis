#!/usr/bin/env python3
"""Accumulate independent native randoms directly at Nside512."""
import argparse
import healpy as hp
from photdis.stage1 import galactic, require_compute
from photdis.photometry import CLEAN_REJECT_MASKBITS
from halo_common import *
from v3_5_dr11_area import random_memmap, tile_map
from v3_9_catalog import OUT as V39, settings
from v3_9_seam import BOUNDARY


def run(side):
    cfg=settings()[side]
    path=Path(cfg['sweep_root']).parent.parent/'randoms'/f'randoms-{side}-1-0.fits'
    stat=path.stat()
    raw,density,names=random_memmap(path)
    inventory=V39/side/'inventory.json'
    files=[s for group in json.loads(inventory.read_text())['groups'] for s in group if not s.get('error')]
    tiles=tile_map(files)
    npix=hp.nside2npix(C['nside'])
    counts=np.zeros((3,npix),'i8')
    depth_sum=np.zeros((3,npix))
    ebv_sum=np.zeros(npix)
    reject=sum(1<<bit for bit in CLEAN_REJECT_MASKBITS)
    limit=10**((22.5-20)/2.5)
    depth_threshold=(2.5/np.log(10)/.1/limit)**2
    for start in range(0,len(raw),300000):
        a=raw[start:start+300000]
        t={key:a[value] for key,value in names.items()}
        _,lat=galactic(t['RA'],t['DEC'])
        own=((t['DEC']>=BOUNDARY)&(lat>0))==(side=='north')
        geo=own&(abs(lat)>30)&(t['EBV']>=0)&(t['EBV']<.03)
        geo&=tiles[(t['RA']%360).astype(int),np.minimum((t['DEC']+90).astype(int),179)]>=0
        geo&=(t['RELEASE']==cfg['release'])&(np.char.strip(t['PHOTSYS'])==(b'N' if side=='north' else b'S'))
        geo&=np.logical_and.reduce([t['NOBS_'+b]>0 for b in 'GRZ'])
        unmasked=geo&((t['MASKBITS'].astype('i8')&reject)==0)
        depth=np.column_stack([t['PSFDEPTH_'+b] for b in 'GRZ'])
        selected=unmasked&(depth>=depth_threshold).all(axis=1)
        pix=hp.ang2pix(C['nside'],t['RA'],t['DEC'],lonlat=True,nest=True)
        for row,keep in enumerate((geo,unmasked,selected)):
            counts[row]+=np.bincount(pix[keep],minlength=npix)
        j=np.flatnonzero(selected)
        if len(j):
            limits=22.5-2.5*np.log10(2.5/np.log(10)/.1/np.sqrt(depth[j]))
            for b in range(3):depth_sum[b]+=np.bincount(pix[j],weights=limits[:,b],minlength=npix)
            ebv_sum+=np.bincount(pix[j],weights=t['EBV'][j],minlength=npix)
    after=path.stat()
    assert (stat.st_size,stat.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    mean=np.divide(depth_sum,counts[2],out=np.full_like(depth_sum,np.nan),where=counts[2]>0)
    ebv=np.divide(ebv_sum,counts[2],out=np.full_like(ebv_sum,np.nan),where=counts[2]>0)
    d=folder('area')/side;d.mkdir(exist_ok=True)
    atomic_npz(d/'randoms.npz',counts=counts,area=counts/density,depth_mag=mean,EBV=ebv)
    csv_rows(d/'area_totals.csv',[dict(native=side,stage=label,randoms=int(n.sum()),area_deg2=n.sum()/density)
        for label,n in zip(('geometric','unmasked','depth_at_least_20mag'),counts)])
    complete(d,dict(inventory=sha(inventory),random_path=str(path),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns),
        density=density,nside=C['nside'],ordering='NESTED',coordsys='C',minimum_depth_mag=20.,
        selection='native unique geometric ownership, |b|>30, 0<=EBV<0.03, clean mask, positive grz observations and 0.10mag sensitivity at 20mag',
        area_scope='shared geometric/depth area for bright and deep actual strict-photometry selections; faint visibility supplied separately')


if __name__=='__main__':
    require_compute()
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('side',choices=('south','north'))
    run(parser.parse_args().side)
