#!/usr/bin/env python3
"""Independent gr geometry across all native imaging, without a depth cut."""
import argparse
import healpy as hp
from photdis.stage1 import require_compute,galactic,tile_bounds
from v3_5_dr11_area import random_memmap,tile_map
from v3_9_seam import BOUNDARY
from sky6_scan import tasks,REJECT,K,regions,domain
from sky6_common import *


def main(side):
    require_compute();d=folder('area')/side;d.mkdir(exist_ok=True);cfg=json.loads((ROOT/C['inputs']['source_roots']).read_text())[side]
    path=Path(cfg['sweep_root']).parent.parent/'randoms'/f'randoms-{side}-1-0.fits';raw,rho,names=random_memmap(path);n=hp.nside2npix(C['nside']);counts=np.zeros((2,n),'i8');sums=np.zeros((7,n));valid=np.zeros((7,n),'i8');domains={}
    tiles=tile_map([dict(filename=Path(p).name,bounds_deg=tile_bounds(p)) for s,p in tasks() if s==side])
    old=STREAM/'full_inventory/raw_to_gr_complete.csv';rows=list(csv.DictReader(old.open()));coef=np.median([[float(r['Ag_per_EBV']),float(r['Ar_per_EBV'])] for r in rows if r['side']==side and r['Ag_per_EBV'] and np.isfinite(float(r['Ag_per_EBV']))],axis=0)
    for start in range(0,len(raw),C['chunk_rows']):
        a=raw[start:start+C['chunk_rows']];t={k:np.asarray(a[v]) for k,v in names.items()};ra=t['RA'];dec=t['DEC'];assert np.isfinite(ra+dec).all() and (abs(dec)<=90).all();lat=galactic(ra,dec)[1];e=t['EBV']
        own=(((dec>=BOUNDARY)&(lat>0))==(side=='north'))&(t['RELEASE']==cfg['release'])&(np.char.strip(t['PHOTSYS'])==(b'N' if side=='north' else b'S'))
        geometry=own&(tiles[(ra%360).astype(int),np.minimum((dec+90).astype(int),179)]>=0)&((t['MASKBITS'].astype('i8')&REJECT)==0)&(t['NOBS_G']>0)&(t['NOBS_R']>0)
        dep=np.column_stack((t['PSFDEPTH_G'],t['PSFDEPTH_R'])).astype(float)
        geometry&=np.isfinite(dep).all(axis=1)&(dep>0).all(axis=1)
        with np.errstate(divide='ignore',invalid='ignore'):
            depth5=22.5-2.5*np.log10(5/np.sqrt(dep));depth0=22.5-2.5*np.log10(K/.1/np.sqrt(dep))-e[:,None]*coef
        pix=hp.ang2pix(C['nside'],ra[geometry],dec[geometry],nest=True,lonlat=True);old_domain=domain(lat,e,depth0)[geometry]
        np.add.at(counts[0],pix,1);np.add.at(counts[1],pix[old_domain],1)
        values=np.column_stack((e,depth5,depth0,e[:,None]*coef))[geometry]
        for j in range(7):
            good=np.isfinite(values[:,j]);np.add.at(sums[j],pix[good],values[good,j]);np.add.at(valid[j],pix[good],1)
        for label,flag in regions(lat,e,depth0).items():domains[label]=domains.get(label,0)+int((geometry&flag).sum())
        if start==0:assert counts[0].sum()==len(pix);print('AREA_INPUT_SMOKE_PASS',side,flush=True)
    values=np.divide(sums,valid,out=np.full_like(sums,np.nan),where=valid>0)
    npz(d/'geometry.npz',counts=counts,area=counts/rho,EBV=values[0],depth5=values[1:3],depth0=values[3:5],extinction=values[5:7],value_counts=valid)
    csv_rows(d/'domains.csv',[dict(domain=k,randoms=v,area_deg2=v/rho) for k,v in domains.items()]);stat=path.stat()
    complete(d,dict(path=str(path),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,coefficient_source=sha(old)),density=rho,extinction_coefficient=coef,area_deg2=float(counts[0].sum()/rho),scope='gr NOBS/positive PSFDEPTH/pixel/native geometry only; no source SNR, morphology, latitude, EBV ceiling, common depth or deblend completeness cut')


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('side',choices=['south','north']);args=a.parse_args();main(args.side)
