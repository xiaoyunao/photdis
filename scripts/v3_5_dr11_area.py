#!/usr/bin/env python3
"""Measure native sweep coverage with the DR11 survey random catalog."""
from collections import Counter
import json
import os
from pathlib import Path

import healpy as hp
import numpy as np

from photdis.photometry import CLEAN_REJECT_MASKBITS
from photdis.stage1 import galactic, require_compute
from v2_1_near import atomic_npz, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_5_dr11_inventory import CONFIG, OUT

NSIDE = 16
AREA = OUT/'area'
RANDOM_FILE = 'randoms-south-1-0.fits'
BANDS = ('G', 'R', 'Z')


def tile_map(sweeps):
    """A one-degree geometry lookup for the integer-edged native sweep tiles."""
    ids = np.full((360,180),-1,dtype='i2')
    for i,s in enumerate(sweeps):
        bounds = np.asarray(s['bounds_deg'])
        if not np.equal(bounds,np.rint(bounds)).all():
            raise ValueError('Coverage lookup requires integer-edged sweep bounds')
        r0,d0,r1,d1 = bounds.astype(int)
        ra = np.arange(r0, r0+((r1-r0)%360 or 360))%360
        dec = np.arange(d0,d1)+90
        ix = np.ix_(ra,dec)
        if (ids[ix]>=0).any():
            raise ValueError(f'Sweep geometry overlaps at {s["filename"]}')
        ids[ix] = i
    return ids


def random_memmap(path):
    from astropy.io import fits

    with fits.open(path,memmap=True) as h:
        t = h[1]
        info, header, dtype = t.fileinfo(), t.header, t.data.dtype
        n, rowlength = header['NAXIS2'], header['NAXIS1']
        density = float(header['DENSITY'])
        if rowlength != dtype.itemsize or density <= 0 or any(dtype[k].hasobject for k in dtype.names):
            raise ValueError('Unusable random catalog metadata')
        names = {k.upper():k for k in dtype.names}
        cols = ['RELEASE','RA','DEC','EBV','PHOTSYS','MASKBITS'] + [
            f'{prefix}_{b}' for prefix in ('NOBS','PSFDEPTH') for b in BANDS]
        required = {k:names[k] for k in cols}
        for k in required.values():
            j = dtype.names.index(k)+1
            if header.get(f'TSCAL{j}',1)!=1 or header.get(f'TZERO{j}',0)!=0:
                raise ValueError(f'Gridded coverage requires decoded {k}')
    return np.memmap(path,dtype=dtype,mode='r',offset=info['datLoc'],shape=(n,)), density, required


def main():
    require_compute()
    marker = OUT/'inventory'/'complete.json'
    inv = json.loads(marker.read_text())
    cfg = json.loads(verified(CONFIG,inv['config_sha256']).read_text())
    source, = [x for x in inv['random_files'] if x['filename']==RANDOM_FILE]
    path = Path(cfg['random_root'])/RANDOM_FILE
    stat = path.stat()
    if (stat.st_size,stat.st_mtime_ns)!=(source['size'],source['mtime_ns']):
        raise ValueError('Random catalog changed after input inventory')
    raw,density,names = random_memmap(path)
    sweeps = inv['sweeps']
    lookup = tile_map(sweeps)
    tiles = np.zeros((len(sweeps),4),dtype='i8')
    npix = hp.nside2npix(NSIDE)
    pixels = np.zeros((len(sweeps),npix,2),dtype='i4')
    releases = Counter()
    reject = sum(1<<bit for bit in CLEAN_REJECT_MASKBITS)
    # PSFDEPTH_R is inverse variance in native flux. The dimensional r=20
    # threshold approximates a 0.1-mag precision at that flux.
    flux_limit = 10**((22.5-cfg['maximum_observed_r'])/2.5)
    depth_limit = (2.5/np.log(10)/cfg['maximum_grz_error_mag']/flux_limit)**2
    for offset in range(0,len(raw),300_000):
        a = raw[offset:offset+300_000]
        t = {key:a[name] for key,name in names.items()}
        ra, dec, dust = t['RA'],t['DEC'],t['EBV']
        if (~np.isfinite(ra) | ~np.isfinite(dec) | (abs(dec)>90)).any():
            raise ValueError('Invalid native random coordinate')
        tile = lookup[(ra%360).astype(int),np.minimum((dec+90).astype(int),179)]
        lat = galactic(ra,dec)[1]
        footprint = (tile>=0) & (abs(lat)>cfg['minimum_abs_galactic_latitude']) & (
            (dust>=0) & (dust<cfg['maximum_ebv']))
        native = footprint & (t['RELEASE']==cfg['native_release']) & (np.char.strip(t['PHOTSYS'])==b'S')
        coverage = native & np.logical_and.reduce([(t[f'NOBS_{b}']>0) for b in BANDS])
        good = coverage & ((t['MASKBITS'].astype('i8')&reject)==0)
        deep = good & (t['PSFDEPTH_R']>=depth_limit)
        k,n = np.unique(t['RELEASE'][footprint],return_counts=True)
        releases.update(dict(zip(k.tolist(),n.tolist())))
        for j,m in enumerate((footprint,coverage,good,deep)):
            tiles[:,j] += np.bincount(tile[m],minlength=len(sweeps))
        i = tile[good]
        h = hp.ang2pix(NSIDE,ra[good],dec[good],lonlat=True,nest=True)
        np.add.at(pixels[:,:,0],(i,h),1)
        np.add.at(pixels[:,:,1],(i[deep[good]],h[deep[good]]),1)
    if not tiles[:,2].sum():
        raise ValueError(f'No eligible native random points. Release breakdown: {releases}')
    np.testing.assert_array_equal(pixels.sum(axis=1),tiles[:,2:])
    AREA.mkdir(parents=True,exist_ok=True)
    names_out = ('coordinate_dust_deg2','grz_imaged_deg2','mask_cleared_deg2','r20_depth_deg2')
    csv_rows(AREA/'by_sweep.csv',[dict(sweep=s['filename'],**dict(zip(names_out,x/density)))
                                 for s,x in zip(sweeps,tiles)])
    i,h = np.nonzero(pixels[:,:,0])
    atomic_npz(AREA/'sweep_healpix.npz',sweep=i,pixel=h,area_deg2=pixels[i,h,0]/density,
               area_r20_deg2=pixels[i,h,1]/density)
    r = dict(job=os.getenv('SLURM_JOB_ID'),source=source,random_rows=len(raw),density_deg2=density,
             healpix_nside=NSIDE,healpix_order='nested',inventory_sha256=sha(marker),
             coordinate_dust_by_release=releases,totals=dict(zip(names_out,tiles.sum(axis=0)/density)),
             semantics='Area traces DECam imaging, foreground dust, hard image masks and a separately labeled r-band depth proxy. Object-specific flux and deblend selection are not encoded in the area.',
             outputs={f.name:sha(f) for f in sorted(AREA.iterdir()) if f.name!='complete.json'})
    save_json(AREA/'complete.json',r)
    print(r['totals'],flush=True)


if __name__ == '__main__':
    main()
