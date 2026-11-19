#!/usr/bin/env python3
"""Keep the actual DR11 Pal5 special-detection circle separate from tail tests."""
from astropy.io import fits
import fitsio
from photdis.stage1 import require_compute,legacy_ids
from stream_common import *
from v3_5_dr11_area import random_memmap


def sep(ra,dec,c):
    r,d=np.deg2rad(ra),np.deg2rad(dec);r0,d0=np.deg2rad([c['ra'],c['dec']])
    return np.rad2deg(np.arctan2(np.hypot(np.cos(d)*np.sin(r-r0),np.cos(d0)*np.sin(d)-np.sin(d0)*np.cos(d)*np.cos(r-r0)),np.sin(d0)*np.sin(d)+np.cos(d0)*np.cos(d)*np.cos(r-r0)))


def main():
    require_compute();d=folder('core');p=ROOT/'configs/stream_inputs/pal5_dr11_core.json';c=json.loads(p.read_text());inputs={str(p.relative_to(ROOT)):sha(p)}
    cfg=json.loads((ROOT/'configs/v3_9_catalog.json').read_text())['south'];root=Path(cfg['sweep_root']);pieces=[];raw_n=0;nonpsf=0;missing=0
    for name in ('sweep-225m005-230p000.fits','sweep-225p000-230p005.fits'):
        path=root/name
        with fits.open(path,memmap=True) as h:
            a=h[1].data;at=np.flatnonzero(sep(a['RA'],a['DEC'],c)<.25)
        with fitsio.FITS(str(path)) as f:
            a=f[1].read(rows=at)
        raw_n+=len(a);typ=np.char.strip(a['TYPE'].astype('U'));nonpsf+=int(((typ!='PSF')&(typ!='')).sum());missing+=int((typ=='').sum());a=a[typ=='PSF'];separation=sep(a['RA'],a['DEC'],c)
        flux=np.column_stack([a['FLUX_'+b] for b in 'GRZ']);iv=np.column_stack([a['FLUX_IVAR_'+b] for b in 'GRZ'])
        with np.errstate(divide='ignore',invalid='ignore'):err=2.5/np.log(10)/flux/np.sqrt(iv)
        pieces.append(dict(TYPE=np.full(len(a),'PSF'),ls_id=legacy_ids(a),RA=a['RA'],DEC=a['DEC'],sep=separation,source_id=np.where(np.char.strip(a['REF_CAT'].astype('U'))=='G3',a['REF_ID'],0),mask=a['MASKBITS'].astype('i8'),err=err,EBV=a['EBV']))
        inputs[name]=dict(bytes=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns)
    t={k:np.concatenate([x[k] for x in pieces]) for k in pieces[0]};assert len(np.unique(t['ls_id']))==len(t['ls_id']);rows=[]
    for label,use in (('actual_cluster_circle',t['sep']<c['radius']),('adjacent_0.13_0.25deg',(t['sep']>=.13)&(t['sep']<.25))):
        rows.append(dict(region=label,PSF=int(use.sum()),gaia_reference=int((use&(t['source_id']>0)).sum()),cluster_bit13=int((use&((t['mask']&(1<<13))!=0)).sum()),gr_err_lt01=int((use&(t['err'][:,:2]>0).all(axis=1)&(t['err'][:,:2]<.1).all(axis=1)).sum()),median_EBV=float(np.median(t['EBV'][use]))))
    csv_rows(d/'actual_core_counts.csv',rows);npz(d/'core_PSF.npz',**t)
    save_json(d/'admission.json',dict(raw_local=raw_n,non_PSF=nonpsf,missing_TYPE=missing,PSF=len(t['ls_id'])))
    path=root.parent.parent/'randoms/randoms-south-1-0.fits';a,density,names=random_memmap(path);rn=np.zeros(2,int)
    for start in range(0,len(a),500000):
        z=a[start:start+500000];ra=z[names['RA']];dec=z[names['DEC']];local=(abs(ra-c['ra'])<.3)&(abs(dec-c['dec'])<.3)
        q=z[local];distance=sep(q[names['RA']],q[names['DEC']],c);covered=(q[names['NOBS_G']]>0)&(q[names['NOBS_R']]>0)
        rn[0]+=int(((distance<c['radius'])&covered).sum());rn[1]+=int(((distance>=.13)&(distance<.25)&covered).sum())
    csv_rows(d/'actual_core_random_area.csv',[dict(region=x,n_random=int(n),density=density,area_deg2=n/density,area_poisson_sigma=np.sqrt(n)/density) for x,n in zip(('actual_cluster_circle','adjacent_0.13_0.25deg'),rn)])
    complete(d,inputs,actual_mask=c,scope='special detection circle is not an independent star-count completeness test; registered Pal5_core track-strip name is historical and does not denote this circle; neither enters recovery criterion',
             interpretation='DR11 turns detection off inside circle, retaining Gaia forced photometry; source counts are conditioned on Gaia catalog in this region')


if __name__=='__main__':main()
