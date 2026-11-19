#!/usr/bin/env python3
"""All available native gr imaging; source and geometric selection kept separate."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
from functools import lru_cache
from astropy.io import fits
import healpy as hp
from photdis.stage1 import require_compute,galactic,legacy_ids,tile_bounds
from v3_9_seam import BOUNDARY
from sky6_common import *

NPIX=hp.nside2npix(C['nside']);K=2.5/np.log(10)
REJECT=sum(1<<b for b in C['gr']['reject_maskbits'])


def tasks():
    cfg=json.loads((ROOT/C['inputs']['source_roots']).read_text())
    return [(side,str(p)) for side in ('south','north') for p in sorted(Path(cfg[side]['sweep_root']).glob('sweep-*.fits'))]


@lru_cache(maxsize=1)
def fixed():
    with np.load(ROOT/C['inputs']['ownership']/'ownership_overrides.npz') as z:seam={k:z[k] for k in ('ls_id','primary_survey_owner')}
    with np.load(OUT/'ridge_inputs/roster.npz') as z:members=z['ls_id']
    return seam,members


def owner(side,ra,dec,lat,release,ids):
    own=(((dec>=BOUNDARY)&(lat>0))==(side=='north'))&(release==(11011 if side=='north' else 11010))
    seam,_=fixed();j=np.searchsorted(seam['ls_id'],ids);k=np.flatnonzero(j<len(seam['ls_id']));k=k[seam['ls_id'][j[k]]==ids[k]];own[k]=seam['primary_survey_owner'][j[k]]
    return own


def domain(lat,ebv,depth):return (abs(lat)>30)&(ebv>=0)&(ebv<.1)&(depth>=23).all(axis=1)


def entry(flux,ivar,tr,nobs,mask,fracmasked,fracin):
    valid=np.isfinite(flux).all(axis=1)&np.isfinite(ivar).all(axis=1)&np.isfinite(tr).all(axis=1)&(flux>0).all(axis=1)&(ivar>0).all(axis=1)&(tr>0).all(axis=1)
    geom=(nobs>0).all(axis=1)&((mask&REJECT)==0)
    with np.errstate(divide='ignore',invalid='ignore'):
        mag=22.5-2.5*np.log10(flux/tr);snr=flux*np.sqrt(ivar);err=K/snr;ext=-2.5*np.log10(tr)
    use=geom&valid&(snr>=5).all(axis=1)&(fracmasked<.4).all(axis=1)&(fracin>.7).all(axis=1)&(mag<23).all(axis=1)
    return use,geom,mag,err,ext


def regions(lat,e,depth):
    return dict(all=np.ones(len(lat),bool),low_latitude=abs(lat)<=30,high_EBV=e>=.1,shallower_23=(depth<23).any(axis=1),old_domain=domain(lat,e,depth))


def scan_one(task):
    side,filename=task;path=Path(filename);d=folder('cache')/(side+'_'+path.stem);d.mkdir(exist_ok=True);marker=d/'complete.json';code=sha(Path(__file__))
    if marker.exists():
        r=json.loads(marker.read_text());assert r['config']==CH and r['scan_code']==code
        return dict(side=side,file=path.name,**r['counts'])
    stat=path.stat();counts={k:0 for k in ('raw','missing_TYPE','non_PSF','PSF','owned','gr_geometry','valid_gr','strict_companion','legacy_entrance')};parts=[];owned=[];members=[];maps=np.zeros((2,NPIX),'i8');reg=[];seam,mid=fixed();coeff=[]
    with fits.open(path,memmap=True) as h:
        raw=h[1].data;names=set(raw.dtype.names)
        if 'TYPE' not in names:raise ValueError(('missing TYPE column, fail admission',filename,len(raw)))
        for start in range(0,len(raw),C['chunk_rows']):
            a=raw[start:start+C['chunk_rows']];typ=np.char.strip(a['TYPE'].astype('U'));counts['raw']+=len(a);counts['missing_TYPE']+=int((typ=='').sum());counts['non_PSF']+=int(((typ!='PSF')&(typ!='')).sum());at=np.flatnonzero(typ=='PSF');counts['PSF']+=len(at)
            def v(key):return np.asarray(a[key])[at]
            ra=v('RA');dec=v('DEC');assert np.isfinite(ra+dec).all() and (abs(dec)<=90).all();lat=galactic(ra,dec)[1];ids=legacy_ids(dict(RELEASE=v('RELEASE'),BRICKID=v('BRICKID'),OBJID=v('OBJID')))
            take=owner(side,ra,dec,lat,v('RELEASE'),ids);at=at[take];ra=ra[take];dec=dec[take];lat=lat[take];ids=ids[take];counts['owned']+=len(ids);owned.append(ids)
            def two(prefix,dtype=float):return np.column_stack([v(prefix+'_'+b) for b in 'GR']).astype(dtype)
            flux=two('FLUX');iv=two('FLUX_IVAR');tr=two('MW_TRANSMISSION');e=v('EBV').astype(float);pixel=hp.ang2pix(C['nside'],ra,dec,lonlat=True,nest=True)
            good,geom,mag,err,ext=entry(flux,iv,tr,two('NOBS'),v('MASKBITS').astype('i8'),two('FRACMASKED'),two('FRACIN'))
            with np.errstate(divide='ignore',invalid='ignore'):depth=22.5-2.5*np.log10(K/.1/np.sqrt(two('PSFDEPTH')))-ext
            strict=(two('FRACFLUX')<.05).all(axis=1)&(two('ANYMASK')==0).all(axis=1);old=domain(lat,e,depth);legacy=good&old&strict&(err<.1).all(axis=1)
            for key,keep in [('gr_geometry',geom),('valid_gr',good),('strict_companion',good&strict),('legacy_entrance',legacy)]:counts[key]+=int(keep.sum())
            maps[0]+=np.bincount(pixel,minlength=NPIX);maps[1]+=np.bincount(pixel[geom],minlength=NPIX)
            for label,flag in regions(lat,e,depth).items():
                reg.append(dict(domain=label,raw_PSF=int(flag.sum()),gr_geometry=int((flag&geom).sum()),valid_gr=int((flag&good).sum()),strict_companion=int((flag&good&strict).sum())))
            dust=(e>1e-4)&np.isfinite(ext).all(axis=1)
            if dust.any():coeff.append(np.median(ext[dust]/e[dust,None],axis=0))
            source=np.where(np.char.strip(v('REF_CAT').astype('U'))=='G3',v('REF_ID'),0)
            parts.append(dict(TYPE=np.full(int(good.sum()),'PSF'),ls_id=ids[good],source_id=source[good],RA=ra[good],DEC=dec[good],lat=lat[good].astype('f4'),EBV=e[good].astype('f4'),m=np.column_stack((mag[good],np.full(good.sum(),np.nan))).astype('f4'),extinction=ext[good].astype('f4'),old_domain=old[good],strict=strict[good],legacy=legacy[good],shallow=(depth[good]<23).any(axis=1)))
            j=np.searchsorted(mid,ids);k=np.flatnonzero(j<len(mid));k=k[mid[j[k]]==ids[k]]
            if len(k):
                f3=np.column_stack([v('FLUX_'+b)[k] for b in 'GRZ']).astype(float);t3=np.column_stack([v('MW_TRANSMISSION_'+b)[k] for b in 'GRZ']).astype(float);i3=np.column_stack([v('FLUX_IVAR_'+b)[k] for b in 'GRZ']).astype(float)
                with np.errstate(divide='ignore',invalid='ignore'):m3=22.5-2.5*np.log10(f3/t3);e3=K/(f3*np.sqrt(i3));a3=-2.5*np.log10(t3)
                members.append(dict(TYPE=np.full(len(k),'PSF'),ls_id=ids[k],m=m3,err=e3,EBV=e[k],extinction=a3,native=np.full(len(k),side=='north','u1')))
    t={k:np.concatenate([p[k] for p in parts]) for k in parts[0]} if parts else dict(TYPE=np.array([],dtype='U3'),ls_id=np.array([],dtype='i8'),source_id=np.array([],dtype='i8'),RA=np.array([]),DEC=np.array([]),lat=np.array([]),EBV=np.array([]),m=np.empty((0,3)),extinction=np.empty((0,2)),old_domain=np.array([],bool),strict=np.array([],bool),legacy=np.array([],bool),shallow=np.array([],bool))
    ids=np.concatenate(owned) if owned else np.array([],dtype='i8');assert len(np.unique(ids))==len(ids);assert (t['TYPE']=='PSF').all();assert (maps.sum(axis=1)==[counts['owned'],counts['gr_geometry']]).all()
    npz(d/'photometry.npz',**t);npz(d/'owned_ids.npz',TYPE=np.array('PSF'),ls_id=ids)
    pix=np.flatnonzero(maps.any(axis=0));npz(d/'geometry.npz',pixels=pix,counts=maps[:,pix]);csv_rows(d/'domains.csv',reg)
    if members:npz(d/'member_photometry.npz',**{k:np.concatenate([p[k] for p in members]) for k in members[0]})
    coefficient=np.median(coeff,axis=0) if coeff else np.array([np.nan,np.nan]);counts.update(Ag_per_EBV=float(coefficient[0]),Ar_per_EBV=float(coefficient[1]))
    save_json(marker,dict(config=CH,scan_code=code,source=dict(path=str(path),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns),counts=counts,outputs={q.name:sha(q) for q in d.iterdir() if q.is_file() and q.name!='complete.json'}))
    return dict(side=side,file=path.name,**counts)


def smoke():
    require_compute();alltasks=tasks();chosen=[]
    for side in ('south','north'):
        tt=[x for x in alltasks if x[0]==side];chosen.append(tt[0]);centers=np.array([[(b[0]+b[2])/2,(b[1]+b[3])/2] for _,p in tt for b in [tile_bounds(p)]])
        latitude=abs(galactic(*centers.T)[1]);chosen.extend([tt[int(latitude.argmin())],tt[int(latitude.argmax())]])
    results=[scan_one(x) for x in dict.fromkeys(chosen)];csv_rows(folder('inventory')/'smoke.csv',results)
    ll,bb=hp.Rotator(coord=('C','G'))(266.4049948,-28.936174,lonlat=True);np.testing.assert_allclose([((ll+180)%360)-180,bb],[0,0],atol=2e-5)
    save_json(OUT/'inventory/smoke.json',dict(passed=True,files=len(results),coordinate='Galactic center, direction/wrap; source/bin/TYPE/native counts checked',scope='data/numerical integrity only; no stream-recovery gate'))


def scan():
    require_compute();assert json.loads((OUT/'inventory/smoke.json').read_text())['passed'];d=folder('inventory');roster=tasks();save_json(d/'tasks.json',dict(tasks=roster,enumeration='all native sweep files; no spatial/dust/depth prefilter'))
    with ProcessPoolExecutor(C['workers'],mp_context=get_context('spawn')) as pool:
        futures=[pool.submit(scan_one,t) for t in roster];rows=[]
        for i,f in enumerate(as_completed(futures)):
            rows.append(f.result())
            if i%50==0:print('SCANNED',i+1,len(roster),flush=True)
    csv_rows(d/'raw_to_display.csv',rows);complete(d,{},sweeps=len(roster),display_scope=C['scope'])


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['smoke','scan']);arg=a.parse_args();globals()[arg.stage]()
