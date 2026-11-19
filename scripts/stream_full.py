#!/usr/bin/env python3
"""Conditional full-footprint gr-only CMD images; frozen distance products untouched."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
from astropy.io import fits
import healpy as hp
from photdis.stage1 import require_compute,galactic,legacy_ids,tile_bounds
from stream_common import *
from stream_extract import K,REJECT
from stream_local import templates,curves,accept,region_masks,uncertainty
from v3_9_seam import BOUNDARY
from v3_5_dr11_area import random_memmap,tile_map

FC=ROOT/'configs/stream_full_20261119.json';F=json.loads(FC.read_text());FH=sha(FC);NPIX=hp.nside2npix(512)


def curve():
    with np.load(OUT/'local/Pal5_tail_redraw.npz') as a:return a['g_centers'],a['sigma_g']


def replay():
    require_compute();d=folder('full_replay');ec=curve();npz(d/'fixed_error_curve.npz',g_centers=ec[0],sigma_g=ec[1]);rows=[];decision={};inputs={'extension_config':FH}
    mag,phase=templates(inputs)['FOS'];cc=curves(mag,phase,'MS_TO')
    density=json.loads((OUT/'local_area/complete.json').read_text())['density']
    for roi in ('Pal5_tail','Pal5_leading','Sgr_north','Sgr_south'):
        cfg=C['local'][roi];mu=10+5*np.log10(cfg['distance_kpc']) if 'distance_kpc' in cfg else cfg['reference_mu']
        t=read(OUT/'local_source',(roi+'.npz','TYPE','ls_id','m','offset','along','selected'),inputs);r=read(OUT/'local_area',(roi+'.npz','offset','along','selected'),inputs)
        g=t['selected'][4];m=t['m'][g];offset=t['offset'][g];along=t['along'][g];sr=region_masks(offset,cfg);rr=region_masks(r['offset'],cfg)
        accepted=accept(m,[mu],cc,ec,'MS_TO')[:,0];passed=0
        for segment in range(cfg['segments']):
            ts=(along>=segment/cfg['segments'])&(along<(segment+1)/cfg['segments']);rs=(r['along']>=segment/cfg['segments'])&(r['along']<(segment+1)/cfg['segments'])
            n=np.array([int((ts&accepted&z).sum()) for z in sr]);random=np.array([int((rs&r['selected'][4]&z).sum()) for z in rr]);area=random/density
            val=np.divide(n,area,out=np.full(4,np.nan),where=area>0);low,high,upper=uncertainty(n[0],n[1],random[0],random[1],density,C['seed']+segment)
            ok=bool(low>0 and area[0]>=.1 and val[0]>val[2:].max());passed+=ok
            rows.append(dict(ROI=roi,segment=segment,mu=mu,n_on=int(n[0]),n_off=int(n[1]),area_on=area[0],area_off=area[1],excess=val[0]-val[1],ci_lo=low,ci_hi=high,upper95=upper,right_density=val[2],left_density=val[3],passed=ok))
        decision[roi]=passed>=2
    both=bool((decision['Pal5_tail'] or decision['Pal5_leading']) and (decision['Sgr_north'] or decision['Sgr_south']))
    csv_rows(d/'common_curve_recovery.csv',rows);save_json(d/'decision.json',dict(per_ROI=decision,both_systems_pass=both,full_config=FH,error_curve_sha=sha(d/'fixed_error_curve.npz'),scope='same fixed measurement-width function transferred from Pal5 pre-CMD errors; no excess-based fitting'))
    complete(d,inputs,full_footprint_authorized_by_registered_condition=both)


def gate():
    r=json.loads((OUT/'full_replay/decision.json').read_text());assert r['both_systems_pass'],'Common-curve transfer did not pass; retain local negative result, no full scan'
    assert r['full_config']==FH;return r


def tasks():
    cfg=json.loads((ROOT/'configs/v3_9_catalog.json').read_text());result=[]
    for side in ('south','north'):
        for p in sorted(Path(cfg[side]['sweep_root']).glob('sweep-*.fits')):
            ra0,de0,ra1,de1=tile_bounds(p)
            if side=='north' and de1<BOUNDARY:continue
            ra,dec=np.meshgrid(np.linspace(ra0,ra1,21),np.linspace(de0,de1,11));b=galactic(ra.ravel(),dec.ravel())[1]
            own=((dec.ravel()>=BOUNDARY)&(b>0))==(side=='north')
            # Conservative <1degree sampled-rectangle bound; exact per-source
            # ownership and latitude remain authoritative at admission.
            if not (own&(abs(b)>29)).any():continue
            result.append((side,str(p)))
    return result


def scan_one(task):
    side,filename=task;path=Path(filename);dest=folder('full_cache')/(side+'_'+path.stem);dest.mkdir(exist_ok=True);marker=dest/'complete.json';code=sha(Path(__file__))
    if marker.exists():
        r=json.loads(marker.read_text());assert r['full_config']==FH and r['source_code']==code
        for name,h in r['outputs'].items():verified(dest/name,h)
        return r['counts']
    counts={};stat=path.stat()
    with fits.open(path,memmap=True) as h:
        a=h[1].data;names=set(a.dtype.names);counts['raw']=len(a)
        if 'TYPE' not in names:raise ValueError(('missing TYPE column',path))
        typ=np.char.strip(a['TYPE'].astype('U'));counts['missing_TYPE']=int((typ=='').sum());counts['non_PSF']=int(((typ!='PSF')&(typ!='')).sum());at=np.flatnonzero(typ=='PSF');counts['PSF']=len(at)
        def v(name):return np.asarray(a[name])[at]
        ra=v('RA');dec=v('DEC');b=galactic(ra,dec)[1];ebv=v('EBV');release=11011 if side=='north' else 11010
        ids=legacy_ids(dict(RELEASE=v('RELEASE'),BRICKID=v('BRICKID'),OBJID=v('OBJID')))
        own=((dec>=BOUNDARY)&(b>0))==(side=='north');own&=v('RELEASE')==release
        seam=ROOT/'data/v3_9/delivery/seam/ownership_overrides.npz'
        with np.load(seam) as z:
            j=np.searchsorted(z['ls_id'],ids);k=np.flatnonzero(j<len(z['ls_id']));k=k[z['ls_id'][j[k]]==ids[k]];own[k]=z['primary_survey_owner'][j[k]]
        keep=own;counts['owned']=int(keep.sum());keep&=(abs(b)>30)&(ebv>=0)&(ebv<.1);counts['latitude_E010']=int(keep.sum());keep&=(v('MASKBITS').astype('i8')&REJECT)==0;counts['clean_mask']=int(keep.sum())
        at=at[keep];ra=ra[keep];dec=dec[keep];ebv=ebv[keep];ids=ids[keep]
        flux=np.column_stack([v('FLUX_'+b) for b in 'GR']);iv=np.column_stack([v('FLUX_IVAR_'+b) for b in 'GR']);tr=np.column_stack([v('MW_TRANSMISSION_'+b) for b in 'GR'])
        keep=(flux>0).all(axis=1)&(iv>0).all(axis=1)&(tr>0).all(axis=1)
        with np.errstate(divide='ignore',invalid='ignore'):
            m=22.5-2.5*np.log10(flux/tr);err=K/(flux*np.sqrt(iv));depth=22.5-2.5*np.log10(K/.1/np.sqrt(np.column_stack([v('PSFDEPTH_'+b) for b in 'GR'])))+2.5*np.log10(tr)
        dust_valid=ebv>1e-4
        if dust_valid.any():
            coefficient=np.median(-2.5*np.log10(tr[dust_valid])/ebv[dust_valid,None],axis=0)
            counts['Ag_per_EBV']=float(coefficient[0]);counts['Ar_per_EBV']=float(coefficient[1])
        for label,cut in (('gr_NOBS',np.column_stack([v('NOBS_'+b)>0 for b in 'GR']).all(axis=1)),('gr_error',np.isfinite(m+err).all(axis=1)&(err<.1).all(axis=1)),('gr_masked',np.column_stack([v('FRACMASKED_'+b)<.4 for b in 'GR']).all(axis=1)),('gr_fracin',np.column_stack([v('FRACIN_'+b)>.7 for b in 'GR']).all(axis=1)),('gr_fracflux',np.column_stack([v('FRACFLUX_'+b)<.05 for b in 'GR']).all(axis=1)),('gr_anymask',np.column_stack([v('ANYMASK_'+b)==0 for b in 'GR']).all(axis=1)),('gr0_lt23',(m<23).all(axis=1)),('gr_complete_depth',(depth>=23).all(axis=1))):
            keep&=cut;counts[label]=int(keep.sum())
        source=np.where(np.char.strip(v('REF_CAT').astype('U'))=='G3',v('REF_ID'),0)
        npz(dest/'photometry.npz',TYPE=np.full(keep.sum(),'PSF'),ls_id=ids[keep],source_id=source[keep],RA=ra[keep],DEC=dec[keep],m=np.column_stack((m[keep],np.full(keep.sum(),np.nan))).astype('f4'),err=err[keep].astype('f4'),native=np.full(keep.sum(),side=='north','u1'))
    save_json(marker,dict(full_config=FH,source_code=code,path=filename,size=stat.st_size,mtime_ns=stat.st_mtime_ns,counts=counts,outputs={'photometry.npz':sha(dest/'photometry.npz')}))
    return counts


def scan():
    require_compute();gate();d=folder('full_inventory');roster=tasks();save_json(d/'tasks.json',dict(tasks=roster,config=FH))
    with ProcessPoolExecutor(F['workers'],mp_context=get_context('spawn')) as pool:
        futures={pool.submit(scan_one,t):t for t in roster};rows=[]
        for i,f in enumerate(as_completed(futures)):
            side,p=futures[f];rows.append(dict(side=side,file=Path(p).name,**f.result()))
            if i%20==0:print('completed sweeps',i+1,len(roster),flush=True)
    csv_rows(d/'raw_to_gr_complete.csv',rows);complete(d,dict(extension_config=FH),sweeps=len(roster),scope=F['scope'])


def identities():
    require_compute();gate();d=folder('full_identities');parts=[];inputs={}
    for p in sorted((OUT/'full_cache').glob('*/complete.json')):
        r=json.loads(p.read_text());z=read_columns(verified(p.with_name('photometry.npz'),r['outputs']['photometry.npz']),('TYPE','ls_id','source_id'));assert (z['TYPE']=='PSF').all();parts.append(z);inputs[str(p.parent.name)]=sha(p)
    ids=np.concatenate([x['ls_id'] for x in parts]);gaia=np.concatenate([x['source_id'] for x in parts]);assert len(np.unique(ids))==len(ids)
    j=np.flatnonzero(gaia>0);order=np.lexsort((ids[j],gaia[j]));jj=j[order];duplicate=np.r_[False,np.diff(gaia[jj])==0];exclude=ids[jj[duplicate]]
    npz(d/'source_alias_exclusions.npz',ls_id=np.sort(exclude),TYPE=np.full(len(exclude),'PSF'));save_json(d/'identity_counts.json',dict(native_unique=len(ids),positive_gaia=len(j),repeated_positive_gaia_aliases=len(exclude),admitted_after_aliases=len(ids)-len(exclude),rule='verified TYPEPSF/native owner first; same positive Gaia reference aliases retained once at minimum native LSID; source_id0 uses unique native IDs, no unsupported positional identity claim'))
    complete(d,inputs)


def count_one(path):
    path=Path(path);r=json.loads((path/'complete.json').read_text());ec=curve();inputs={};t=read_columns(verified(path/'photometry.npz',r['outputs']['photometry.npz']),('TYPE','ls_id','m','RA','DEC','native'))
    with np.load(OUT/'full_identities/source_alias_exclusions.npz') as a:keep=~np.isin(t['ls_id'],a['ls_id'])
    t={k:v[keep] for k,v in t.items()};native=int(path.name.startswith('north_'));temp=read(OLD/'native',('template.npz','south','north','phase','north_support'),inputs)
    use=(temp['phase']==0)&(temp['north_support'] if native else True);m=temp[('south','north')[native]];cc=[]
    ii=np.flatnonzero(use)
    from halo_filtering import envelope
    for seg in np.split(ii,np.flatnonzero(np.diff(ii)>1)+1):
        if len(seg)>1:cc.append(envelope(m[seg,0],m[seg,0]-m[seg,1]))
    mu=np.round(np.arange(14,20.001,.1),1);accepted=accept(t['m'],mu,cc,ec,'MS_TO');pix=hp.ang2pix(512,t['RA'],t['DEC'],lonlat=True,nest=True);unique,inv=np.unique(pix,return_inverse=True)
    counts=np.array([np.bincount(inv[k],minlength=len(unique)) for k in accepted.T],dtype='i4')
    dest=folder('full_nodes')/path.name;dest.mkdir(exist_ok=True);npz(dest/'nodes.npz',pixels=unique,counts=counts,node_totals=accepted.sum(axis=0),native=np.array(native),sources=np.array(len(pix)),union=np.array(int(accepted.any(axis=1).sum())))
    save_json(dest/'complete.json',dict(config=FH,error_curve=sha(OUT/'full_replay/fixed_error_curve.npz'),source=sha(path/'complete.json'),code=sha(Path(__file__)),outputs={'nodes.npz':sha(dest/'nodes.npz')}))
    return str(dest)


def count():
    require_compute();gate();paths=sorted((OUT/'full_cache').glob('*/complete.json'));d=folder('full_count');counts=np.zeros((2,61,NPIX),'i4');totals=np.zeros((2,61),'i8');rows=[]
    with ProcessPoolExecutor(F['workers'],mp_context=get_context('spawn')) as pool:
        futures=[pool.submit(count_one,str(p.parent)) for p in paths]
        for i,f in enumerate(as_completed(futures)):
            p=Path(f.result())
            with np.load(p/'nodes.npz') as a:
                code=int(a['native']);counts[code][:,a['pixels']]+=a['counts'];totals[code]+=a['node_totals'];rows.append(dict(part=p.name,sources=int(a['sources']),union=int(a['union']),acceptances=int(a['node_totals'].sum())))
            if i%40==0:print('counted',i+1,len(paths),flush=True)
    np.testing.assert_array_equal(counts.sum(axis=2),totals);npz(d/'native_nodes.npz',counts=counts,mu=np.round(np.arange(14,20.001,.1),1),totals=totals);csv_rows(d/'source_node_counts.csv',rows);complete(d,dict(extension_config=FH),conserved=True)


def area(side):
    require_compute();gate();d=folder('full_area')/side;d.mkdir(exist_ok=True);cfg=json.loads((ROOT/'configs/v3_9_catalog.json').read_text())[side];path=Path(cfg['sweep_root']).parent.parent/'randoms'/f'randoms-{side}-1-0.fits';stat=path.stat();raw,density,names=random_memmap(path);counts=np.zeros(NPIX,'i4');dust=np.zeros(NPIX);deps=np.zeros((2,NPIX))
    files=[dict(filename=Path(p).name,bounds_deg=tile_bounds(p)) for s,p in tasks() if s==side];tiles=tile_map(files)
    # Native transmission coefficients measured from the actual raw source
    # records, rather than assuming the same band response across cameras.
    summary=list(csv.DictReader((OUT/'full_inventory/raw_to_gr_complete.csv').open()))
    coefficient=np.median([[float(r['Ag_per_EBV']),float(r['Ar_per_EBV'])] for r in summary if r['side']==side and r.get('Ag_per_EBV','')],axis=0)
    for start in range(0,len(raw),500000):
        a=raw[start:start+500000];t={k:np.asarray(a[v]) for k,v in names.items()};lat=galactic(t['RA'],t['DEC'])[1]
        own=((t['DEC']>=BOUNDARY)&(lat>0))==(side=='north');keep=own&(abs(lat)>30)&(t['EBV']>=0)&(t['EBV']<.1)&(t['RELEASE']==cfg['release'])
        keep&=np.char.strip(t['PHOTSYS'])==(b'N' if side=='north' else b'S')
        keep&=tiles[(t['RA']%360).astype(int),np.minimum((t['DEC']+90).astype(int),179)]>=0
        keep&=(t['MASKBITS'].astype('i8')&REJECT)==0;keep&=(t['NOBS_G']>0)&(t['NOBS_R']>0)
        dep=np.column_stack((t['PSFDEPTH_G'],t['PSFDEPTH_R']))
        with np.errstate(divide='ignore',invalid='ignore'):lim=22.5-2.5*np.log10(K/.1/np.sqrt(dep))-t['EBV'][:,None]*coefficient
        keep&=(lim>=23).all(axis=1);pix=hp.ang2pix(512,t['RA'][keep],t['DEC'][keep],lonlat=True,nest=True);counts+=np.bincount(pix,minlength=NPIX).astype('i4');dust+=np.bincount(pix,weights=t['EBV'][keep],minlength=NPIX)
        for band in range(2):deps[band]+=np.bincount(pix,weights=lim[keep,band],minlength=NPIX)
    npz(d/'randoms.npz',counts=counts,area=counts/density,EBV=np.divide(dust,counts,out=np.full_like(dust,np.nan),where=counts>0),depth=np.divide(deps,counts,out=np.full_like(deps,np.nan),where=counts>0));complete(d,dict(path=str(path),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,extension_config=FH),density=density,extinction_coefficient=coefficient,area_deg2=float(counts.sum()/density),scope='native imaging gr complete23mag geometry; no star-detection/deblend completeness claim')


def assemble():
    require_compute();gate();d=folder('full_atlas');inputs={};z=read(OUT/'full_count',('native_nodes.npz','counts','mu','totals'),inputs);area=np.array([read(OUT/'full_area'/side,('randoms.npz','area'),inputs)['area'] for side in ('south','north')]);from halo_atlas import smooth
    pixel=hp.nside2pixarea(512,degrees=True);raw=area.sum(axis=0);mask=(raw/pixel>=.1)&(smooth(raw,.3)/pixel>=.7)
    native_area=np.repeat(area[:,None,:],3,axis=1);count=np.zeros((2,1,61,NPIX),'i4');count[1,0]=z['counts'].sum(axis=0)
    h=fits.Header();h['NSIDE']=512;h['ORDERING']='NESTED';h['COORDSYS']='C';h['PHASE']='MS/TO';h['CONFIG']=FH;h['PRODUCT']='CMD counts, not distances'
    fits.HDUList([fits.PrimaryHDU(header=h),fits.ImageHDU(count,name='COUNTS'),fits.ImageHDU(z['counts'],name='NATIVE_COUNTS'),fits.ImageHDU(z['mu'],name='MU'),fits.ImageHDU(native_area,name='NATIVE_AREAS'),fits.ImageHDU(mask.astype('u1'),name='DISPLAY_MASK')]).writeto(d/'cmd_filter_cube.fits',overwrite=False)
    complete(d,inputs,source_counts=int(z['totals'].sum()),eligible_area_deg2=float(raw.sum()),display_area_deg2=float(mask.sum()*pixel),phase='MS_TO only',interpretation=F['role'],north_validation=F['scope'],error_curve=sha(OUT/'full_replay/fixed_error_curve.npz'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('replay','smoke','scan','identities','count','south','north','assemble'));a=p.parse_args()
    if a.stage in ('south','north'):area(a.stage)
    elif a.stage=='smoke':
        require_compute();gate();roster=tasks()
        for side in ('south','north'):print(side,scan_one(next(t for t in roster if t[0]==side)),flush=True)
    else:globals()[a.stage]()
