#!/usr/bin/env python3
"""Fixed local imaging experiment, before any photometric or population cut."""
from concurrent.futures import ThreadPoolExecutor
import fitsio
from photdis.stage1 import require_compute,galactic,legacy_ids,tile_bounds
from photdis.photometry import CLEAN_REJECT_MASKBITS
from stream_common import *
from v3_5_dr11_area import random_memmap
from v3_9_seam import BOUNDARY

ROIS=[k for k,v in C['local'].items() if isinstance(v,dict) and 'track' in v]
VARS=C['local']['variants']
REJECT=sum(1<<b for b in CLEAN_REJECT_MASKBITS)
K=2.5/np.log(10)


def geometry(name,ra,dec):
    cfg=C['local'][name];lo,hi=cfg['ra_range'];off,along=profile(name,ra,dec)
    use=(ra>=lo)&(ra<hi)&(abs(off)<cfg['off_abs_deg'][1]+1)
    if 'exclude_M5_radius_deg' in cfg:
        distance=np.hypot((ra-229.6384)*np.cos(np.deg2rad(dec)),dec-2.0810)
        use&=distance>cfg['exclude_M5_radius_deg']
    return use,off,along


def footprints(t,random=False):
    high=abs(galactic(t['RA'],t['DEC'])[1])>30
    base=(t['EBV']>=0)&high&((t['mask']&REJECT)==0)
    obs=t['nobs']>0
    gr_geo=base&obs[:,:2].all(axis=1)
    grz_geo=gr_geo&obs[:,2]
    e03=t['EBV']<.03;e10=t['EBV']<.10
    complete=(t['depth0'][:,:2]>=23).all(axis=1)
    # Randoms describe imaging coverage, not object deblending or PSF purity.
    if random:
        return np.array([grz_geo&e03,grz_geo&e10,gr_geo&e10,grz_geo&e10,
                         gr_geo&e10&complete,grz_geo&e10&complete])
    good=t['valid']&(t['err']<.1)&(t['fracmasked']<.4)&(t['fracin']>.7)
    strict=good.all(axis=1)&(t['fracflux']<.5).all(axis=1)&(t['observed']<23).all(axis=1)
    gr=good[:,:2].all(axis=1)&(t['fracflux'][:,:2]<.05).all(axis=1)
    gr&=(t['anymask'][:,:2]==0).all(axis=1)&(t['m'][:,:2]<23).all(axis=1)
    z=good[:,2]&(t['fracflux'][:,2]<.5)&(t['observed'][:,2]<23)
    return np.array([grz_geo&e03&strict,grz_geo&e10&strict,gr_geo&e10&gr,
                     grz_geo&e10&gr&z,gr_geo&e10&gr&complete,grz_geo&e10&gr&complete&z])


def process_file(path):
    chunks=[];flow=[]
    with fitsio.FITS(str(path)) as f:
        names=set(f[1].get_colnames());rc=f[1].read(columns=['RA','DEC'])
        spatial=np.logical_or.reduce([geometry(n,rc['RA'],rc['DEC'])[0] for n in ROIS])
        rows=np.flatnonzero(spatial)
        if not len(rows):return None,[],dict(path=str(path),size=path.stat().st_size,rows=f[1].get_nrows(),local=0)
        cols=['RELEASE','BRICKID','OBJID','TYPE','RA','DEC','EBV','REF_ID','REF_CAT','MASKBITS']
        cols += [p+'_'+b for b in 'GRZ' for p in ('FLUX','FLUX_IVAR','MW_TRANSMISSION','NOBS','FRACMASKED','FRACIN','FRACFLUX','PSFDEPTH','ANYMASK')]
        if 'TYPE' not in names:
            for n in ROIS:flow.append(dict(roi=n,file=path.name,stage='missing_TYPE',n=int(geometry(n,rc['RA'][rows],rc['DEC'][rows])[0].sum())))
            return None,flow,dict(path=str(path),size=path.stat().st_size,missing_TYPE=len(rows))
        assert set(cols)<=names, sorted(set(cols)-names)
        a=f[1].read(columns=cols,rows=rows)
    typ=np.char.strip(a['TYPE'].astype('U'));psf=typ=='PSF'
    for n in ROIS:
        g=geometry(n,a['RA'],a['DEC'])[0]
        for stage,keep in (('raw_local',g),('missing_TYPE',g&(typ=='')),('non_PSF',g&~psf&(typ!='')),('TYPE_PSF',g&psf)):
            flow.append(dict(roi=n,file=path.name,stage=stage,n=int(keep.sum())))
    a=a[psf];lat=galactic(a['RA'],a['DEC'])[1]
    own=~((a['DEC']>=BOUNDARY)&(lat>0))&(a['RELEASE']==11010)
    a=a[own]
    def three(p):return np.column_stack([a[p+'_'+b] for b in 'GRZ'])
    flux,iv,trans=three('FLUX'),three('FLUX_IVAR'),three('MW_TRANSMISSION')
    valid=(flux>0)&(iv>0)&(trans>0)&np.isfinite(flux+iv+trans)
    with np.errstate(divide='ignore',invalid='ignore'):
        observed=22.5-2.5*np.log10(flux);m=observed+2.5*np.log10(trans)
        err=K/(flux*np.sqrt(iv));dep=22.5-2.5*np.log10(K/.1/np.sqrt(three('PSFDEPTH')))+2.5*np.log10(trans)
    source=np.where(np.char.strip(a['REF_CAT'].astype('U'))=='G3',a['REF_ID'],0)
    t=dict(TYPE=np.full(len(a),'PSF'),ls_id=legacy_ids(a),source_id=source,RA=a['RA'],DEC=a['DEC'],EBV=a['EBV'],
           mask=a['MASKBITS'].astype('i8'),native=np.zeros(len(a),'u1'),m=m.astype('f4'),err=err.astype('f4'),observed=observed.astype('f4'),
           depth0=dep.astype('f4'),nobs=three('NOBS'),valid=valid,fracflux=three('FRACFLUX'),fracmasked=three('FRACMASKED'),fracin=three('FRACIN'),anymask=three('ANYMASK'))
    return t,flow,dict(path=str(path),size=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns,raw_local=len(rows),PSF_owned=len(a))


def sources():
    require_compute();d=folder('local_source');inputs={};root=Path(json.loads((ROOT/'configs/v3_9_catalog.json').read_text())['south']['sweep_root'])
    selected=[]
    for p in sorted(root.glob('sweep-*.fits')):
        ra0,dec0,ra1,dec1=tile_bounds(p)
        if any(ra0<C['local'][n]['ra_range'][1] and ra1>C['local'][n]['ra_range'][0] and dec0<35 and dec1> -40 for n in ROIS):selected.append(p)
    payload=[];flow=[];receipts=[]
    with ThreadPoolExecutor(8) as pool:
        for t,ff,receipt in pool.map(process_file,selected):
            if t is not None and len(t['ls_id']):payload.append(t)
            flow.extend(ff);receipts.append(receipt)
    t={k:np.concatenate([p[k] for p in payload]) for k in payload[0]}
    _,idx=np.unique(t['ls_id'],return_index=True);duplicates=len(t['ls_id'])-len(idx);t={k:v[idx] for k,v in t.items()}
    assert (t['TYPE']=='PSF').all();npz(d/'photometry.npz',**t)
    for n in ROIS:
        g,off,along=geometry(n,t['RA'],t['DEC']);a={k:v[g] for k,v in t.items()};select=footprints(a)
        npz(d/(n+'.npz'),**a,offset=off[g],along=along[g],selected=select)
        flow.append(dict(roi=n,file='unique_owned',stage='PSF_owned',n=int(g.sum())))
        for v,s in zip(VARS,select):flow.append(dict(roi=n,file='unique_owned',stage=v,n=int(s.sum())))
    csv_rows(d/'source_cutflow.csv',flow);save_json(d/'raw_inputs.json',receipts)
    complete(d,inputs,deduplicated_ls_ids=duplicates,TYPE_required=True,native='DECam; all local declinations outside north ownership',
             source_quality_scope='gr source quality is separate from geometric random coverage; z only required in named contrasts')
    print('sources',len(t['ls_id']),flush=True)


def randoms():
    require_compute();d=folder('local_area');cfg=json.loads((ROOT/'configs/v3_9_catalog.json').read_text())['south']
    path=Path(cfg['sweep_root']).parent.parent/'randoms/randoms-south-1-0.fits';stat=path.stat();raw,density,names=random_memmap(path)
    pieces={n:[] for n in ROIS};nlocal=0
    for start in range(0,len(raw),500000):
        a=raw[start:start+500000];ra=a[names['RA']];dec=a[names['DEC']]
        prelim=np.logical_or.reduce([(ra>=C['local'][n]['ra_range'][0])&(ra<C['local'][n]['ra_range'][1])&(dec> -40)&(dec<35) for n in ROIS])
        j=np.flatnonzero(prelim)
        if not len(j):continue
        a=a[j];t={key:np.asarray(a[val]) for key,val in names.items()};lat=galactic(t['RA'],t['DEC'])[1]
        own=~((t['DEC']>=BOUNDARY)&(lat>0))&(t['RELEASE']==11010)&(np.char.strip(t['PHOTSYS'])==b'S')
        t={k:v[own] for k,v in t.items()};dep=np.column_stack([t['PSFDEPTH_'+b] for b in 'GRZ'])
        # Schlafly/Finkbeiner coefficients used by native DR11 transmissions.
        transmission=10**(-.4*t['EBV'][:,None]*np.array([3.214,2.165,1.211]))
        with np.errstate(divide='ignore',invalid='ignore'):
            depth=22.5-2.5*np.log10(K/.1/np.sqrt(dep))+2.5*np.log10(transmission)
        v=dict(RA=t['RA'],DEC=t['DEC'],EBV=t['EBV'],mask=t['MASKBITS'].astype('i8'),depth0=depth,nobs=np.column_stack([t['NOBS_'+b] for b in 'GRZ']))
        for n in ROIS:
            g,off,along=geometry(n,v['RA'],v['DEC'])
            if g.any():
                p={k:z[g] for k,z in v.items()};p.update(offset=off[g],along=along[g],selected=footprints(p,True));pieces[n].append(p);nlocal+=int(g.sum())
        if start%10000000==0:print('random progress',start,len(raw),nlocal,flush=True)
    rows=[]
    for n,pp in pieces.items():
        t={k:np.concatenate([p[k] for p in pp],axis=1 if k=='selected' else 0) for k in pp[0]}
        npz(d/(n+'.npz'),**t)
        for var,keep in zip(VARS,t['selected']):rows.append(dict(roi=n,variant=var,n_random=int(keep.sum()),area_deg2=keep.sum()/density))
    csv_rows(d/'areas.csv',rows);complete(d,dict(path=str(path),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns),density=density,
        scope='independent DR11 random geometry/mask/nobs/dereddened depth, no invented source-quality completeness',dust_coefficients=[3.214,2.165,1.211])


def audit():
    require_compute();d=folder('local_source');rows=[]
    for roi in ROIS:
        with np.load(d/(roi+'.npz')) as a:t={k:a[k] for k in a.files}
        high=abs(galactic(t['RA'],t['DEC'])[1])>30
        for relaxed in (False,True):
            keep=np.ones(len(high),bool)
            cuts=[('owned_unique_PSF',keep.copy()),('latitude',high),('finite_nonnegative_EBV',np.isfinite(t['EBV'])&(t['EBV']>=0)),
                  ('EBV_limit',t['EBV']<(.1 if relaxed else .03)),('clean_maskbits',(t['mask']&REJECT)==0),
                  ('gr_nobs',(t['nobs'][:,:2]>0).all(axis=1)),('gr_flux_transmission_ivar',t['valid'][:,:2].all(axis=1)),
                  ('gr_error',(t['err'][:,:2]<.1).all(axis=1)),('gr_fracmasked',(t['fracmasked'][:,:2]<.4).all(axis=1)),
                  ('gr_fracin',(t['fracin'][:,:2]>.7).all(axis=1)),('gr_fracflux',(t['fracflux'][:,:2]<.05).all(axis=1)),
                  ('gr_anymask',(t['anymask'][:,:2]==0).all(axis=1)),('gr_dered_mag',(t['m'][:,:2]<23).all(axis=1)),
                  ('gr_complete_depth',(t['depth0'][:,:2]>=23).all(axis=1))]
            for stage,cut in cuts:
                before=int(keep.sum());keep&=cut;rows.append(dict(roi=roi,route='gr_E010' if relaxed else 'gr_E003',stage=stage,before=before,retained=int(keep.sum()),excluded=before-int(keep.sum())))
    csv_rows(d/'sequential_cutflow.csv',rows)
    rec=json.loads((d/'complete.json').read_text());rec['cutflow_audit_code']=sha(Path(__file__));rec['outputs']['sequential_cutflow.csv']=sha(d/'sequential_cutflow.csv');save_json(d/'complete.json',rec)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('sources','randoms','audit'));a=p.parse_args();globals()[a.stage]()
