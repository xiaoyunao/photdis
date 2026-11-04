#!/usr/bin/env python3
"""Allocated native checkpoint consolidation and targeted zero-Gaia-ID recovery.

No predictions. Original sweep outputs and source evidence remain immutable.
"""
import concurrent.futures
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'.deps')]
import fitsio
import numpy as np
from scipy.spatial import cKDTree
from photdis.assembly_v1_4 import epoch_positions,unit_vectors,positional_join,representative_rows,uppercase_fields
from photdis.photometry import derive_photometry
from photdis.reference_v1_4 import optional_photometry
from photdis.source_evidence_v1_4 import vac_evidence
from baseline_v1_4_source_audit import VAC_COLUMNS
from photdis.stage1 import require_compute,write_json,legacy_ids
from baseline_v1_4_extract import sha,atomic_npz,RAW_COLUMNS

OUT=ROOT/'data/baseline_v1_4'
SOUTH=Path('/global/cfs/projectdirs/cosmo/data/legacysurvey/dr11/south')
KEYS=['RELEASE','BRICKID','OBJID','REF_ID','REF_CAT','RA','DEC','REF_EPOCH','PMRA','PMDEC','EBV',
      'g0','r0','z0','w10','w20','r','sigma_g','sigma_r','sigma_z','sigma_w1','sigma_w2',
      'grz_base','grz_clean','w1_clean','w2_clean','snr_w1','snr_w2',
      'i0','sigma_i','i_usable','w1_measurement_state','w2_measurement_state','i_measurement_state']
_TARGET_VECTORS=None


def projected(raw):
    d={k:raw[k] for k in raw.dtype.names}
    d.update(derive_photometry(raw,w2_snr_min=5));d.update(optional_photometry(raw))
    return {k:d[k] for k in KEYS}


def consolidate():
    output=OUT/'native_photometry.npz';marker=output.with_suffix('.json')
    if marker.exists():
        assert sha(output)==json.loads(marker.read_text())['sha256'];return
    assert (OUT/'extraction_complete.json').exists()
    parts=[];inputs={}
    for i,path in enumerate(sorted((OUT/'photometry').glob('*.json'))):
        m=json.loads(path.read_text());f=path.with_suffix('.npz')
        assert sha(f)==m['sha256'],f
        inputs[path.stem]=m['sha256']
        if m['selected_rows']:
            with np.load(f) as z:parts.append({k:z[k] for k in KEYS})
        if i%200==0:print('verified sweep',i,flush=True)
    assert len(inputs)==1600
    d={k:np.concatenate([p[k] for p in parts]) for k in KEYS};del parts
    lsid=(d['RELEASE'].astype('i8')<<40)|(d['BRICKID'].astype('i8')<<20)|d['OBJID'].astype('i8')
    _,first=np.unique(lsid,return_index=True)
    # Sweep files should have disjoint primary objects. Refuse conflicting copies.
    if len(first)!=len(lsid):raise RuntimeError('Duplicate native ls_id across extracted sweeps; inspect before selecting')
    d['ls_id']=lsid
    d['source_id']=np.where(np.char.strip(d['REF_CAT'].astype('U'))=='G3',d['REF_ID'],0).astype('i8')
    atomic_npz(output,**d)
    write_json(marker,dict(rows=len(lsid),sha256=sha(output),sweep_hashes=inputs))
    print('Consolidated native',len(lsid),flush=True)


def brick_candidates(task):
    path,indices=task
    destination=OUT/'zero_candidates'/ (Path(path).stem+'.npz')
    marker=destination.with_suffix('.json')
    if marker.exists():
        saved=json.loads(marker.read_text());assert sha(destination)==saved['sha256'];return destination
    with fitsio.FITS(path) as f:
        names=[k.upper() for k in f[1].get_colnames()]
        cols=['RA','DEC','REF_EPOCH','PMRA','PMDEC']+(['BRICK_PRIMARY'] if 'BRICK_PRIMARY' in names else [])
        pos=uppercase_fields(f[1].read(columns=cols))
        vector,_=epoch_positions(*[pos[k] for k in ['RA','DEC','REF_EPOCH','PMRA','PMDEC']])
        good=np.isfinite(vector).all(axis=1)
        if 'BRICK_PRIMARY' in names:good &= pos['BRICK_PRIMARY']
        q=np.flatnonzero(good);dist=cKDTree(_TARGET_VECTORS[indices]).query(vector[q])[0]
        q=q[dist<2*np.sin(np.deg2rad(3/3600)/2)]
        raw=uppercase_fields(f[1].read(columns=RAW_COLUMNS,rows=q))
    d=projected(raw) if len(raw) else {'REF_ID':np.empty(0,'i8')}
    atomic_npz(destination,**d)
    write_json(marker,dict(path=path,rows=len(raw),sha256=sha(destination)))
    return destination


def recover_zero():
    global _TARGET_VECTORS
    output=OUT/'zero_match.npz';marker=output.with_suffix('.json')
    if marker.exists():
        assert sha(output)==json.loads(marker.read_text())['sha256'];return
    with np.load(OUT/'source_evidence.npz') as z:
        ids=z['identity'];qualified=(z['source_id']<=0)&(z['class_label']>=0)
    class_rows=int(qualified.sum())
    with np.load(OUT/'sources.npz') as z:targets=z['rvs_targetid'];source_ids=z['source_id']
    # Zero-Gaia distance evidence can exist without qualified class labels.
    for kind,columns in VAC_COLUMNS.items():
        with np.load(OUT/(kind+'_raw.npz')) as z:
            zero=(z['source_id']<=0)|(z['source_id']==999999)
            v={k:z[k][zero] for k in columns}
        q=np.arange(len(v['targetid']),dtype='i8')+1
        valid=vac_evidence(q,q,dict(v,targetid=q,source_id=q),kind)[kind+'_valid']
        qualified|=(source_ids<=0)&np.isin(targets,v['targetid'][valid&(v['targetid']>0)])
    with np.load(OUT/'sources.npz') as z:
        cols=['identity','source_id','original_row','target_target_ra','target_target_dec','target_ref_epoch','target_pmra','target_pmdec','rvs_primary','rvs_sn_r']
        s={k:z[k][qualified] for k in cols}
    ix=representative_rows(s['identity'],s['rvs_primary'],s['rvs_sn_r']);s={k:v[ix] for k,v in s.items()}
    q={k:s[v] for k,v in zip(['ra','dec','epoch','pmra','pmdec'],['target_target_ra','target_target_dec','target_ref_epoch','target_pmra','target_pmdec'])}
    _TARGET_VECTORS,epoch_known=epoch_positions(*[q[k] for k in ['ra','dec','epoch','pmra','pmdec']])
    good=np.isfinite(_TARGET_VECTORS).all(axis=1)&(abs(q['dec'])<=90)
    # Do not interpret bad coordinates as absence of southern imaging.
    if not good.all():raise RuntimeError('Invalid qualified zero-ID coordinates require explicit handling')
    bricks=uppercase_fields(fitsio.read(str(SOUTH/'survey-bricks-dr11-south.fits.gz'),columns=['BRICKNAME','RA','DEC']))
    # Centers within0.4deg cover the 0.25deg native bricks, including boundaries.
    neighbors=cKDTree(unit_vectors(bricks['RA'],bricks['DEC'])).query_ball_point(_TARGET_VECTORS,2*np.sin(np.deg2rad(.4)/2))
    owners={}
    for i,nn in enumerate(neighbors):
        for j in nn:owners.setdefault(j,[]).append(i)
    tasks=[];missing=0
    for j,indices in owners.items():
        name=str(bricks['BRICKNAME'][j].decode() if isinstance(bricks['BRICKNAME'][j],bytes) else bricks['BRICKNAME'][j]).strip()
        path=SOUTH/'tractor'/name[:3]/('tractor-'+name+'.fits')
        if not path.exists():missing+=1;continue
        tasks.append((str(path),indices))
    (OUT/'zero_candidates').mkdir(exist_ok=True)
    print('Zero-ID qualified identities',len(ix),'known epochs',int(epoch_known.sum()),'targeted bricks',len(tasks),'missing bricks',missing,flush=True)
    paths=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=8,mp_context=mp.get_context('fork')) as pool:
        for i,path in enumerate(pool.map(brick_candidates,tasks)):
            paths.append(path)
            if i%250==0:print('targeted bricks completed',i,flush=True)
    parts=[]
    for path in paths:
        with np.load(path) as z:
            if len(z['REF_ID']):parts.append({k:z[k] for k in KEYS})
    if not parts:raise RuntimeError('No zero-ID candidates; inspect coordinate/brick inputs')
    d={k:np.concatenate([p[k] for p in parts]) for k in KEYS}
    lsid=(d['RELEASE'].astype('i8')<<40)|(d['BRICKID'].astype('i8')<<20)|d['OBJID'].astype('i8')
    _,first=np.unique(lsid,return_index=True);d={k:v[first] for k,v in d.items()};d['ls_id']=lsid[first]
    d['source_id']=np.where(np.char.strip(d['REF_CAT'].astype('U'))=='G3',d['REF_ID'],0).astype('i8')
    c={k:d[v] for k,v in zip(['ra','dec','epoch','pmra','pmdec'],['RA','DEC','REF_EPOCH','PMRA','PMDEC'])}
    match,sep,unknown,amb=positional_join(q,c)
    atomic_npz(OUT/'zero_photometry.npz',**d)
    atomic_npz(output,identity=s['identity'],source_row=s['original_row'],candidate_row=match,separation_arcsec=sep,unknown_epoch=unknown,ambiguous=amb)
    write_json(marker,dict(qualified_rows=int(qualified.sum()),class_qualified_rows=class_rows,identities=len(ix),matched=int((match>=0).sum()),
        ambiguous=int(amb.sum()),unmatched=int((match<0).sum()),matched_unknown_epoch=int((unknown&(match>=0)).sum()),
        known_source_epochs=int(epoch_known.sum()),bricks=len(tasks),missing_bricks=missing,
        policy='unique native object within2arcsec; accept<=1arcsec both epochs known else<=0.5arcsec static, flagged; one-to-one',
        sha256=sha(output),photometry_sha256=sha(OUT/'zero_photometry.npz')))
    print('Zero-ID matching complete',marker.read_text(),flush=True)


def main():
    require_compute()
    for path,h in json.loads((OUT/'freeze.json').read_text())['hashes'].items():assert sha(ROOT/path)==h
    assert sha(OUT/'source_evidence.npz')==json.loads((OUT/'source_evidence.json').read_text())['output_sha256']
    consolidate();recover_zero()

if __name__=='__main__':main()
