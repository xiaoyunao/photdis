#!/usr/bin/env python3
"""Native DR11 qualification of external reference-availability inventory.

No predictions, residuals or fitted correction. Exact G3 identity joins only;
final prospective roles require a separate all-history audit and freeze.
"""
import concurrent.futures
import json
import multiprocessing as mp
import os
from pathlib import Path
import fitsio
import numpy as np
from scipy.spatial import cKDTree
from v2_1_near import ROOT,load,sha,save_json,atomic_npz,csv_rows
from photdis.stage1 import require_compute
from photdis.assembly_v1_4 import unit_vectors,uppercase_fields
from photdis.reference_v1_4 import unique_join
from v3_0_native import RAW,native_payload

SOUTH=Path('/global/cfs/projectdirs/cosmo/data/legacysurvey/dr11/south')
_IDS=_OUT=None


def brick(task):
    name,indices=task;path=SOUTH/'tractor'/name[:3]/('tractor-'+name+'.fits')
    marker=_OUT/'bricks'/f'{name}.json';destination=marker.with_suffix('.npz')
    if not path.exists():return dict(brick=name,missing_file=True)
    stat=path.stat();stamp=dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns)
    if marker.exists():
        old=json.loads(marker.read_text());assert old['native_stat']==stamp
        if old['matched_primary']:assert sha(destination)==old['sha256']
        return old
    with fitsio.FITS(str(path)) as f:
        names={x.upper():x for x in f[1].get_colnames()}
        mandatory=['REF_ID','REF_CAT','BRICK_PRIMARY']
        if not all(k in names for k in mandatory):raise RuntimeError('Missing exact identity/primary columns: '+str(path))
        identity=uppercase_fields(f[1].read(columns=[names[k] for k in mandatory]))
        exact=(np.char.strip(identity['REF_CAT'].astype('U'))=='G3')&np.isin(identity['REF_ID'],_IDS[indices])
        primary=exact&identity['BRICK_PRIMARY'];rows=np.flatnonzero(primary)
        if 'TYPE' not in names:
            result=dict(brick=name,native_path=str(path),native_stat=stamp,exact=int(exact.sum()),
                matched_primary=0,missing_TYPE=int(primary.sum()),nonprimary=int((exact&~primary).sum()))
        elif len(rows):
            raw=uppercase_fields(f[1].read(columns=[names[k] for k in RAW if k in names],rows=rows))
            payload=native_payload(raw,'DECam_DR11');payload['native_row']=rows
            atomic_npz(destination,**payload)
            result=dict(brick=name,native_path=str(path),native_stat=stamp,exact=int(exact.sum()),
                matched_primary=len(rows),missing_TYPE=int((payload['TYPE']=='').sum()),
                nonprimary=int((exact&~primary).sum()),sha256=sha(destination))
        else:result=dict(brick=name,native_path=str(path),native_stat=stamp,exact=int(exact.sum()),matched_primary=0,missing_TYPE=0,nonprimary=int(exact.sum()))
    after=path.stat();assert stamp==dict(size=after.st_size,mtime_ns=after.st_mtime_ns)
    save_json(marker,result);return result


def main():
    global _IDS,_OUT
    require_compute();source=ROOT/'data/v3_1/external_inventory_finite_sky/acquisition.npz'
    inventory=json.loads(source.with_name('complete.json').read_text());assert sha(source)==inventory['outputs'][source.name]
    bricks_path=SOUTH/'survey-bricks-dr11-south.fits.gz'
    paths=[source,source.with_name('complete.json'),bricks_path,ROOT/'scripts/v3_1_external_native.py',ROOT/'scripts/v3_0_native.py',
        ROOT/'src/photdis/photometry.py',ROOT/'src/photdis/v3/data.py']
    configs=sorted((ROOT/'configs').glob('field*.json'));paths+=configs
    inputs={str(p):sha(p) for p in paths};_OUT=ROOT/'data/v3_1/external_native';_OUT.mkdir(exist_ok=True)
    freeze=_OUT/'inputs.json';marker=_OUT/'complete.json'
    if freeze.exists():assert json.loads(freeze.read_text())==inputs
    else:save_json(freeze,inputs)
    if marker.exists():
        saved=json.loads(marker.read_text());assert saved['inputs']==inputs
        for name,digest in saved['outputs'].items():assert sha(_OUT/name)==digest
        return
    acquisition=load(source);_IDS=acquisition['source_id'];assert (_IDS>0).all() and len(np.unique(_IDS))==len(_IDS)
    positions=unit_vectors(acquisition['ra'],acquisition['dec']);assert np.isfinite(positions).all()
    b=uppercase_fields(fitsio.read(str(bricks_path),columns=['BRICKNAME','RA','DEC']))
    neighbors=cKDTree(unit_vectors(b['RA'],b['DEC'])).query_ball_point(positions,2*np.sin(np.deg2rad(.4)/2))
    owners={}
    for i,near in enumerate(neighbors):
        for j in near:owners.setdefault(int(j),[]).append(i)
    tasks=[(str(b['BRICKNAME'][j].decode() if isinstance(b['BRICKNAME'][j],bytes) else b['BRICKNAME'][j]).strip(),np.array(ix)) for j,ix in sorted(owners.items())]
    (_OUT/'bricks').mkdir(exist_ok=True)
    print('EXTERNAL NATIVE',len(_IDS),'inventory identities',len(tasks),'targeted bricks',flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=16,mp_context=mp.get_context('fork')) as pool:
        records=[]
        for i,r in enumerate(pool.map(brick,tasks)):
            records.append(r)
            if i%250==0:print('BRICKS',i,flush=True)
    save_json(_OUT/'native_sources.json',records)
    pieces=[load(_OUT/'bricks'/f"{r['brick']}.npz") for r in records if r.get('matched_primary',0)]
    if not pieces:raise RuntimeError('No exact primary native matches; inventory and provenance retained')
    data={k:np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
    assert all(set(p)==set(data) for p in pieces)
    # Primary brick ownership should be unique; fail rather than silently pick.
    assert len(np.unique(data['ls_id']))==len(data['ls_id']),'Duplicate primary native owner'
    ids,inverse,counts=np.unique(data['identity'],return_inverse=True,return_counts=True)
    distinct=counts[inverse]==1;psf=data['TYPE']=='PSF';verified=data['RELEASE']==11010
    selected=distinct&psf&verified&data['photometric_entry']
    # Reapply all historical field footprints at native coordinates.
    configs=sorted((ROOT/'configs').glob('field*.json'))
    bounds=[json.loads(p.read_text()).get('bounds_deg') for p in configs]
    bounds=[x for x in bounds if x]+[[10,-45,15,-40],[350,-45,355,-40]]
    footprint=np.zeros(len(selected),bool)
    for lo,de,hi,dh in bounds:
        inside=((data['RA']>=lo)&(data['RA']<=hi)) if lo<=hi else ((data['RA']>=lo)|(data['RA']<=hi))
        footprint|=inside&(data['DEC']>=de)&(data['DEC']<=dh)
    selected&=~footprint
    stages=dict(exact_primary=np.ones(len(selected),bool),unique_identity=distinct,
        native_DR11_PSF=distinct&psf&verified,photometric_domain=distinct&psf&verified&data['photometric_entry'],
        outside_recorded_fields=selected)
    rows=[]
    for stage,m in stages.items():
        rows.append(dict(stage=stage,n=int(m.sum()),cells=len(np.unique(data['cell'][m])),
            nonPSF=int((m&~psf&(data['TYPE']!='')).sum()),missing_TYPE=int((m&(data['TYPE']=='')).sum())))
    csv_rows(_OUT/'admission.csv',rows)
    native={k:v[selected] for k,v in data.items()};reference_rows=unique_join(native['identity'],_IDS);assert (reference_rows>=0).all()
    reference={k:v[reference_rows] for k,v in acquisition.items()};reference['TYPE']=native['TYPE'].copy()
    reference['identity']=native['identity'];reference['ls_id']=native['ls_id'];reference['native_cell']=native['cell']
    assert (native['TYPE']=='PSF').all() and (reference['TYPE']=='PSF').all()
    atomic_npz(_OUT/'photometry.npz',**native);atomic_npz(_OUT/'references.npz',**reference)
    missing_type=sum(r.get('missing_TYPE',0) for r in records)
    save_json(marker,dict(inputs=inputs,outputs={name:sha(_OUT/name) for name in ['photometry.npz','references.npz','admission.csv','native_sources.json']},
        field_config_hashes={str(p.relative_to(ROOT)):sha(p) for p in configs},job=os.getenv('SLURM_JOB_ID'),
        acquisition_rows=len(_IDS),matched_unique_ids=len(ids),unmatched_inventory=int((~np.isin(_IDS,ids)).sum()),
        missing_TYPE_exclusions=missing_type,nonPSF_exclusions=int(((data['TYPE']!='')&~psf).sum()),
        duplicate_source_exclusions=int((~distinct).sum()),scientific_PSF_admissions=int(selected.sum()),
        targeted_bricks=len(tasks),missing_brick_files=sum(r.get('missing_file',False) for r in records),
        final_test_role_assigned=False,predictions_read=False,
        limits='Exact positive G3 ID only, within0.4deg brick-center search at APOGEE positions; absent match is not proof of absent imaging. Historical source/system ancestry and final role freeze pending. Reference availability selection does not represent full target population.'))
    print('EXTERNAL NATIVE COMPLETE',int(selected.sum()),flush=True)


if __name__=='__main__':main()
