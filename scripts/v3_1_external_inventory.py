#!/usr/bin/env python3
"""Allocated APOGEE availability census, with zero native-unverified admissions."""
import json
import os
from pathlib import Path
import fitsio
import numpy as np
from v2_1_near import ROOT,sha,save_json,atomic_npz,csv_rows
from v3_1_inventory import projected
from photdis.stage1 import require_compute,galactic
from photdis.v3_1.sky import qualified_sky
from photdis.spectroscopy import APOGEE_COLUMNS,apogee_best_rows,project_apogee


def main():
    require_compute(); headers=json.loads((ROOT/'manifests/v3_1/external_reference_headers.json').read_text())
    paths=[Path(r['path']) for r in headers]; old=ROOT/'data/v3_0/references.npz'
    configs=sorted((ROOT/'configs').glob('field*.json'))
    provenance=paths+[old,ROOT/'scripts/v3_1_external_inventory.py',ROOT/'src/photdis/spectroscopy.py',ROOT/'docs/v3_1_external_sources.md',ROOT/'src/photdis/v3_1/sky.py']+configs
    inputs={str(p):sha(p) for p in provenance}
    out=ROOT/'data/v3_1/external_inventory_finite_sky';out.mkdir(parents=True,exist_ok=True);marker=out/'complete.json'
    if marker.exists():
        record=json.loads(marker.read_text());assert record['inputs']==inputs
        for name,digest in record['outputs'].items():assert sha(out/name)==digest
        return
    allstar=fitsio.read(str(paths[0]),ext=1,columns=APOGEE_COLUMNS+['FIELD'])
    nn=fitsio.read(str(paths[1]),ext=1,columns=['APOGEE_ID','RA_APOGEE','DEC_APOGEE','source_id','dist','dist_error','nn_parallax'])
    assert len(allstar)==len(nn)
    name_match=np.char.strip(allstar['APOGEE_ID'].astype('U'))==np.char.strip(nn['APOGEE_ID'].astype('U'))
    csv_rows(out/'alignment_exclusions.csv',[dict(row=int(i),allstar=str(allstar['APOGEE_ID'][i]),astroNN=str(nn['APOGEE_ID'][i]),
        reason='Unequal exact APOGEE_ID; astroNN distance excluded, no suffix correction') for i in np.flatnonzero(~name_match)])
    np.testing.assert_allclose(allstar['RA'],nn['RA_APOGEE'],atol=1e-7,rtol=0,equal_nan=True)
    np.testing.assert_allclose(allstar['DEC'],nn['DEC_APOGEE'],atol=1e-7,rtol=0,equal_nan=True)
    assert nn['source_id'].dtype.kind in 'iu','Never round floating-point Gaia IDs'
    representative=apogee_best_rows(allstar);a=allstar[representative];n=nn[representative];s=project_apogee(a)
    ids=s['source_id'];same_id=(n['source_id']==ids)&(ids>0)&name_match[representative]
    reference=projected(old,['identity','cell']);seen_id=np.isin(ids,reference['identity'])
    sky=qualified_sky(s['ra'],s['dec']);cell=sky['cell']
    seen_sky=np.isin(cell,reference['cell']);bounds=[json.loads(p.read_text()).get('bounds_deg') for p in configs]
    bounds=[x for x in bounds if x]+[[10,-45,15,-40],[350,-45,355,-40]]
    footprint=np.zeros(len(ids),bool)
    for lo,de,hi,dh in bounds:
        inside=((s['ra']>=lo)&(s['ra']<=hi)) if lo<=hi else ((s['ra']>=lo)|(s['ra']<=hi))
        footprint |= inside & (s['dec']>=de) & (s['dec']<=dh)
    highlat=sky['valid_coordinates']&(abs(sky['latitude'])>30)
    distance=n['dist']/1000.;error=n['dist_error']/1000.
    available=same_id&np.isfinite(distance)&(distance>0)&np.isfinite(error)&(error>0)
    relative=np.divide(error,distance,out=np.full(len(ids),np.inf),where=available)
    precise=available&(relative<.2)
    # Distance availability does not require a qualified FeH/class label.
    base=(s['quality']|precise)&(ids>0)&highlat
    stages=dict(unique_inventory=np.ones(len(ids),bool),qualified_highlat=base,
        qualified_new_identity=base&~seen_id,qualified_new_identity_outside_fields=base&~seen_id&~footprint,
        qualified_new_identity_sky_fields=base&~seen_id&~seen_sky&~footprint)
    rows=[];coverage=[]
    for name,mask in stages.items():
        rows.append(dict(stage=name,n=int(mask.sum()),cells=len(np.unique(cell[mask])),apogee_fields=len(np.unique(a['FIELD'][mask])),
            dwarfs=int((mask&s['label_dwarf']).sum()),giants=int((mask&s['label_giant']).sum()),
            poor_giants=int((mask&s['label_giant']&(s['feh'] < -1.5)).sum()),
            spectroscopic_distance=int((mask&available).sum()),distance_relative_error_lt20percent=int((mask&precise).sum()),
            native_TYPE_unverified=int(mask.sum()),admitted_science=0))
        for kind,label in [('all',np.ones(len(ids),bool)),('dwarf',s['label_dwarf']),('giant',s['label_giant']),('poor_giant',s['label_giant']&(s['feh'] < -1.5))]:
            for lo,hi in zip([0,2,5,10,20,50],[2,5,10,20,50,np.inf]):
                m=mask&label&precise&(distance>=lo)&(distance<hi)
                coverage.append(dict(stage=name,reference_class=kind,low_kpc=lo,high_kpc=None if np.isinf(hi) else hi,n=int(m.sum()),
                    cells=len(np.unique(cell[m])),fields=len(np.unique(a['FIELD'][m]))))
    csv_rows(out/'availability.csv',rows);csv_rows(out/'distance_coverage.csv',coverage)
    # Acquisition inventory only: no TYPE inference from APOGEE/Gaia classification.
    use=stages['qualified_new_identity_outside_fields'];candidate={k:s[k][use] for k in s.dtype.names}
    candidate.update(original_row=representative[use],apogee_field=np.char.strip(a['FIELD'][use].astype('U')),
        cell=cell[use],seen_reference_sky=seen_sky[use],distance_kpc=distance[use],distance_error_kpc=error[use],
        distance_available=available[use],distance_precise=precise[use],TYPE=np.full(use.sum(),'',dtype='U4'))
    atomic_npz(out/'acquisition.npz',**candidate)
    members=fitsio.read(str(paths[2]),ext=1,columns=['GC_NAME','GAIAEDR3_SOURCE_ID','RA','DEC'])
    cluster=[]
    for name in np.unique(members['GC_NAME']):
        m=members['GC_NAME']==name; sid=members['GAIAEDR3_SOURCE_ID'][m];positive=sid>0
        cluster.append(dict(name=str(name.decode() if isinstance(name,bytes) else name).strip(),rows=int(m.sum()),
            positive_unique_ids=len(np.unique(sid[positive])),outside_v3_ids=len(np.unique(sid[positive&~np.isin(sid,reference['identity'])])),
            native_TYPE_unverified=int(m.sum()),admitted_science=0,membership='local v1.0 membership definition not yet certified'))
    csv_rows(out/'cluster_inventory.csv',cluster)
    save_json(marker,dict(inputs=inputs,outputs={p.name:sha(p) for p in out.iterdir() if p.name!='complete.json'},job=os.getenv('SLURM_JOB_ID'),
        source_rows=len(allstar),unique_identities=len(ids),invalid_coordinate_exclusions=int((~sky['valid_coordinates']).sum()),
        high_latitude_inventory=int(highlat.sum()),position_alignment_passed=True,exact_name_mismatches=int((~name_match).sum()),
        distance_requires_exact_name_and_positive_Gaia_identity=True,
        positive_Gaia_astroNN_mismatch=int(((ids>0)&~same_id).sum()),
        acquisition_rows=int(use.sum()),scientific_admissions=0,
        scope='Reference availability only; native TYPE/photometry and full historical source/system audit precede final role freeze. Distances are pc converted to kpc from producer code, never weighted_dist. Gaia-trained teacher dependence retained. No prediction residuals read.'))
    print('V3.1 EXTERNAL INVENTORY COMPLETE',int(use.sum()),flush=True)


if __name__=='__main__':main()
