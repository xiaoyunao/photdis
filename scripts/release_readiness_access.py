#!/usr/bin/env python3
"""Validate a native release partition and show how to access unchanged PDFs."""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from photdis.stage1 import require_compute
from v2_1_near import ROOT, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_6_vac_distances import read_columns
from v3_9_unified import owner_rows
from final_science_catalog import indices
from release_readiness_sample import DEST

PDF = ('ls_id','TYPE','qrf_quantiles_kpc','qrf_mu_edges','qrf_mu_mass','qrf_bins')
SCHEMA = ROOT/'configs/photdis_dr11_release.json'


def get_partition(part, overrides):
    frozen=part['region']=='south_v35'
    extras=('RA','DEC') if frozen else ('north_primary_region',)
    p=read_columns(verified(ROOT/part['photo_file'],part['photo_sha256']),('ls_id','TYPE',*extras))
    own,_=owner_rows(part['side'],p,overrides,frozen=frozen)
    if not part['pdf_file']:
        raise ValueError('This partition has no archived PDFs')
    q=read_columns(verified(ROOT/part['pdf_file'],part['pdf_sha256']),PDF+(() if frozen else ('photo_row',)))
    j=np.arange(len(p['ls_id'])) if frozen else q['photo_row']
    np.testing.assert_array_equal(q['ls_id'],p['ls_id'][j])
    np.testing.assert_array_equal(q['TYPE'],p['TYPE'][j])
    if not frozen:
        state=read_columns(verified(ROOT/part['state_file'],part['state_sha256']),('ls_id','TYPE','state'))
        np.testing.assert_array_equal(state['ls_id'],p['ls_id'])
        np.testing.assert_array_equal(state['TYPE'],p['TYPE'])
        np.testing.assert_array_equal(j,np.flatnonzero(state['state']==3))
    result={k:v[own[j]] for k,v in q.items()}
    assert len(result['ls_id'])==int(part['owned_pdf_rows'])
    check_pdf(result)
    return result


def check_pdf(q):
    if not np.all(q['TYPE']=='PSF') or not (np.diff(q['qrf_mu_edges'],axis=1)>0).all():
        raise ValueError('Invalid native PDF rows')
    np.testing.assert_allclose(q['qrf_mu_mass'].sum(axis=1),1,atol=1e-10,rtol=0)
    np.testing.assert_allclose(q['qrf_bins'].sum(axis=1),1,atol=1e-10,rtol=0)


def verify():
    schema=json.loads(SCHEMA.read_text())
    parts,overrides,inputs=indices()
    totals={k:sum(int(p[k]) for p in parts) for k in ('owned_rows','owned_pdf_rows')}
    if dict(**totals,partitions=len(parts))!=schema['native_census']:
        raise ValueError('Machine-readable source statistics failed against the original release')
    for side in ('north','south'):
        area=ROOT/schema['area']['table_'+side]
        inputs[f'{side}_area']=sha(area)
    rows=[]
    for region in ('south_v35','north','south_low','south_faint'):
        part=max((p for p in parts if p['region']==region),key=lambda p:int(p['owned_pdf_rows']))
        raw=get_partition(part,overrides)
        rows.append(dict(region=region,product='raw',partition=part['pdf_file'],
                         rows=len(raw['ls_id']),example_ls_id=int(raw['ls_id'][0]),
                         example_d50_kpc=float(raw['qrf_quantiles_kpc'][0,2])))
    cal=ROOT/schema['calibrated_near_pdf']['receipt']
    cm=json.loads(cal.read_text())
    assert cm['rows']==schema['calibrated_near_pdf']['raw_sidecar_rows_before_release_ownership']
    shard=cm['shards'][0]
    mark=verified(ROOT/shard['receipt'],shard['receipt_sha256'])
    r=json.loads(mark.read_text())
    p=next(k for k in r['outputs'] if k.endswith('.npz'))
    q=read_columns(verified(mark.with_name(p),r['outputs'][p]),PDF)
    check_pdf(q)
    part=next(x for x in parts if x['region']=='south_v35' and
              p==x['sweep']+'_'+Path(x['photo_file']).name)
    raw=get_partition(part,overrides)
    ids,ix=np.unique(raw['ls_id'],return_index=True)
    common,i,j=np.intersect1d(ids,q['ls_id'],return_indices=True)
    assert len(common)
    np.testing.assert_allclose(raw['qrf_quantiles_kpc'][ix[i],2],q['qrf_quantiles_kpc'][j,2],atol=1e-10)
    np.testing.assert_array_equal(raw['qrf_mu_mass'][ix[i]],q['qrf_mu_mass'][j])
    rows.append(dict(region='south_v35',product='calibrated_near',partition=str(mark.with_name(p).relative_to(ROOT)),
                     rows=len(common),example_ls_id=int(common[0]),
                     example_d50_kpc=float(q['qrf_quantiles_kpc'][j[0],2])))
    p=DEST/'access'
    p.mkdir(exist_ok=True,parents=True)
    out=p/'partition_examples.csv'
    csv_rows(out,rows)
    save_json(p/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),schema=sha(SCHEMA),inputs=inputs,
                                   calibrated_receipt=sha(cal),outputs={out.name:sha(out)}))


if __name__=='__main__':
    require_compute()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partition',help='Exact photo_file entry from unified_parts.csv')
    parser.add_argument('--ls-id',type=int,help='Show the archived original PDF for this native PSF')
    a=parser.parse_args()
    if a.partition:
        parts,overrides,_=indices()
        part,=(p for p in parts if p['photo_file']==a.partition)
        t=get_partition(part,overrides)
        j,=np.flatnonzero(t['ls_id']==a.ls_id)
        print(json.dumps({k:v[j].tolist() for k,v in t.items()}))
    else:
        verify()
