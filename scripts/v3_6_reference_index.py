#!/usr/bin/env python3
"""Associate the archived reference parents with the verified v3.5 catalog."""
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from photdis.reference_v1_4 import unique_join
from photdis.stage1 import require_compute
from v2_1_near import ROOT, atomic_npz, csv_rows, load, save_json, sha
from v3_3_sed import verified
from v3_4_near_reference_support import separation
from v3_5_dr11_production import FREEZE, PRODUCTION
from v3_5_dr11_spatial import DELIVERY


OUT = ROOT/'data/v3_6/reference_index'
STATUSES = ('no_supported_match','conflicting_keys','position_mismatch','legacy_key','gaia_key')


def identities(receipt):
    dest = PRODUCTION/Path(receipt['sweep']).stem
    path = verified(dest/'complete.json',receipt['receipt_sha256'])
    record = json.loads(path.read_text())
    if not record['all_PSF'] or record['source'] != receipt['sweep']:
        raise ValueError(f'Unverified native PSF identity roster: {path}')
    data = load(verified(dest/'identities.npz',record['outputs']['identities.npz']))
    if len(data['ls_id']) != record['counts'].get('rows',0):
        raise ValueError(f'Catalog roster size differs from receipt: {path}')
    return data


def association(query, catalog):
    legacy = unique_join(query['ls_id'],catalog['ls_id'])
    source = np.flatnonzero(catalog['source_id']>0)
    gaia = np.full(len(legacy),-1,'i8')
    if len(source):
        match = unique_join(query['source_id'],catalog['source_id'][source])
        valid = match>=0
        gaia[valid] = source[match[valid]]
    conflict = (legacy>=0)&(gaia>=0)&(legacy!=gaia)
    row = np.where(legacy>=0,legacy,gaia)
    status = np.where(row<0,0,np.where(legacy>=0,3,4)).astype('u1')
    status[conflict] = 1
    return row,status


def check_coordinates(query, matches, offsets, receipts):
    row,status = matches
    candidates = np.flatnonzero(status>=3)
    owner = np.searchsorted(offsets[1:],row[candidates],side='right')
    def check_sweep(index):
        refs = candidates[owner==index]
        within = row[refs]-offsets[index]
        assert (within>=0).all() and (within<offsets[index+1]-offsets[index]).all()
        dest = PRODUCTION/Path(receipts[index]['sweep']).stem
        record = json.loads((dest/'complete.json').read_text())
        start = 0
        for name,digest in sorted(record['outputs'].items()):
            if not name.startswith('part_'):
                continue
            with np.load(verified(dest/name,digest)) as z:
                n = len(z['ls_id'])
                in_part = (within>=start)&(within<start+n)
                if in_part.any():
                    query_rows = refs[in_part]
                    j = within[in_part]-start
                    if not (z['TYPE'][j]=='PSF').all():
                        raise ValueError(f'Native output morphology changed: {dest/name}')
                    close = separation(query['RA'][query_rows],query['DEC'][query_rows],
                                       z['RA'][j],z['DEC'][j])
                    status[query_rows[~close]] = 2
            start += n
        if start != offsets[index+1]-offsets[index]:
            raise ValueError(f'Partition row order does not reconcile: {dest}')
    with ThreadPoolExecutor(max_workers=workers()) as pool:
        list(pool.map(check_sweep,np.unique(owner)))
    return np.where(status>=3,row,-1),status


def workers():
    return min(12,int(os.getenv('SLURM_CPUS_PER_TASK','1')))


def main():
    require_compute()
    delivery_file = DELIVERY/'complete.json'
    delivery = json.loads(delivery_file.read_text())
    freeze = json.loads(verified(FREEZE,delivery['production_freeze_sha256']).read_text())
    with verified(DELIVERY/'sweeps.csv',delivery['outputs']['sweeps.csv']).open() as stream:
        receipts = list(csv.DictReader(stream))
    if [r['sweep'] for r in receipts] != [n for group in freeze['groups'] for n in group]:
        raise ValueError('Delivery and production have different sweep order')
    with ThreadPoolExecutor(max_workers=workers()) as pool:
        chunks = list(pool.map(identities,receipts))
    offsets = np.r_[0,np.cumsum([len(t['ls_id']) for t in chunks],dtype='i8')]
    catalog = {k:np.concatenate([t[k] for t in chunks]) for k in ('ls_id','source_id')}
    if len(np.unique(catalog['ls_id'])) != delivery['totals']['rows']:
        raise ValueError('Global Legacy identities changed')
    preparation = ROOT/'data/v3_0/preparation.json'
    provenance = ROOT/'data/v3_0/preparation_freeze.json'
    archive = ROOT/'data/v3_0/references.npz'
    older = ROOT/'data/baseline_v1_4/references.npz'
    sources = {'v3_0':verified(archive,json.loads(preparation.read_text())['output_sha256']),
               'baseline_v1_4':verified(older,json.loads(provenance.read_text())[str(older.relative_to(ROOT))])}
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'complete.json').exists():
        raise FileExistsError(OUT/'complete.json')
    summaries = []
    for name,path in sources.items():
        with np.load(path) as f:
            query = {k:f[k] for k in ('ls_id','source_id','RA','DEC')}
        row,status = check_coordinates(query,association(query,catalog),offsets,receipts)
        atomic_npz(OUT/f'{name}.npz',reference_row=np.arange(len(row)),catalog_row=row,match_status=status)
        summaries.append(dict(reference=name,raw_rows=len(row),**{
            label:int((status==i).sum()) for i,label in enumerate(STATUSES)}))
        print(summaries[-1],flush=True)
    csv_rows(OUT/'association.csv',summaries)
    atomic_npz(OUT/'catalog_rows.npz',**catalog,sweep_offset=offsets,
               sweep=np.asarray([r['sweep'] for r in receipts]))
    save_json(OUT/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),supported_rows=int(offsets[-1]),
              inputs={str(p.relative_to(ROOT)):sha(p) for p in (delivery_file,FREEZE,Path(__file__),
                                                               preparation,provenance,*sources.values())},
              statuses=STATUSES,outputs={p.name:sha(p) for p in sorted(OUT.iterdir()) if p.name!='complete.json'}))


if __name__ == '__main__':
    main()
