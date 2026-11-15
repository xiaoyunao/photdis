#!/usr/bin/env python3
"""Project the frozen v3.5 posterior parts once for full-catalog comparisons."""
import argparse
import csv
import fcntl
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from photdis.stage1 import require_compute
from v2_1_near import ROOT, atomic_npz, csv_rows, load, save_json, sha
from v3_3_sed import verified
from v3_5_dr11_production import CUTS, FREEZE, PRODUCTION
from v3_5_dr11_spatial import DELIVERY
from v3_5_dr11_validate import join_parts
from v3_6_reference_index import OUT as INDEX, workers


OUT = ROOT/'data/v3_6/catalog_projection'
EDGES = np.array([0.,2.,5.,10.,np.inf])
KEYS = ('ls_id','source_id','identity','TYPE','RA','DEC','observed_r','dereddened_g_r',
        'parent_row','stratum','band_state','dwarf_selection_score','nominal_qnear',
        'nominal_evaluated','qrf_quantiles_kpc','qrf_qnear','qrf_bins','primary','strict')


def project(name,receipt,tile,offset,expected):
    work = PRODUCTION/Path(name).stem
    marker = json.loads(verified(work/'complete.json',receipt).read_text())
    chunks = []
    for number,(part,digest) in enumerate(sorted((k,v) for k,v in marker['outputs'].items()
                                                 if k.startswith('part_'))):
        with np.load(verified(work/part,digest),allow_pickle=False) as z:
            p = {k:z[k] for k in KEYS}
            p.update(sweep_number=np.full(len(p['ls_id']),tile,'i2'),
                     part_number=np.full(len(p['ls_id']),number,'i2'),
                     part_row=np.arange(len(p['ls_id']),dtype='i4'))
            chunks.append(p)
    if not chunks:
        if len(expected):
            raise ValueError(f'Missing forecast parts in {name}')
        return {}
    p = join_parts(chunks)
    np.testing.assert_array_equal(p['ls_id'],expected)
    if not (p['TYPE']=='PSF').all():
        raise ValueError(f'Missing native PSF in {name}')
    np.testing.assert_array_equal(p['nominal_evaluated'],np.isfinite(p['nominal_qnear']))
    np.testing.assert_allclose(p['qrf_bins'].sum(axis=1),1.,atol=1e-12,rtol=0)
    p['catalog_row'] = offset+np.arange(len(expected),dtype='i8')
    return p


def summarize(p,cuts):
    nominal,qrf,dwarf = (p[k] for k in ('nominal_qnear','qrf_qnear','dwarf_selection_score'))
    d50 = p['qrf_quantiles_kpc'][:,2]
    hard = np.searchsorted(EDGES,d50,side='right')-1
    if not ((hard>=0)&(hard<len(EDGES)-1)).all():
        raise ValueError('QRF median outside delivered distance support')
    q = p['qrf_bins']
    shells = np.column_stack((q[:,:3],q[:,3:].sum(axis=1)))
    rows = []
    for group,select in (('all',np.ones(len(d50),bool)),('primary',p['primary']),('strict',p['strict'])):
        for band,observed in (('all',np.ones(len(d50),bool)),('W1_usable',p['band_state'][:,4]==4),
                              ('W1_not_usable',p['band_state'][:,4]!=4)):
            use = select&observed
            compared = use&p['nominal_evaluated']
            values = {}
            for cut,gates in cuts.items():
                prob = gates['minimum_near_probability']
                select_dwarf = dwarf>=gates['minimum_dwarf_score']
                values[f'{cut}_dwarf_score_pass'] = int((use&select_dwarf).sum())
                values[f'{cut}_nominal_near_qrf_below_gate'] = int((compared&(nominal>=prob)&(qrf<prob)).sum())
                values[f'{cut}_qrf_near_nominal_below_gate'] = int((compared&(qrf>=prob)&(nominal<prob)).sum())
            rows.append(dict(selection=group,W1=band,rows=int(use.sum()),
                nominal_evaluated=int(compared.sum()),nominal_qnear_sum=float(nominal[compared].sum()),
                qrf_qnear_sum=float(qrf[use].sum()),qrf_qnear_sum_on_nominal_rows=float(qrf[compared].sum()),
                dwarf_score_sum=float(dwarf[use].sum()),qrf_d50_sum_kpc=float(d50[use].sum()),
                modal_shell_differs_from_median_n=int((use&(hard!=np.argmax(shells,axis=1))).sum()),
                **{f'posterior_shell_{k}':float(n) for k,n in enumerate(shells[use].sum(axis=0))},
                **{f'median_shell_{k}':int(n) for k,n in enumerate(np.bincount(hard[use],minlength=4))},**values))
    return rows


def main(group):
    require_compute()
    dest = OUT/f'{group:02d}'
    dest.mkdir(parents=True,exist_ok=True)
    with (dest/'writer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        project_group(group,dest)


def project_group(group,dest):
    delivery_path = DELIVERY/'complete.json'
    delivery = json.loads(delivery_path.read_text())
    freeze = json.loads(verified(FREEZE,delivery['production_freeze_sha256']).read_text())
    group_names = freeze['groups'][group]
    index_path = INDEX/'complete.json'
    inputs = {str(x.relative_to(ROOT)):sha(x) for x in (delivery_path,index_path,FREEZE,CUTS)}
    receipt = dest/'complete.json'
    if receipt.exists():
        record = json.loads(receipt.read_text())
        if (record['group']!=group or set(record['sweeps'])!=set(group_names)
                or any(record['inputs'].get(k)!=v for k,v in inputs.items())):
            raise ValueError(f'Existing projection has different frozen inputs: {receipt}')
        for name in ('rows.npz','summary.csv'):
            verified(dest/name,record['outputs'][name])
        print(f'{group:02d}: verified existing {record["rows"]} projected rows',flush=True)
        return
    ix = json.loads(index_path.read_text())
    catalog = load(verified(INDEX/'catalog_rows.npz',ix['outputs']['catalog_rows.npz']))
    with verified(DELIVERY/'sweeps.csv',delivery['outputs']['sweeps.csv']).open() as stream:
        receipts = {r['sweep']:r['receipt_sha256'] for r in csv.DictReader(stream)}
    tile = {name:i for i,name in enumerate(catalog['sweep'])}
    off = catalog['sweep_offset']
    def read(name):
        i = tile[name]
        return project(name,receipts[name],i,off[i],catalog['ls_id'][off[i]:off[i+1]])
    with ThreadPoolExecutor(max_workers=workers()) as pool:
        p = join_parts([a for a in pool.map(read,group_names) if a])
    want = sum(off[tile[n]+1]-off[tile[n]] for n in group_names)
    if len(p['ls_id'])!=want:
        raise ValueError('Projected catalog group lost supported rows')
    atomic_npz(dest/'rows.npz',**p)
    cfg = json.loads(CUTS.read_text())
    csv_rows(dest/'summary.csv',summarize(p,{x:cfg[x] for x in ('primary','strict')}))
    save_json(receipt,dict(job=os.getenv('SLURM_JOB_ID'),group=group,rows=len(p['ls_id']),
              sweeps={n:receipts[n] for n in group_names},
              inputs={**inputs,str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))},
              outputs={x:sha(dest/x) for x in ('rows.npz','summary.csv')}))
    print(f'{group:02d}: {len(p["ls_id"])} projected native PSF rows',flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('group',type=int,choices=range(48))
    main(parser.parse_args().group)
