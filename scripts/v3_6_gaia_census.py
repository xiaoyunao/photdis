#!/usr/bin/env python3
"""Join Gaia DR3 once by source-ID tile across the full supported DR11 frame."""
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from photdis.gaia import NPIX, SOURCE_ID_HEALPIX_FACTOR, derive_gaia_labels, join_gaia
from photdis.stage1 import require_compute
from photdis.v3.metrics import distance_summary
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_4_near_reference_support import separation
from v3_4_near_validate import GAIA, geometry, tile_inventory
from v3_5_dr11_production import CUTS
from v3_5_dr11_validate import join_parts
from v3_6_catalog_projection import OUT as PROJECTION
from v3_6_vac_distances import DISTANCE_EDGES, read_columns


OUT = ROOT/'data/v3_6/gaia_census'
FIELDS = ('catalog_row','ls_id','source_id','RA','DEC','TYPE','primary','strict',
          'observed_r','dereddened_g_r','band_state','qrf_quantiles_kpc','qrf_bins',
          'nominal_qnear','qrf_qnear','dwarf_selection_score')
STATES = ('no_valid_native_gaia_id','dr3_key_not_found','dr3_position_mismatch',
          'unqualified_astrometry','weak_signed_parallax','intermediate_signed_parallax',
          'geometry_reference')
_STATE = None


def distances(p,qualified,flags,gaia,cfg):
    close = separation(p['RA'],p['DEC'],gaia['ra'],gaia['dec'])
    usable = qualified&close&flags['astro_quality']&flags['zpt_valid']&flags['finite_parallax']
    d,lo,hi = geometry(flags['p_corr'],gaia['parallax_error'],usable,cfg)
    stat = np.zeros(len(d),'u1')
    stat[qualified] = 1
    stat[gaia['gaia_match']] = 2
    stat[gaia['gaia_match']&close] = 3
    stat[usable] = 5
    stat[usable&(flags['snr_corr']<5)] = 4
    stat[np.isfinite(d)] = 6
    return stat,d,lo,hi


def worker(item):
    shard,rows = item
    marker = OUT/f'{shard:02d}.json'
    frozen = sha(OUT/'freeze.json')
    if marker.exists():
        previous = json.loads(marker.read_text())
        if previous['freeze_sha256']!=frozen or previous['rows']!=len(rows):
            raise ValueError(f'Gaia shard already uses a different input freeze: {shard}')
        for name,digest in previous['outputs'].items():
            verified(OUT/name,digest)
        return shard,len(rows)
    catalog,cfg = _STATE
    p = {k:v[rows] for k,v in catalog.items()}
    ids = p['source_id']
    if not (p['TYPE']=='PSF').all():
        raise ValueError('Full Gaia census requires native PSFs')
    valid = (ids>0)&(ids<NPIX*SOURCE_ID_HEALPIX_FACTOR)
    gaia = join_gaia(np.where(valid,ids,0),GAIA)
    labels = derive_gaia_labels(gaia)
    state,center,lo,hi = distances(p,valid,labels,gaia,cfg)
    summary = []
    ref = np.full(len(ids),np.nan)
    good = np.isfinite(center)
    ref[good] = 10+5*np.log10(center[good])
    mu = 10+5*np.log10(p['qrf_quantiles_kpc'])
    blue = (p['dereddened_g_r']>=.5)&(p['dereddened_g_r']<.8)
    for selection,selected in (('supported',np.ones(len(ids),bool)),('primary',p['primary']),
                               ('strict',p['strict'])):
        for cohort,group in (('all',np.ones(len(ids),bool)),('blue',blue),
                             ('faint_blue',blue&(p['observed_r']>=18))):
            keep = selected&group
            for k,name in enumerate(STATES):
                use = keep&(state==k)
                summary.append(dict(selection=selection,cohort=cohort,gaia_state=name,n=int(use.sum()),
                                    negative_corrected=int((use&labels['negative_corrected_parallax']).sum())))
    performance = []
    for selection,selected in (('supported',np.ones(len(ids),bool)),('primary',p['primary']),
                               ('strict',p['strict'])):
        for a,b in zip(DISTANCE_EDGES[:-1],DISTANCE_EDGES[1:]):
            use = selected&(center>=a)&(center<b)
            performance.append(dict(shard=shard,selection=selection,reference_lo_kpc=a,
                                    reference_hi_kpc=b,**distance_summary(mu[use],ref[use])))
    out = OUT/f'{shard:02d}.npz'
    audit = OUT/f'{shard:02d}_counts.csv'
    metrics = OUT/f'{shard:02d}_geometry.csv'
    retained_gaia = ('parallax','parallax_error','ruwe','ra','dec','ref_epoch','pmra','pmdec',
                     'gaia_file_missing','phot_g_mean_mag','bp_rp')
    retained_flags = ('p_corr','zpt_mas','snr_corr','astro_quality','zpt_valid','gaia_match')
    atomic_npz(out,**p,gaia_state=state,reference_kpc=center,
               reference_lo_kpc=lo,reference_hi_kpc=hi,
               **{f'gaia_{k}':gaia[k] for k in retained_gaia},**{k:labels[k] for k in retained_flags})
    csv_rows(audit,summary)
    csv_rows(metrics,performance)
    save_json(marker,dict(shard=shard,rows=len(ids),freeze_sha256=frozen,
                          outputs={x.name:sha(x) for x in (out,audit,metrics)}))
    print(f'Gaia ID region {shard}: {len(ids)} rows, {int((state>=4).sum())} signed references',flush=True)
    return shard,len(ids)


def main():
    global _STATE
    require_compute()
    source = PROJECTION/'complete.json'
    receipt = json.loads(source.read_text())
    parts = []
    for name,digest in receipt['child_receipts'].items():
        marker = verified(ROOT/name,digest)
        child = json.loads(marker.read_text())
        path = verified(marker.with_name('rows.npz'),child['outputs']['rows.npz'])
        parts.append(read_columns(path,FIELDS))
    catalog = join_parts(parts)
    del parts
    assert len(catalog['ls_id'])==receipt['rows']
    np.testing.assert_array_equal(np.sort(catalog['catalog_row']),np.arange(receipt['rows']))
    ids = catalog['source_id']
    valid = (ids>0)&(ids<NPIX*SOURCE_ID_HEALPIX_FACTOR)
    if len(np.unique(ids[valid]))!=valid.sum():
        raise ValueError('Gaia aliases require resolved ownership before joint counts')
    zones = np.where(valid,ids//SOURCE_ID_HEALPIX_FACTOR//512,24)
    order = np.argsort(zones,kind='stable')
    groups = np.split(order,np.flatnonzero(np.diff(zones[order]))+1)
    jobs = [(int(zones[j[0]]),j) for j in groups if len(j)]
    OUT.mkdir(parents=True,exist_ok=True)
    frozen = OUT/'freeze.json'
    record = dict(inputs={str(p.relative_to(ROOT)):sha(p) for p in
                         (source,Path(__file__),CUTS,ROOT/'src/photdis/gaia.py')},
                  tiles=tile_inventory(ids[valid]),statuses=STATES,shards={str(k):len(v) for k,v in jobs})
    if frozen.exists():
        if json.loads(frozen.read_text())!=json.loads(json.dumps(record)):
            raise ValueError('Gaia census inputs differ from the existing freeze')
    else:
        save_json(frozen,record)
    _STATE = catalog,json.loads(CUTS.read_text())
    with ProcessPoolExecutor(max_workers=min(25,int(os.getenv('SLURM_CPUS_PER_TASK','1'))),
                             mp_context=mp.get_context('fork')) as pool:
        result = list(pool.map(worker,jobs))
    if sum(n for _,n in result)!=receipt['rows']:
        raise ValueError('Gaia regions do not reconcile to the full supported frame')
    save_json(OUT/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),rows=receipt['rows'],
        freeze_sha256=sha(frozen),statuses=STATES,child_receipts={
            f'{k:02d}.json':sha(OUT/f'{k:02d}.json') for k,_ in result}))


if __name__=='__main__':
    main()
