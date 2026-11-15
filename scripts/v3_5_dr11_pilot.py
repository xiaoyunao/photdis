#!/usr/bin/env python3
"""Census a native DR11 sweep and apply the frozen cuts on a stratified pilot."""
import argparse
from collections import Counter
from itertools import islice
import json
import os
from pathlib import Path

import numpy as np

from photdis.baseline_v1 import support
from photdis.stage1 import galactic, require_compute
from photdis.v3.controls import optical_magnitudes
from photdis.v3.data import stable_rank
from v2_1_near import ROOT, atomic_npz, save_json, sha
from v3_0_cohorts import INPUT_KEYS
from v3_0_native import native_payload, read_native, subset
from v3_3_sed import verified
from v3_4_near_forecast import forecast, prepare
from v3_4_near_validation_inputs import CONFIG as CUTS
from v3_5_dr11_inventory import CONFIG, OUT

PILOT = OUT/'pilot'
ARCHIVE = ROOT/'data/v3_4/near_catalog'
LEDGER = ('outside_tile', 'other_release', 'missing_type', 'non_psf', 'sky_dust',
          'outside_r', 'optical_quality', 'outside_qrf_support', 'model_supported')


def in_tile(ra, dec, bounds):
    r0, d0, r1, d1 = bounds
    width = (r1-r0) % 360 or 360
    return (np.isfinite(ra) & np.isfinite(dec) & ((ra-r0) % 360 < width)
            & (dec >= d0) & ((dec < d1) | ((d1 == 90) & (dec == 90))))


def scan_rows(raw, bounds, cfg, model, first_row=0):
    """Return supported rows plus an exclusive account of input records."""
    status = np.full(len(raw), len(LEDGER)-1, dtype='i1')
    kind = np.char.strip(raw['TYPE'].astype('U'))
    keep = in_tile(raw['RA'], raw['DEC'], bounds)
    tests = (keep, raw['RELEASE'] == cfg['native_release'], kind != '', kind == 'PSF')
    for index, valid in enumerate(tests):
        status[(status == len(LEDGER)-1) & ~valid] = index
    p = np.flatnonzero(status == len(LEDGER)-1)
    latitude, dust = galactic(raw['RA'][p], raw['DEC'][p])[1], raw['EBV'][p]
    ok = (abs(latitude) > cfg['minimum_abs_galactic_latitude']) & (dust >= 0) & (dust < cfg['maximum_ebv'])
    status[p[~ok]] = 4
    p = p[ok]
    flux = raw['FLUX_R'][p]
    with np.errstate(divide='ignore', invalid='ignore'):
        observed = (22.5-2.5*np.log10(flux.astype('f8'))).astype('f4')
    ok = np.isfinite(observed) & (observed < cfg['maximum_observed_r'])
    status[p[~ok]] = 5
    p = p[ok]
    t = native_payload(raw[p], 'DECam_DR11')
    grz = [0, 1, 3]
    usable = (t['photometric_entry'] & t['band_mask'][:,grz].all(axis=1)
              & np.isfinite(t['optical_error']) & (t['optical_error'] < cfg['maximum_grz_error_mag']))
    status[p[~usable]] = 6
    p, t = p[usable], subset(t, usable)
    if len(p):
        mag, err = optical_magnitudes(t)
        supported = (err.max(axis=1) < cfg['maximum_grz_error_mag']) & support(model, np.pad(mag,((0,0),(0,2))))
    else:
        supported = np.empty(0, bool)
    status[p[~supported]] = 7
    q = {k:t[k][supported] for k in INPUT_KEYS}
    q.update(parent_row=(first_row+p[supported]).astype('i8'), source_id=t['source_id'][supported],
             observed_r=t['r'][supported], dereddened_g_r=(t['g0']-t['r0'])[supported])
    return q, status


def strata(photo, cfg):
    r = np.searchsorted(cfg['validation_r_edges'], photo['observed_r'], side='right')-1
    color = np.searchsorted(cfg['validation_color_edges'], photo['dereddened_g_r'], side='right')-1
    nr, nc = (len(cfg[k])-1 for k in ('validation_r_edges','validation_color_edges'))
    if np.any((r < 0) | (r >= nr) | (color < 0) | (color >= nc)):
        raise ValueError('Supported object outside pilot stratum edges')
    return (r*nc+color)*6+photo['band_state'][:,4]


def sample(photo, groups, pop, cfg):
    """Partition-stable ranks give each retained row a known inclusion weight."""
    rows, weights = [], []
    cap = cfg['pilot_maximum_rows_per_stratum']
    rank = stable_rank(photo['ls_id'], cfg['pilot_rank_seed'])
    for key in sorted(pop):
        ix = np.flatnonzero(groups == key)
        selected = ix[np.argsort(rank[ix],kind='stable')[:cap]]
        if len(selected):
            rows.extend(selected)
            weights.extend(np.full(len(selected),len(ix)/len(selected)))
    return np.asarray(rows,dtype=int), np.asarray(weights)


def prior_sources():
    marker = ARCHIVE/'complete.json'
    catalog = json.loads(marker.read_text())
    ids = []
    for name, digest in catalog['child_receipts'].items():
        receipt = verified(ROOT/name,digest)
        for part, expected in json.loads(receipt.read_text())['outputs'].items():
            if part.startswith('part_'):
                with np.load(verified(receipt.parent/part,expected)) as z:
                    ids.append(z['ls_id'])
    ids = np.concatenate(ids)
    if len(np.unique(ids)) != catalog['selected']:
        raise ValueError('Frozen v3.4 catalog identity inventory failed')
    return np.sort(ids), sha(marker)


def examine(source, cfg, models, probe, old_ids, old_digest, inputs):
    path = Path(cfg['sweep_root'])/source['filename']
    stat = path.stat()
    if (stat.st_size,stat.st_mtime_ns) != (source['size'],source['mtime_ns']):
        raise ValueError(f'Inventory changed for {path.name}')
    destination = PILOT/path.stem
    complete = destination/'complete.json'
    if complete.exists():
        r = json.loads(complete.read_text())
        if r['inputs'] != inputs or r['source_stat'] != [source['size'],source['mtime_ns']]:
            raise ValueError(f'Pilot inputs changed for {path.name}')
        for file, digest in r['outputs'].items():
            verified(destination/file,digest)
        return
    raw = read_native(path)
    if len(raw) != source['n_rows']:
        raise ValueError('Inventory row count changed')
    parts, counts = [], np.zeros(len(LEDGER),dtype='i8')
    for start in range(0,len(raw),250_000):
        photo, status = scan_rows(raw[start:start+250_000],source['bounds_deg'],cfg,models['qrf'],start)
        counts += np.bincount(status,minlength=len(LEDGER))
        parts.append(photo)
    del raw
    whole = {k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
    if counts.sum()!=source['n_rows'] or len(np.unique(whole['ls_id']))!=len(whole['ls_id']):
        raise ValueError('Parent ledger or native identities do not reconcile')
    groups = strata(whole,cfg)
    reused = np.isin(whole['ls_id'],old_ids)
    roster = {k:whole[k] for k in ('ls_id','source_id','RA','DEC','parent_row')}
    roster.update(reuse_v34=reused,stratum=groups)
    whole, groups = subset(whole,~reused), groups[~reused]
    pop = Counter(groups.tolist())
    rows, weight = sample(whole,groups,pop,cfg)
    chosen = subset(whole,rows)
    cut = json.loads(CUTS.read_text())
    predictions = []
    size = cfg['inference_batch_rows']
    for i in range(0,len(rows),size):
        predictions.append(forecast(subset(chosen,slice(i,i+size)),cut,models,probe,retain_curve=False))
    destination.mkdir(parents=True,exist_ok=True)
    count = dict(primary=0, strict=0)
    if predictions:
        combined = {k:np.concatenate([p[k] for p in predictions]) for k in predictions[0]}
        if not combined['qrf_supported'].all():
            raise ValueError('QRF input and forecast support differ')
        count = {k:int(combined[k].sum()) for k in ('primary','strict')}
        atomic_npz(destination/'prediction.npz',**combined,source_id=chosen['source_id'],
                   observed_r=chosen['observed_r'],dereddened_g_r=chosen['dereddened_g_r'],
                   RA=chosen['RA'],DEC=chosen['DEC'],EBV=chosen['EBV'],
                   band_state=chosen['band_state'],stratum=groups[rows],design_weight=weight)
    atomic_npz(destination/'supported_roster.npz',**roster)
    save_json(complete,dict(job=os.getenv('SLURM_JOB_ID'),source=source['filename'], bounds=source['bounds_deg'],
        n_parent=source['n_rows'],exclusive_counts=dict(zip(LEDGER,counts.tolist())),
        reused_v34=int(reused.sum()),survey_n=len(groups),sample_n=len(rows),**count,
        old_catalog_sha256=old_digest,stratum_counts=pop,all_PSF=True,
        inputs=inputs,source_stat=[source['size'],source['mtime_ns']],
        new_outcomes_read=False,outputs={p.name:sha(p) for p in sorted(destination.glob('*.npz'))}))
    print('PILOT',path.name,'support',counts[-1],'sample',len(rows),count,flush=True)


def main():
    require_compute()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('group',type=int)
    p.add_argument('number_of_groups',type=int)
    args=p.parse_args()
    inv_path=OUT/'inventory'/'complete.json'
    inv=json.loads(inv_path.read_text())
    verified(CONFIG,inv['config_sha256'])
    cfg=json.loads(CONFIG.read_text())
    all_sweeps={r['filename']:r for r in inv['sweeps']}
    models,probe,paths=prepare()
    inputs={str(p.relative_to(ROOT)):sha(p) for p in (CONFIG,CUTS,inv_path,Path(__file__),*paths)}
    old,receipt=prior_sources()
    for name in islice(inv['pilot'],args.group,None,args.number_of_groups):
        examine(all_sweeps[name],cfg,models,probe,old,receipt,inputs)


if __name__ == '__main__':
    main()
