#!/usr/bin/env python3
"""Resumable full native DECam inference with exact v3.4 catalog reuse."""
import argparse
from collections import Counter
from functools import lru_cache
from itertools import chain
import json
import os
from pathlib import Path

import healpy as hp
import numpy as np

from photdis.stage1 import require_compute
from photdis.v3.controls import apply_temperature, optical_magnitudes, probe_features
from photdis.v3_3 import frozen_qrf
from v2_1_near import ROOT, atomic_npz, load, save_json, sha
from v3_0_native import read_native, subset
from v3_3_sed import verified
from v3_4_near_forecast import forecast, prepare, qrf_bins
from v3_5_dr11_area import AREA, NSIDE, tile_map
from v3_5_dr11_inventory import CONFIG, OUT
from v3_5_dr11_pilot import ARCHIVE, CUTS, LEDGER, scan_rows, strata
from v3_5_dr11_pilot_evidence import PRODUCT as PILOT_EVIDENCE

PRODUCTION = OUT/'catalog'
FREEZE = PRODUCTION/'freeze.json'
INDEX = PRODUCTION/'archive_index.npz'
ARCHIVE_FIELDS = ('identity','ls_id','TYPE','qrf_supported','dwarf_selection_score','nominal_qnear',
    'nominal_numerical_flag','qrf_quantiles_kpc','qrf_qnear','qrf_bins','qrf_mu_edges','qrf_mu_mass',
    'primary','strict')
CHOICES = ROOT/'configs/v3_4_near_release_domain.json'


def setup(n_groups):
    """Bind every sweep and the reusable archival row ownership map."""
    if n_groups<1:
        raise ValueError('At least one processing group is required')
    if FREEZE.exists():
        raise FileExistsError(FREEZE)
    inventory = OUT/'inventory/complete.json'
    inv = json.loads(inventory.read_text())
    area = AREA/'complete.json'
    pilot = PILOT_EVIDENCE/'complete.json'
    if json.loads(area.read_text())['inventory_sha256'] != sha(inventory):
        raise ValueError('Coverage and source roster refer to different sweeps')
    pilot_report = json.loads(pilot.read_text())
    for name,digest in pilot_report['outputs'].items():
        verified(pilot.parent/name,digest)
    old = ARCHIVE/'complete.json'
    record = json.loads(old.read_text())
    index, payload = [], []
    for key, expected in record['child_receipts'].items():
        path = verified(ROOT/key,expected)
        for name, h in json.loads(path.read_text())['outputs'].items():
            if not name.startswith('part_'):
                continue
            part = verified(path.parent/name,h)
            with np.load(part) as z:
                ids = z['ls_id']
                if not (z['TYPE']=='PSF').all():
                    raise ValueError('Non-PSF v3.4 rows cannot be reused')
            index.append((ids,np.full(len(ids),len(payload),dtype='i4'),np.arange(len(ids),dtype='i4')))
            payload.append(dict(path=str(part.relative_to(ROOT)),sha256=h))
    with np.load(ROOT/payload[0]['path']) as old_sample:
        ra, dec = float(old_sample['RA'][0]), float(old_sample['DEC'][0])
    old_tile = int(tile_map(inv['sweeps'])[int(ra%360),min(int(dec+90),179)])
    if old_tile < 0:
        raise ValueError('An archived v3.4 source lacks sweep coverage')
    smoke = inv['sweeps'][old_tile]['filename']
    ids, owner, row = [np.concatenate(x) for x in zip(*index)]
    if len(np.unique(ids))!=record['selected']:
        raise ValueError('v3.4 reusable identity ownership failed')
    order = np.argsort(ids)
    PRODUCTION.mkdir(parents=True,exist_ok=True)
    atomic_npz(INDEX,ls_id=ids[order],owner=owner[order],row=row[order])
    groups = [[] for _ in range(n_groups)]
    totals = np.zeros(n_groups,'i8')
    sweeps = [x for x in inv['sweeps'] if x['possibly_high_latitude']]
    omitted = [s for s in sweeps if 'error' in s]
    for s in sorted((x for x in sweeps if 'error' not in x),key=lambda x:-x['size']):
        i = totals.argmin()
        groups[i].append(s['filename'])
        totals[i] += s['size']
    save_json(FREEZE,dict(job=os.getenv('SLURM_JOB_ID'),archive_index_sha256=sha(INDEX),archive_parts=payload,
        excluded_bad_headers=omitted,groups=groups,group_bytes=totals,smoke=smoke,
        inputs={str(p.relative_to(ROOT)):sha(p) for p in (CONFIG,CUTS,CHOICES,inventory,area,pilot,old,Path(__file__),
            ROOT/'scripts/v3_5_dr11_pilot.py',ROOT/'scripts/v3_4_near_forecast.py')}))
    print('Frozen production groups',len(groups),'sweeps',sum(map(len,groups)),'bytes',totals.sum(),flush=True)


def archived_positions(ls_ids,index):
    ix = np.searchsorted(index['ls_id'],ls_ids)
    within = ix < len(index['ls_id'])
    within[within] &= index['ls_id'][ix[within]] == ls_ids[within]
    return np.where(within,ix,-1)


@lru_cache(maxsize=2)
def archive_part(path,digest):
    with np.load(verified(ROOT/path,digest)) as z:
        return {k:z[k] for k in ARCHIVE_FIELDS}


def copied_predictions(t, positions, index, old):
    files = index['owner'][positions]
    for key in np.unique(files):
        use = files==key
        rows = index['row'][positions[use]]
        entry = old[key]
        pred = subset(archive_part(entry['path'],entry['sha256']),rows)
        np.testing.assert_array_equal(pred['ls_id'],t['ls_id'][use])
        yield subset(t,use),pred,np.ones(len(rows),bool),True


def qrf_only(t, models, dwarf):
    """Rows below the primary dwarf score cannot pass either nominal cut."""
    n = len(dwarf)
    mags,errors = optical_magnitudes(t)
    q = frozen_qrf(models['qrf'],mags,errors,t['identity'])
    bins = qrf_bins(q['mu_edges'],q['mass'])
    return dict(identity=t['identity'],ls_id=t['ls_id'],TYPE=t['TYPE'],qrf_supported=q['supported'],
                dwarf_selection_score=dwarf,nominal_qnear=np.full(n,np.nan),nominal_numerical_flag=np.zeros(n,bool),
                qrf_quantiles_kpc=q['distance'],qrf_qnear=bins[:,:3].sum(axis=1),qrf_bins=bins,
                qrf_mu_edges=q['mu_edges'],qrf_mu_mass=q['mass'],primary=np.zeros(n,bool),strict=np.zeros(n,bool))


def new_predictions(t,cuts,models,probe,batch):
    for i in range(0,len(t['ls_id']),batch):
        photo = subset(t,slice(i,i+batch))
        dwarf = apply_temperature(probe['model'].predict_proba(probe_features(photo,'grz')),probe['temperature'])[:,0]
        needs_gate = dwarf >= cuts['primary']['minimum_dwarf_score']
        for use,gated in ((needs_gate,True),(~needs_gate,False)):
            if use.any():
                rows = subset(photo,use)
                p = forecast(rows,cuts,models,probe,retain_curve=False) if gated else qrf_only(rows,models,dwarf[use])
                np.testing.assert_allclose(p['dwarf_selection_score'],dwarf[use],atol=1e-12,rtol=0)
                yield rows,p,np.full(len(p['ls_id']),gated,bool),False


def check_pdf(p):
    if not (p['TYPE']=='PSF').all() or not p['qrf_supported'].all() or (p['strict'] & ~p['primary']).any():
        raise ValueError('Prediction failed native PSF, support or selection checks')
    mass,edges = p['qrf_mu_mass'],p['qrf_mu_edges']
    if (mass<0).any() or not np.isfinite(edges).all() or (np.diff(edges,axis=1)<=0).any():
        raise ValueError('Invalid native QRF histogram')
    np.testing.assert_allclose(mass.sum(axis=1),1,atol=1e-12,rtol=0)
    np.testing.assert_allclose(p['qrf_bins'].sum(axis=1),1,atol=1e-12,rtol=0)
    np.testing.assert_allclose(p['qrf_qnear'],p['qrf_bins'][:,:3].sum(axis=1),atol=1e-12,rtol=0)


def process(source,cfg,cuts,choices,models,probe,index,plan,bind):
    path = Path(cfg['sweep_root'])/source['filename']
    stat = path.stat()
    if (stat.st_size,stat.st_mtime_ns)!=(source['size'],source['mtime_ns']):
        raise ValueError(f'Native input changed: {path.name}')
    dest = PRODUCTION/path.stem
    marker = dest/'complete.json'
    if marker.exists():
        x = json.loads(marker.read_text())
        if x['binding']!=bind:
            raise ValueError(f'Changed production inputs: {path.name}')
        for name,h in x['outputs'].items():
            verified(dest/name,h)
        return
    dest.mkdir(parents=True,exist_ok=True)
    started = dest/'started.json'
    if started.exists():
        if json.loads(started.read_text())!=bind:
            raise ValueError(f'Changed production retry: {path.name}')
    else:
        with started.open('x') as f:
            json.dump(bind,f)
    raw = read_native(path)
    ledger = np.empty(len(raw),dtype='i1')
    if len(ledger)!=source['n_rows']:
        raise ValueError('Sweep count differs from inventory')
    part, totals, output = 0, Counter(), {}
    ids, native_gaia = [], []
    spatial = np.zeros((2,3,6,hp.nside2npix(NSIDE)))
    selected_at_sky = np.zeros((2,3,hp.nside2npix(NSIDE)),dtype='i8')
    for first in range(0,len(raw),250_000):
        photo,status = scan_rows(raw[first:first+250_000],source['bounds_deg'],cfg,models['qrf'],first)
        ledger[first:first+len(status)] = status
        pos = archived_positions(photo['ls_id'],index)
        reuse = pos>=0
        old = copied_predictions(subset(photo,reuse),pos[reuse],index,plan['archive_parts'])
        new = new_predictions(subset(photo,~reuse),cuts,models,probe,cfg['inference_batch_rows'])
        for t,p,gate,reused in chain(old,new):
            check_pdf(p)
            r,c,w1 = t['observed_r'],t['dereddened_g_r'],t['band_state'][:,4]
            r0,r1 = choices['observed_r']; c0,c1 = choices['dereddened_g_r']
            bright_red = (r>=r0)&(r<r1)&(c>=c0)&(c<c1)&np.isin(w1,choices['w1_states'])
            name = f'part_{part:05d}.npz'
            result = dest/name
            atomic_npz(result,**p,nominal_evaluated=gate,source_id=t['source_id'],RA=t['RA'],DEC=t['DEC'],
                EBV=t['EBV'],parent_row=t['parent_row'],photometric_system=t['photometric_system'],
                observed_r=r,dereddened_g_r=c,band_state=t['band_state'],band_mask=t['band_mask'],
                stratum=strata(t,cfg),analog_v34_observable=bright_red,reused_v34=np.full(len(r),reused,bool))
            pix = hp.ang2pix(NSIDE,t['RA'],t['DEC'],lonlat=True,nest=True)
            faint_blue = (r>=18)&(c>=.5)&(c<.8)
            for selection, cut in enumerate(('primary','strict')):
                for k, region in enumerate((np.ones(len(r),bool),bright_red,faint_blue)):
                    keep = p[cut]&region
                    selected_at_sky[selection,k] += np.bincount(pix[keep],minlength=spatial.shape[-1])
                    for shell in range(6):
                        spatial[selection,k,shell] += np.bincount(
                            pix[keep],weights=p['qrf_bins'][keep,shell],minlength=spatial.shape[-1])
            output[name] = sha(result)
            part += 1
            ids.append(p['ls_id'])
            native_gaia.append(t['source_id'])
            totals.update(dict(rows=len(r),primary=int(p['primary'].sum()),strict=int(p['strict'].sum()),
                numerical=int(p['nominal_numerical_flag'].sum()),gate=int(gate.sum()),reused=len(r)*int(reused)))
    counts = np.bincount(ledger,minlength=len(LEDGER))
    legacy = np.concatenate(ids) if ids else np.empty(0,'i8')
    if len(np.unique(legacy))!=totals['rows'] or totals['rows']!=counts[-1] or counts.sum()!=len(ledger):
        raise ValueError('Unique partition identities or exclusive parent ledger failed')
    if set(p.name for p in dest.glob('part_*.npz'))!=set(output):
        raise ValueError(f'Unexpected partition files in {dest}')
    atomic_npz(dest/'parent_status.npz',status=ledger)
    atomic_npz(dest/'identities.npz',ls_id=legacy,source_id=np.concatenate(native_gaia) if ids else legacy)
    np.testing.assert_allclose(spatial.sum(axis=2),selected_at_sky,atol=1e-7,rtol=0)
    atomic_npz(dest/'spatial.npz',mass=spatial,selected=selected_at_sky)
    output.update({k:sha(dest/k) for k in ('parent_status.npz','identities.npz','spatial.npz')})
    save_json(marker,dict(job=os.getenv('SLURM_JOB_ID'),source=source['filename'],binding=bind,
        outputs=output,exclusive_counts=dict(zip(LEDGER,counts)),counts=totals,all_PSF=True))
    print('v3.5',path.name,dict(totals),flush=True)


def run(group,lane,lanes,*,smoke=False):
    plan = json.loads(FREEZE.read_text())
    for key,digest in plan['inputs'].items():
        verified(ROOT/key,digest)
    inventory = json.loads((OUT/'inventory/complete.json').read_text())
    sources = {s['filename']:s for s in inventory['sweeps']}
    cfg,cuts,choices = [json.loads(p.read_text()) for p in (CONFIG,CUTS,CHOICES)]
    model,probe,paths = prepare()
    bind = dict(freeze_sha256=sha(FREEZE),models={str(p.relative_to(ROOT)):sha(p) for p in paths})
    index = load(verified(INDEX,plan['archive_index_sha256']))
    names = [plan['smoke']] if smoke else plan['groups'][group][lane::lanes]
    for name in names:
        process(sources[name],cfg,cuts,choices['domains'][0],model,probe,index,plan,bind)


def main():
    require_compute()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('setup','smoke','run'))
    p.add_argument('group',type=int)
    p.add_argument('lane',type=int,nargs='?',default=0)
    p.add_argument('lanes',type=int,nargs='?',default=1)
    a=p.parse_args()
    if a.stage=='setup':
        setup(a.group)
    else:
        run(a.group,a.lane,a.lanes,smoke=a.stage=='smoke')


if __name__=='__main__':
    main()
