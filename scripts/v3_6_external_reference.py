#!/usr/bin/env python3
"""Recover the external spectroscopic inventories before both DR11 cuts."""
import csv
import json
import os
from pathlib import Path

import fitsio
import numpy as np

from photdis.reference_v1_4 import unique_join
from photdis.stage1 import require_compute
from photdis.v3.metrics import distance_summary
from v2_1_near import ROOT, atomic_npz, csv_rows, load, save_json, sha
from v3_2_reference_evidence import (DIST_COLS, MINESWEEPER_COLS, STAR_COLS, channels,
                                    minesweeper_channels, projected)
from v3_2_reference_inventory import ARCHIVES, text
from v3_3_sed import verified
from v3_4_near_reference_support import separation
from v3_4_near_validate import ARCHIVE, MINE
from v3_5_dr11_production import CUTS
from v3_5_dr11_spatial import DELIVERY
from v3_5_dr11_validate import OUT as VALIDATION, acquire_spectral_metadata, take_rows, validation_history
from v3_6_reference_index import OUT as INDEX
from v3_6_vac_distances import DISTANCE_EDGES, read_columns
from v3_6_vac_flow import gather_forecasts, selection_stages


OUT = ROOT/'data/v3_6/external_reference'
SHELLS = ('0–2','2–5','5–10','10–20','20–50','>50')
CLASSES = ('dwarf','giant','other','unknown')


def read_reference(name,index):
    if name=='DR19_Astra':
        base = ARCHIVE/'dr19_astra_0.6.0'
        keyed = dict(index,identity=index['source_id'])
        dist = projected(base/ARCHIVES['distance'],keyed,index['distance_row'],DIST_COLS)
        star = projected(base/ARCHIVES['stellar'],keyed,index['stellar_row'],STAR_COLS)
        return dict(**channels(dist,star),**{f'raw_distance_{k}':v for k,v in dist.items()},
                    **{f'raw_stellar_{k}':v for k,v in star.items()})
    obs = 'APO' if name=='DR20_APO' else 'LCO'
    rows = index['reference_row' if obs=='APO' else 'distance_row']
    order = np.argsort(rows)
    with fitsio.FITS(str(MINE)) as source:
        names = {key.lower():key for key in source[1].get_colnames()}
        data = source[1].read(rows=rows[order],columns=[names[k] for k in MINESWEEPER_COLS])
    restore = np.argsort(order)
    table = {k:data[names[k]][restore] for k in MINESWEEPER_COLS}
    if table['source_id'].dtype.kind not in 'iu' or (text(table['obs'])!=obs).any():
        raise ValueError('DR20 source rows changed')
    np.testing.assert_array_equal(table['source_id'],index['source_id'])
    return dict(**minesweeper_channels(table),**{f'raw_{k}':v for k,v in table.items()})


def records(p,survey):
    all_rows = np.ones(len(p['ls_id']),bool)
    cls = p['class_label']
    d,err = p['distance_kpc'],p['distance_error_kpc']
    pred = p['qrf_quantiles_kpc']
    use_distance = p['distance_available']&np.isfinite(d)&(d>0)&np.isfinite(err)&(err>0)
    r,c = p['observed_r'],p['dereddened_g_r']
    with np.errstate(divide='ignore',invalid='ignore'):
        ref_mu = 10+5*np.log10(d)
        pred_mu = 10+5*np.log10(pred)
    numerator = np.digitize(pred[:,2],DISTANCE_EDGES)-1
    output,migration = [],[]
    for cut,eligible in (('supported',all_rows),('primary',p['primary']),('strict',p['strict'])):
        for label,selected in (('all',all_rows),('outside_recorded_history',~p['historical_evidence'])):
            for name,physical in (('all',all_rows),*[(lab,cls==j) for j,lab in enumerate(CLASSES[:3])],
                                  ('unknown',cls<0)):
                target = eligible&selected&physical
                for condition,observed in (('all',all_rows),('r14_18',(r>=14)&(r<18)),
                    ('r18_20',(r>=18)&(r<20)),('blue',(c>=.5)&(c<.8)),
                    ('r18_20_blue',(r>=18)&(r<20)&(c>=.5)&(c<.8))):
                    for quality,dq in (('available',use_distance),('precise',p['distance_precise'])):
                        for j,(lo,hi) in enumerate(zip(DISTANCE_EDGES[:-1],DISTANCE_EDGES[1:])):
                            keep = target&observed&dq&(d>=lo)&(d<hi)
                            output.append(dict(survey=survey,selection=cut,history=label,reference_class=name,
                                observed=condition,quality=quality,distance_lo_kpc=lo,distance_hi_kpc=hi,
                                matched=int((target&observed).sum()),
                                **distance_summary(pred_mu[keep],ref_mu[keep])))
                            if condition=='all' and quality=='precise':
                                for k,predicted_shell in enumerate(SHELLS):
                                    migration.append(dict(survey=survey,selection=cut,history=label,
                                        reference_class=name,reference_shell_kpc=SHELLS[j],
                                        assigned_shell_kpc=predicted_shell,hard_count=int((keep&(numerator==k)).sum()),
                                        qrf_weighted_count=float(p['qrf_bins'][keep,k].sum())))
    return output,migration


def main():
    require_compute()
    index = INDEX/'complete.json'
    receipt = json.loads(index.read_text())
    path = verified(INDEX/'catalog_rows.npz',receipt['outputs']['catalog_rows.npz'])
    catalog = load(path)
    delivery = DELIVERY/'complete.json'
    run = json.loads(delivery.read_text())
    with verified(DELIVERY/'sweeps.csv',run['outputs']['sweeps.csv']).open() as f:
        digests = {r['sweep']:r['receipt_sha256'] for r in csv.DictReader(f)}
    validation = VALIDATION/'complete.json'
    original = json.loads(validation.read_text())
    hashes = {str(p.relative_to(ROOT)):sha(p) for p in (index,delivery,validation,CUTS,Path(__file__))}
    previous,legacy = validation_history(hashes,None)
    old_spectra = verified(VALIDATION/'spectra_reference_roster.npz',
                           original['outputs']['spectra_reference_roster.npz'])
    old = read_columns(old_spectra,('ls_id',))['ls_id']
    inventory = acquire_spectral_metadata(hashes)
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'complete.json').exists():
        raise FileExistsError(OUT/'complete.json')
    cfg = json.loads(CUTS.read_text())
    flow,metrics,migration = [],[],[]
    for survey,roster in inventory.items():
        match = unique_join(roster['source_id'],catalog['source_id'])
        found = np.flatnonzero(match>=0)
        status = np.zeros(len(match),'u1')
        if not len(found):
            atomic_npz(OUT/f'{survey}_inventory.npz',**roster,match_status=status,catalog_row=match)
            flow.append(dict(survey=survey,inventory=len(match),no_supported_match=len(match),
                             position_rejected=0,verified_supported=0,primary=0,strict=0,precise_distance=0))
            continue
        p = gather_forecasts(match[found],catalog,digests)
        verified_position = separation(p['RA'],p['DEC'],roster['RA'][found],roster['DEC'][found])
        accept = found[verified_position]
        p = take_rows(p,verified_position)
        r = take_rows(roster,accept)
        np.testing.assert_array_equal(p['source_id'],r['source_id'])
        status[found] = 1
        status[accept] = 2
        atomic_npz(OUT/f'{survey}_inventory.npz',**roster,match_status=status,catalog_row=match)
        if not len(accept):
            flow.append(dict(survey=survey,inventory=len(match),no_supported_match=int((status==0).sum()),
                             position_rejected=len(found),verified_supported=0,primary=0,strict=0,precise_distance=0))
            continue
        ref = read_reference(survey,r)
        p.update(**ref,catalog_row=match[accept],inventory_row=accept,
                 ref_ra=r['RA'],ref_dec=r['DEC'],
                 historical_evidence=np.isin(p['source_id'],previous)|np.isin(p['identity'],previous)
                 |np.isin(p['ls_id'],legacy)|np.isin(p['ls_id'],old))
        p.update({f'reference_{k}':v for k,v in r.items() if k not in ('RA','DEC','source_id')})
        p.update(selection_stages(p,cfg))
        atomic_npz(OUT/f'{survey}_matched.npz',**p)
        m,q = records(p,survey)
        metrics.extend(m)
        migration.extend(q)
        row = dict(survey=survey,inventory=len(match),no_supported_match=int((status==0).sum()),
                   position_rejected=int((status==1).sum()),verified_supported=len(accept),
                   primary=int(p['primary'].sum()),strict=int(p['strict'].sum()),
                   precise_distance=int(p['distance_precise'].sum()))
        flow.append(row)
        print(row,flush=True)
    for path,rows in zip(('association.csv','distance.csv','migration.csv'),(flow,metrics,migration)):
        csv_rows(OUT/path,rows)
    save_json(OUT/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=hashes,summary=flow,
        scope='Previously compiled external metadata inventories, exact native Gaia ID and angular verification. Overlapping surveys can contain the same star. The source inventories are targeted and keep their release-specific exclusions.',
        outputs={p.name:sha(p) for p in OUT.iterdir() if p.name!='complete.json'}))


if __name__=='__main__':
    main()
