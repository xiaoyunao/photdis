#!/usr/bin/env python3
"""Export a bright, dwarf-enhanced 2–10 kpc working-sample index."""
import argparse
import csv
import json
import os
from pathlib import Path

import joblib
import numpy as np

from photdis.stage1 import require_compute
from photdis.v3.controls import apply_temperature, probe_features
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_6_vac_distances import read_columns
from v3_9_catalog import OUT as EXPANSION, subset
from v3_9_delivery import DEST as V39
from v3_9_references import join, predictor_features
from v3_9_unified import owner_rows
from v3_10_selection import DEST as CLASSIFIERS

DEST = ROOT/'data/v3_10/working_sample'
SCORE = .80
RADIAL_MASS = .50
DOMAINS = {'north':'north','south':'south_low'}
SHELLS = 10+5*np.log10([2.,5.,10.])


def shell_mass(edges,mass,cuts=SHELLS):
    width = np.diff(edges,axis=1)
    cdf = np.column_stack([np.sum(mass*np.clip((m-edges[:,:-1])/width,0,1),axis=1) for m in cuts])
    return np.diff(cdf,axis=1)


def inputs(side):
    root = CLASSIFIERS/side
    fit = root/'complete.json'
    release = V39/'complete.json'
    seam = V39/'seam/complete.json'
    inventory = EXPANSION/side/'inventory.json'
    rel = json.loads(release.read_text())
    s = json.loads(verified(seam,rel['sources'][str(seam.relative_to(ROOT))]).read_text())
    overrides = read_columns(verified(seam.parent/'ownership_overrides.npz',s['outputs']['ownership_overrides.npz']),
                             ('ls_id','TYPE','primary_survey_owner'))
    model = joblib.load(verified(root/'model.joblib',json.loads(fit.read_text())['outputs']['model.joblib']))
    index = verified(V39/'unified_parts.csv',rel['outputs']['unified_parts.csv'])
    return ({str(p.relative_to(ROOT)):sha(p) for p in (fit,release,seam,inventory)},
            model,overrides,inventory,index)


def read_part(part, model, overrides):
    side = part['side']
    extra = ('mag_grz',) if side=='north' else ('flux','flux_error','transmission','band_mask')
    p = read_columns(verified(ROOT/part['photo_file'],part['photo_sha256']),
                     ('TYPE','ls_id','row_state','north_primary_region','r','g0','r0','z0','RA','DEC',
                      'MASKBITS','EBV','PSFDEPTH_G','PSFDEPTH_R','PSFDEPTH_Z','error_grz',*extra))
    owns,_ = owner_rows(side,p,overrides)
    k = owns & (p['r']<20)
    if not k.any():
        return {}
    p = subset(p,k)
    number = len(p['ls_id'])
    state = np.zeros(number,'u1')
    strict = p['row_state']==7
    state[strict] = 1
    p['score'] = np.full(number,np.nan,'f4')
    p['pdf_rows'] = np.full(number,-1,'i4')
    p['radial_mass'] = np.full((number,2),np.nan,'f4')
    p['qrf_median_kpc'] = np.full(number,np.nan,'f4')
    if part['pdf_file']:
        cols = ('ls_id','TYPE','photo_row','qrf_mu_edges','qrf_mu_mass','qrf_quantiles_kpc')
        z = read_columns(verified(ROOT/part['pdf_file'],part['pdf_sha256']),cols)
        j = np.searchsorted(np.flatnonzero(k),z['photo_row'])
        photo_row = np.flatnonzero(k)
        i = np.flatnonzero(j<number)
        i = i[(photo_row[j[i]]==z['photo_row'][i])&strict[j[i]]]
        a = j[i]
        if len(a):
            np.testing.assert_array_equal(z['ls_id'][i],p['ls_id'][a])
            np.testing.assert_array_equal(z['TYPE'][i],p['TYPE'][a])
            data = subset(p,a)
            x = predictor_features(data['mag_grz']) if side=='north' else probe_features(data,'grz')
            score = apply_temperature(model['model'].predict_proba(x),model['temperature'])[:,0]
            radial = shell_mass(z['qrf_mu_edges'][i],z['qrf_mu_mass'][i])
            if (radial<-1e-10).any() or (radial.sum(axis=1)>1+1e-10).any():
                raise ValueError('Invalid radial mass')
            p['score'][a] = score
            p['radial_mass'][a] = radial
            p['qrf_median_kpc'][a] = z['qrf_quantiles_kpc'][i,2]
            p['pdf_rows'][a] = i
            state[a] = 2
            state[a[score>=SCORE]] = 3
            state[a[(score>=SCORE)&(radial.sum(axis=1)>=RADIAL_MASS)]] = 4
    p['state'] = state
    return {key:p[key] for key in ('ls_id','TYPE','r','g0','r0','z0','RA','DEC','MASKBITS','EBV',
                'PSFDEPTH_G','PSFDEPTH_R','PSFDEPTH_Z','error_grz','row_state','state','score',
                'radial_mass','qrf_median_kpc','pdf_rows')}


def run(side, group):
    bound,model,overrides,inventory,index = inputs(side)
    sources = json.loads(inventory.read_text())['groups'][group]
    sweeps = {Path(s['filename']).stem for s in sources}
    with index.open() as f:
        parts = [r for r in csv.DictReader(f) if r['region']==DOMAINS[side] and r['sweep'] in sweeps]
    root = DEST/side
    root.mkdir(parents=True,exist_ok=True)
    tables,parts_out = [],[]
    for part in parts:
        rows = read_part(part,model,overrides)
        if not rows:
            continue
        n = len(rows['ls_id'])
        rows['part_index'] = np.full(n,len(parts_out),'i4')
        tables.append(rows)
        counts = np.bincount(rows['state'],minlength=5)
        parts_out.append(dict(part_index=len(parts_out),sweep=part['sweep'],
                              file=part['pdf_file'],pdf_sha256=part['pdf_sha256'],
                              **{f'state_{i}':int(k) for i,k in enumerate(counts)}))
    t = join(tables)
    file = root/f'{group:02d}.npz'
    if t:
        if len(np.unique(t['ls_id']))!=len(t['ls_id']) or not np.all(t['TYPE']=='PSF'):
            raise ValueError('A native key is repeated in the working sample')
        atomic_npz(file,**t)
    table = root/f'{group:02d}.csv'
    csv_rows(table,parts_out)
    outputs = [table] + ([file] if t else [])
    save_json(root/f'{group:02d}.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=bound,
        side=side,group=group,outputs={p.name:sha(p) for p in outputs},
        rows=len(t.get('ls_id',[])),selected=int((t['state']==4).sum()) if t else 0))


if __name__=='__main__':
    require_compute()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('side',choices=tuple(DOMAINS))
    p.add_argument('--shard',type=int,default=int(os.getenv('SLURM_ARRAY_TASK_ID','0')))
    p.add_argument('--groups-per-job',type=int,default=1)
    a = p.parse_args()
    start = a.shard*a.groups_per_job
    for group in range(start,start+a.groups_per_job):
        run(a.side,group)
