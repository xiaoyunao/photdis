#!/usr/bin/env python3
"""Validate the v3.10 science delivery and render the working-sample Aitoff plate."""
import csv
import json
import os

import numpy as np

from photdis.stage1 import require_compute
from v2_1_near import ROOT, save_json, sha
from v3_3_sed import verified
from v3_6_vac_distances import read_columns
from v3_8_direct_figures import BLUE, STYLE, aitoff, plt
from v3_10_figures import check_templates, sky_coord
from v3_10_working_sample import SCORE, RADIAL_MASS

VERSION = ROOT/'data/v3_10'
DEST = VERSION/'delivery'
CHECKPOINTS = (
    'figures/regression','figures/north','figures/south_faint','figures/south_low','figures/combined',
    'evidence/north','evidence/south_faint','evidence/south_low','selection/north','selection/south',
    'redrock','redrock/comparison','selection_sed/north','selection_sed/south',
    'branch_cases','branch_cases/verified_plate','branches','dust/replay','dust/low_latitude',
    'coverage/north','coverage/south','science')


def product(receipt):
    r = json.loads(receipt.read_text())
    for name,digest in r['outputs'].items():
        verified(receipt.parent/name,digest)
    return r


def working_plot(science):
    states = []
    sky = []
    for side in ('north','south'):
        root = VERSION/'working_sample'/side
        total,rows,ids = np.zeros(5,'i8'),0,[]
        for i in range(48):
            receipt = root/f'{i:02d}.json'
            if sha(receipt)!=science['inputs'][str(receipt.relative_to(ROOT))]:
                raise ValueError('Working sample changed after scientific evaluation')
            r = product(receipt)
            rows += r['rows']
            if not r['rows']:
                continue
            p = read_columns(root/f'{i:02d}.npz',('ls_id','TYPE','r','RA','DEC','state','score',
                             'qrf_median_kpc','radial_mass','row_state','pdf_rows'))
            if not np.all((p['TYPE']=='PSF')&(p['r']<20)) or np.any(p['state']>4):
                raise ValueError('Working sample scope and morphology mismatch')
            valid = p['state']>=2
            if not np.array_equal(p['pdf_rows']>=0,valid):
                raise ValueError('Working sample state lost a production PDF pointer')
            live = p['state']==4
            check = valid&(p['score']>=SCORE-1e-7)&(p['radial_mass'].sum(axis=1)>=RADIAL_MASS-1e-7)
            if np.any(live&~check) or int(live.sum())!=r['selected']:
                raise ValueError('Working sample threshold or membership mismatch')
            if not np.all(p['row_state'][valid]==7):
                raise ValueError('A lower quality row entered the working-sample PDF tier')
            ids.append(p['ls_id'])
            l,b = sky_coord(p['RA'][live],p['DEC'][live])
            sky.append(np.stack([l,b,p['qrf_median_kpc'][live]]))
            total += np.bincount(p['state'],minlength=5)
        keys = np.concatenate(ids)
        if len(keys)!=rows or len(np.unique(keys))!=rows:
            raise ValueError('Working sample raw identity conservation failed')
        states.append(dict(side=side,rows=rows,counts=total.tolist()))
    if sum(s['counts'][4] for s in states)!=science['working_rows']:
        raise ValueError('Working-sample science and export counts differ')
    l,b,d = np.concatenate(sky,axis=1)
    le,be = np.linspace(-np.pi,np.pi,145),np.linspace(-np.pi/2,np.pi/2,73)
    grid = np.histogram2d(l,b,bins=(le,be))[0].T
    extent = (.5*(le[1:]+le[:-1]),.5*(be[1:]+be[:-1]),grid>0)
    with plt.rc_context(STYLE):
        aitoff(DEST,'working_sample_aitoff','Dwarf-enhanced working sample · Galactic longitude increases to the left',
               d,l,b,np.isfinite(d),extent,BLUE)
    return states


def main():
    require_compute()
    lock = check_templates()
    DEST.mkdir(parents=True,exist_ok=True)
    resources = [VERSION/sub/'complete.json' for sub in CHECKPOINTS]+[VERSION/'seam/compare.json']
    records = {str(p.parent.relative_to(VERSION)):product(p) for p in resources}
    for path in resources:
        r = records[str(path.parent.relative_to(VERSION))]
        if 'style_lock_sha256' in r and r['style_lock_sha256']!=lock:
            raise ValueError(f'Mixed drawing styles in {path}')
    branch = records['branches']
    t = read_columns(VERSION/'branches/distance_pdf.npz',('ls_id','TYPE','pdf','pdf_by_branch','mu_edges'))
    if len(t['ls_id'])!=branch['rows'] or not np.all(t['TYPE']=='PSF') or np.any(np.diff(t['mu_edges'])<=0):
        raise ValueError('Branch export replaced a native identity or a PDF grid')
    if not np.isfinite(t['pdf']).all() or np.any(t['pdf']<0):
        raise ValueError('Branch export contains invalid densities')
    np.testing.assert_allclose(t['pdf'].sum(axis=-1),1.,rtol=0,atol=1e-9)
    np.testing.assert_allclose(t['pdf_by_branch'].sum(axis=1),t['pdf'][-1],rtol=0,atol=1e-11)
    states = working_plot(records['science'])
    with (DEST/'working_sample_aitoff.csv').open() as f:
        rows = list(csv.DictReader(f))
        if int(rows[-1]['n'])!=records['science']['working_rows']:
            raise ValueError('The plot and working-sample counts differ')
    save_json(DEST/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),style_lock_sha256=lock,working_sample=states,
        paired_production_seam=records['seam']['pairs'],luminosity_case_rows=branch['rows'],
        inputs={str(p.relative_to(ROOT)):sha(p) for p in resources},
        outputs={f.name:sha(f) for f in DEST.iterdir() if f.name!='complete.json'}))


if __name__=='__main__':
    main()
