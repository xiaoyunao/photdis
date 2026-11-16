#!/usr/bin/env python3
"""Verify the comparison cohorts and render the locked reference plates."""

import csv
import json
import os

import numpy as np
from PIL import Image

from photdis.stage1 import require_compute
from v2_1_near import ROOT, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_6_vac_distances import read_columns
from v3_8_direct_figures import STYLE, plt, render_reference_figures
from v3_10_figures import check_templates
from v3_12_reference import DEST as REFERENCE
from v3_12_structure import DEST as STRUCTURE

DEST = ROOT/'data/v3_12/delivery'


def read_table(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def bind(path, inputs):
    receipt = json.loads(path.read_text())
    for file,digest in receipt['outputs'].items():
        verified(path.parent/file,digest)
    inputs[str(path.relative_to(ROOT))] = sha(path)
    return receipt


def plots(side,data):
    grz = np.column_stack([data['g0'],data['r0']])
    simple = data['simple_mu']
    support = np.isfinite(simple).all(axis=1)
    for key,label in (('qrf_mu','Frozen QRF'),('simple_mu','Native color relation')):
        q = 10**((data[key]-10)/5)
        result = {}
        for name,ref,ok in (
            ('Gaia',data['reference_kpc'],data['gaia_state']==6),
            ('SpecDis',data['specdis_kpc'],data['specdis_quality']),
            ('SPDist',data['spdist_kpc'],data['spdist_quality']),
        ):
            use = support&ok&np.isfinite(ref)&(ref>0)
            result[name] = (ref[use],q[use],grz[use])
        out = DEST/side/key
        out.mkdir(parents=True,exist_ok=True)
        with plt.rc_context(STYLE):
            render_reference_figures(out,result,
                distance_title=f'{side.title()} · {label} · same reference PSFs')


def accepted_distance(rows,side):
    out = []
    for a in rows:
        if a['reference']!='SpecDis' or a['axis']!='all' or a['group']!='blue_dwarf' or a['d_lo']!='5.0':
            continue
        if not a.get('weighted_nmad'):
            continue
        n,e = int(a['available']),float(a['effective_n'])
        bias,nmad,severe = (float(a[f'weighted_{f}']) for f in ('bias','nmad','severe'))
        out.append(dict(side=side,method=a['method'],available=n,effective_n=e,
            effective_retention=e/n,bias=bias,nmad=nmad,severe=severe,
            retention_pass=e/n>=.5,distance_pass=(abs(bias)<.10 and nmad<=.20 and severe<.10),
            scope='fixed targeted SpecDis reference cohort'))
    return out


def run():
    lock = check_templates()
    inputs,distances,structure = {},[],[]
    for side in ('north','south'):
        root = REFERENCE/side
        orig = bind(root/'complete.json',inputs)
        t = read_columns(root/'fixed_rows.npz',(
            'TYPE','ls_id','g0','r0','reference_kpc','gaia_state','specdis_kpc','specdis_quality',
            'spdist_kpc','spdist_quality','simple_mu','qrf_mu'))
        if (not np.all(t['TYPE']=='PSF') or len(np.unique(t['ls_id']))!=len(t['ls_id'])
                or len(t['ls_id'])!=orig['held_out']):
            raise ValueError('Changed held-out reference cohort')
        rows = read_table(root/'distance.csv')
        if any(int(a['selected'])>int(a['available']) or float(a['effective_n'])>int(a['available'])+1e-6 for a in rows):
            raise ValueError('Retention exceeds fixed roster')
        distances.extend(accepted_distance(rows,side))
        plots(side,t)
    for group in ('north','m30'):
        rec = bind(STRUCTURE/f'{group}.json',inputs)
        if rec['inputs']['style']!=lock:
            raise ValueError('Unexpected figure style')
        for name in rec['results']:
            for part in (0,1):
                rows = read_table(STRUCTURE/f'{name.lower()}_{part}.csv')
                for r in rows:
                    if r['peak_bias'] and r['reference_window_poisson_z']:
                        r['peak_pass'] = (abs(float(r['peak_bias']))<=.10 and float(r['reference_window_poisson_z'])>=3.)
                    structure.append(r)
    for p in [*DEST.rglob('*.png'),*STRUCTURE.glob('*.png')]:
        with Image.open(p) as img:
            if img.width<1000 or img.height<1000 or abs(img.info['dpi'][0]-300)>1:
                raise ValueError('Unexpected plot format')
            img.verify()
        with p.with_suffix('.pdf').open('rb') as f:
            if f.read(5)!=b'%PDF-':
                raise ValueError('Missing vector PDF')
    csv_rows(DEST/'distance_acceptance.csv',distances)
    csv_rows(DEST/'structure_acceptance.csv',structure)
    save_json(DEST/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,style=lock,
        outputs={str(f.relative_to(DEST)):sha(f) for f in DEST.rglob('*') if f.is_file() and f.name!='complete.json'}))


if __name__=='__main__':
    require_compute()
    run()
