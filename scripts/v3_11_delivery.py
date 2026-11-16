#!/usr/bin/env python3
"""Verify the source-bound v3.11 science outputs before delivery."""

import csv
import json
import os

import numpy as np
from PIL import Image

from photdis.stage1 import require_compute
from v2_1_near import ROOT, save_json, sha
from v3_3_sed import verified
from v3_10_figures import check_templates

BASE = ROOT/'data/v3_11'
STAGES = ('selection/north','selection/south','layers/north','layers/south',
          'branches','seam','structure','depth','figures')


def table(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def run():
    lock = check_templates()
    results,inputs = {},{}
    png,pdf = 0,0
    for stage in STAGES:
        folder = BASE/stage
        receipt = folder/'complete.json'
        origin = json.loads(receipt.read_text())
        inputs[str(receipt.relative_to(ROOT))] = sha(receipt)
        for fname,digest in origin['outputs'].items():
            f = verified(folder/fname,digest)
            if f.suffix=='.png':
                with Image.open(f) as plate:
                    if plate.width<1000 or plate.height<1000 or abs(plate.info['dpi'][0]-300)>1.:
                        raise ValueError(f'Invalid figure dimensions or DPI: {f}')
                    plate.verify()
                png += 1
            elif f.suffix=='.pdf':
                with f.open('rb') as stream:
                    if stream.read(5)!=b'%PDF-':
                        raise ValueError(f'Invalid PDF figure: {f}')
                pdf += 1
        if stage.startswith('selection/'):
            if any(b['candidate_pass']>b['baseline_pass'] for b in origin['calibration_leakage'].values()):
                raise ValueError('Calibration passes exceed bound')
            rows = table(folder/'distances.csv')
            if not all(int(r['n'])<=int(r['eligible']) for r in rows):
                raise ValueError('Selection count exceeds the fixed cohort')
        elif stage.startswith('layers/'):
            rows = table(folder/'composition.csv')
            for label,expected in (('supported',origin['role4']),):
                k = [r for r in rows if r['view']==label and r['component']=='role4_catalog'
                     and r['r_min']=='-inf' and r['r_max']=='20.0'
                     and r['gr_min']=='-inf' and r['gr_max']=='inf']
                if len(k)!=4 or sum(int(r['hard_n']) for r in k)!=expected:
                    raise ValueError('Hard distance shell membership fails conservation')
                if abs(sum(float(r['pdf_n']) for r in k)-expected)>.1:
                    raise ValueError('PDF shell membership fails conservation')
        elif stage=='branches':
            rows = table(folder/'evidence.csv')
            pairs = {(r['ls_id'],r['test'],r['mode']) for r in rows}
            if len(pairs)!=len(rows) or any(r['TYPE']!='PSF' for r in rows):
                raise ValueError('Unexpected band comparison identities')
            for r in rows:
                if r['mode']=='plus_band' and (r['ls_id'],r['test'],'optical') not in pairs:
                    raise ValueError('Unpaired band comparison')
        elif stage=='seam':
            with np.load(folder/'same_star.npz',allow_pickle=False) as p:
                if len(np.unique(p['ls_id']))!=origin['paired'] or not np.all(p['TYPE']=='PSF'):
                    raise ValueError('Native seam join lost identity')
        elif stage in ('figures','structure'):
            if origin['inputs']['figure_templates']!=lock:
                raise ValueError('A figure used a different style source')
        results[stage] = dict(job=origin['job'],outputs=len(origin['outputs']))
    if png!=pdf or png<8:
        raise ValueError('Incomplete pairs of English scientific figures')
    save_json(BASE/'delivery.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,
        stages=results,figure_style_lock=lock,png=png,pdf=pdf))


if __name__=='__main__':
    require_compute()
    run()
