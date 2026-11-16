#!/usr/bin/env python3
"""Freeze the scientific products and reconcile the counts from each verified stage."""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from photdis.stage1 import require_compute
from v2_1_near import ROOT, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_9_catalog import OUT
from v3_9_dust import OUT as DUST
from v3_9_existing import GATES, PDF
from v3_9_branches import OUT as BRANCHES

DEST = ROOT/'data/v3_9/delivery'


def receipt(path, *, verify_outputs=True):
    result = json.loads(path.read_text())
    if verify_outputs:
        for name,digest in result['outputs'].items():
            verified(path.parent/name,digest)
    return result


def calibration(shards):
    accum = np.zeros((2,6))
    rows = checks = 0
    for i in range(shards):
        path = PDF/f'{i:02d}'/'complete.json'
        r = receipt(path)
        if r['shard']!=i or r['shards']!=shards:
            raise ValueError('Incomplete or overlapping near-PDF shards')
        with (path.parent/'shell_counts.csv').open() as stream:
            counts = [a for a in csv.DictReader(stream) if 'rows' in a]
        n = sum(int(a['rows']) for a in counts)
        if n!=r['rows']:
            raise ValueError('Calibrated per-sweep counts do not match receipt')
        rows += n
        checks += len(counts)
        for k,kind in enumerate(('original','calibrated')):
            accum[k] += np.array([sum(float(a[f'{kind}_{j}']) for a in counts) for j in range(6)])
    np.testing.assert_allclose(accum.sum(axis=1),rows,atol=1e-5,rtol=0)
    return rows,checks,accum


def run(shards):
    DEST.mkdir(parents=True,exist_ok=True)
    science = {
        'gates':GATES/'complete.json','branch_control':BRANCHES/'complete.json',
        'bayestar19':DUST/'bayestar19'/'complete.json',
        'faint_noise_control':OUT/'noise_control/complete.json',
        'north_native_fit':OUT/'north/native_model/complete.json',
        'paired_photometry':OUT/'validation/overlap/complete.json',
        **{name:OUT/'figures'/name/'complete.json' for name in ('south_faint','south_low','north')}}
    records = {k:receipt(v) for k,v in science.items()}
    rows,parts,mass = calibration(shards)
    summaries = [dict(region=name,precise_survey_owned_rows=records[name]['projected_rows'],
                     reference_rows=records[name]['reference_rows']) for name in ('south_faint','south_low','north')]
    csv_rows(DEST/'expansion.csv',summaries)
    edges = [0,2,5,10,20,50,float('inf')]
    csv_rows(DEST/'calibrated_shell_counts.csv',
        [dict(lo_kpc=edges[i],hi_kpc=edges[i+1],frozen=mass[0,i],calibrated=mass[1,i]) for i in range(6)])
    census = {}
    for side in ('south','north'):
        status = []
        catalog = OUT/side/'inventory.json'
        inventory = json.loads(catalog.read_text())
        for group in inventory['groups']:
            for source in group:
                stem = Path(source['filename']).stem
                p = OUT/side/'photo'/stem/'complete.json'
                q = OUT/side/'predict'/stem/'complete.json'
                a,b = receipt(p,verify_outputs=False), receipt(q,verify_outputs=False)
                if sha(p)!=b['binding']['photo']:
                    raise ValueError(f'Broken census to prediction binding in {stem}')
                if a.get('error') or a.get('missing_fields'):
                    counts = {}
                else:
                    counts = a['counts']
                    if a['rows'] != sum(counts.values()):
                        raise ValueError(f'Unaccounted-for FITS rows in {stem}')
                status.append(dict(sweep=stem,raw_count=a['rows'],photo_sha256=sha(p),
                    predict_sha256=sha(q),error=a.get('error') or '',
                    missing_fields=' '.join(a.get('missing_fields',[])),
                    **counts,**b['counts']))
        target = DEST/f'{side}_sweeps.csv'
        csv_rows(target,status)
        census[side] = dict(sweeps=len(status),bad=sum(bool(z['error'] or z['missing_fields']) for z in status))
    gate = records['gates']
    if gate['rows']!=336678 or not rows:
        raise ValueError('Missing complete gate or near-PDF counts')
    save_json(DEST/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),census=census,
        gate_rows=gate['rows'],gate_passed=gate['passes'],gate_reused=gate['from_redrock'],
        near_pdf_rows=rows,near_pdf_parts=parts,
        sources={str(p.relative_to(ROOT)):sha(p) for p in science.values()},
        outputs={p.name:sha(p) for p in DEST.iterdir() if p.name!='complete.json'}))
    print('v3.9 sources and PDF counts validated',census,gate['rows'],rows,flush=True)


if __name__=='__main__':
    require_compute()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--near-pdf-shards',type=int,default=16)
    run(ap.parse_args().near_pdf_shards)
