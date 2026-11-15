#!/usr/bin/env python3
"""Reduce the 48 projected DR11 groups and check against the delivery totals."""
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

from photdis.stage1 import require_compute
from v2_1_near import ROOT, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_5_dr11_production import FREEZE
from v3_5_dr11_spatial import DELIVERY
from v3_6_catalog_projection import OUT


def reduce_rows(groups):
    slices = defaultdict(list)
    for rows in groups:
        for r in rows:
            slices[r['selection'],r['W1']].append(r)
    total = []
    for (cut,band),rows in slices.items():
        fields = set(rows[0])-{'selection','W1'}
        if len(rows)!=len(groups) or any(set(r)-{'selection','W1'}!=fields for r in rows):
            raise ValueError(f'Incomplete catalog summary for {cut}/{band}')
        summed = {}
        for key in rows[0]:
            if key in fields:
                values = [r[key] for r in rows]
                summed[key] = sum(map(int,values)) if all(v.isdecimal() for v in values) else math.fsum(map(float,values))
        n,compared = summed['rows'],summed['nominal_evaluated']
        hard = sum(summed[f'median_shell_{k}'] for k in range(4))
        prob = sum(summed[f'posterior_shell_{k}'] for k in range(4))
        if hard!=n or not math.isclose(prob,n,abs_tol=1.e-7,rel_tol=0):
            raise ValueError(f'Shell counts fail conservation for {cut}/{band}')
        means = {'qrf_qnear_mean':summed['qrf_qnear_sum']/n if n else None,
                 'dwarf_score_mean':summed['dwarf_score_sum']/n if n else None,
                 'qrf_d50_mean_kpc':summed['qrf_d50_sum_kpc']/n if n else None,
                 'nominal_qnear_mean':summed['nominal_qnear_sum']/compared if compared else None,
                 'qrf_qnear_mean_on_nominal_rows':summed['qrf_qnear_sum_on_nominal_rows']/compared if compared else None}
        disagreements = {f'{k}_fraction_of_nominal_evaluated':v/compared if compared else None
                         for k,v in summed.items() if '_below_gate' in k}
        total.append(dict(selection=cut,W1=band,**summed,**means,**disagreements))
    return total


def main():
    require_compute()
    delivery = DELIVERY/'complete.json'
    record = json.loads(delivery.read_text())
    freeze = json.loads(verified(FREEZE,record['production_freeze_sha256']).read_text())
    parts,summary = {},[]
    for group,names in enumerate(freeze['groups']):
        path = OUT/f'{group:02d}'/'complete.json'
        r = json.loads(path.read_text())
        if r['group']!=group or set(r['sweeps'])!=set(names):
            raise ValueError(f'Projection receipt has incorrect sweep membership: {path}')
        with verified(path.with_name('summary.csv'),r['outputs']['summary.csv']).open() as stream:
            summary.append(list(csv.DictReader(stream)))
        parts[str(path.relative_to(ROOT))] = sha(path)
    rows = reduce_rows(summary)
    for row in rows:
        if row['W1']=='all':
            key = 'rows' if row['selection']=='all' else row['selection']
            if row['rows']!=record['totals'][key]:
                raise ValueError(f'Delivered {key} count differs from catalog summary')
    dest = OUT/'complete.json'
    if dest.exists():
        raise FileExistsError(dest)
    table = OUT/'summary.csv'
    csv_rows(table,rows)
    save_json(dest,dict(job=os.getenv('SLURM_JOB_ID'),rows=record['totals']['rows'],
        child_receipts=parts,inputs={str(p.relative_to(ROOT)):sha(p) for p in (delivery,Path(__file__))},
        outputs={table.name:sha(table)},
        interpretation='QRF posterior sums and medians are model-derived. Nominal disagreements use only rows with computed nominal near probabilities; rows without a nominal score have no comparison.'))
    print(f'All {len(parts)} groups reconcile to {record["totals"]["rows"]} rows',flush=True)


if __name__=='__main__':
    main()
