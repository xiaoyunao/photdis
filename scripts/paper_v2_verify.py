#!/usr/bin/env python3
"""Validate the v2 statistical and rendering receipts."""
import csv
import json
import os

import numpy as np
from PIL import Image

from photdis.stage1 import require_compute
from v2_1_near import save_json,sha
from v3_3_sed import verified
from paper_v2_scan import OUT,REFS


def csv_table(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def main():
    require_compute()
    stages=('reference','figures','filter','spatial')
    marks={}
    reports={}
    for stage in stages:
        p=OUT/stage/'complete.json'
        r=json.loads(p.read_text())
        for k,h in r['outputs'].items():
            verified(p.with_name(k),h)
        marks[stage]=sha(p)
        reports[stage]=r
    t=csv_table(OUT/'reference/coverage.csv')
    for ref in REFS:
        rows=[r for r in t if r['reference']==ref]
        assert sum(int(r['parent']) for r in rows)==72289386
        assert sum(int(r['parent']) for r in rows if r['state']=='3')==43347612
        for r in rows:
            n,m,v,q,h=map(lambda k:int(r[k]),('parent','recorded','valid','qualified','heldout'))
            assert n>=m>=v>=q>=h and n==m+int(r['unrecorded'])
    t=csv_table(OUT/'reference/benchmark_replay.csv')
    q=[r for r in t if r['frame']=='frozen_F15']
    assert [int(r['n']) for r in q]==[9355,8927]
    np.testing.assert_allclose([float(r['nmad']) for r in q],[.09771549,.10383555],atol=1e-6)
    t=csv_table(OUT/'spatial/distance_layers.csv')
    for method in ('Median distance','Integrated PDF mass'):
        np.testing.assert_allclose(sum(float(r['n_or_mass']) for r in t if r['region']=='all' and
                                      r['statistic']==method),43347612,atol=3e-6,rtol=0)
    plates=[]
    for stage in ('figures','spatial'):
        d=OUT/stage
        for a in sorted(d.glob('*.png')):
            with Image.open(a) as x:
                x.verify()
            with Image.open(a) as x:
                assert x.width>=2000 and all(abs(v-300.)<.02 for v in x.info['dpi'])
            assert a.with_suffix('.pdf').read_bytes().startswith(b'%PDF')
            plates.append(a.stem)
    assert len(plates)>=8
    adopted=reports['filter']['passed']
    if adopted:
        assert (OUT/'spatial/aitoff_matched_filter.png').exists()
    save_json(OUT/'delivery.json',dict(job=os.getenv('SLURM_JOB_ID'),
        inputs=marks,plates=plates,parent=72289386,pdfs=43347612,filter_gate=adopted))


if __name__=='__main__':
    main()
