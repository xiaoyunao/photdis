#!/usr/bin/env python3
"""Verify the full catalogue outputs and close the observational validation ledger."""
import csv
import json
import os
from collections import defaultdict

from PIL import Image

from photdis.stage1 import require_compute
from v2_1_near import ROOT, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_10_figures import check_templates
from final_catalogue_raw import DEST
from release_readiness_archive import PRODUCTS


def rows(p):
    with p.open() as f:
        return list(csv.DictReader(f))


def main():
    require_compute()
    inputs={'style':check_templates()}
    out=DEST/'delivery'
    out.mkdir(exist_ok=True,parents=True)
    figures=[]
    for name in ('atlas','population','benchmark/results'):
        d=DEST/name
        marker=d/'complete.json'
        m=json.loads(marker.read_text())
        inputs[name]=sha(marker)
        for k,v in m['outputs'].items():
            p=verified(d/k,v)
            if p.suffix=='.png':
                with Image.open(p) as im:
                    assert min(im.size)>1000 and abs(im.info['dpi'][0]-300)<.1
                    figures.append(dict(stage=name,figure=p.stem,width=im.width,height=im.height))
                    im.verify()
                assert p.with_suffix('.pdf').read_bytes().startswith(b'%PDF')
    assert len(figures)==9
    a=rows(DEST/'atlas/F18_catalogue_flow.csv')
    assert [int(x['n']) for x in a][0]==3967618454
    assert [int(x['n']) for x in a][-3:]==[72289386,43347612,2281023]
    b=DEST/'benchmark'
    q=json.loads((b/'parent.json').read_text())
    for name in PRODUCTS:
        d=b/'archives'/name
        p=d/'complete.json'
        z=json.loads(p.read_text())
        assert z['rows']==q['unique_gaia'] and z['inputs']['ids_sha256']==q['outputs']['ids.npy']
        for k,h in z['batches'].items():
            verified(d/f'{k}.json',h)
        inputs[name]=sha(p)
    for r in rows(b/'results/coverage.csv'):
        assert int(r['valid68'])<=int(r['valid'])<=int(r['qualified_link'])<=int(r['matched'])
        assert int(r['matched'])<=int(r['queried'])<=int(r['n'])
    pairs=defaultdict(set)
    for r in rows(b/'results/same_star.csv'):
        pairs[tuple(r[k] for k in ('group','reference','lo','hi','pair'))].add(int(r['n']))
    assert all(len(x)==1 for x in pairs.values())
    groups=defaultdict(lambda:[0,0,0])
    for r in rows(DEST/'population/validation_by_cell.csv'):
        if r['scenario']!='unassessed':
            a=groups[r['region'],r['reference'],r['scenario']]
            a[0]+=1
            a[1]+=int(r['covered'])
            a[2]+=int(r['tests'])
    tests=[]
    for (region,reference,scenario),(cells,n,total) in sorted(groups.items()):
        tests.append(dict(region=region,reference=reference,scenario=scenario,
            cells=cells,within95=n,tests=total,fraction=n/total,passes_90pct=n/total>=.90))
    csv_rows(out/'p1_test_summary.csv',tests)
    csv_rows(out/'figure_checks.csv',figures)
    save_json(out/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,
          figures=len(figures),pairs=len(pairs),parent_sources=q['n'],atlas_partitions=28917,
          outputs={p.name:sha(p) for p in out.glob('*.csv')}))


if __name__=='__main__':
    main()
