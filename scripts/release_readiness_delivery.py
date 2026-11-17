#!/usr/bin/env python3
"""Check source/interval counts and original data lineage in release products."""
import csv
import json
import os

import numpy as np
from PIL import Image

from photdis.stage1 import require_compute
from v2_1_near import ROOT, save_json, sha
from v3_3_sed import verified
from v3_6_vac_distances import read_columns
from v3_10_figures import check_templates
from release_readiness_archive import PRODUCTS
from release_readiness_sample import DEST
from release_readiness_access import SCHEMA


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def main():
    require_compute()
    inputs={'style':check_templates(),'schema':sha(SCHEMA)}
    receipt=json.loads((DEST/'cohort_receipt.json').read_text())
    ids=np.load(verified(DEST/'gaia_ids.npy',receipt['outputs']['gaia_ids.npy']),allow_pickle=False)
    n=len(ids)
    for name in PRODUCTS:
        root=DEST/'archives'/name
        path=root/'complete.json'
        a=json.loads(path.read_text())
        bsize=a['inputs']['batch_size']
        assert a['rows']==n and a['inputs']['ids_sha256']==receipt['outputs']['gaia_ids.npy']
        assert len(a['batches'])==(n+bsize-1)//bsize
        counted=0
        for key,digest in a['batches'].items():
            b=json.loads(verified(root/f'{key}.json',digest).read_text())
            i=int(key)*bsize
            j=min(i+bsize,n)
            assert (b['first'],b['last'],b['input_rows'])==(ids[i],ids[j-1],j-i)
            assert 0<=b['matched']<=b['input_rows']
            counted+=b['matched']
        assert counted==a['matched']
        inputs[name]=sha(path)
    for folder in ('population','comparison','access'):
        marker=DEST/folder/'complete.json'
        x=json.loads(marker.read_text())
        for k,h in x['outputs'].items():
            verified(marker.with_name(k),h)
        inputs[folder]=sha(marker)
    coverage=rows(DEST/'comparison/coverage.csv')
    summary=[x for x in coverage if x['region']=='all' and x['group']=='all']
    assert len(summary)==5
    for r in summary:
        assert int(r['n'])==n and int(r['not_queried'])==0
        assert int(r['valid68'])<=int(r['valid_median'])<=int(r['archive_row'])<=n
    pairs=rows(DEST/'comparison/paired_distances.csv')
    counts={}
    for r in pairs:
        if r['match'].startswith('pair_') or r['match']=='all_methods_common':
            key=tuple(r[k] for k in ('region','group','reference','ref_lo','ref_hi','match'))
            counts.setdefault(key,int(r['n']))
            assert counts[key]==int(r['n'])
    p=rows(DEST/'population/retention_migration.csv')
    if not all(sum(int(x[k]) for k in ('median_0_2','median_2_5','median_5_10','median_10_plus'))==int(x['n'])
               for x in p):
        raise ValueError('Population migration does not close to the direct statistics')
    schema=json.loads(SCHEMA.read_text())
    f=ROOT/schema['selected_index']['path']
    release=json.loads(f.with_name('complete.json').read_text())
    t=read_columns(verified(f,release['outputs'][f.name]),('ls_id','TYPE','science','q'))
    assert len(np.unique(t['ls_id']))==len(t['ls_id']) and (t['TYPE']=='PSF').all()
    for k,v in schema['selected_index']['tiers'].items():
        assert int((t['science']==int(k)).sum())==v['rows']
    for code,(a,b) in enumerate(((0.,2.),(2.,5.),(5.,10.)),start=1):
        d=t['q'][t['science']==code,2]
        assert ((d>=a)&(d<b)).all()
    png=DEST/'comparison/F12_published_distances.png'
    with Image.open(png) as img:
        assert abs(img.info['dpi'][0]-300)<.1 and min(img.size)>1000
        img.verify()
    out=DEST/'delivery'
    out.mkdir(exist_ok=True,parents=True)
    save_json(out/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),rows=n,inputs=inputs,
        all_products=int(summary[0]['all_products']),paired_views=len(counts),
        population_rows=len(p),publication_rows=len(t['ls_id'])))


if __name__=='__main__':
    require_compute()
    main()
