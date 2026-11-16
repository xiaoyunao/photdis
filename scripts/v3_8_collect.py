#!/usr/bin/env python3
"""Verify and publish compact v3.8 science products from a staged transfer."""
import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT/'figures/v3_8'
TABLES = ROOT/'manifests/v3_8'
SCIENCE = ('candidate','candidate_index','branches','reliability','north','north_report','redrock')
FIGURE_STAGES = ('reference','overview','sky')


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def collect(folder,stage):
    mark = folder/'complete.json'
    receipts = json.loads(mark.read_text())['outputs']
    dest = TABLES/stage
    dest.mkdir(parents=True,exist_ok=True)
    FIGS.mkdir(parents=True,exist_ok=True)
    n = 0
    for f,h in receipts.items():
        path = folder/f
        if path.suffix not in ('.png','.pdf','.csv','.json'):
            continue
        if digest(path)!=h:
            raise ValueError(f'Invalid transferred product {path}')
        target = FIGS/f if path.suffix in ('.png','.pdf') else dest/f
        shutil.copyfile(path,target)
        n += 1
    shutil.copyfile(mark,dest/mark.name)
    return n


def table(stage,name):
    with (TABLES/stage/name).open() as file:
        return list(csv.DictReader(file))


def verify():
    from PIL import Image
    checked = 0
    for stage in (*SCIENCE,*FIGURE_STAGES):
        folder = TABLES/stage
        for f,h in json.loads((folder/'complete.json').read_text())['outputs'].items():
            suffix = Path(f).suffix
            if suffix not in ('.png','.pdf','.csv','.json'):
                continue
            src = FIGS/f if suffix in ('.png','.pdf') else folder/f
            assert digest(src)==h,src
            if suffix=='.png':
                with Image.open(src) as im:
                    assert abs(im.info['dpi'][0]-300)<.1,src
                    if f.startswith('F04_'):
                        assert im.size[0]==6600,src
            checked += 1
    for marker in (TABLES/'figures').glob('*.json'):
        for f,h in json.loads(marker.read_text())['outputs'].items():
            path = marker.parent/f if f.endswith('.csv') else FIGS/f
            assert digest(path)==h,path
            checked += 1
    for f in (TABLES/'sky').glob('F04_*.csv'):
        with f.open() as stream:
            n = [int(x['n']) for x in csv.DictReader(stream)]
        assert len(n)==4 and sum(n[:3])==n[3],f
        checked += 1
    distant = {r['source']:int(r['n']) for r in table('reference','F02_direct_distances.csv')}
    for row in table('reference','F03_same_star_cmd.csv'):
        assert int(row['rows'])==distant[row['source']],row
        checked += 1
    data = table('candidate','parallax_distribution.csv')
    for v,s in {(r['selection'],r['scope']) for r in data}:
        rows = [r for r in data if r['selection']==v and r['scope']==s]
        for k in ('observed','qrf','broad'):
            np.testing.assert_allclose(sum(float(r[k]) for r in rows),1.,atol=1e-9)
            checked += 1
    for r in table('reliability','reference_gap.csv'):
        assert int(r['rows'])==int(r['qualified_geometry_or_spec'])+int(r['without_qualified_match']),r
        checked += 1
    for name in ('published_primary','confirmed_addition','missing_nominal_gate','evaluated_reject'):
        a = [r for r in table('candidate_index','membership.csv') if r['status']==name]
        assert len(a)==4 and int(a[0]['rows'])==sum(int(r['shell_rows']) for r in a),name
        checked += 1
    pilot = json.loads((TABLES/'north/complete.json').read_text())
    north = table('north','distance.csv')
    n = int(north[0]['n'])
    assert n<=pilot['test'] and all(int(r['n'])==n for r in north)
    assert sum(int(r['qualified_native_pairs']) for r in
               table('north_report','classification_comparison.csv'))==n
    checked += 2
    names = ('anchors.csv','cases.csv')
    groups = [{r['ls_id'] for r in table('branches',name) if r['TYPE']=='PSF'} for name in names]
    assert len(groups[0])==32 and len(groups[1])==48 and not groups[0]&groups[1]
    checked += 1
    return checked


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    p.add_argument('stages',nargs='*',choices=(*SCIENCE,*FIGURE_STAGES))
    p.add_argument('--verify',action='store_true')
    args = p.parse_args()
    for stage in args.stages:
        folder = args.source/stage if stage in SCIENCE else args.source/'direct_figures'/stage
        print(f'{stage}: {collect(folder,stage)} products')
    if args.verify:
        print(f'Validated {verify()} product and science-integrity checks')


if __name__=='__main__':
    main()
