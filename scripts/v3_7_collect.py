#!/usr/bin/env python3
"""Validate and copy compact Slurm science products for the paper."""
import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
CSV = {
    'reference_comparison': ('qualified_migration.csv','overlap.csv','paired_distances.csv'),
    'lf_odds': ('odds_leakage.csv',),
    'selection': ('reference.csv','catalog.csv'),
    'weak_gaia': ('scores.csv','stress.csv'),
    'blue_neighborhood': ('neighborhoods.csv',),
    'classifier_sky': ('development.csv','retention.csv'),
    'lf_known_branches': ('case_probabilities.csv',),
    'use_domains': ('observable_domains.csv',),
}
SMALL = {
    'paired_distances.csv': lambda r: r['history']=='all' and (
        r['axis']=='all' or r['anchor']=='Gaia' and r['overlap']=='Gaia+SpecDis+SPDist'),
    'retention.csv': lambda r: r['healpix_nside4']=='-1',
}


def collect(stage,source,root):
    receipt = source/stage/'complete.json'
    checksums = json.loads(receipt.read_text())['outputs']
    output = root/stage
    output.mkdir(parents=True,exist_ok=True)
    compact = {}
    for name in CSV[stage]:
        path = source/stage/name
        digest = sha256(path.read_bytes()).hexdigest()
        if checksums[name]!=digest:
            raise ValueError(f'Checksum mismatch: {path}')
        if name in SMALL:
            with path.open() as f, (output/name).open('w') as g:
                reader = csv.DictReader(f)
                writer = csv.DictWriter(g,fieldnames=reader.fieldnames,lineterminator="\n")
                writer.writeheader()
                writer.writerows(r for r in reader if SMALL[name](r))
            compact[name] = dict(complete_sha256=digest,
                                 compact_sha256=sha256((output/name).read_bytes()).hexdigest())
        else:
            if path.stat().st_size>5*1024**2:
                raise ValueError(f'Provide a compact projection of {path}')
            shutil.copyfile(path,output/name)
    shutil.copyfile(receipt,output/receipt.name)
    if compact:
        (output/'compact.json').write_text(json.dumps(compact,indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('inputs',type=Path)
    p.add_argument('stages',nargs='*',choices=tuple(CSV))
    args = p.parse_args()
    for stage in args.stages or CSV:
        collect(stage,args.inputs,ROOT/'manifests/v3_7')


if __name__=='__main__':
    main()
