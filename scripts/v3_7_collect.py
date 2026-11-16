#!/usr/bin/env python3
"""Validate and copy compact Slurm science products for the paper."""
import argparse
from collections import defaultdict
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


def predicted_composition(path,destination):
    """Transpose reference-shell counts into the two predicted near-far bins."""
    totals = defaultdict(lambda: [0,0.,0,0.])
    with path.open() as f:
        for r in csv.DictReader(f):
            if r['reference_class']=='all':
                continue
            source_shell = int(r['reference_shell'] in ('10–20','20–50','>50'))
            group = tuple(r[k] for k in ('source','selection','domain','reference_class','history'))
            for j,name in ((1,'2–5'),(2,'5–10')):
                item = totals[(*group,name)]
                hard,soft = int(r[f'hard_{j}']),float(r[f'pdf_{j}'])
                item[0] += hard
                item[1] += soft
                item[2] += hard*source_shell
                item[3] += soft*source_shell
    with destination.open('w') as f:
        writer = csv.writer(f,lineterminator="\n")
        writer.writerow(('source','selection','domain','reference_class','history','predicted_shell',
                         'reference_hard_count','reference_pdf_mass','reference_gt10_hard_count',
                         'reference_gt10_pdf_mass'))
        writer.writerows((*group,*sums) for group,sums in totals.items())


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
    if stage=='selection':
        target = output/'predicted_composition.csv'
        original = output/'reference.csv'
        predicted_composition(original,target)
        (output/'derived.json').write_text(json.dumps(dict(
            inputs={original.name:sha256(original.read_bytes()).hexdigest()},
            outputs={target.name:sha256(target.read_bytes()).hexdigest()}),indent=2)+'\n')
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
