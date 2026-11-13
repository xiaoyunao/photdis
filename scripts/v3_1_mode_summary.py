#!/usr/bin/env python3
"""Compact design-weighted mode/support summary; full weak-mode rows stay remote."""
import csv
import json
import os
from collections import defaultdict
from v2_1_near import ROOT,sha,save_json,csv_rows
from photdis.stage1 import require_compute
from photdis.v3_1.comparison import paired_mean


def main():
    require_compute(); base=ROOT/'data/v3_1/diagnostics/diagnostic'
    report=base/'stratified_report'; source=report/'mode_reference_support.csv'
    marker=json.loads((report/'complete.json').read_text()); assert sha(source)==marker['outputs'][source.name]
    complete=json.loads((base/'complete.json').read_text()); metadata={}
    for name,digest in complete['objects'].items():
        path=base/name; assert sha(path)==digest
        for r in json.loads(path.read_text())['results']:
            metadata[r['identity'],r['control']]=r
    stars=defaultdict(lambda:dict(total=0,significant=0,weak_mass=0.,low_support_mode_mass=0.,
        missing_label_mode_mass=0.,classes=set(),single=False,binary=False))
    raw_rows=0
    with source.open() as handle:
        for r in csv.DictReader(handle):
            assert r['TYPE']=='PSF'; raw_rows+=1
            key=int(r['identity']),r['control']; item=stars[key]; mass=float(r['mass'])
            item['total']+=1
            if mass <= .05:item['weak_mass']+=mass
            else:
                item['significant']+=1
                classes=[float(r[k]) for k in ['class_dwarf_probability','class_evolved_probability','class_other_probability']]
                item['classes'].add(max(range(3),key=classes.__getitem__))
                item['single'] |= float(r['binary_probability']) < .1
                item['binary'] |= float(r['binary_probability']) > .9
            if float(r['dominant_parent_effective_reference_sources']) < 10:item['low_support_mode_mass']+=mass
            if float(r['dominant_parent_missing_class_fraction']) > .5:item['missing_label_mode_mass']+=mass
    output=[]
    for control in sorted({k[1] for k in stars}):
        for label in [None,-1,0,1,2]:
            keys=[k for k in stars if k[1]==control and (label is None or metadata[k]['class_label']==label)]
            metrics={'mode_count':lambda x:x['total'],'significant_mode_count':lambda x:x['significant'],
                'multimodal_fraction':lambda x:float(x['significant']>1),
                'multiple_dominant_classes_fraction':lambda x:float(len(x['classes'])>1),
                'single_and_binary_dominated_modes_fraction':lambda x:float(x['single'] and x['binary']),
                'weak_mode_mass':lambda x:x['weak_mass'],
                'mass_in_modes_dominated_by_lt10_effective_sources':lambda x:x['low_support_mode_mass'],
                'mass_in_modes_dominated_by_majority_missing_class_kernel':lambda x:x['missing_label_mode_mass']}
            for name,fn in metrics.items():
                output.append(dict(control=control,reference_class='all' if label is None else str(label),metric=name,
                    **paired_mean([fn(stars[k]) for k in keys],[metadata[k]['design_weight'] for k in keys],[metadata[k]['cell'] for k in keys])))
    out=base/'mode_support_summary';out.mkdir(exist_ok=True)
    csv_rows(out/'summary.csv',output)
    save_json(out/'complete.json',dict(inputs={str(p.relative_to(ROOT)):sha(p) for p in [source,report/'complete.json',base/'complete.json',ROOT/'scripts/v3_1_mode_summary.py',ROOT/'src/photdis/v3_1/comparison.py']},
        outputs={'summary.csv':sha(out/'summary.csv')},raw_mode_rows=raw_rows,star_controls=len(stars),job=os.getenv('SLURM_JOB_ID'),
        scope='Design-weighted per-star summaries; significant mass>.05; weak basins retained, not assumed numerical artifacts. Dominant-kernel source support is not posterior source ESS; binary dominance is model attribution, not an observed binary classification.'))


if __name__=='__main__':main()
