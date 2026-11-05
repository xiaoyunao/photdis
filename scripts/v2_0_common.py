"""Shared paths, CSV output and audited reads for the frozen v2.0 report."""
import csv
import json
from pathlib import Path
import numpy as np
from baseline_v1_4_experiments import ROOT, OUT, save_json, load, subset, truth
from baseline_v1_4_extract import sha, atomic_npz
from photdis.report_v2 import percent_metrics, ranges
from photdis.reference_v1_4 import unique_join
from photdis.stage1 import galactic

DEST=ROOT/'data/v2_0'
TABLES=DEST/'manifests'
FIG=DEST/'figures'
CFG=json.loads((ROOT/'configs/v2_0_snapshot.json').read_text())
INPUTS={}


def audited(path, expected=None):
    path=Path(path)
    digest=sha(path)
    if expected is not None:
        assert digest==expected,str(path)
    INPUTS[str(path.relative_to(ROOT))]=digest
    return path


def csv_rows(path,rows):
    rows=list(rows)
    if not rows:
        rows=[dict(status='no rows')]
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with open(path,'w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def strata(t):
    out={'all':np.ones(len(t['identity']),bool)}
    for k,label in [(-1,'unknown'),(0,'dwarf'),(1,'evolved'),(2,'other')]:
        out[label]=t['class_label']==k
    for lo,hi in [(-np.inf,18),(18,20),(20,22),(22,25)]:
        mask=(t['r']>=lo)&(t['r']<hi)
        label=f'r{lo}_{hi}'
        out[label]=mask
        out[label+':no_W1']=mask&~t['B']
    out.update(no_W1=~t['B'],W1_only=t['B']&~t['C'],W1W2=t['C'])
    for lo,hi in [(0,.03),(.03,.1),(.1,np.inf)]:
        out[f'error{lo}_{hi}']=(t['optical_error']>=lo)&(t['optical_error']<hi)
    for name in ('supported','ambiguous','outside_train_precision'):
        if name in t and (name+'_available' not in t or t[name+'_available'].any()):
            out[name]=t[name]
            out['not_'+name]=~t[name]
    if 'TYPE' in t:
        for name in np.unique(t['TYPE']):
            out['TYPE_'+name]=t['TYPE']==name
    for cell in np.unique(t['cell']):
        out[f'cell{cell}']=t['cell']==cell
    return out
