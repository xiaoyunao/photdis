"""Versioned v1.1 IO and reference score helpers."""
from baseline_v1_common import *
CFG=json.load(open(ROOT/'configs/baseline_v1_1.json'))
OUT=ROOT/CFG['output_dir'];EXPERT=ROOT/CFG['expert_dir']
VARIANTS=('wide','oracle','oracle_support','oracle_full','natural','balanced','evidence')

def reference_rows(t,name):
    valid=(t['label_source']>0)|t['specdis_valid']|t['spdist_valid']
    return np.flatnonzero(np.isin(t['split'],[1,2])&valid&available(t,MODELS[name][1]))

def evidence_rows(t,name):
    use=(t['split']==0)&t['fit_eligible']&available(t,MODELS[name][1])
    rng=np.random.default_rng(CFG['seed']);rows=[];weights=[]
    for field in sorted(np.unique(t['field'][use])):
        for label in (-1,0,1,2):
            pool=np.flatnonzero(use&(t['field']==field)&(t['class_label']==label))
            if not len(pool):continue
            chosen=np.sort(rng.choice(pool,min(len(pool),CFG['evidence_per_field_class']),replace=False))
            rows.extend(chosen);weights.extend([len(pool)/len(chosen)]*len(chosen))
    order=np.argsort(rows)
    return np.asarray(rows)[order],np.asarray(weights)[order],np.flatnonzero(use)
