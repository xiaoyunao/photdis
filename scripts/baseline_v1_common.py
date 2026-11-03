"""Shared versioned IO and scoring for v1, without modifying v0 products."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'.deps')]
import numpy as np
from photdis.baseline_v0 import BANDS,ERR,metrics
from kgiant_distance_inventory import write_json,sha
from baseline_v0_fit import truth_mu,strata,available,MODELS
CFG=json.load(open(ROOT/'configs/baseline_v1.json'))
OLD=ROOT/CFG['input_dir'];OUT=ROOT/CFG['output_dir']

def load(path):
    with np.load(path) as f:return {k:f[k] for k in f.files}

def arrays(t,rows):
    return np.column_stack([t[k][rows] for k in BANDS]),np.column_stack([t[k][rows] for k in ERR]),t['identity'][rows]

def identity_hash(ids):return hashlib.sha256(np.asarray(ids,dtype='i8').tobytes()).hexdigest()

def score(t,rows,mu):
    return {s:{k:metrics(mu[v],truth_mu(t,s)[rows][v]) for k,v in strata(t,rows).items()}
            for s in ('Gaia','specdis','spdist','chosen')}

def concat(parts):return {k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
