#!/usr/bin/env python3
"""Predict only the fixed small parent and held-Gaia samples."""
from baseline_v1_1_common import *
import joblib,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from photdis.stage1 import require_compute
from photdis.baseline_v1_1 import integrate
_T=None;_MODEL=None;_GATE=None

def worker(rows):
    m,e,ids=arrays(_T,rows)
    result=integrate(_MODEL,m,e,ids,('wide','evidence'),gate=_GATE,draws=CFG['draws'])
    raw=_GATE.predict_proba(m)
    result['evidence']['raw_weights']=raw
    result['evidence']['gate_entropy']=-np.sum(raw*np.log(np.maximum(raw,1e-300)),axis=1)
    return result

def main(kind,name):
    global _T,_MODEL,_GATE
    require_compute();_T=load(OUT/f'{kind}_sample.npz')
    _MODEL=joblib.load(EXPERT/f'{name}.joblib');_GATE=joblib.load(OUT/f'{name}_gate.joblib')
    rows=np.flatnonzero(available(_T,name));batches=[rows[j:j+CFG['chunk_size']] for j in range(0,len(rows),CFG['chunk_size'])]
    with ProcessPoolExecutor(max_workers=8,mp_context=mp.get_context('fork')) as pool:parts=list(pool.map(worker,batches))
    for variant in ('wide','evidence'):
        p=concat([x[variant] for x in parts]);p.update(rows=rows,identity=_T['identity'][rows],edges=_MODEL.edges,centers=_MODEL.centers)
        np.savez_compressed(OUT/f'{kind}_{name}_{variant}.npz',**p)
    print(kind,name,len(rows),flush=True)
if __name__=='__main__':main(*sys.argv[1:])
