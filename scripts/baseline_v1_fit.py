#!/usr/bin/env python3
"""One fixed population family per band/control; no test-dependent fitting."""
from baseline_v1_common import *
import joblib,time
from concurrent.futures import ProcessPoolExecutor
from photdis.stage1 import require_compute
from photdis.baseline_v1 import PopulationModel,integrate

VARIANTS=('wide','natural','balanced','oracle')
_EVAL_MODEL=None;_EVAL_T=None;_EVAL_POLICY=None

def evaluate_chunk(ix):
    m,e,ids=arrays(_EVAL_T,ix);low=_EVAL_T['optical_error'][ix]<.03
    chunk={};used_draws=np.zeros(len(ix),np.int16)
    for label,mask in [('low',low),('high',~low)]:
        if not mask.any():continue
        draws=_EVAL_POLICY[label]['draws'];used_draws[mask]=draws
        values=integrate(_EVAL_MODEL,m[mask],e[mask],ids[mask],draws,
            variants=VARIANTS,oracle_labels=_EVAL_T['class_label'][ix][mask])
        for v,p in values.items():
            if v not in chunk:chunk[v]={k:np.empty((len(ix),)+a.shape[1:],a.dtype) for k,a in p.items()}
            for k,a in p.items():chunk[v][k][mask]=a
    return chunk,used_draws

def train(name):
    require_compute();OUT.mkdir(exist_ok=True,parents=True)
    path=OUT/(name+'.joblib')
    if path.exists():raise FileExistsError(path)
    t=load(OLD/'references.npz');dim,tier=MODELS[name]
    use=(t['split']==0)&t['fit_eligible']&available(t,tier)
    ids=t['identity'][use];old=json.load(open(OLD/'fit_result.json'))['models'][name]
    assert identity_hash(ids)==old['train_identity_sha256']
    assert not np.intersect1d(ids,t['identity'][t['split']>0]).size
    start=time.time();wide=joblib.load(OLD/(name+'.joblib'))
    model=PopulationModel(wide).fit(arrays(t,use)[0],t['Mg'][use],t['class_label'][use],CFG['seed'])
    joblib.dump(model,path)
    write_json(OUT/(name+'_train.json'),dict(n_train=int(use.sum()),
        classes={str(k):int((t['class_label'][use]==k).sum()) for k in (-1,0,1,2)},
        train_identity_sha256=identity_hash(ids),model_sha256=sha(path),
        v0_model_sha256=sha(OLD/(name+'.joblib')),seconds=time.time()-start))
    print('Trained',name,int(use.sum()),time.time()-start,flush=True)

def evaluate(name):
    global _EVAL_MODEL,_EVAL_T,_EVAL_POLICY
    require_compute();t=load(OLD/'references.npz');model=joblib.load(OUT/(name+'.joblib'))
    policy=json.load(open(OUT/'numerical.json'))['selection'];tier=MODELS[name][1]
    valid=(t['label_source']>0)|t['specdis_valid']|t['spdist_valid']
    rows=np.flatnonzero(np.isin(t['split'],[1,2])&valid&available(t,tier))
    chunks={v:[] for v in VARIANTS};all_draws=np.zeros(len(rows),np.int16);start=time.time()
    # Forked workers share read-only models/references. Ordered map preserves IDs.
    _EVAL_MODEL=model;_EVAL_T=t;_EVAL_POLICY=policy
    batches=[rows[j:j+CFG['chunk_size']] for j in range(0,len(rows),CFG['chunk_size'])]
    import multiprocessing as mp
    with ProcessPoolExecutor(max_workers=8,mp_context=mp.get_context('fork')) as pool:
        offset=0
        for chunk,draws in pool.map(evaluate_chunk,batches):
            all_draws[offset:offset+len(draws)]=draws;offset+=len(draws)
            for v in VARIANTS:chunks[v].append(chunk[v])
            print('Evaluate',name,offset,len(rows),'seconds',time.time()-start,flush=True)
    summaries={}
    for v in VARIANTS:
        p=concat(chunks[v]);p.update(rows=rows,draws=all_draws,centers=model.centers)
        np.savez_compressed(OUT/f'{name}_{v}_evaluation.npz',**p)
        summaries[v]={tag:score(t,rows[t['split'][rows]==sp],p['mu'][t['split'][rows]==sp]) for tag,sp in [('cal',1),('test',2)]}
    write_json(OUT/(name+'_scores.json'),dict(variants=summaries,seconds=time.time()-start,n=len(rows),model_sha256=sha(OUT/(name+'.joblib'))))

if __name__=='__main__':
    if sys.argv[1]=='train':train(sys.argv[2])
    else:evaluate(sys.argv[2])
