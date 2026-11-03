#!/usr/bin/env python3
"""Training-only cross-fit evidence, one fixed gate, and reference comparisons."""
from baseline_v1_1_common import *
import joblib,time,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from scipy.special import ndtr
from photdis.stage1 import require_compute
from photdis.baseline_v0 import HistogramQRF,features
from photdis.baseline_v1 import PopulationModel,support,qmc_noise,shift_pdf
from photdis.baseline_v1_1 import EvidenceGate,integrate
_T=None;_MODEL=None;_GATE=None

def likelihood_chunk(rows):
    m,e,ids=arrays(_T,rows);y=_T['Mg'][rows];sigma=_T['sigma_mu'][rows]
    wide=_MODEL.wide
    assert np.all(np.isfinite(sigma)&(sigma>0))
    z=(wide.edges[None,:]-y[:,None])/sigma[:,None]
    a,b=z[:,:-1],z[:,1:]
    kernel=np.where(a>0,ndtr(-a)-ndtr(-b),ndtr(b)-ndtr(a))/np.diff(wide.edges)
    likelihood=np.zeros((len(rows),4))
    experts=[_MODEL.branches.get(k) for k in range(3)]+[wide]
    bs=np.column_stack([support(b,m) if b is not None else np.zeros(len(rows),bool) for b in experts[:3]])
    for d in range(CFG['draws']):
        delta=qmc_noise(ids,d)*np.nan_to_num(e,nan=0,posinf=0)
        x=features(m+delta,wide.dim)
        for k,b in enumerate(experts):
            if b is not None:
                p=shift_pdf(b.distribution(x)[0],delta[:,0],wide.edges)
                likelihood[:,k]+=np.sum(p*kernel,axis=1)
    likelihood/=CFG['draws']
    effective=likelihood.copy()
    for k in range(3):effective[~bs[:,k],k]=likelihood[~bs[:,k],3]
    return dict(rows=rows,likelihood=likelihood,effective=effective,branch_support=bs)

def evaluation_chunk(rows):
    m,e,ids=arrays(_T,rows)
    return integrate(_MODEL,m,e,ids,VARIANTS,labels=_T['class_label'][rows],gate=_GATE,draws=CFG['draws'])

def parallel(fn,rows):
    batches=[rows[j:j+CFG['chunk_size']] for j in range(0,len(rows),CFG['chunk_size'])]
    with ProcessPoolExecutor(max_workers=8,mp_context=mp.get_context('fork')) as pool:
        return list(pool.map(fn,batches))

def train(name):
    global _T,_MODEL
    require_compute();OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/(name+'_gate.joblib')
    if path.exists():raise FileExistsError(path)
    _T=t=load(OLD/'references.npz');rows,weight,trainrows=evidence_rows(t,name)
    dim,tier=MODELS[name];parts=[];folds=[];start=time.time()
    for fold,fields in enumerate(CFG['inner_folds']):
        hold=rows[np.isin(t['field'][rows],fields)]
        fitrows=trainrows[~np.isin(t['field'][trainrows],fields)]
        assert len(hold) and not np.intersect1d(t['identity'][hold],t['identity'][fitrows]).size
        assert np.all(t['split'][fitrows]==0) and np.all(t['split'][hold]==0)
        m=arrays(t,fitrows)[0];y=t['Mg'][fitrows];labels=t['class_label'][fitrows]
        wide=HistogramQRF(dim,seed=20261102,n_estimators=128,min_samples_leaf=32,max_depth=20,bin_width=.1).fit(m,y,labels)
        _MODEL=PopulationModel(wide).fit(m,y,labels,CFG['seed'])
        result=concat(parallel(likelihood_chunk,hold));result['fold']=np.full(len(hold),fold,'i1');parts.append(result)
        folds.append(dict(fold=fold,held_fields=fields,n_fit=len(fitrows),n_evidence=len(hold),
            train_identity_sha256=identity_hash(t['identity'][fitrows]),held_identity_sha256=identity_hash(t['identity'][hold])))
        # Exact identities are inspectable; validation re-derives both partitions.
        np.savez_compressed(OUT/f'{name}_fold{fold}_identities.npz',train_rows=fitrows,held_rows=hold)
        print(name,'fold',fold,'evidence',len(hold),'seconds',time.time()-start,flush=True)
    p=concat(parts);order=np.argsort(p['rows']);p={k:v[order] for k,v in p.items()}
    assert np.array_equal(p['rows'],rows)
    p.update(identity=t['identity'][rows],sample_weight=weight)
    np.savez_compressed(OUT/(name+'_evidence.npz'),**p)
    # Normalize likelihoods per row for conditioning; this adds only a fixed
    # objective constant. It does not alter relative expert evidence.
    empty=int(np.sum(p['effective'].max(axis=1)<=1e-300))
    likelihood=np.maximum(p['effective'],1e-300);likelihood/=likelihood.max(axis=1)[:,None]
    gate=EvidenceGate(dim,CFG['gate_l2'],CFG['gate_class_loss_weight'],CFG['gate_maxiter'])
    gate.fit(arrays(t,rows)[0],likelihood,t['class_label'][rows],weight)
    joblib.dump(gate,path)
    write_json(OUT/(name+'_train.json'),dict(n_train=len(trainrows),n_evidence=len(rows),folds=folds,
        train_identity_sha256=identity_hash(t['identity'][trainrows]),evidence_identity_sha256=identity_hash(t['identity'][rows]),
        evidence_sha256=sha(OUT/(name+'_evidence.npz')),gate_sha256=sha(path),
        frozen_expert_sha256=sha(EXPERT/(name+'.joblib')),config_sha256=sha(ROOT/'configs/baseline_v1_1.json'),
        optimization=gate.optimization,empty_evidence_rows=empty,seconds=time.time()-start))
    print(name,'gate',gate.optimization,flush=True)

def evaluate(name):
    global _T,_MODEL,_GATE
    require_compute();_T=t=load(OLD/'references.npz');rows=reference_rows(t,name)
    _MODEL=joblib.load(EXPERT/(name+'.joblib'));_GATE=joblib.load(OUT/(name+'_gate.joblib'))
    start=time.time();chunks=parallel(evaluation_chunk,rows);scores={}
    for v in VARIANTS:
        p=concat([chunk[v] for chunk in chunks]);p.update(rows=rows,identity=t['identity'][rows],centers=_MODEL.centers,edges=_MODEL.edges)
        np.savez_compressed(OUT/f'{name}_{v}_evaluation.npz',**p)
        scores[v]={tag:score(t,rows[t['split'][rows]==sp],p['mu'][t['split'][rows]==sp]) for tag,sp in [('cal',1),('test',2)]}
    write_json(OUT/(name+'_scores.json'),dict(variants=scores,n=len(rows),identity_sha256=identity_hash(t['identity'][rows]),seconds=time.time()-start))
    print(name,'evaluated',len(rows),flush=True)

if __name__=='__main__':
    name=sys.argv[2]
    if sys.argv[1]=='train':train(name)
    elif sys.argv[1]=='evaluate':evaluate(name)
    else:train(name);evaluate(name)
