#!/usr/bin/env python3
"""Two fixed classifiers and shared frozen-expert cache; allocated compute only."""
from baseline_v1_1_common import *
import joblib,time,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from photdis.stage1 import require_compute
from photdis.baseline_v0 import features
from photdis.baseline_v1 import support,qmc_noise,shift_pdf
from photdis.baseline_v1_1 import transfer_unsupported
from photdis.baseline_v1_2 import (DirectClassifier,balanced_weights,distance_summary,
                                  mix_cached,main_component)
PREV=OUT;CFG=json.load(open(ROOT/'configs/baseline_v1_2.json'));OUT=ROOT/CFG['output_dir']
VARS=('natural','balanced','evidence','quadratic','forest')
_T=None;_MODEL=None


def cache_chunk(rows):
    m,e,ids=arrays(_T,rows);model=_MODEL;wide=model.wide;n=len(rows)
    bs=np.column_stack([support(model.branches[k],m) if k in model.branches else np.zeros(n,bool) for k in range(3)])
    branch=np.zeros((n,4,len(wide.centers)));raw=np.zeros((n,2,4));weights=np.zeros((n,2,4));mix=np.zeros((n,2,len(wide.centers)))
    e=np.nan_to_num(e,nan=0,posinf=0)
    for d in range(CFG['draws']):
        delta=qmc_noise(ids,d)*e;x=features(m+delta,model.dim)
        broad,cl=wide.distribution(x);experts=[model.branches[k].distribution(x)[0] if k in model.branches else broad for k in range(3)]+[broad]
        for k,p in enumerate(experts):branch[:,k]+=shift_pdf(p,delta[:,0],wide.edges)
        gp=model.gate.predict_proba(x)
        for vi,v in enumerate(('natural','balanced')):
            g=gp.copy()
            if v=='balanced':g/=np.maximum(model.frequencies,1e-10)
            g/=g.sum(axis=1)[:,None];raw[:,vi,:3]+=g
            w,_=transfer_unsupported(np.column_stack([g*(1-cl[:,3,None]),cl[:,3]]),bs)
            p=sum(w[:,k,None]*experts[k] for k in range(4))
            mix[:,vi]+=shift_pdf(p,delta[:,0],wide.edges);weights[:,vi]+=w
    branch/=branch.sum(axis=2)[:,:,None];mix/=mix.sum(axis=2)[:,:,None]
    return dict(branch_pdf=branch.astype('f4'),raw_old=raw/CFG['draws'],weights_old=weights/CFG['draws'],
                mix_old=mix.astype('f4'),branch_support=bs,supported=support(wide,m))


def main(name):
    global _T,_MODEL
    require_compute();OUT.mkdir(exist_ok=True,parents=True);start=time.time()
    t=load(OLD/'references.npz');dim,tier=MODELS[name]
    train=np.flatnonzero((t['split']==0)&t['fit_eligible']&(t['class_label']>=0)&available(t,tier))
    y=t['class_label'][train];w=balanced_weights(y);m=arrays(t,train)[0]
    records={};classifiers={}
    for kind in ('quadratic','forest'):
        path=OUT/f'{name}_{kind}.joblib'
        if path.exists():raise FileExistsError(path)
        clf=DirectClassifier(dim,kind,CFG).fit(m,y,w)
        joblib.dump(clf,path)
        if kind=='forest':clf.model.n_jobs=1
        classifiers[kind]=clf
        records[kind]=dict(n=len(train),counts=np.bincount(y,minlength=3).tolist(),
            class_weight_sums=[float(w[y==k].sum()) for k in range(3)],
            identity_sha256=identity_hash(t['identity'][train]),weight_sha256=hashlib.sha256(w.tobytes()).hexdigest(),
            model_sha256=sha(path),iterations=clf.model.n_iter_.tolist() if kind=='quadratic' else None)
    np.savez_compressed(OUT/f'{name}_train.npz',rows=train,identity=t['identity'][train],weight=w)
    _MODEL=joblib.load(EXPERT/f'{name}.joblib');evidence=joblib.load(PREV/f'{name}_gate.joblib')
    records['expert_sha256']=sha(EXPERT/f'{name}.joblib')
    for kind in (('reference','parent','parallax') if name in 'ABC' else ('reference',)):
        _T=t if kind=='reference' else load(PREV/f'{kind}_sample.npz')
        rows=reference_rows(_T,name) if kind=='reference' else np.flatnonzero(available(_T,tier))
        if kind=='reference':assert not np.intersect1d(t['identity'][train],t['identity'][rows]).size
        batches=[rows[j:j+CFG['chunk_size']] for j in range(0,len(rows),CFG['chunk_size'])]
        with ProcessPoolExecutor(max_workers=CFG['workers'],mp_context=mp.get_context('fork')) as pool:parts=list(pool.map(cache_chunk,batches))
        cache=concat(parts);del parts
        cache.update(rows=rows,identity=_T['identity'][rows],edges=_MODEL.edges,centers=_MODEL.centers)
        np.savez_compressed(OUT/f'{kind}_{name}_cache.npz',**cache)
        m=arrays(_T,rows)[0];g=m[:,0]
        for variant in VARS:
            if variant in ('natural','balanced'):
                vi=('natural','balanced').index(variant)
                raw=cache['raw_old'][:,vi];weights=cache['weights_old'][:,vi];pdf=cache['mix_old'][:,vi]
            else:
                raw=(evidence if variant=='evidence' else classifiers[variant]).predict_proba(m)
                pdf,weights,_=mix_cached(cache['branch_pdf'],raw,cache['branch_support'])
                if raw.shape[1]==3:raw=np.column_stack([raw,np.zeros(len(raw))])
            main,index=main_component(cache['branch_pdf'],weights)
            p=dict(rows=rows,identity=_T['identity'][rows],edges=_MODEL.edges,pdf=pdf.astype('f4'),
                mu=distance_summary(pdf,_MODEL.edges,g),main_mu=distance_summary(main,_MODEL.edges,g),
                main_index=index,raw_weights=raw,weights=weights,supported=cache['supported'],branch_support=cache['branch_support'])
            np.savez_compressed(OUT/f'{kind}_{name}_{variant}.npz',**p)
        records[kind]=dict(n=len(rows),identity_sha256=identity_hash(_T['identity'][rows]))
        print(name,kind,len(rows),'seconds',round(time.time()-start,1),flush=True)
    records['seconds']=time.time()-start;write_json(OUT/f'{name}_fit.json',records)
if __name__=='__main__':main(sys.argv[1])
