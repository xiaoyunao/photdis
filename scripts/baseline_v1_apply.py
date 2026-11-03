#!/usr/bin/env python3
"""Disjoint bounded FITS shards, then deterministic whole-field assembly."""
from baseline_v1_common import *
import fitsio,os,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from baseline_v1_predict import load_selected,predict_all
from photdis.stage1 import require_compute

_LOCK=None;_MODELS=None
BASE=['identity','ls_id','source_id','ra','dec','ebv','r',*BANDS,*ERR,'B','C','w1_clean','w2_clean','snr_w1','snr_w2','release','optical_error','nonstellar_risk','field']

def worker(task):
    global _LOCK,_MODELS
    require_compute();field,start,stop=task;path=OUT/'shards'/f'field{field:02}_{start:08}.fits'
    if path.exists():raise FileExistsError(path)
    if _LOCK is None:_LOCK,_MODELS=load_selected()
    with fitsio.FITS(str(OLD/f'catalog_{field:02}.fits')) as f:
        raw=f[1].read(rows=np.arange(start,stop),columns=BASE)
    dtype=raw.dtype.descr+[('route','i1'),('supported','?'),('ambiguous','?'),('internal_uncalibrated','?'),
        ('outside_train_precision','?'),('research_quality','?'),('numerical_converged','?'),('draws','i2'),('version','i2'),
        ('mu','f4',(5,)),('mu_cal','f4',(5,)),('distance_kpc','f4',(5,)),('branch_weights','f4',(4,)),('disagreement','f4')]
    for k in 'ABC':dtype += [(k+'_mu','f4',(5,)),(k+'_modes_mu','f4',(4,)),(k+'_weights','f4',(4,)),(k+'_supported','?'),(k+'_ambiguous','?')]
    tick=time.time()
    for j in range(0,len(raw),CFG['chunk_size']):
        a=raw[j:j+CFG['chunk_size']];t={k:a[k] for k in BASE};out=np.zeros(len(a),dtype=dtype)
        for k in out.dtype.names:
            if out[k].dtype.kind=='f':out[k]=np.nan
        for k in BASE:out[k]=a[k]
        out['route']=-1;out['version']=1;out['internal_uncalibrated']=True;out['outside_train_precision']=a['optical_error']>=.03
        if field!=13:
            p,models=predict_all(t,_LOCK,_MODELS)
            for k in ('route','supported','ambiguous','mu','mu_cal','draws','numerical_converged','research_quality','disagreement'):out[k]=p[k]
            out['distance_kpc']=p['distance'];out['branch_weights']=p['weights']
            for k,b in models.items():
                out[k+'_mu']=b['mu'];out[k+'_weights']=b['weights'];out[k+'_supported']=b['supported'];out[k+'_ambiguous']=b['ambiguous']
                modes=b['modes'].copy();modes[:,:2]=a['g0'][:,None]-modes[:,:2];out[k+'_modes_mu']=modes
        with fitsio.FITS(str(path),'rw') as f:
            if len(f)==1:f.write(out,extname='BASELINE_V1')
            else:f[1].append(out)
    return dict(field=field,start=start,stop=stop,n=stop-start,path=str(path),sha256=sha(path),seconds=time.time()-tick)

def main():
    global _LOCK,_MODELS
    require_compute();(OUT/'shards').mkdir(exist_ok=True)
    tasks=[]
    for field in range(14):
        with fitsio.FITS(str(OLD/f'catalog_{field:02}.fits')) as f:n=f[1].get_nrows()
        tasks.extend((field,a,min(a+100000,n)) for a in range(0,n,100000))
    if (OUT/'application.json').exists():raise FileExistsError('v1 application exists')
    # One run, many independent shards: no repeated full-parent scoring or shared writers.
    records=[]
    # Fork shares large read-only forest tables rather than reading them per worker.
    _LOCK,_MODELS=load_selected()
    import multiprocessing as mp
    with ProcessPoolExecutor(max_workers=int(os.environ.get('BASELINE_WORKERS','32')),
                             mp_context=mp.get_context('fork')) as pool:
        futures=[pool.submit(worker,task) for task in tasks]
        for future in as_completed(futures):
            records.append(future.result());write_json(OUT/'application_progress.json',records)
            print('Shard',records[-1],flush=True)
    fields=[]
    for field in range(14):
        path=OUT/f'catalog_{field:02}.fits'
        if path.exists():raise FileExistsError(path)
        records0=sorted((r for r in records if r['field']==field),key=lambda r:r['start'])
        counts=dict(n=0,supported=0,ambiguous=0,research_quality=0,numerical_converged=0,routes={str(k):0 for k in (-1,0,1,2)})
        for rec in records0:
            a=fitsio.read(rec['path']);counts['n']+=len(a)
            for k in ('supported','ambiguous','research_quality','numerical_converged'):counts[k]+=int(a[k].sum())
            for k in counts['routes']:counts['routes'][k]+=int((a['route']==int(k)).sum())
            with fitsio.FITS(str(path),'rw') as f:
                if len(f)==1:f.write(a,extname='BASELINE_V1')
                else:f[1].append(a)
        fields.append(dict(field=field,**counts,sha256=sha(path),size=path.stat().st_size))
        print('Assembled',fields[-1],flush=True)
    write_json(OUT/'application.json',dict(fields=fields,n=sum(f['n'] for f in fields),
        selection_sha256=sha(OUT/'selection.json'),shards=records))
if __name__=='__main__':main()
