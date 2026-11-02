#!/usr/bin/env python3
"""Apply frozen training-only v0 models to all unique optical parents by field."""
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'.deps')]
import numpy as np
import fitsio,joblib
from photdis.stage1 import require_compute
from photdis.baseline_v0 import BANDS,ERR,distance_quantiles
from kgiant_distance_inventory import write_json,sha
CFG=json.load(open('configs/baseline_v0.json'));OUT=Path(CFG['output_dir'])

def main():
    require_compute();field=int(sys.argv[1]);start=time.time()
    with np.load(OUT/f'parent_{field:02}.npz') as f:t={k:f[k] for k in f.files}
    owner=np.load(OUT/f'owner_{field:02}.npy');t={k:v[owner] for k,v in t.items()};n=len(t['g0'])
    path=OUT/f'catalog_{field:02}.fits'
    if path.exists():raise FileExistsError(path)
    fit=json.load(open(OUT/'fit_result.json'));north=field in CFG['north_fields']
    models={} if north else {k:joblib.load(OUT/(k+'.joblib')) for k in ('A','B','C')}
    dtype=[(k,t[k].dtype) for k in ['identity','ls_id','source_id','ra','dec','ebv','r','g0','r0','z0','w10','w20','sigma_g','sigma_r','sigma_z','sigma_w1','sigma_w2','B','C','w1_clean','w2_clean','snr_w1','snr_w2','release','optical_error','nonstellar_risk']]
    dtype += [('field','i2'),('route','i1'),('supported','?'),('ambiguous','?'),('internal_uncalibrated','?'),('outside_train_precision','?'),('mu','f4',(5,)),('distance_kpc','f4',(5,)),('mu_cal','f4',(5,))]
    for k in ('A','B','C'):dtype += [(k+'_mu','f4',(5,)),(k+'_modes_mu','f4',(4,)),(k+'_class_weights','f4',(4,)),(k+'_supported','?'),(k+'_ambiguous','?')]
    totals=dict(n_unique=n,n_input_records=len(owner),n_no_native_model=n if north else 0,n_recommended_supported=0,n_ambiguous=0,routes={str(k):0 for k in (-1,0,1,2)})
    # App histograms are actual unique optical objects, not spectroscopy selection.
    hist={}
    for key,value,edges in [('r',t['r'],[12,16,18,20,22,25]),('gr',t['g0']-t['r0'],[-5,.4,.8,1.2,1.6,5]),('error',t['optical_error'],[0,.03,.06,.12,.25])]:
        hist[key]=dict(edges=edges,counts={k:np.histogram(value[m],edges)[0].tolist() for k,m in [('A',np.ones(n,bool)),('B',t['B']),('C',t['C'])]})
    totals['error_cumulative']={str(cut):int((t['optical_error']<cut).sum()) for cut in [.03,.06,.12,.25]}
    totals['availability']={'A':n,'B':int(t['B'].sum()),'C':int(t['C'].sum())};totals['histograms']=hist
    if n == 0:
        fitsio.write(str(path),np.zeros(0,dtype=dtype),extname='BASELINE_V0')
    for j in range(0,n,CFG['chunk_size']):
        ix=slice(j,min(j+CFG['chunk_size'],n));size=len(t['g0'][ix]);out=np.zeros(size,dtype=dtype)
        for k in out.dtype.names:
            if out[k].dtype.kind=='f':out[k]=np.nan
        for k in t:
            if k in out.dtype.names:out[k]=t[k][ix]
        out['field']=field;out['route']=-1;out['internal_uncalibrated']=True;out['outside_train_precision']=out['optical_error']>=.03
        mags=np.column_stack([t[k][ix] for k in BANDS]);errors=np.column_stack([t[k][ix] for k in ERR])
        for code,k in enumerate(('A','B','C')):
            if north:continue
            use=np.ones(size,bool) if k=='A' else t[k][ix]
            if not use.any():continue
            p=models[k].predict(mags[use],errors[use],t['identity'][ix][use],CFG['photometry_draws']);mu,d=distance_quantiles(t['g0'][ix][use],p['q_Meffective'])
            out[k+'_mu'][use]=mu;out[k+'_supported'][use]=p['supported'];out[k+'_ambiguous'][use]=p['ambiguous'];out[k+'_class_weights'][use]=p['class_weights']
            modes=p['modes_Meffective'].copy();modes[:,:2]=t['g0'][ix][use,None]-modes[:,:2];out[k+'_modes_mu'][use]=modes
            upgrade=p['supported']&(fit['route'][k]['adopt']|~out['supported'][use]) if k!='A' else np.ones(use.sum(),bool)
            dest=np.flatnonzero(use)[upgrade];out['route'][dest]=code;out['mu'][dest]=mu[upgrade];out['distance_kpc'][dest]=d[upgrade];out['supported'][dest]=p['supported'][upgrade];out['ambiguous'][dest]=p['ambiguous'][upgrade]
            calibrated=mu[upgrade].copy();c=fit['models'][k]['cqr_expand68_95'];calibrated[:,1]-=c[0];calibrated[:,3]+=c[0];calibrated[:,0]-=c[1];calibrated[:,4]+=c[1];calibrated[:,0]=np.minimum(calibrated[:,0],calibrated[:,1]);calibrated[:,4]=np.maximum(calibrated[:,4],calibrated[:,3]);out['mu_cal'][dest]=calibrated
        with fitsio.FITS(str(path),'rw') as f:
            if len(f)==1:f.write(out,extname='BASELINE_V0')
            else:f[1].append(out)
        totals['n_recommended_supported']+=int(out['supported'].sum());totals['n_ambiguous']+=int(out['ambiguous'].sum())
        for code in totals['routes']:totals['routes'][code]+=int((out['route']==int(code)).sum())
        if j%100000==0:print('field',field,'rows',j+size,'of',n,'seconds',time.time()-start,flush=True)
    assert sum(totals['routes'].values())==n
    totals.update(field=field,elapsed_seconds=time.time()-start,catalog=str(path),catalog_size=path.stat().st_size,catalog_sha256=sha(path),model_sha256={k:fit['models'][k]['model_sha256'] for k in models})
    write_json(OUT/f'catalog_{field:02}.json',totals)
if __name__=='__main__':main()
