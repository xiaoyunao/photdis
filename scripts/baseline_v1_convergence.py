#!/usr/bin/env python3
from baseline_v1_common import *
import joblib
from scipy.special import ndtr
from photdis.stage1 import require_compute
from photdis.baseline_v0 import HistogramQRF,quantiles,distance_quantiles
from photdis.baseline_v1 import integrate

def compare(p,ref,mask):
    d=abs(p['mu'][mask][:,[0,2,4]]-ref['mu'][mask][:,[0,2,4]])
    return dict(n=int(mask.sum()),p50_abs_mag=np.median(d,axis=0).tolist(),
        p95_abs_mag=np.quantile(d,.95,axis=0).tolist(),
        ambiguity_flip=float(np.mean(p['ambiguous'][mask]!=ref['ambiguous'][mask])))

def predict(h,t,rows,draws,method='sobol',replicate=0):
    chunks=[]
    for j in range(0,len(rows),200):
        m,e,ids=arrays(t,rows[j:j+200])
        p=integrate(h,m,e,ids,draws,method,replicate)['wide']
        # Retain the old bin-center estimator for the IID convergence comparison.
        p['old_mu']=distance_quantiles(m[:,0],quantiles(p['pdf'],h.centers))[0]
        p.pop('pdf');chunks.append(p)
    return concat(chunks)

def run(name):
    require_compute();t=load(OUT/'sample.npz');rows=np.flatnonzero(available(t,name));h=joblib.load(OLD/(name+'.joblib'))
    ref=predict(h,t,rows,512,replicate=2);saved={'rows':rows,'reference_mu':ref['mu']};results={}
    layers={'low':t['optical_error'][rows]<.03,'high':t['optical_error'][rows]>=.03}
    for method,draws,rep in [('iid',8,0),('iid',32,0),('iid',128,0),('sobol',32,0),('sobol',128,0),('sobol',128,1)]:
        tag=f'{method}_{draws}_{rep}';p=predict(h,t,rows,draws,method,rep)
        results[tag]={k:compare(p,ref,m) for k,m in layers.items() if m.any()}
        if method=='iid':
            results[tag]['old_centers']={k:compare(dict(mu=p['old_mu'],ambiguous=p['ambiguous']),ref,m) for k,m in layers.items() if m.any()}
        saved[tag+'_mu']=p['mu'];saved[tag+'_ambiguous']=p['ambiguous']
        print(name,tag,results[tag],flush=True)
    write_json(OUT/f'convergence_{name}.json',dict(n=len(rows),comparisons=results,model_sha256=sha(OLD/(name+'.joblib'))))
    np.savez(OUT/f'convergence_{name}.npz',**saved)

def analytic():
    m=np.tile([2.,1.,0.,0.,0.],(40,1));h=HistogramQRF(2,n_estimators=2,min_samples_leaf=8).fit(m,np.zeros(40),np.zeros(40,int))
    ids=np.arange(5000);m=np.tile(m[0],(len(ids),1));e=np.zeros_like(m);e[:,0]=.2
    idx=np.argmax(h.distribution(np.array([[1.,1.]]))[0][0]);lo,hi=h.edges[idx:idx+2]
    def primitive(x):return x*ndtr(x/.2)+.2*np.exp(-.5*(x/.2)**2)/np.sqrt(2*np.pi)
    def cdf(x):return (primitive(x-lo)-primitive(x-hi))/(hi-lo)
    result={}
    for method,n in [('iid',8),('iid',32),('iid',128),('sobol',32),('sobol',128)]:
        p=integrate(h,m,e,ids,n,method)['wide'];q=2-p['mu'][:,::-1]
        old=quantiles(p['pdf'],h.centers)
        result[f'{method}_{n}']=dict(mean_probability95=float(np.mean(cdf(q[:,4])-cdf(q[:,0]))),old_center_probability95=float(np.mean(cdf(old[:,4])-cdf(old[:,0]))))
    return result

def finalize():
    require_compute();allr={n:json.load(open(OUT/f'convergence_{n}.json')) for n in 'ABC'}
    selection={};tolerance=np.array(CFG['quantile_p95_tolerance_mag'])
    for layer in ('low','high'):
        checks={}
        for n in (32,128):
            vals=[r['comparisons'][f'sobol_{n}_0'][layer] for r in allr.values() if layer in r['comparisons'][f'sobol_{n}_0']]
            checks[str(n)]=bool(all(np.all(np.asarray(v['p95_abs_mag'])<=tolerance) and v['ambiguity_flip']<=CFG['branch_flip_tolerance'] for v in vals))
        selection[layer]=dict(draws=32 if checks['32'] else 128,converged=checks['32'] or checks['128'],checks=checks)
    write_json(OUT/'numerical.json',dict(selection=selection,analytic=analytic(),convergence=allr,
        caveat='Numerical counterexample probability is not real-catalog coverage; unresolved layers retain flag.'))
    print('Numerical selection',selection,flush=True)
if __name__=='__main__':
    if sys.argv[1]=='finalize':finalize()
    else:run(sys.argv[1])
