#!/usr/bin/env python3
"""Point and shared-refit conditional quantiles, then external parameter mixtures."""
from nat8_common import *
from nat8_math import fast_quantiles
from nat8_statsmath import hist_curves,quantiles_from_curve,sparse_CDF_components
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from photdis.stage1 import require_compute


@lru_cache(maxsize=6)
def cached(p):return load(p)


def worker(task):
    s,k,points=task;t=cached(OUT/'prepare/roster.npz');z=cached(OUT/'prepare'/f'{s}_E.npz');draws=cached(OUT/'fit/parameter_draws.npz');i=z['ii'][k];label=str(t['fold'][i]) if s=='M5' else 'full';a=np.array(points[label]);aa=draws[label+'_a'];mm=z['means'][k];lw=z['logw'][k];sigma=z['sigma'][k];x=z['x'];means=mm-a[0]-a[1]*x;bootmeans=mm[None]-aa[:,0,None]-aa[:,1,None]*x[None];q=fast_quantiles(means,lw,sigma);qb=fast_quantiles(bootmeans,lw,sigma)
    design=json.loads((OUT/'prepare/fit_design.json').read_text());bound=next(r for r in design if str(r['prediction_fold'])==(str(t['fold'][i]) if s=='M5' else 'external'));nodec=x+C['pivot'];outside=(nodec<bound['gr_min'])|(nodec>bound['gr_max']);pout=float(np.exp(lw[outside]).sum());domain=0 if pout<=.05 else (1 if pout<=.5 else 2);result=dict(i=int(i),q=q,qboot=qb,pout=pout,domain=domain)
    if s!='M5':
        # One shared calibration-parameter index for every target member.
        mix,w=sparse_CDF_components(bootmeans,lw);weights=np.tile(w/len(aa),len(aa));flat=mix.ravel();h=.00125;gx,cdf,crps,pdf=hist_curves(flat,weights,sigma,h);mq=quantiles_from_curve(gx,cdf)[0];result.update(marginal_q=mq,marginal_CDF_error_bound='Gaussian mean-grid deposition, checked independently')
        target=folder('marginal_raw')/f'{int(i):05d}.npz';npz(target,mu=gx,cdf=cdf[0],density=pdf[0],crps=crps[0])
    return result


def main():
    require_compute();d=folder('predict');t=load(OUT/'prepare/roster.npz');parameters=json.loads((OUT/'fit/parameters.json').read_text());points={label:parameters[label]['point']['a'] for label in parameters};inputs={'fit':sha(OUT/'fit/complete.json'),'prepare':sha(OUT/'prepare/complete.json'),'numerics':sha(ROOT/'configs/native_luminosity_numerics_20261119.json')};statuses=[]
    for s in C['systems']:
        z=load(OUT/'prepare'/f'{s}_E.npz');tasks=[(s,k,points) for k in range(len(z['ii']))]
        with ProcessPoolExecutor(max_workers=C['workers']) as pool:r=list(pool.map(worker,tasks,chunksize=2))
        np.testing.assert_array_equal([a['i'] for a in r],z['ii']);data=dict(ii=z['ii'],TYPE=t['TYPE'][z['ii']],ls_id=t['ls_id'][z['ii']],support=z['support'],post=z['post'],q=np.array([a['q'] for a in r]),qboot=np.array([a['qboot'] for a in r]),pout=np.array([a['pout'] for a in r]),domain=np.array([a['domain'] for a in r]))
        if s!='M5':data['marginal_q']=np.array([a['marginal_q'] for a in r])
        npz(d/f'{s}_calibrated.npz',**data)
        for k,i in enumerate(z['ii']):statuses.append(dict(TYPE=t['TYPE'][i],ls_id=int(t['ls_id'][i]),system=s,native=int(t['native'][i]),original_E_supported=bool(z['support'][k]),calibrated_supported=bool(data['support'][k]),post_mass=float(data['post'][k]),outside_color_mass=float(data['pout'][k]),domain=('overlap','transition','extrapolation')[data['domain'][k]],prediction_fit=str(t['fold'][i]) if s=='M5' else 'fullM5',extrapolation_does_not_change_support=True))
        print('Predictions',s,len(r),flush=True)
    csv_rows(d/'support_domain.csv',statuses);complete(d,inputs,color_likelihood_weights_support_post_mass_unchanged=True,conditional_M5='out-of-fold parameters only; no M5 parameter-mixture forecast',marginal_external='empirical bootstrap parameter mixture, all target members share one draw index; target-reference kernel not yet added')


if __name__=='__main__':main()
