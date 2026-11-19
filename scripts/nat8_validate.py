#!/usr/bin/env python3
"""Actual-source direct-tail, half-grid, invariance and legacy preservation QA."""
from nat8_common import *
from nat8_statsmath import hist_curves,curve_values,sparse_CDF_components,direct_queries,quantiles_from_curve
from nat8_math import fast_quantiles
from seq7_math import log_density_grid
from photdis.stage1 import require_compute
from scipy.special import ndtr,logsumexp
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from concurrent.futures import ProcessPoolExecutor


def one(task):
    s,k=task;t=load(OUT/'prepare/roster.npz');z=load(OUT/'prepare'/f'{s}_E.npz');p=load(OUT/'predict'/f'{s}_calibrated.npz');d=load(OUT/'fit/parameter_draws.npz');par=json.loads((OUT/'fit/parameters.json').read_text());i=z['ii'][k];label=str(t['fold'][i]) if s=='M5' else 'full';a=np.array(par[label]['point']['a']);aa=d[label+'_a'];mm=z['means'][k];lw=z['logw'][k];xx=z['x'];sigma=z['sigma'][k];ref=10+5*np.log10(t['ref'][i]);rs=0 if s=='M5' else float(t['reference_sigma_'+s]);queries=np.linspace(ref-1.5,ref+1.5,31);rows=[]
    for state,params in [('point',a[None]),('draw0',aa[:1]),('draw127',aa[127:128]),('draw255',aa[255:256])]+([('marginal',aa)] if s!='M5' else []):
        means=(mm[None]-params[:,0,None]-params[:,1,None]*xx).ravel();weights=np.tile(lw-np.log(len(params)),len(params));effective_sigma=np.hypot(sigma,rs);exact_lp=logsumexp(weights[None]-.5*((queries[:,None]-means)/effective_sigma)**2,axis=1)-np.log(effective_sigma*np.sqrt(2*np.pi));exact_cf=(np.exp(weights)[None]*ndtr((queries[:,None]-means)/effective_sigma)).sum(axis=1)
        gx=np.arange(queries.min()-.02,queries.max()+.021,.00125);lp=log_density_grid(gx,means,weights,effective_sigma);log_error=float(abs(CubicSpline(gx,lp)(queries)-exact_lp).max());active,w=sparse_CDF_components(means,weights);cx,cf,cr,_=hist_curves(active,w,effective_sigma,.0025,query=queries);fx,ff,fr,_=hist_curves(active,w,effective_sigma,.00125,query=queries);cdf_error=float(abs(curve_values(cx,cf,queries)[0]-exact_cf).max());crps_error=float(abs(curve_values(cx,cr,queries)-curve_values(fx,fr,queries)).max());max_quantile_error=0.
        if state=='point':
            mapped=mm-a[0]-a[1]*xx;actual=fast_quantiles(mapped,lw,sigma);roots=np.array([brentq(lambda y:(np.exp(lw)*ndtr((y-mapped)/sigma)).sum()-q,mapped.min()-11*sigma,mapped.max()+11*sigma,xtol=1e-11) for q in [.025,.16,.5,.84,.975]]);max_quantile_error=float(abs(actual-roots).max());np.testing.assert_allclose(actual,p['q'][k],atol=1e-12)
        if state=='marginal':
            raw=load(OUT/'marginal_raw'/f'{i:05d}.npz');active,w=sparse_CDF_components(means,weights);qx,qc,_,_=hist_curves(active,w,sigma,.000625);fine_q=quantiles_from_curve(qx,qc)[0];max_quantile_error=float(abs(fine_q-p['marginal_q'][k]).max())
        assert log_error<.001 and cdf_error<.0002 and crps_error<.0001 and max_quantile_error<.0025,(s,k,state,log_error,cdf_error,crps_error,max_quantile_error)
        rows.append(dict(TYPE=t['TYPE'][i],ls_id=int(t['ls_id'][i]),system=s,state=state,logscore_error=log_error,CDF_error=cdf_error,CRPS_error=crps_error,quantile_error=max_quantile_error))
    return rows


def main():
    require_compute();d=folder('validation');tasks=[];counts=[];t=load(OUT/'prepare/roster.npz')
    for s in C['systems']:
        z=load(OUT/'prepare'/f'{s}_E.npz');p=load(OUT/'predict'/f'{s}_calibrated.npz');assert (p['TYPE']=='PSF').all();np.testing.assert_array_equal(z['support'],p['support']);np.testing.assert_array_equal(z['post'],p['post']);np.testing.assert_array_equal(z['ii'],p['ii']);assert len(p['ii'])==C['parent_N'][s];use=np.flatnonzero(z['support']);tasks.extend((s,int(k)) for k in use[np.linspace(0,len(use)-1,3).astype(int)]);counts.append(dict(system=s,parent=len(p['ii']),E_supported=int(z['support'].sum()),support_changes=0,branch_probability_changes=0))
    with ProcessPoolExecutor(max_workers=8) as pool:allrows=list(pool.map(one,tasks))
    csv_rows(d/'direct_halfgrid.csv',[r for rows in allrows for r in rows]);csv_rows(d/'invariants.csv',counts)
    receipts={}
    for rel,expected in C['source_receipts'].items():
        p=verified(ROOT/rel,expected);record=json.loads(p.read_text());stage=OLD/p.parent.name
        for name,h in record['outputs'].items():verified(stage/name,h)
        receipts[rel]=expected
    # The public v7 support table has a documented TYPE/native export transform;
    # source receipts above use the untouched private original, when applicable.
    sky=ROOT/'figures/sky_sequence_20261119/selected_channels/rgb_selected_fixed_gain.png';assert sha(sky)=='411b0086e9d4c6f9f208805f02f064e2c24bb117d61293083b9bb51dc7faaed5'
    from v3_10_figures import regression,STYLE
    target=d/'style_regression';target.mkdir(exist_ok=True)
    with plt.rc_context(STYLE):result=regression(target)
    save_json(d/'style_regression.json',dict(original_v38_pixel_identical=result,style_lock=check_templates()));save_json(d/'old_preservation.json',dict(receipts=receipts,sky=sha(sky),old_science_sources_unchanged=True));complete(d,{'prepare':sha(OUT/'prepare/complete.json'),'fit':sha(OUT/'fit/complete.json'),'predict':sha(OUT/'predict/complete.json')},actual_source_direct_and_half_grid=True,no_scientific_retuning=True);print('Numerical, invariant and style checks passed',flush=True)


if __name__=='__main__':main()
