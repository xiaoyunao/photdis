#!/usr/bin/env python3
"""Held-field Gaia checks with observed-SNR-conditioned CDF and simulations."""
from baseline_v1_1_common import *
from photdis.stage1 import require_compute
from photdis.baseline_parallax import parallax_check,simulate_selected


def summarize(t,rows,pit,center,sim,mask):
    use=mask&np.isfinite(pit)&np.isfinite(center)&np.isfinite(sim).all(axis=1)
    n=int(use.sum())
    if not n:return dict(n=0)
    w=t['sample_weight'][rows][use];w=w/w.sum();y=t['p_corr'][rows][use]
    observed=float(w@(y-center[use]));draw=w@(sim[use]-center[use,None])
    return dict(n=n,n_negative=int((y<0).sum()),n_effective=float(1/np.sum(w*w)),
        mean_pit=float(w@pit[use]),low05=float(w@(pit[use]<.05)),high95=float(w@(pit[use]>.95)),
        mean_center_residual_mas=observed,conditional_simulation_mean_residual_ci95=np.quantile(draw,[.025,.975]).tolist(),
        conditional_simulation_p=float((1+np.sum(abs(draw)>=abs(observed)))/(len(draw)+1)),
        pit_histogram=np.histogram(pit[use],bins=np.linspace(0,1,11),weights=w)[0].tolist())


def main():
    require_compute();t=load(OUT/'parallax_sample.npz');result={}
    for name in 'ABC':
        result[name]={}
        for variant in ('wide','evidence'):
            p=load(OUT/f'parallax_{name}_{variant}.npz');rows=p['rows'];parts=[];draws=[]
            rng=np.random.default_rng(CFG['seed']+77)
            for start in range(0,len(rows),200):
                j=np.arange(start,min(start+200,len(rows)));r=rows[j]
                args=(p['pdf'][j],p['edges'],t['g0'][r],t['p_corr'][r],t['gaia_parallax_error'][r],t['snr_lo'][r],t['snr_hi'][r])
                a=parallax_check(*args);parts.append(a)
                draws.append(simulate_selected(args[0],args[1],args[2],args[4],args[5],args[6],rng))
            a=concat(parts);sim=np.concatenate(draws)
            groups={'all':np.ones(len(rows),bool),'no_W1':~t['B'][rows],'W1_only':t['B'][rows]&~t['C'][rows],'W1W2':t['C'][rows]}
            for k in range(5):groups[f'snr{k}']=t['snr_bin'][rows]==k
            for lo,hi in [(12,16),(16,18),(18,20),(20,22),(22,25)]:groups[f'r{lo}_{hi}']=(t['r'][rows]>=lo)&(t['r'][rows]<hi)
            color=t['g0'][rows]-t['r0'][rows]
            for lo,hi in [(-np.inf,.5),(.5,1.),(1.,1.5),(1.5,np.inf)]:groups[f'gr{lo}_{hi}']=(color>=lo)&(color<hi)
            groups['nonstellar_risk']=t['nonstellar_risk'][rows]
            data={k:summarize(t,rows,a['pit'],a['conditional_mean'],sim,mask) for k,mask in groups.items()}
            # Fixed numerical check: independent higher-order quadrature, not new fitting.
            ix=np.arange(min(256,len(rows)));r=rows[ix]
            check=parallax_check(p['pdf'][ix],p['edges'],t['g0'][r],t['p_corr'][r],t['gaia_parallax_error'][r],t['snr_lo'][r],t['snr_hi'][r],order=16)
            data['quadrature_pit_max_difference']=float(np.nanmax(abs(check['pit']-a['pit'][ix])))
            data['selection_underflow_rows']=int(np.sum(~np.isfinite(a['pit'])))
            data['prediction_sha256']=sha(OUT/f'parallax_{name}_{variant}.npz')
            result[name][variant]=data
            np.savez_compressed(OUT/f'parallax_{name}_{variant}_checks.npz',rows=rows,identity=t['identity'][rows],**a)
            print('Parallax',name,variant,'n',len(rows),flush=True)
    write_json(OUT/'parallax.json',dict(results=result,sample_sha256=sha(OUT/'parallax_sample.npz'),
        scope='quality- and zero-point-valid Gaia test matches, SNR/WISE/field probability sample; no positive-parallax cut; each SNR interval conditioned explicitly',
        limitation='conditional per-star empirical PDFs; simulations do not model shared astrophysical or Gaia systematics, and are not an independent population calibration',simulation_repeats=256))
if __name__=='__main__':main()
