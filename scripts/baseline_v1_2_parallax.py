#!/usr/bin/env python3
"""Complete and SNR-conditional probability checks on the frozen Gaia sample."""
from baseline_v1_2_run import *
from photdis.baseline_parallax import parallax_check
from photdis.baseline_v1_2 import full_parallax_log_density
SNR=np.array([-np.inf,0,5,10,20,np.inf])


def summarize(t,rows,a,mask):
    use=mask&np.isfinite(a['full_pit'])
    if not use.any():return dict(n=0)
    w=t['sample_weight'][rows][use];w=w/w.sum()
    hist=lambda x:np.histogram(x[use],np.linspace(0,1,11),weights=w)[0].tolist()
    actual=t['snr_bin'][rows][use];pred=a['bin_probability'][use]
    return dict(n=int(use.sum()),n_negative=int((t['p_corr'][rows][use]<0).sum()),effective_n=float(1/(w@w)),n_conditional_valid=int(np.isfinite(a['conditional_pit'][use]).sum()),
        full_pit_mean=float(w@a['full_pit'][use]),full_low05=float(w@(a['full_pit'][use]<.05)),full_high95=float(w@(a['full_pit'][use]>.95)),
        full_pit_histogram=hist(a['full_pit']),conditional_pit_histogram=hist(a['conditional_pit']),
        mean_log_density=float(w@a['full_log_density'][use]),mean_conditional_log_density=float(w@a['conditional_log_density'][use]),
        mean_observed_bin_log_probability=float(w@a['bin_log_probability'][use]),
        mean_observed_bin_probability=float(w@a['observed_bin_probability'][use]),
        weighted_observed_snr_fraction=[float(w@(actual==k)) for k in range(5)],
        weighted_predicted_snr_fraction=np.sum(w[:,None]*pred,axis=0).tolist(),
        snr_brier=float(w@np.sum((pred-np.eye(5)[actual])**2,axis=1)),
        mean_unconditional_residual_mas=float(w@(t['p_corr'][rows][use]-a['full_mean'][use])),
        mean_conditional_residual_mas=float(w@(t['p_corr'][rows][use]-a['conditional_mean'][use])))


def evaluate(t,rows,pdf,edges):
    # Stored float32 bin masses are normalized before probability factorization.
    pdf=np.asarray(pdf,float);pdf=pdf/pdf.sum(axis=1)[:,None]
    parts=[]
    for start in range(0,len(rows),200):
        j=np.arange(start,min(start+200,len(rows)));r=rows[j];n=len(r)
        args=(pdf[j],edges,t['g0'][r],t['p_corr'][r],t['gaia_parallax_error'][r])
        full=parallax_check(*args,np.full(n,-np.inf),np.full(n,np.inf))
        cond=parallax_check(*args,t['snr_lo'][r],t['snr_hi'][r])
        bins=np.column_stack([parallax_check(*args,np.full(n,lo),np.full(n,hi))['selection_probability'] for lo,hi in zip(SNR[:-1],SNR[1:])])
        density=full['conditional_density'];prob=cond['selection_probability'];cd=cond['conditional_density']
        logdensity=full_parallax_log_density(*args)
        # Save underflows rather than allowing a floor to hide them.
        a=dict(full_pit=full['pit'],conditional_pit=cond['pit'],full_mean=full['conditional_mean'],
            conditional_mean=cond['conditional_mean'],full_density=density,conditional_density=cd,
            observed_bin_probability=prob,bin_probability=bins,
            full_log_density=logdensity,conditional_log_density=logdensity-np.log(np.maximum(prob,1e-300)),
            bin_log_probability=np.log(np.maximum(prob,1e-300)))
        parts.append(a)
    a=concat(parts);j=np.arange(min(256,len(rows)));r=rows[j]
    high=parallax_check(pdf[j],edges,t['g0'][r],t['p_corr'][r],t['gaia_parallax_error'][r],np.full(len(r),-np.inf),np.full(len(r),np.inf),order=16)
    checks=dict(bin_mass_max_error=float(np.max(abs(a['bin_probability'].sum(axis=1)-1))),
        observed_bin_mass_max_error=float(np.max(abs(a['bin_probability'][np.arange(len(rows)),t['snr_bin'][rows]]-a['observed_bin_probability']))),
        density_factorization_max_relative_error=float(np.nanmax(abs(a['full_density']-a['conditional_density']*a['observed_bin_probability'])/np.maximum(a['full_density'],1e-280))),
        full_density_underflow=int((a['full_density']<=1e-300).sum()),bin_probability_underflow=int((a['observed_bin_probability']<=1e-300).sum()),
        quadrature_full_pit_max_difference=float(np.max(abs(high['pit']-a['full_pit'][j]))))
    return a,checks


def main():
    require_compute();t=load(PREV/'parallax_sample.npz');result={}
    for name in 'ABC':
        cache=load(OUT/f'parallax_{name}_cache.npz');rows=cache['rows'];result[name]={}
        for variant in ('wide','evidence','balanced','quadratic','forest'):
            old=variant in ('wide','evidence')
            p=load(PREV/f'parallax_{name}_{variant}.npz') if old else load(OUT/f'parallax_{name}_{variant}.npz')
            assert np.array_equal(rows,p['rows'])
            for decision in (('mixture',) if variant=='wide' else ('mixture','main')):
                if decision=='mixture':pdf=p['pdf']
                else:
                    # v1.1 evidence did not save main_index; use the same final weights.
                    pdf=cache['branch_pdf'][np.arange(len(rows)),np.argmax(p['weights'],axis=1)]
                a,checks=evaluate(t,rows,pdf,p['edges'])
                groups={'all':np.ones(len(rows),bool),'no_W1':~t['B'][rows],'W1_only':t['B'][rows]&~t['C'][rows],'W1W2':t['C'][rows]}
                for k in range(5):groups[f'snr{k}']=t['snr_bin'][rows]==k
                for lo,hi in [(12,16),(16,18),(18,20),(20,22),(22,25)]:groups[f'r{lo}_{hi}']=(t['r'][rows]>=lo)&(t['r'][rows]<hi)
                color=t['g0'][rows]-t['r0'][rows]
                for lo,hi in [(-np.inf,.5),(.5,1.),(1.,1.5),(1.5,np.inf)]:groups[f'gr{lo}_{hi}']=(color>=lo)&(color<hi)
                key=variant+'_'+decision
                result[name][key]=dict(groups={k:summarize(t,rows,a,mask) for k,mask in groups.items()},checks=checks)
                np.savez_compressed(OUT/f'parallax_{name}_{key}_checks.npz',rows=rows,identity=t['identity'][rows],**a)
                print('Full parallax',name,key,checks,flush=True)
    write_json(OUT/'parallax.json',dict(results=result,sample_sha256=sha(PREV/'parallax_sample.npz'),
        scope='Inverse-probability-weighted quality/zero-point eligible Gaia test parent, no observed SNR truncation in primary PIT/log-density',
        limitation='No r>=22 Gaia objects; reference selection and shared systematics not corrected by sampling weights; development diagnostics, not independent calibration'))
if __name__=='__main__':main()
