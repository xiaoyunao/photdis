#!/usr/bin/env python3
"""Four frozen cluster responses and shared-distance prototype on allocated CPU."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import numpy as np
import fitsio
from scipy.special import logsumexp, ndtri
from scipy.stats import qmc
from speclite.filters import load_filters
from photdis.cluster_rgb_conditional import grid_from_bins
from photdis.cluster_rgb_lower import probabilities
from photdis.cluster_rgb_external import identity_hash, score, verify_hashes
from photdis.cluster_rgb_scales import (
    LEGACY_R, band_extinction, change_scale, color_geometry, reference_covariance)
from photdis.cluster_rgb_shared_distance import (
    member_log_likelihood, prior_member_log_likelihood, posterior_summary)
from photdis.stage1 import require_compute, write_json

CFG = Path('configs/cluster_rgb_scales_v1.json')
OUT = Path('data/cluster_rgb_scales_v1')


def read_cluster(name, settings):
    path = Path(f'data/cluster_rgb_gr_v2/{name}.fits')
    whole = fitsio.read(path)
    mg = whole['g0'].astype(float)-settings['mu']
    mask = ((whole['pmember']>.9) & (whole['phase'].astype('U')=='lower_rgb')
            & whole['grz_clean_same_row'] & (mg>=.5) & (mg<3))
    mask &= np.isfinite(sum(whole[k].astype(float) for k in
        ('g0','gr0','rz0','sigma_g','sigma_r','sigma_z')))
    tab = whole[mask]
    if len(tab) != settings['expected_n']:
        raise ValueError(f'{name} identity count changed: {len(tab)}')
    tab = tab[np.argsort(tab['source_id'])]
    ids = identity_hash(tab['source_id'])
    if name == 'ngc4147':
        locked = json.loads(Path('manifests/cluster_rgb_ngc4147_blind_coverage_v1.json').read_text())
        if ids != locked['common_ids_sha256']: raise ValueError('NGC4147 identity changed')
    elif name == 'ngc5466':
        old = np.load('data/cluster_rgb_lower_v2/ngc5466_diagnostic.npz')
        if ids != identity_hash(old['source_id']):
            raise ValueError('NGC5466 diagnostic identity changed')
    else:
        frozen = json.loads(Path('manifests/cluster_rgb_lower_model_v2.json').read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest() != frozen['input_sha256'][name]:
            raise ValueError('Development source catalog changed')
    if np.any(tab['release'] != 11010):
        raise ValueError('Unexpected natural photometric system')
    return tab, dict(n=len(tab),ids_sha256=ids,input_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def filters_and_extinction(settings, ebv, temperature=4500):
    return np.array([band_extinction(f.wavelength,f.response,ebv,settings['rv'],temperature)
                     for f in load_filters('decam2014-g','decam2014-r','decam2014-z')])


def responses(cfg, fits):
    grid = grid_from_bins([[.5,3]],.02)
    results = dict(config_sha256=hashlib.sha256(CFG.read_bytes()).hexdigest(),
                   model_sha256=cfg['model_sha256'],compute_host=socket.gethostname(),clusters={})
    for name,s in cfg['clusters'].items():
        tab, meta = read_cluster(name,s)
        colors = np.column_stack([tab['gr0'],tab['rz0']]).astype(float)
        err = np.column_stack([tab['sigma_g'],tab['sigma_r'],tab['sigma_z']]).astype(float)
        g = tab['g0'].astype(float); nominal_m = g-s['mu']
        old_a = tab['ebv'][:,None]*LEGACY_R
        eproxy = .86*float(np.median(tab['ebv']))
        coefficients = {str(t):(filters_and_extinction(s,.001,t)/.001).tolist()
                        for t in (4000,4500,5000)}
        meta.update(raw_sfd_median=float(np.median(tab['ebv'])),sf11_proxy_ebv=eproxy,
                    legacy_A_median=np.median(old_a,axis=0).tolist(),ccm_small_E_coefficients=coefficients)
        scenarios = {}
        arrays = dict(source_id=tab['source_id'],nominal_mg=nominal_m)
        for label in cfg['scenarios']:
            if label == 'paper_ccm_spectroscopic_4147' and name != 'ngc4147': continue
            e = eproxy if label.startswith('sf11') else s['paper_ebv']
            a = old_a if label == 'legacy' else filters_and_extinction(s,e)
            de = 0 if label in ('legacy','paper_ccm','paper_ccm_spectroscopic_4147') else e-s['paper_ebv']
            rr = np.array(s['rr_extinction_response_components'])
            dmu = -float(np.mean(rr))*de
            dz = .1 if label.endswith('plus_0p1') else 0
            dmu += s['rr_dmu_dfeh']*dz
            z = s['feh']+dz; zs = s['feh_sigma']
            if label == 'paper_ccm_spectroscopic_4147': z,zs=-1.84,np.hypot(.02,.1)
            ng,nc,nm = change_scale(g,colors,old_a,a,s['mu'],dmu)
            row = dict(physical_ebv=None if label=='legacy' else e,
                       A_median=np.median(a,axis=0).tolist() if a.ndim==2 else a.tolist(),
                       rr_mu=s['mu']+dmu,rr_mu_band_weight_range=sorted((s['mu']-rr*de+s['rr_dmu_dfeh']*dz).tolist()),
                       feh=z,feh_sigma=zs,models={})
            for modelname in ('robust_gr','robust_grz'):
                ncolor = 1 if modelname=='robust_gr' else 2
                prob = probabilities(fits[modelname]['models'],grid,
                    [z-np.sqrt(2)*zs,z,z+np.sqrt(2)*zs],[.25,.5,.25],nc[:,:ncolor],err)
                stats, quantiles, residual = score(grid,prob,nm,bin_reference=nominal_m)
                row['models'][modelname] = stats
                arrays[label+'__'+modelname+'_residual'] = residual
            residual,along,perp = color_geometry(fits['robust_grz']['models'][0],nm,z,nc,err)
            row['geometry'] = dict(color_median=np.median(residual,axis=0).tolist(),
                along_median=float(np.median(along)),perpendicular_median=float(np.median(perp)),
                perpendicular_abs_gt2=int(np.sum(abs(perp)>2)),
                projected_reference_outside_n=int(np.sum((nm<.5)|(nm>=3))))
            arrays[label+'__color_residual'] = residual
            arrays[label+'__along'] = along; arrays[label+'__perpendicular'] = perp
            scenarios[label] = row
            print('response',name,label,row['models']['robust_grz']['overall'],flush=True)
        meta['scenarios'] = scenarios
        results['clusters'][name] = meta
        np.savez(OUT/f'{name}_response.npz',**arrays)
    corr = {}
    for rho in (0.,.5,1.):
        cov = reference_covariance([.017,.024],[.036,.033],rho)
        avg,diff = np.array([.5,.5]),np.array([1.,-1.])
        # These are properties of the reference budgets, not re-fit RGB posteriors.
        zcov = reference_covariance([np.sqrt(.2**2-.1**2),0],[.1,.1],rho)
        pcov = reference_covariance([0,0],[.01,.01],rho)
        corr[str(rho)] = dict(distance_cov=cov.tolist(),
            mean_mu_sigma=float(np.sqrt(avg@cov@avg)),difference_mu_sigma=float(np.sqrt(diff@cov@diff)),
            feh_cov=zcov.tolist(),difference_feh_sigma=float(np.sqrt(diff@zcov@diff)),
            photometric_difference_sigma=float(np.sqrt(diff@pcov@diff)))
    results['reference_correlation_cases'] = corr
    write_json(OUT/'responses.json',results)


def shared_distance(cfg,fits,name):
    s = cfg['clusters'][name]; setup = cfg['joint']
    tab, meta = read_cluster(name,s)
    g = tab['g0'].astype(float)
    colors = np.column_stack([tab['gr0'],tab['rz0']]).astype(float)
    errors = np.column_stack([tab['sigma_g'],tab['sigma_r'],tab['sigma_z']]).astype(float)
    mu = np.arange(*setup['mu_prior'],setup['step'])
    u = qmc.Sobol(6,scramble=True,seed=setup['qmc_seed']).random_base2(setup['qmc_power'])
    normal = ndtri(np.clip(u[:,:5],1e-10,1-1e-10))
    counts = sorted(set(setup['subsample_sizes']+[len(g)]))
    half = len(u)//2
    rband = filters_and_extinction(s,.001)/.001
    results, curves = {}, {'mu':mu}
    for key in ('prior_only','robust_gr','robust_grz'):
        nc = 1 if key=='robust_gr' else 2
        joint = np.zeros((len(u),len(counts),len(mu)))
        individual = np.full((len(g),len(mu)),-np.inf)
        halfway = None
        for j,(uu,zz) in enumerate(zip(u,normal)):
            de = zz[1]*setup['physical_ebv_sigma']
            offset = zz[2:5]*setup['zero_point_sigma']-rband*de
            ng = g+offset[0]
            if key=='prior_only':
                ll = prior_member_log_likelihood(mu,ng,errors[:,0])
            else:
                models = fits[key]['models']
                model = models[min(int(uu[5]*len(models)),len(models)-1)]
                cc = colors+np.array([offset[0]-offset[1],offset[1]-offset[2]])
                ll = member_log_likelihood(model,mu,ng,cc[:,:nc],errors,
                                          s['feh']+s['feh_sigma']*zz[0])
            csum = np.cumsum(ll,axis=0)
            for k,n in enumerate(counts): joint[j,k] = csum[n-1]
            individual = np.logaddexp(individual,ll)
            if j+1==half: halfway=joint[:half].copy()
            if (j+1)%128==0: print('joint',name,key,j+1,flush=True)
        result = {}
        wrong_csum = np.cumsum(individual-np.log(len(u)),axis=0)
        for k,n in enumerate(counts):
            proper = logsumexp(joint[:,k],axis=0)-np.log(len(u))
            wrong = wrong_csum[n-1]
            small = logsumexp(halfway[:,k],axis=0)-np.log(half)
            summary,p = posterior_summary(mu,proper)
            bad,bp = posterior_summary(mu,wrong)
            previous,_ = posterior_summary(mu,small)
            # Prior mass integrated over mu for each nuisance draw, posterior ESS.
            logweight = logsumexp(joint[:,k],axis=1)
            weight = np.exp(logweight-logsumexp(logweight))
            summary.update(wrong_independent=bad,half_draws=previous,
                half_vs_full_median_delta=summary['q'][2]-previous['q'][2],
                half_vs_full_width68_delta=summary['width68']-previous['width68'],
                nuisance_ess=float(1/np.sum(weight**2)),
                # RR reference is used only here, after computing the posterior.
                offset_from_rr=summary['q'][2]-s['mu'])
            result[str(n)] = summary
            curves[f'{key}_{n}'] = p
            curves[f'{key}_{n}_wrong'] = bp
        results[key] = result
        print('summary',name,key,result[str(len(g))],flush=True)
    meta.update(results=results,mu_prior=setup['mu_prior'],rr_mu_comparison_only=s['mu'],
                n_shared_draws=len(u),frozen_model_target_scale_independence=True,
                selection_role='conditional_on_reference_selected_Mg_range',
                compute_host=socket.gethostname())
    write_json(OUT/f'{name}_joint.json',meta)
    np.savez(OUT/f'{name}_joint.npz',**curves)


def main():
    require_compute()
    parser=argparse.ArgumentParser();parser.add_argument('task',choices=['response','joint'])
    parser.add_argument('--cluster');args=parser.parse_args()
    cfg=json.loads(CFG.read_text());verify_hashes(cfg['model_sha256'])
    fits=json.loads(Path('manifests/cluster_rgb_lower_robust_v2.json').read_text())['results']
    OUT.mkdir(parents=True,exist_ok=True)
    if args.task=='response': responses(cfg,fits)
    else: shared_distance(cfg,fits,args.cluster)


if __name__=='__main__': main()
