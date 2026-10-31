#!/usr/bin/env python3
"""Run the registered local color experiment on an allocated NERSC CPU."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import numpy as np
from scipy.stats import truncnorm
from cluster_rgb_scales_run import read_cluster,filters_and_extinction
from photdis.cluster_rgb_external import identity_hash,verify_hashes
from photdis.cluster_rgb_shared_sampling import effective_covariance,metropolis_chains,chain_diagnostics
from photdis.cluster_rgb_local_color import (
    LocalColorTarget,expand_abundance_atoms,brightness_log_marginal,edge_crossing_bound,
    component_bin_mass,weighted_joint,conditional_mass,summarize_local)
from photdis.stage1 import require_compute,write_json

CFG=Path('configs/cluster_rgb_local_color_v1.json')
OUT=Path('data/cluster_rgb_local_color_v1')


def source_hashes():
    paths=[CFG,Path('scripts/cluster_rgb_local_color_run.py')]
    paths += [Path('src/photdis')/(n+'.py') for n in
        ('cluster_rgb_local_color','cluster_rgb_shared_sampling','cluster_rgb_shared_distance','cluster_rgb_lower')]
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def fit_case(cfg,settings,g,colors,errors,models,rband,name,modelname,prior,se,draws,denom,metadata,serial):
    nc=colors.shape[1]
    if prior=='unknown':
        models,labels=expand_abundance_atoms(models,cfg['unknown_feh_components'])
        feh=-1.7;sz=0.;weights=cfg['unknown_weights']
    else:
        labels=np.zeros(len(models),int);feh=settings['feh'];sz=settings['feh_sigma']
        weights={'reference':[1.]}
    bounds=settings['mu']+np.array(cfg['delta_mu_bounds'])
    covs=[effective_covariance(m,rband,se,cfg['zero_point_sigma'],sz) for m in models]
    target=LocalColorTarget(models,g,colors,errors,feh,covs,len(g),bounds)
    # The reference-centered window is intentionally disclosed, not a blind recovery.
    start=np.r_[settings['mu'],np.zeros(nc)]
    seed=cfg['seed']+serial*10007
    x,k,lp,accept=metropolis_chains(target,start,len(models),chains=cfg['chains'],
        warmup=cfg['warmup'],draws=draws,seed=seed)
    center,sigma=target.conditional_mu(x,k)
    mu_samples=truncnorm.rvs((bounds[0]-center)/sigma,(bounds[1]-center)/sigma,
        loc=center,scale=sigma,random_state=np.random.default_rng(seed+1))
    traces={'mu':mu_samples,'theta':x[:,:,0],'log_target':lp}
    traces.update({f'color_{i}':x[:,:,i+1] for i in range(nc)})
    diagnostics={key:chain_diagnostics(a) for key,a in traces.items()}
    atom=labels[k];atom_mixing=True
    if prior=='unknown':
        if np.ptp(atom)==0:
            atom_mixing=False
            diagnostics['atom']={'constant':True,'value':int(atom.flat[0]),'rhat':None,'bulk_ess':None,'tail_ess':None}
        else:diagnostics['atom']=chain_diagnostics(atom)
    finite=[v for v in diagnostics.values() if v.get('rhat') is not None]
    sampling_pass=atom_mixing and all(v['rhat']<cfg['diagnostics']['rhat_max'] and
        v['bulk_ess']>cfg['diagnostics']['ess_min'] and v['tail_ess']>cfg['diagnostics']['ess_min'] for v in finite)
    fine_edges=settings['mu']+np.linspace(*cfg['delta_mu_bounds'],401)
    edges=fine_edges[::2]
    chain_component=np.array([component_bin_mass(fine_edges,center[j],sigma[j],atom[j],len(weights[next(iter(weights))]))
                              for j in range(cfg['chains'])])
    total=chain_component.mean(axis=0)
    curves={};stats={}
    for key,w in weights.items():
        fine_joint=weighted_joint(total,w)
        coarse_joint=fine_joint.reshape(-1,2).sum(axis=1)
        fine_p=conditional_mass(fine_joint,denom['fine'])
        p=conditional_mass(coarse_joint,denom['coarse'])
        row=summarize_local(edges-settings['mu'],p)
        check=summarize_local(fine_edges-settings['mu'],fine_p)
        chain_stats=[]
        for mass in chain_component:
            joint=weighted_joint(mass,w).reshape(-1,2).sum(axis=1)
            chain_stats.append(summarize_local(edges-settings['mu'],conditional_mass(joint,denom['coarse'])))
        qchange=float(np.max(abs(np.array(row['q'])-check['q'])))
        reweight=np.asarray(w)[atom]*len(w)
        row.update(grid_max_quantile_change=qchange,chain_summaries=chain_stats,
            max_chain_median_spread=float(np.ptp([c['q'][2] for c in chain_stats])),
            reweight_ess_fraction=float(reweight.mean()**2/np.mean(reweight**2)),
            numerical_pass=bool(sampling_pass and qchange<cfg['diagnostics']['grid_quantile_tolerance_mag']
                                and denom['order_delta']<cfg['diagnostics']['denominator_log_tolerance']))
        stats[key]=row;curves[key]=p
    result=dict(metadata,cluster=name,model=modelname,abundance=prior,sigma_E=se,
        n=len(g),draws_per_chain=draws,warmup=cfg['warmup'],chains=cfg['chains'],seed=seed,
        diagnostics=diagnostics,sampling_pass=bool(sampling_pass),stats=stats,
        atom_occupancies_by_chain=[np.bincount(a,minlength=int(labels.max())+1).tolist() for a in atom],
        acceptance_local_model_curve=accept,
        interpretation='exposed local information, normalized ratio AFTER shared marginalization')
    stem=f'{name}__{modelname}__{prior}__e{se:.3f}__n{draws}'
    write_json(OUT/(stem+'.json'),result)
    np.savez(OUT/(stem+'.npz'),edges=edges-settings['mu'],fine_edges=fine_edges-settings['mu'],
             chain_component_mass=chain_component,log_brightness=denom['coarse'],**curves)
    np.savez(OUT/(stem+'_chains.npz'),state=x,model_index=k,log_target=lp,mu=mu_samples)
    print(stem,'N',len(g),'pass',sampling_pass,'results',
        {key:(v['q'][2],v['width68'],v['edge_flag']) for key,v in stats.items()},flush=True)


def main():
    require_compute();OUT.mkdir(parents=True,exist_ok=True)
    parser=argparse.ArgumentParser();parser.add_argument('--cluster',required=True)
    parser.add_argument('--draws',type=int);parser.add_argument('--only',nargs='*')
    args=parser.parse_args();cfg=json.loads(CFG.read_text());hashes=source_hashes()
    basepath=Path('configs/cluster_rgb_scales_v1.json')
    assert hashlib.sha256(basepath.read_bytes()).hexdigest()==cfg['base_scale_config_sha256']
    base=json.loads(basepath.read_text());settings=base['clusters'][args.cluster]
    verify_hashes(cfg['model_sha256']);parent,meta=read_cluster(args.cluster,settings)
    nominal=parent['g0'].astype(float)-settings['mu']
    take=(nominal>=cfg['nominal_interior_mg'][0])&(nominal<=cfg['nominal_interior_mg'][1])
    tab=parent[take];g=tab['g0'].astype(float)
    colors=np.column_stack([tab['gr0'],tab['rz0']]).astype(float)
    errors=np.column_stack([tab['sigma_g'],tab['sigma_r'],tab['sigma_z']]).astype(float)
    metadata=dict(parent_n=meta['n'],parent_ids_sha256=meta['ids_sha256'],input_sha256=meta['input_sha256'],
        ids_sha256=identity_hash(tab['source_id']),source_hashes=hashes,model_sha256=cfg['model_sha256'],
        host=socket.gethostname(),n=len(g),mu_reference_for_local_design=settings['mu'],
        nominal_mg_range=[float(g.min()-settings['mu']),float(g.max()-settings['mu'])] if len(g) else [],
        sigma_g_quantiles=np.quantile(errors[:,0],[0,.5,.95,1]).tolist() if len(g) else [])
    if len(g)<cfg['minimum_fit_n']:
        metadata['status']='too few interior members under fixed design'
        write_json(OUT/f'{args.cluster}_selection.json',metadata);return
    rband=filters_and_extinction(settings,.001)/.001
    draws=args.draws or cfg['draws'];assert draws in (cfg['draws'],cfg['refinement_draws'])
    models=json.loads(Path('manifests/cluster_rgb_lower_robust_v2.json').read_text())['results']
    brightness={};arrays={'source_id':tab['source_id']}
    fine_edges=settings['mu']+np.linspace(*cfg['delta_mu_bounds'],401)
    fine_mu=(fine_edges[:-1]+fine_edges[1:])/2
    coarse_mu=(fine_edges[::2][:-1]+fine_edges[::2][1:])/2
    check_mu=np.r_[fine_edges[0],coarse_mu,fine_edges[-1]]
    serial=cfg['clusters'].index(args.cluster)*8
    for se in cfg['physical_ebv_sigmas']:
        sg=np.hypot(cfg['zero_point_sigma'],rband[0]*se)
        lower=brightness_log_marginal(check_mu,g,errors[:,0],sg,cfg['gh_orders'][0])
        upper=brightness_log_marginal(check_mu,g,errors[:,0],sg,cfg['gh_orders'][1])
        fine=brightness_log_marginal(fine_mu,g,errors[:,0],sg,cfg['gh_orders'][1])
        denom=dict(coarse=upper[1:-1],fine=fine,order_delta=float(np.max(abs(lower-upper))))
        log_range=float(np.ptp(upper));cross=edge_crossing_bound(check_mu,g,errors[:,0],sg)
        brightness[f'{se:.3f}']=dict(shared_g_sigma=float(sg),log_range=log_range,
            boundary_flag=log_range>cfg['diagnostics']['brightness_log_range_flag'],
            quadrature_order_delta=denom['order_delta'],max_union_crossing_bound=float(cross.max()),
            log_B_endpoints=[float(upper[0]),float(upper[-1])],
            min_margin_over_total_sigma=float(np.min(np.minimum(g-settings['mu']-.5-.5,
                3.-(g-settings['mu'])-.5)/np.hypot(errors[:,0],sg))))
        arrays[f'log_B_{se:.3f}']=upper[1:-1]
        for modelname in cfg['models']:
            nc=1 if modelname=='robust_gr' else 2
            for prior in cfg['priors']:
                key=f'{modelname}__{prior}__e{se:.3f}';serial+=1
                if args.only and key not in args.only:continue
                dest=OUT/f'{args.cluster}__{key}__n{draws}.json'
                if dest.exists():
                    old=json.loads(dest.read_text())
                    if old['source_hashes']!=hashes:raise ValueError(f'Existing output has different code: {dest}')
                    print('Retain completed',dest,flush=True);continue
                fit_case(cfg,settings,g,colors[:,:nc],errors,models[modelname]['models'],rband,
                    args.cluster,modelname,prior,se,draws,denom,metadata,serial)
    metadata['brightness_checks']=brightness
    write_json(OUT/f'{args.cluster}_selection.json',metadata)
    np.savez(OUT/f'{args.cluster}_selection.npz',delta_mu=coarse_mu-settings['mu'],**arrays)
    print('Completed',args.cluster,'selection',metadata,flush=True)


if __name__=='__main__':main()
