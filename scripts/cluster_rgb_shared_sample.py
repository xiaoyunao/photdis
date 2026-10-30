#!/usr/bin/env python3
"""Numerical cross-check of the frozen primary grz shared-distance integral."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import numpy as np
from cluster_rgb_scales_run import CFG, OUT, read_cluster, filters_and_extinction
from photdis.cluster_rgb_external import verify_hashes
from photdis.cluster_rgb_shared_distance import posterior_summary
from photdis.cluster_rgb_shared_sampling import (
    effective_covariance,CollapsedTarget,metropolis_chains,chain_diagnostics,conditional_density)
from photdis.stage1 import require_compute,write_json


def main():
    require_compute()
    parser=argparse.ArgumentParser();parser.add_argument('--cluster',required=True)
    parser.add_argument('--draws',type=int,default=12000)
    parser.add_argument('--counts',type=int,nargs='+',help='Numerical refinement of specified frozen subsets')
    parser.add_argument('--tag',default='mcmc',help='Separate numerical output tag')
    args=parser.parse_args()
    if not args.tag.replace('_','').isalnum():raise ValueError('Invalid output tag')
    cfg=json.loads(CFG.read_text());verify_hashes(cfg['model_sha256'])
    models=json.loads(Path('manifests/cluster_rgb_lower_robust_v2.json').read_text())['results']['robust_grz']['models']
    s=cfg['clusters'][args.cluster];setup=cfg['joint']
    tab,metadata=read_cluster(args.cluster,s)
    g=tab['g0'].astype(float);colors=np.column_stack([tab['gr0'],tab['rz0']]).astype(float)
    errors=np.column_stack([tab['sigma_g'],tab['sigma_r'],tab['sigma_z']]).astype(float)
    rband=filters_and_extinction(s,.001)/.001
    covariances=[effective_covariance(m,rband,setup['physical_ebv_sigma'],
        setup['zero_point_sigma'],s['feh_sigma']) for m in models]
    counts=sorted(set(setup['subsample_sizes']+[len(g)]))
    if args.counts:
        if not set(args.counts).issubset(counts):raise ValueError('Subset not in frozen registration')
        counts=sorted(set(args.counts))
    grid=np.arange(*setup['mu_prior'],setup['step'])
    results={};curves={'mu':grid}
    for n in counts:
        target=CollapsedTarget(models,g,colors,errors,s['feh'],covariances,n,setup['mu_prior'])
        # Initial location depends only on the selected photometry, not RR distance.
        start=np.r_[np.median(g[:n])-1.75,0.,0.]
        x,k,lp,accept=metropolis_chains(target,start,len(models),draws=args.draws,seed=103015+n)
        center,sigma=target.conditional_mu(x,k)
        p=conditional_density(grid,center,sigma,setup['mu_prior'])
        summary,_=posterior_summary(grid,np.log(np.maximum(p,1e-300)))
        diagnostics={label:chain_diagnostics(trace) for label,trace in (
            ('theta',x[:,:,0]),('color_gr',x[:,:,1]),('color_rz',x[:,:,2]),
            ('conditional_mu_mean',center),('log_target',lp))}
        per_chain=[]
        for ci in range(4):
            cp=conditional_density(grid,center[ci],sigma[ci],setup['mu_prior'])
            cs,_=posterior_summary(grid,np.log(np.maximum(cp,1e-300)))
            per_chain.append(cs)
        occupancy=np.array([np.bincount(row,minlength=len(models))/len(row) for row in k])
        summary.update(diagnostics=diagnostics,chain_summaries=per_chain,
            acceptance_local_model_curve=accept,calibration_occupancy_by_chain=occupancy.tolist(),
            max_chain_median_spread=float(np.ptp([v['q'][2] for v in per_chain])),
            max_occupancy_total_variation=float(max(.5*np.sum(abs(a-b)) for a in occupancy for b in occupancy)),
            conditional_g_sigma_range=[float(sigma.min()),float(sigma.max())],
            numerical_pass=bool(all(v['rhat']<1.01 and v['bulk_ess']>400 and v['tail_ess']>400
                                    for v in diagnostics.values())),
            offset_from_rr=summary['q'][2]-s['mu'])
        results[str(n)]=summary;curves[f'robust_grz_{n}']=p
        np.savez(OUT/f'{args.cluster}_{args.tag}_chain_n{n}.npz',state=x,model_index=k,log_target=lp)
        print(args.cluster,n,'median',summary['q'][2],'width68',summary['width68'],
              'pass',summary['numerical_pass'],'diagnostics',diagnostics,flush=True)
    metadata.update(results={'robust_grz':results},config_sha256=hashlib.sha256(CFG.read_bytes()).hexdigest(),
        method='exact Gaussian collapse; four post-warmup fixed-proposal MH chains; analytic grey-offset marginal',
        proposal_scheme='55% local Gaussian, 25% calibration model, 20% reversible RGB-distance shear (sigma=.3 mag)',
        chains=4,warmup_per_chain=3000,draws_per_chain=args.draws,compute_host=socket.gethostname(),
        model_sha256=cfg['model_sha256'],rr_mu_comparison_only=s['mu'],
        mu_prior=setup['mu_prior'],selection_role='conditional_on_reference_selected_Mg_range')
    write_json(OUT/f'{args.cluster}_joint_{args.tag}.json',metadata)
    np.savez(OUT/f'{args.cluster}_joint_{args.tag}.npz',**curves)


if __name__=='__main__':main()
