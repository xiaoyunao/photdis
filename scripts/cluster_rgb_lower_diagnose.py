#!/usr/bin/env python3
"""Post-unblinding explanation and registered internal recovery controls."""
import hashlib
import json
from pathlib import Path
import socket
import fitsio
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from photdis.cluster_rgb_lower import mean, probabilities, summarize
from photdis.cluster_rgb_conditional import grid_from_bins
from photdis.stage1 import require_compute, write_json

MODEL=Path('manifests/cluster_rgb_lower_model_v2.json')
OUT=Path('manifests/cluster_rgb_lower_diagnostic_v2r1.json')


def prior_exact(ref):
    q=np.tile([.5625,.9,1.75,2.6,2.9375],(len(ref),1))
    def stat(mask):
        r=ref[mask]-1.75
        if not len(r):return {'n':0}
        return dict(n=len(r),median=float(np.median(r)),nmad=float(1.4826*np.median(abs(r-np.median(r)))),
            tail=float(np.mean(abs(r)>.5)),cover68=float(np.mean((ref[mask]>=.9)&(ref[mask]<=2.6))),
            cover95=float(np.mean((ref[mask]>=.5625)&(ref[mask]<=2.9375))),width68=1.7,log_density=float(-np.log(2.5)))
    return {'overall':stat(np.ones(len(ref),bool)),
            'bins':{f'{lo:.1f}_{lo+.5:.1f}':stat((ref>=lo)&(ref<lo+.5)) for lo in np.arange(.5,3,.5)}},q


def recovery(models,grid,nc):
    out={}
    for mode in ('mean_curve','scatter_and_photometry','shared_calibration'):
        rng=np.random.default_rng(730235)
        refs=[]; probs=[]; truths=[]; cluster_metrics=[];indices=[]
        for cid in range(12):
            m=np.linspace(.51,2.99,100) if mode=='mean_curve' else rng.uniform(.5,3,100)
            # Select a calibration realization once per synthetic cluster.
            # Each realization came from coherent mu/E/FeH/zero-point draws.
            index=int(rng.integers(len(models))) if mode=='shared_calibration' else 0
            indices.append(index)
            gen=models[index]
            colors=mean(gen,m,-1.7)
            band_error=np.zeros((len(m),3))
            if mode!='mean_curve':
                scatter=rng.multivariate_normal(np.zeros(nc),np.asarray(gen['intrinsic']),len(m))
                if 'contamination' in gen:
                    scatter*=np.where(rng.random(len(m))<.05,5.,1.)[:,None]
                colors+=scatter
                band_error=rng.normal(0,.003,(len(m),3))
                colors[:,0]+=band_error[:,0]-band_error[:,1]
                if nc==2:colors[:,1]+=band_error[:,1]-band_error[:,2]
            infer=models if mode=='shared_calibration' else [models[0]]
            p=probabilities(infer,grid,[-1.7],[1],colors,np.full((len(m),3),.003))
            # The same observed g is used in final mu and in observed colors.
            ref=m+band_error[:,0]
            stats,_,_=summarize(grid,p,ref)
            cluster_metrics.append(stats['overall'])
            refs.extend(ref);truths.extend(m);probs.extend(p)
        stats,_,_=summarize(grid,np.asarray(probs),refs,bin_reference=truths)
        # Cluster rather than star bootstrap for shared-calibration uncertainty.
        boot=np.random.default_rng(835)
        for metric in ('cover68','cover95','median'):
            v=np.array([s[metric] for s in cluster_metrics])
            draws=np.mean(v[boot.integers(12,size=(1000,12))],axis=1)
            stats['mean_cluster_'+metric+'_bootstrap95']=np.quantile(draws,[.025,.975]).tolist()
        stats['cluster_realization_indices']=indices
        stats['per_cluster']=cluster_metrics
        stats['interpretation']='deterministic contraction, not coverage test' if mode=='mean_curve' else 'coverage under declared generator and Gaussian observation noise'
        out[mode]=stats
    return out


def main():
    require_compute()
    if OUT.exists():raise FileExistsError(OUT)
    frozen=json.loads(MODEL.read_text())
    robust=Path('manifests/cluster_rgb_lower_robust_v2.json')
    if robust.exists():
        extra=json.loads(robust.read_text())
        if extra['parent_sha256']!=hashlib.sha256(MODEL.read_bytes()).hexdigest():raise ValueError('Robust parent mismatch')
        frozen['results'].update(extra['results'])
    grid=grid_from_bins(frozen['support'],.02)
    if not np.allclose([grid.min(),grid.max()],[.51,2.99]):raise ValueError('Revisit prior-only analytic range')
    t=fitsio.read('data/cluster_rgb_gr_v2/ngc5466.fits')
    mg=t['g0']-16.02
    mask=(t['pmember']>.9)&t['grz_clean_same_row']&(t['phase'].astype('U')=='lower_rgb')&(mg>=.5)&(mg<3)
    mask &= np.isfinite(t['g0']+t['gr0']+t['rz0']+t['sigma_g']+t['sigma_r']+t['sigma_z'])
    t=t[mask];ref=mg[mask]
    old=fitsio.read('data/cluster_rgb_conditional_v1/ngc5466_scores.fits')
    old=old[(old['tier'].astype('U')=='grz_common')&(old['condition'].astype('U')=='known_rrab')]
    if set(t['source_id'])!=set(old['source_id']):raise ValueError('Original common identity changed')
    prior,prior_q=prior_exact(ref)
    colors=np.column_stack([t['gr0'],t['rz0']])
    errors=np.column_stack([t['sigma_g'],t['sigma_r'],t['sigma_z']])
    zoff=np.sqrt(2)*.23
    conditions={'known':([-2.16-zoff,-2.16,-2.16+zoff],[.25,.5,.25]),
                'unknown_equal':([-1.46,-2.],[.5,.5]),
                'unknown_rich':([-1.46,-2.],[.8,.2]),'unknown_poor':([-1.46,-2.],[.2,.8])}
    results={};simulations={};rows={};pair={}
    fig,ax=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    ax[0].scatter(ref,ref-1.75,color='0.6',s=9,label='prior only')
    for name,result in frozen['results'].items():
        nc=2 if name.endswith('grz') else 1
        results[name]={}
        for label,(zs,weights) in conditions.items():
            p=probabilities(result['models'],grid,zs,weights,colors[:,:nc],errors)
            stats,q,r=summarize(grid,p,ref)
            stats['log_score_gain_over_prior']=stats['overall']['log_density']+np.log(2.5)
            stats['width68_fraction_of_prior']=stats['overall']['width68']/1.7
            results[name][label]=stats
            if label=='known':
                rows[name]=r
                ax[0].scatter(ref,r,s=9,alpha=.65,label=name)
                ax[1].plot(ref[np.argsort(ref)],q[np.argsort(ref),2],'.',ms=3,label=name)
        simulations[name]=recovery(result['models'],grid,nc)
        x=np.linspace(.5,3,200)
        ax[2].plot(x,mean(result['models'][0],x,-1.7)[:,0],label=name)
        print(name,results[name]['known']['overall'],flush=True)
    for kind in ('quadratic','monotone','robust'):
        if kind+'_gr' not in rows:continue
        a,b=rows[kind+'_gr'],rows[kind+'_grz']
        rng=np.random.default_rng(734)
        ix=rng.integers(len(a),size=(1000,len(a)))
        nmad=lambda x:1.4826*np.median(abs(x-np.median(x,axis=-1,keepdims=True)),axis=-1)
        pair[kind]={'gain':float(nmad(a)-nmad(b)),
                    'source_bootstrap95':np.quantile(nmad(a[ix])-nmad(b[ix]),[.025,.975]).tolist()}
    ax[0].axhline(0,color='k',lw=.7);ax[0].set(xlabel='Reference Mg',ylabel='Predicted minus reference modulus')
    ax[1].axhline(1.75,color='0.6',ls=':',label='prior only');ax[1].plot([.5,3],[.5,3],'k--',lw=.7);ax[1].set(xlabel='Reference Mg',ylabel='Posterior median Mg')
    ax[2].set(xlabel='Mg',ylabel='Mean g-r at FeH=-1.7')
    for a in ax:a.legend(fontsize=6);a.grid(alpha=.2)
    figure=Path('figures/cluster_rgb/07_lower_rgb_controls_v2r1.png');fig.savefig(figure,dpi=160);plt.close(fig)
    data=Path('data/cluster_rgb_lower_v2');data.mkdir(exist_ok=True,parents=True)
    np.savez(data/'ngc5466_diagnostic.npz',source_id=t['source_id'],mg_ref=ref,**rows)
    write_json(OUT,dict(role='post_unblinding_explanation_not_independent_validation',
        model_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(),n_common=len(t),
        same_as_original_common_ids=True,prior_only=prior,results=results,paired=pair,
        recovery=simulations,figure=str(figure),compute_host=socket.gethostname()))

if __name__=='__main__':main()
