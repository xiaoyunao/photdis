#!/usr/bin/env python3
"""Re-score immutable PDFs and independently predict native two-color distances."""
import argparse
import csv
import json

import numpy as np
from scipy.special import ndtr

from cluster_depth_inputs import C, CONFIG, OLD, OLD_CONFIG, OUT, ROOT
from environment_dust import modulus_quantiles
from environment_results import histogram_scores
from photdis.stage1 import require_compute
from seq7_math import evaluate_mix, mixture, quantiles
from v2_1_near import atomic_npz, csv_rows, load, save_json, sha

MODELS = ('mist','ridge_M5','ridge_M30')
OPTICAL_VETO = sum(1<<bit for bit in (0,1,2,3,4,5,6,7,10,11,12))


def column(t,name):
    return np.column_stack([t[name+'_'+b] for b in 'GRZ'])


def imaging_arrays(t):
    s=C['quality']
    obs=t['mag_grz']-2.5*np.log10(t['transmission'][:,[0,1,3]])
    np.testing.assert_allclose(obs[:,1],t['r'],atol=3e-5,rtol=0)
    depth = column(t,'PSFDEPTH')
    with np.errstate(divide='ignore',invalid='ignore'):
        margin=np.min(22.5-2.5*np.log10(5/np.sqrt(depth))-obs,axis=1)
    band_mask=(column(t,'ALLMASK')!=0).any(1)
    bits=t['MASKBITS'].astype('i8')
    m=dict(fracflux=column(t,'FRACFLUX').max(1),fracmasked=column(t,'FRACMASKED').max(1),
           fracin=column(t,'FRACIN').min(1),sigma=t['error_grz'].max(1),depth_margin=margin,
           nobs=column(t,'NOBS').min(1),masked=((bits&OPTICAL_VETO)!=0)|band_mask,
           masked_any_bit=(bits!=0)|band_mask)
    flags=dict(outer=t['radius']>=s['outer_arcmin'],low_crowd=m['fracflux']<=s['max_fracflux'],
               phot_precise=m['sigma']<=s['max_sigma'],deep=np.isfinite(margin)&(margin>=s['min_depth_margin']),
               high_membership=t['member_p']>s['strict_membership'],imaging_unmasked=~m['masked']&
                  (m['fracmasked']<=s['max_fracmasked'])&(m['fracin']>=s['min_fracin']))
    flags['clean']=np.logical_and.reduce(list(flags.values()))
    flags['clean_all_bits']=flags['clean']&~m['masked_any_bit']
    return m,flags


def resample(points):
    x=points[:,1]
    order=np.argsort(x)
    x=x[order]
    assert (np.diff(x)>0).all()
    step=C['model']['step_Mr_mag']
    n=int(np.ceil((x[-1]-x[0])/step))
    edges=np.linspace(x[0],x[-1],n+1)
    mid=(edges[1:]+edges[:-1])/2
    return np.column_stack([np.interp(mid,x,points[order,j]) for j in range(3)]),np.diff(edges)


def tracks(key):
    paths=ROOT/'configs'/CONFIG.stem
    metadata=json.loads((paths/'mist_sources.json').read_text())['systems'][key]
    path=paths/(key+'_mist_ms.csv')
    assert sha(path)==metadata['sha256']
    x=np.loadtxt(path,delimiter=',',skiprows=1)
    m=x[:,1:]
    m=m[(m[:,1]>=4.5)&(m[:,1]<=8.5)]
    result={'mist':resample(m)}
    with (ROOT/'manifests/sky_sequence_20261119/ridge/ridge.csv').open() as stream:
        ridges=list(csv.DictReader(stream))
    for name in ('M5','M30'):
        rows=[r for r in ridges if r['system']==name and r['native']=='south' and r['usable']=='True']
        a=np.array([[float(r[k]) for k in ('Mg_lo','Mg','gr','rz')] for r in rows])
        a=a[(a[:,1]-a[:,2])>=4.5]
        segments=np.split(a,np.flatnonzero(np.diff(a[:,0])>.251)+1)
        chunks=[]
        for segment in segments:
            if len(segment)>1:
                mg,gr,rz=segment[:,1:].T
                chunks.append(resample(np.column_stack((mg,mg-gr,mg-gr-rz))))
        nodes,weight=[np.concatenate(b) for b in zip(*chunks)]
        result['ridge_'+name]=nodes,weight
    return result


def normal_abs(mean,sigma):
    z=mean/sigma
    return 2*sigma*np.exp(-.5*z*z)/np.sqrt(2*np.pi)+mean*(2*ndtr(z)-1)


def gaussian_crps(means,weights,sigma,ref):
    first=np.sum(weights*normal_abs(means-ref[:,None],sigma[:,None]),axis=1)
    delta=means[:,:,None]-means[:,None,:]
    pair=normal_abs(delta,np.sqrt(2)*sigma[:,None,None])
    second=np.einsum('ijk,ij,ik->i',pair,weights,weights)
    return first-second/2


def metrics(q,crps,log,ref):
    return dict(median=q[:,2],bias=q[:,2]-ref,crps=crps,log=log,width68=q[:,3]-q[:,1],
                width95=q[:,4]-q[:,0],c68=(q[:,1]<=ref)&(ref<=q[:,3]),
                c95=(q[:,0]<=ref)&(ref<=q[:,4]),severe=abs(q[:,2]-ref)>1.)


def independent(m,e,nodes,w,ref,ref_sigma):
    settings=C['model']
    prior=w/w.sum()
    # The existing native-color likelihood includes a 0.10 mag gray dispersion.
    assert settings['gray_sigma_mag']==.10
    means,logw,sigma,chi=mixture(m,np.hypot(e,settings['cov_floor_mag']),nodes,prior)
    colors=np.diff(-m,axis=1)
    reference_colors=np.diff(-nodes,axis=1)
    supported=((colors>=reference_colors.min(0))&(colors<=reference_colors.max(0))).all(1)
    supported &= chi<=settings['support_color_chi2']
    q=quantiles(means,logw,sigma)
    _,log=evaluate_mix(ref,means,logw,np.hypot(sigma,ref_sigma))
    crps=gaussian_crps(means,np.exp(logw),sigma,ref)
    assert np.isfinite(log).all() and (crps>=-1e-9).all()
    return metrics(q,crps,log,ref),dict(means=means,logw=logw,sigma=sigma,
                  color_chi2=chi,supported=supported,q=q)


def arms(key):
    receipt=json.loads((OUT/'inputs.json').read_text())
    assert receipt['config']==sha(CONFIG)
    prior=json.loads(OLD_CONFIG.read_text())['cluster_candidates'][key]
    native=OUT/(key+'_native.npz')
    pdf=OLD/(key+'_fixed_pdfs.npz')
    assert sha(native)==receipt['files'][key]['native'] and sha(pdf)==receipt['files'][key]['pdf']
    t=load(native)
    p=load(pdf)
    np.testing.assert_array_equal(t['ls_id'],p['ls_id'])
    assert (t['TYPE']=='PSF').all() and (p['TYPE']=='PSF').all()
    np.testing.assert_allclose(p['mass'].sum(1),1.,atol=1e-10)
    m=t['mag_grz'];e=t['error_grz']
    ref=np.full(len(m),prior['rrl_mu'])
    sigma=np.full(len(m),np.hypot(prior['rrl_sigma_mag'],C['model']['score_ref_common_mag']))
    q=modulus_quantiles(p['mass'],p['edges'])
    log,crps=histogram_scores(p['edges'],p['mass'],ref,sigma)
    qrf=metrics(q,crps,log,ref)
    gr,rz=np.diff(-m,axis=1).T
    mr=m[:,1]-ref
    faint=(t['r']>=20)&(t['r']<22)
    lower=(mr>=5)&(gr>=.4)
    with (ROOT/'manifests/faint_signed_20261120/results/cluster_scale_summary.csv').open() as f:
        old={r['subset']:r for r in csv.DictReader(f) if r['system']==key}
    for label,use in [('all',ref>0),('bright',t['r']<20),('faint_lower_sequence',faint&lower)]:
        assert use.sum()==int(old[label]['n'])
        np.testing.assert_allclose(np.median(qrf['bias'][use]),float(old[label]['bias']),atol=1e-9)
    quality,flags=imaging_arrays(t)
    a={k:t[k] for k in ('ls_id','TYPE','r','radius','member_p','EBV','MASKBITS')}
    a.update(gr=gr,rz=rz,Mr=mr,faint=faint,lower=lower,block=(t['ls_id']>>20)&((1<<20)-1),
             **quality,**flags)
    a.update({'qrf_'+k:v for k,v in qrf.items()})
    series=tracks(key)
    metadata=[]
    for name in MODELS:
        nodes,w=series[name]
        all_metrics,all_cube=[],[]
        for j in range(0,len(m),96):
            sl=slice(j,j+96)
            metric,cache=independent(m[sl],e[sl],nodes,w,ref[sl],sigma[sl])
            all_metrics.append(metric)
            all_cube.append(cache)
        model_metrics={k:np.concatenate([x[k] for x in all_metrics]) for k in all_metrics[0]}
        cache={k:np.concatenate([x[k] for x in all_cube]) for k in all_cube[0]}
        a.update({name+'_'+k:v for k,v in model_metrics.items()})
        a[name+'_support']=cache['supported']
        # Compare age/FeH sequence color placement at the external luminosity
        # independently of the distance prediction (diagnostic only).
        order=np.argsort(nodes[:,1]);rr=nodes[order,1]
        for color,cc in [('gr',np.diff(-nodes,axis=1)[:,0]),('rz',np.diff(-nodes,axis=1)[:,1])]:
            obs=a[color]
            values=np.interp(mr,rr,cc[order],left=np.nan,right=np.nan)
            ix=np.clip(np.searchsorted(rr,mr),1,len(rr)-1)
            values[rr[ix]-rr[ix-1]>1.7*C['model']['step_Mr_mag']]=np.nan
            a[name+'_residual_'+color]=obs-values
        np.testing.assert_allclose(np.exp(cache['logw']).sum(1),1.,atol=1e-10)
        atomic_npz(OUT/(key+'_'+name+'_posterior.npz'),ls_id=t['ls_id'],TYPE=t['TYPE'],nodes=nodes,**cache)
        colors=np.diff(-nodes,axis=1)
        metadata.append(dict(system=key,model=name,nodes=len(nodes),supported=int(cache['supported'].sum()),
                             Mr_min=nodes[:,1].min(),Mr_max=nodes[:,1].max(),
                             gr_min=colors[:,0].min(),gr_max=colors[:,0].max(),
                             rz_min=colors[:,1].min(),rz_max=colors[:,1].max()))
    atomic_npz(OUT/(key+'_diagnostics.npz'),**a)
    csv_rows(OUT/(key+'_model_support.csv'),metadata)
    print('Compared all model and QRF distributions',key,len(m),flush=True)


if __name__=='__main__':
    require_compute()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('system',choices=C['systems'])
    arms(parser.parse_args().system)
