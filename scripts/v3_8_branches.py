#!/usr/bin/env python3
"""Identify deficient luminosity branches and test a small independent anchor patch."""
import copy
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

from photdis.stage1 import require_compute
from photdis.v3.inference import kernel_loglike
from photdis.v3.likelihood import trapezoid_weights
from photdis.v3.model import inference_model
from photdis.v3_3 import class_log_likelihood
from photdis_figure_style import BLUE, GRAY, GREEN, ORANGE, STYLE
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_3_catalog import FREEZE, read_frozen
from v3_3_sed import verified
from v3_6_distance_evidence import OUT as EVIDENCE
from v3_6_lf_sensitivity import GRID
from v3_6_reference_photometry import OUT as PHOTO
from v3_6_vac_distances import read_columns
from v3_7_lf_known_branches import BLUE as BLUE_SOURCE, examples
from v3_8_candidate import read_cache

OUT = ROOT/'data/v3_8/branches'
USE_BANDS = np.array([True,True,False,True,False,False])
LABEL = {'faint_blue_near':2,'blue_2_to_5':0,'far_giant':1,'normal':0}
NAMES = ('Dwarf','Giant','Other')
MASS = .03


def choose():
    evidence = json.loads((EVIDENCE/'complete.json').read_text())
    blue = json.loads((BLUE_SOURCE/'complete.json').read_text())
    excluded = {}
    rows = list(examples(evidence,blue,excluded))
    cohorts = {}
    for name in LABEL:
        if name=='normal':
            continue
        rows1 = [r for r in rows if r['cohort']==name]
        unique = {r['ls_id']:r for r in rows1}
        cohorts[name] = sorted(unique.values(),key=lambda r: np.uint64(r['ls_id']) % np.uint64(86069))
    # Use a fixed identity order for training/test partition and representative examples.
    anchors = cohorts['faint_blue_near'][:32]
    cohorts['faint_blue_near'] = cohorts['faint_blue_near'][32:44]
    cohorts['blue_2_to_5'] = cohorts['blue_2_to_5'][:12]
    cohorts['far_giant'] = cohorts['far_giant'][:12]
    a = read_columns(read_cache(EVIDENCE,'Gaia.npz'),
        ('ls_id','TYPE','reference_kpc','qrf_quantiles_kpc','primary','high_quality','qrf_training',
         'consensus_class','observed_r','dereddened_g_r'))
    q = a['qrf_quantiles_kpc'][:,2]
    d = a['reference_kpc']
    clean = ((a['TYPE']=='PSF')&a['primary']&a['high_quality']&~a['qrf_training']&
             (a['consensus_class']==0)&(d>.15)&(d<2.)&(abs(q/d-1)<.1)&
             (a['dereddened_g_r']>=.8)&(a['dereddened_g_r']<1.4))
    j = np.flatnonzero(clean)
    j = j[np.argsort(a['ls_id'][j]%86069)[:12]]
    cohorts['normal'] = [dict(ls_id=int(a['ls_id'][i]),reference_kpc=d[i],
                              qrf_median_kpc=q[i],source='Gaia',TYPE='PSF',cohort='normal') for i in j]
    test = sum(cohorts.values(),[])
    all_ids = [r['ls_id'] for r in anchors+test]
    if len(set(all_ids))!=len(all_ids):
        raise ValueError('Branch training and test identities must be strictly disjoint')
    return anchors,test,excluded


def observation(photo,index):
    with np.errstate(divide='ignore'):
        err = 1/np.sqrt(photo['flux_ivar'][index])
    return dict(flux=photo['flux'][index],flux_error=err,band_mask=photo['band_mask'][index],
                transmission=photo['transmission'][index],EBV=photo['EBV'][index],
                photometric_system='DECam_DR11')


def patch(model,obs,refs):
    """Fixed empirical low-luminosity component built on 32 separate anchors."""
    intrinsic = np.stack([x['flux']/np.where(x['transmission']>0,x['transmission'],1.)*
                         r['reference_kpc']**2 for x,r in zip(obs,refs)])
    if not np.isfinite(intrinsic[:,USE_BANDS]).all() or (intrinsic[:,USE_BANDS]<=0).any():
        raise ValueError('Anchor optical flux must be positive')
    k = len(intrinsic)
    new = copy.deepcopy(model)
    means = np.zeros((k,model.means.shape[1]))
    means[:,:6][:,USE_BANDS] = intrinsic[:,USE_BANDS]
    cov = np.zeros((k,*model.covariances.shape[1:]))
    di = np.arange(cov.shape[-1])
    cov[:,di,di] = np.maximum((.15*abs(means))**2,1e-8)
    new.means = np.concatenate([model.means,means])
    new.covariances = np.concatenate([model.covariances,cov])
    new.weights = np.r_[model.weights*(1-MASS),np.full(k,MASS/k)]
    new.classes = np.vstack([model.classes,np.tile([0.,0.,1.],(k,1))])
    return new


def evaluate(model,obs,d,pred,cls):
    likelihood = class_log_likelihood(model,{**obs,'band_mask':obs['band_mask']&USE_BANDS},GRID)
    prior = np.log(model.weights@model.classes)[:,None]+norm.logpdf(GRID,15.,5.)
    w = trapezoid_weights(GRID)
    joint = likelihood+prior
    q = np.exp(joint-logsumexp(joint+np.log(w)))
    mu = 10+5*np.log10(d)
    wrong_mu = 10+5*np.log10(pred)
    correct = abs(GRID-mu)<=.5
    wrong = abs(GRID-wrong_mu)<=.5
    score = dict(correct_mass=float(np.sum(q[:,correct]*w[correct])),
                 alternative_mass=float(np.sum(q[:,wrong]*w[wrong])),
                 correct_class_joint_mass=float(q[cls,correct]@w[correct]),
                 correct_peak_log_likelihood=float(likelihood[:,correct].max()) if correct.any() else None,
                 alternative_peak_log_likelihood=float(likelihood[:,wrong].max()) if wrong.any() else None)
    return q,likelihood,prior,score


def main():
    require_compute()
    OUT.mkdir(parents=True,exist_ok=True)
    marker = OUT/'complete.json'
    if marker.exists():
        raise FileExistsError(marker)
    anchors,test,excluded = choose()
    rows = anchors+test
    photo = read_columns(read_cache(PHOTO,'photometry.npz'),
        ('ls_id','TYPE','flux','flux_ivar','band_mask','transmission','EBV'))
    key = np.argsort(photo['ls_id'])
    ids = np.asarray([r['ls_id'] for r in rows])
    j = key[np.searchsorted(photo['ls_id'][key],ids)]
    np.testing.assert_array_equal(photo['ls_id'][j],ids)
    if not (photo['TYPE'][j]=='PSF').all():
        raise ValueError('Require native PSFs for branch decomposition')
    obs = [observation(photo,i) for i in j]
    entry = read_frozen()['models']['nominal']
    model = inference_model(joblib.load(verified(ROOT/entry['path'],entry['sha256'])))
    alternate = patch(model,obs[:len(anchors)],anchors)
    stats,pdf,like,prior = [],[],[],[]
    for r,o in zip(test,obs[len(anchors):]):
        cls = LABEL[r['cohort']]
        d,pred = r['reference_kpc'],r['qrf_median_kpc']
        q,logl,logp,controls = evaluate(model,o,d,pred,cls)
        v,_,_,patched = evaluate(alternate,o,d,pred,cls)
        u = 10+5*np.log10(d)
        kernels,_ = kernel_loglike(model,{**o,'band_mask':o['band_mask']&USE_BANDS},np.array([u]))
        support = np.flatnonzero((model.classes[:,cls]>.5)&~model.discrepancy_channel)
        if len(support):
            best = support[np.argmax(kernels[:,:,0].max(axis=0)[support])]
            score = float(kernels[:,best,0].max())
            effective = getattr(model,'effective_reference_sources',None)
            effective_n = float(effective[best]) if effective is not None and np.ndim(effective[best])==0 else None
        else:
            best,score,effective_n = -1,None,None
        stats.append(dict(**r,best_original_component=int(best),component_peak_log_likelihood=score,
            effective_reference_sources=effective_n,
            **{k+'_original':x for k,x in controls.items()},**{k+'_patch':x for k,x in patched.items()}))
        pdf.append(np.stack([q,v]))
        like.append(logl)
        prior.append(logp)
    csv_rows(OUT/'cases.csv',stats)
    csv_rows(OUT/'anchors.csv',anchors)
    atomic_npz(OUT/'branch_curves.npz',ls_id=ids[len(anchors):],TYPE=np.full(len(test),'PSF'),
               grid_mu=GRID,pdf=np.stack(pdf),log_likelihood=np.stack(like),log_prior=np.stack(prior))
    with plt.rc_context(STYLE):
        f,ax = plt.subplots(4,3,figsize=(16,15),layout='constrained',sharex=True)
        for k,cohort in enumerate(LABEL):
            i = next(j for j,t in enumerate(test) if t['cohort']==cohort)
            for j,legend in enumerate(NAMES):
                ax[k,0].plot(GRID,like[i][j]-np.max(like[i]),label=legend,color=(BLUE,ORANGE,GRAY)[j])
                ax[k,1].plot(GRID,prior[i][j],color=(BLUE,ORANGE,GRAY)[j])
            ax[k,2].plot(GRID,pdf[i][0].sum(axis=0),color=BLUE,label='Original grz SED')
            ax[k,2].plot(GRID,pdf[i][1].sum(axis=0),color=GREEN,label='With low-luminosity anchors')
            for p in ax[k]:
                p.axvline(10+5*np.log10(test[i]['reference_kpc']),color=GRAY,ls='--',lw=1.2)
                p.axvline(10+5*np.log10(test[i]['qrf_median_kpc']),color=ORANGE,ls=':',lw=1.2)
                p.set(xlim=(2,20))
            ax[k,0].set_ylim(-35,1)
            ax[k,0].set_ylabel(('Faint blue nearby','Blue 2–5 kpc','Far giant','Accurate dwarf')[k])
        for a,t in zip(ax[0],('Class likelihood · relative log scale','Nominal class and distance log prior','Distance posterior')):
            a.set_title(t,fontsize=14)
        ax[0,0].legend(fontsize=9)
        ax[0,2].legend(fontsize=9)
        for a in ax[-1]:
            a.set_xlabel('Distance modulus (mag)')
        f.suptitle('Representative luminosity branches · gray: reference · orange dotted: QRF median')
        for ext in ('png','pdf'):
            f.savefig(OUT/f'F08_luminosity_branches.{ext}',dpi=300)
        plt.close(f)
    paths = (EVIDENCE/'complete.json',PHOTO/'complete.json',BLUE_SOURCE/'complete.json',FREEZE,Path(__file__).resolve())
    save_json(marker,dict(inputs={str(x.relative_to(ROOT)):sha(x) for x in paths},
        exclusions=excluded,n_test=len(test),n_anchors=len(anchors),patch_mass=MASS,
        outputs={x.name:sha(x) for x in OUT.iterdir() if x.is_file() and x!=marker}))


if __name__=='__main__':
    main()
