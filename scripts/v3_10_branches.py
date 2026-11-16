#!/usr/bin/env python3
"""Test explicit luminosity branches against frozen production forecasts."""
import concurrent.futures
import json
import multiprocessing
import os

import joblib
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

from photdis.reference_v1_4 import unique_join
from photdis.stage1 import galactic, require_compute
from photdis.v3.inference import kernel_loglike
from photdis.v3.model import inference_model
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_3_catalog import read_frozen
from v3_3_sed import verified
from v3_6_lf_priors import giant_luminosity_weights, radial_profiles
from v3_6_vac_distances import read_columns
from v3_8_branches import USE_BANDS, patch
from v3_9_branches import histogram_on, window
from v3_9_catalog import OUT as EXPANSION
from v3_9_references import choose, join, subset
from v3_10_branch_cases import DEST as CASES
from v3_10_evidence import DEST as EVIDENCE

DEST = ROOT/'data/v3_10/branches'
EDGES = np.linspace(-5.,25.,201)
MU = (EDGES[1:]+EDGES[:-1])/2
WD_PRIORS = np.array([.001,.01,.05])
OBS = ('flux','flux_error','band_mask','transmission','EBV')
CONTEXT = None


def cohorts(inputs):
    marker = CASES/'complete.json'
    meta = json.loads(marker.read_text())
    base = ('ls_id','TYPE','RA','DEC','sky_role','reference_kpc','qrf_quantiles_kpc','source_id')
    faint = read_columns(verified(CASES/'faint_gaia.npz',meta['outputs']['faint_gaia.npz']),
                          (*base,'clean_low_luminosity'))
    f = faint.pop('clean_low_luminosity')
    faint['cohort'] = np.where(f,'faint_low_luminosity','faint_other')
    sources = [faint]
    inputs[str(marker.relative_to(ROOT))] = sha(marker)
    for region in ('south_faint','south_low'):
        root = EVIDENCE/region
        receipt = root/'complete.json'
        meta = json.loads(receipt.read_text())
        d = read_columns(verified(root/'references.npz',meta['outputs']['references.npz']),
            (*base,'r','g0','r0','row_state','evaluation_row','class_label',
             'specdis_kpc','spdist_kpc','specdis_quality','spdist_quality'))
        ref = np.where(d['specdis_quality'],d['specdis_kpc'],
                       np.where(d['spdist_quality'],d['spdist_kpc'],np.nan))
        c = d['class_label']
        color = d['g0']-d['r0']
        qrf = d['qrf_quantiles_kpc'][:,2]
        eligible = ((d['row_state']==7)&d['evaluation_row']&(d['sky_role']>=3)&
                    np.isfinite(qrf)&np.isfinite(ref)&(ref>0))
        blue = (color>=.5)&(color<.8)
        variants = (
            ('faint_far_dwarfs',(c==0)&(d['r']>=20)&(d['r']<21)&(ref>=5)&(ref<10)),
            ('blue_2_5',(c==0)&(d['r']<20)&blue&(ref>=2)&(ref<5)),
            ('ordinary_dwarfs',(c==0)&(d['r']<20)&~blue&(ref>=2)&(ref<10)),
            ('far_giants',(c==1)&(ref>=5)&(ref<25)),
            ('other_stars',(c==2)&(ref>=2)&(ref<10)))
        for i,(label,k) in enumerate(variants):
            j = choose(d['ls_id'],eligible&k,128,31030+i)
            if not len(j):
                continue
            sub = {name:d[name][j] for name in base}
            sub['reference_kpc'] = ref[j]
            sub['cohort'] = np.full(len(j),label)
            sources.append(sub)
        inputs[str(receipt.relative_to(ROOT))] = sha(receipt)
    result = join(sources)
    sort = np.argsort(result['ls_id'])
    if np.any(np.diff(result['ls_id'][sort])<=0) or not np.all(result['TYPE']=='PSF'):
        raise ValueError('Branch case cohorts overlap or contain unverified sources')
    return subset(result,sort)


def photos(t, inputs):
    ref = EXPANSION/'south/reference'
    result = []
    for marker in sorted(ref.glob('[0-9][0-9].json')):
        record = json.loads(marker.read_text())
        for name,digest in record['outputs'].items():
            p = read_columns(verified(ref/name,digest),('ls_id','TYPE','sweep',*OBS))
            use = np.isin(p['ls_id'],t['ls_id'])
            if use.any():
                result.append(subset(p,use))
        inputs[str(marker.relative_to(ROOT))] = sha(marker)
    p = join(result)
    j = unique_join(t['ls_id'],p['ls_id'])
    if np.any(j<0) or not np.all(p['TYPE'][j]=='PSF'):
        raise ValueError('Missing unique native branch photometry')
    return subset(p,j)


def qrf_pdf(t, p, inputs):
    qrf = np.full((len(t['ls_id']),len(MU)),np.nan)
    for sweep in np.unique(p['sweep']):
        use = np.flatnonzero(p['sweep']==sweep)
        folder = EXPANSION/'south/predict'/sweep
        marker = folder/'complete.json'
        record = json.loads(marker.read_text())
        for name,digest in record['outputs'].items():
            if name.startswith('state_'):
                continue
            with np.load(verified(folder/name,digest),allow_pickle=False) as z:
                j = unique_join(t['ls_id'][use],z['ls_id'])
                a = use[j>=0]
                j = j[j>=0]
                if not len(j):
                    continue
                np.testing.assert_array_equal(z['TYPE'][j],t['TYPE'][a])
                np.testing.assert_array_equal(z['qrf_quantiles_kpc'][j],t['qrf_quantiles_kpc'][a])
                for i,x,m in zip(a,z['qrf_mu_edges'][j],z['qrf_mu_mass'][j]):
                    qrf[i] = histogram_on(EDGES,x,m)
        inputs[str(marker.relative_to(ROOT))] = sha(marker)
    np.testing.assert_allclose(qrf.sum(axis=1),1.,rtol=0,atol=1e-9)
    return qrf


def observation(p,i):
    obs = {k:p[k][i] for k in OBS}
    obs['photometric_system'] = 'DECam_DR11'
    return obs


def likelihood(i):
    model,weights,lf_weights,p,lon,lat = CONTEXT
    obs = observation(p,i)
    ll,_ = kernel_loglike(model,obs,MU,bands=USE_BANDS)
    frac = model.binary_reference_fraction
    flux = logsumexp(ll+np.log([1-frac,frac])[:,None,None],axis=0)
    with np.errstate(divide='ignore'):
        ordinary = logsumexp(flux[:,:,None]+np.log(weights)[:,None,:],axis=0).T
        lf = logsumexp(flux[:,:,None]+np.log(lf_weights)[:,None,:],axis=0).T
    disk,halo = radial_profiles(MU,lon[i],lat[i])
    # Nominal branch evidence uses the existing broad distance prior. The
    # LF view uses the RGB weighting and disk/halo ratios as sensitivity.
    nominal = ordinary+norm.logpdf(MU,15.,5.)[None,:]
    profiles = np.stack([.8*disk+.2*halo,.4*disk+.6*halo,.8*disk+.2*halo,disk])
    return nominal,lf+np.log(profiles)


def summarize(t,pdf):
    ref = t['reference_kpc']
    grid = np.cumsum(pdf,axis=-1)
    median = 10**((MU[(grid>=.5).argmax(axis=-1)]-10)/5)
    close = np.array([[window(EDGES,m,d) for m,d in zip(a,ref)] for a in pdf])
    delta = 5*np.log10(median/ref[None,:])
    return median,close,delta


def run():
    global CONTEXT
    inputs = {}
    t = cohorts(inputs)
    p = photos(t,inputs)
    base = qrf_pdf(t,p,inputs)
    record = read_frozen()['models']['nominal']
    model = inference_model(joblib.load(verified(ROOT/record['path'],record['sha256'])))
    raw = model.weights[:,None]*model.classes
    priors = raw.sum(axis=0)
    original = raw/priors
    rgb,_ = giant_luminosity_weights(model,.32,-1.4,.7)
    ids = np.where(t['source_id']>0,t['source_id'],-t['ls_id'])
    solo = unique_join(ids,ids)>=0
    train = (t['cohort']=='faint_low_luminosity')&(t['sky_role']<3)&solo
    anchors = choose(t['ls_id'],train,48,31034)
    if len(anchors)<16:
        raise ValueError('Low-luminosity anchors require an independent training group')
    made = patch(model,[observation(p,i) for i in anchors],
                 [dict(reference_kpc=t['reference_kpc'][i]) for i in anchors])
    n = len(model.weights)
    weights = np.zeros((len(made.weights),4))
    weights[:n,:3] = original
    weights[n:,3] = 1/len(anchors)
    alt = weights.copy()
    alt[:n,:3] = rgb
    lon,lat = galactic(t['RA'],t['DEC'])
    CONTEXT = made,weights,alt,p,lon,lat
    cpus = int(os.environ.get('SLURM_CPUS_PER_TASK',1))
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(cpus,16),
            mp_context=multiprocessing.get_context('fork')) as pool:
        ll = np.asarray(list(pool.map(likelihood,range(len(lon)),chunksize=12)))
    CONTEXT = None
    old = ll[:,0,:3]+np.log(priors)[None,:,None]
    original_pdf = np.exp(logsumexp(old,axis=1)-logsumexp(old,axis=(1,2))[:,None])
    candidates,classes = [],[]
    for low in WD_PRIORS:
        prior = np.r_[priors*(1-low),low]
        raw = ll[:,1]+np.log(prior)[None,:,None]
        joint = np.exp(raw-logsumexp(raw,axis=(1,2))[:,None,None])
        candidates.append(joint.sum(axis=1))
        classes.append(joint)
    _,close,_ = summarize(t,candidates)
    cal = (t['sky_role']==3)&solo
    groups = [cal&(t['cohort']==name) for name in np.unique(t['cohort'])]
    balanced_score = [np.mean([np.log(np.maximum(c[k],1e-12)).mean() for k in groups if k.any()])
                      for c in close]
    best = int(np.argmax(balanced_score))
    chosen_prior = np.r_[priors*(1-WD_PRIORS[best]),WD_PRIORS[best]]
    patched = ll[:,0]+np.log(chosen_prior)[None,:,None]
    patched_pdf = np.exp(logsumexp(patched,axis=1)-logsumexp(patched,axis=(1,2))[:,None])
    pdf = np.stack([base,original_pdf,patched_pdf,candidates[best]])
    labels = ('frozen_qrf','unpatched_sed','patched_nominal_prior','explicit_branches')
    medians,close,delta = summarize(t,pdf)
    stats = []
    for scope,take in (('calibration',cal),('held_out',(t['sky_role']==4)&solo)):
        for name in np.unique(t['cohort']):
            k = take&(t['cohort']==name)
            if k.any():
                for i,label in enumerate(labels):
                    stats.append(dict(scope=scope,cohort=name,model=label,n=int(k.sum()),
                        median_relative=float(np.median(medians[i,k]/t['reference_kpc'][k]-1)),
                        median_window_mass=float(np.median(close[i,k])),
                        mean_window_log_mass=float(np.log(np.maximum(close[i,k],1e-12)).mean()),
                        severe_fraction=float((abs(delta[i,k])>1).mean())))
    np.testing.assert_allclose(pdf.sum(axis=-1),1.,rtol=0,atol=1e-9)
    DEST.mkdir(parents=True,exist_ok=True)
    csv_rows(DEST/'comparisons.csv',stats)
    atomic_npz(DEST/'distance_pdf.npz',ls_id=t['ls_id'],TYPE=t['TYPE'],cohort=t['cohort'],
               reference_kpc=t['reference_kpc'],sky_role=t['sky_role'],training=train,
               anchor=np.isin(t['ls_id'],t['ls_id'][anchors]),
               mu_edges=EDGES,methods=np.array(labels),pdf=pdf,pdf_by_branch=classes[best])
    save_json(DEST/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,rows=len(lon),
        frozen_sed_sha256=record['sha256'],anchors=len(anchors),prior_alternatives=WD_PRIORS,
        prior_balanced_calibration_scores=balanced_score,chosen_prior=float(WD_PRIORS[best]),
        outputs={f.name:sha(f) for f in DEST.iterdir() if f.name!='complete.json'}))


if __name__=='__main__':
    require_compute()
    run()
