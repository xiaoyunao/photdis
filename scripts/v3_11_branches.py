#!/usr/bin/env python3
"""Locate the loss of likelihood or prior weight for held-out luminosity branches."""

import concurrent.futures
import csv
import json
import multiprocessing
import os

import joblib
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

from photdis.reference_v1_4 import unique_join
from photdis.stage1 import require_compute
from photdis.v3.inference import kernel_loglike
from photdis.v3.model import inference_model
from v2_1_near import ROOT, csv_rows, save_json, sha
from v3_3_catalog import read_frozen
from v3_3_sed import verified
from v3_6_vac_distances import read_columns
from v3_8_branches import USE_BANDS, patch
from v3_10_branches import DEST as V310, cohorts, observation, photos

DEST = ROOT/'data/v3_11/branches'
BANDS = ('grz', 'i', 'W1', 'W2')
OPTIONAL = (2,4,5)
STEP = .10
MU = np.arange(-5.+STEP/2,26.,STEP)
WINDOW = .5
CONTEXT = None


def band_model(original, p, t, anchors, band):
    model = patch(original,[observation(p,i) for i in anchors],
                  [dict(reference_kpc=t['reference_kpc'][i]) for i in anchors])
    if band is not None:
        flux = p['flux'][anchors,band] / p['transmission'][anchors,band]
        scale = np.square(t['reference_kpc'][anchors])
        intrinsic_error = p['flux_error'][anchors,band] / p['transmission'][anchors,band] * scale
        model.means[-len(anchors):,band] = flux*scale
        model.covariances[-len(anchors):,band,band] = intrinsic_error**2 + (.15*flux*scale)**2 + 1e-8
    return model


def odds(model, obs, bands, truth, alternative, low, truth_low, mu):
    ll,_ = kernel_loglike(model,obs,mu,bands=bands)
    binary = model.binary_reference_fraction
    kernels = logsumexp(ll+np.log([1-binary,binary])[:,None,None],axis=0)
    n = len(model.weights)-low
    weights = model.weights[:n]/model.weights[:n].sum()
    branch = np.stack([logsumexp(kernels[:n]+np.log(weights)[:,None],axis=0),
                       logsumexp(kernels[n:],axis=0)-np.log(low)])
    prior_class = np.log([.99,.01])[:,None]
    prior_distance = norm.logpdf(mu,15.,5.)
    post = branch + prior_class + prior_distance
    correct = np.abs(mu-truth) <= WINDOW
    wrong = np.abs(mu-alternative) <= WINDOW
    offsets = np.log(mu[1]-mu[0])
    true_branch = int(truth_low)
    correct_likelihood = logsumexp(branch[true_branch,correct])+offsets
    all_evidence = logsumexp(post)
    correct_log_post = logsumexp(post[true_branch,correct])-all_evidence
    correct_distance_mass = np.exp(logsumexp(post[:,correct])-all_evidence)
    correct_kernels = kernels[n:] if truth_low else kernels[:n]
    best = correct_kernels[:,correct].max(axis=1)
    result = [correct_likelihood,np.nan,np.nan,np.nan,
              np.exp(correct_log_post),correct_distance_mass,
              best.max(),np.mean(best>=best.max()-2.)]
    if abs(truth-alternative)>2*WINDOW and wrong.any():
        wrong_likelihood = logsumexp(branch[0,wrong])+offsets
        post_odds = correct_log_post-(logsumexp(post[0,wrong])-all_evidence)
        odds_likelihood = correct_likelihood-wrong_likelihood
        result[1:4] = [wrong_likelihood,odds_likelihood,post_odds-odds_likelihood]
    return result


def evaluate(i):
    p,t,models,measured,label = CONTEXT
    truth = 10+5*np.log10(t['reference_kpc'][i])
    alt = 10+5*np.log10(t['qrf_quantiles_kpc'][i,2])
    obs = observation(p,i)
    primary,low = models['grz']
    out = [('grz','optical',odds(primary,obs,USE_BANDS,truth,alt,low,label[i],MU))]
    for name,band in zip(BANDS[1:],OPTIONAL):
        if name not in models or not measured[i,band]:
            continue
        m,n = models[name]
        extra = USE_BANDS.copy()
        extra[band] = True
        out.append((name,'optical',odds(m,obs,USE_BANDS,truth,alt,n,label[i],MU)))
        out.append((name,'plus_band',odds(m,obs,extra,truth,alt,n,label[i],MU)))
    return i,out


def run():
    global CONTEXT
    inputs = {}
    marker = V310/'complete.json'
    branch = read_columns(verified(V310/'distance_pdf.npz',json.loads(marker.read_text())[
        'outputs']['distance_pdf.npz']),('TYPE','ls_id','anchor'))
    inputs[str(marker.relative_to(ROOT))] = sha(marker)
    t = cohorts(inputs)
    np.testing.assert_array_equal(t['ls_id'],branch['ls_id'])
    np.testing.assert_array_equal(t['TYPE'],branch['TYPE'])
    p = photos(t,inputs)
    path = ROOT/'manifests/v3_10/branch_cases/wd/wd_evidence.csv'
    with path.open() as f:
        rows = list(csv.DictReader(f))
    inputs[str(path.relative_to(ROOT))] = sha(path)
    high = {int(r['ls_id']) for r in rows if r['published_pwd'] and float(r['published_pwd'])>.75}
    if not high:
        raise ValueError('No published high-confidence WD entries in the preserved source table')
    wd = np.isin(t['ls_id'],list(high)) & (t['cohort']=='faint_low_luminosity')
    confirmed = {int(r['ls_id']) for r in rows if r['spectroscopic_wd']=='True'}
    spec_wd = np.isin(t['ls_id'],list(confirmed)) & (t['cohort']=='faint_low_luminosity')
    wd |= spec_wd
    group = np.where(spec_wd,'spectroscopic_wd',np.where(wd,'published_wd_candidate',t['cohort']))
    use = (t['sky_role']==4)&(wd | np.isin(group,['blue_2_5','faint_far_dwarfs','ordinary_dwarfs']))
    identity = np.where(t['source_id']>0,t['source_id'],-t['ls_id'])
    use &= unique_join(identity,identity)>=0
    observed = p['band_mask'] & np.isfinite(p['flux']) & np.isfinite(p['flux_error'])
    observed &= (p['flux_error']>0)&np.isfinite(p['transmission'])&(p['transmission']>0)
    record = read_frozen()['models']['nominal']
    original = inference_model(joblib.load(verified(ROOT/record['path'],record['sha256'])))
    anchors = np.flatnonzero(branch['anchor'])
    models = {'grz':(band_model(original,p,t,anchors,None),len(anchors))}
    counts = {'grz':len(anchors)}
    for name,band in zip(BANDS[1:],OPTIONAL):
        a = anchors[observed[anchors,band]]
        counts[name] = len(a)
        if len(a)>=8:
            models[name] = band_model(original,p,t,a,band),len(a)
    CONTEXT = p,t,models,observed,wd
    columns = ('correct_log_evidence','alternative_log_evidence','log_likelihood_odds','log_prior_odds_effect',
               'correct_joint_mass','correct_distance_mass','best_template_log_likelihood','template_fraction_near_best')
    rows = []
    cpus = int(os.environ.get('SLURM_CPUS_PER_TASK',1))
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(cpus,24),
            mp_context=multiprocessing.get_context('fork')) as pool:
        for i,out in pool.map(evaluate,np.flatnonzero(use),chunksize=4):
            for test,mode,values in out:
                rows.append(dict(ls_id=int(t['ls_id'][i]),TYPE=t['TYPE'][i],cohort=group[i],
                                 test=test,mode=mode,**dict(zip(columns,values))))
    CONTEXT = None
    stats = []
    for band,j in zip(BANDS[1:],OPTIONAL):
        for name in np.unique(group[use]):
            k = use & (group==name)
            stats.append(dict(test=band,cohort=name,eligible=int(k.sum()),measured=int((k&observed[:,j]).sum()),
                              anchors=counts[band],test_ran=band in models))
    DEST.mkdir(parents=True,exist_ok=True)
    csv_rows(DEST/'evidence.csv',rows)
    csv_rows(DEST/'coverage.csv',stats)
    save_json(DEST/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,
        wd_candidates=int(wd.sum()),held_out=int(use.sum()),anchor_counts=counts,
        prior_wd=.01,reference_prior_mu=[15.,5.],window_half_width_mag=WINDOW,
        frozen_sed=record['sha256'],outputs={f:sha(DEST/f) for f in ('evidence.csv','coverage.csv')}))


if __name__=='__main__':
    require_compute()
    run()
