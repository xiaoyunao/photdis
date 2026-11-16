#!/usr/bin/env python3
"""Separate branch shape from component abundance and export a PDF control."""
import json
import os
from pathlib import Path

import joblib
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

from photdis.reference_v1_4 import unique_join
from photdis.stage1 import require_compute
from photdis.v3.inference import kernel_loglike
from photdis.v3.model import inference_model
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_3_catalog import FREEZE, read_frozen
from v3_3_sed import verified
from v3_5_dr11_spatial import DELIVERY
from v3_5_dr11_validate import join_parts
from v3_6_catalog_projection import OUT as PROJECTION
from v3_6_vac_distances import read_columns
from v3_6_weak_gaia import native_qrf
from v3_8_branches import LABEL, OUT as V38, PHOTO, USE_BANDS, choose, observation, patch
from v3_8_candidate import read_cache

OUT = ROOT/'data/v3_9/branches'
BANDS = ('g','r','z')


def colors(mag):
    return mag[:,:-1]-mag[:,1:]


def production_pdf(ids):
    pointer = ('ls_id','TYPE','sweep_number','part_number','part_row')
    parents = json.loads((PROJECTION/'complete.json').read_text())['child_receipts']
    parts = []
    for key,digest in parents.items():
        folder = verified(ROOT/key,digest).parent
        p = read_columns(read_cache(folder,'rows.npz'),pointer)
        use = np.isin(p['ls_id'],ids)
        if use.any():
            parts.append({k:v[use] for k,v in p.items()})
    p = join_parts(parts)
    if len(p.get('ls_id',[]))!=len(ids):
        raise ValueError('The branch sample must map once to the native production PDF')
    j = unique_join(ids,p['ls_id'])
    if np.any(j<0) or not (p['TYPE'][j]=='PSF').all():
        raise ValueError('Invalid production PDF ownership')
    return native_qrf({k:v[j] for k,v in p.items()},DELIVERY/'complete.json')


def histogram_on(edges, old_edges, old_mass):
    cumulative = np.cumsum(old_mass)
    cdf = np.interp(edges,old_edges,np.r_[0.,cumulative/cumulative[-1]],left=0.,right=1.)
    return np.maximum(np.diff(cdf),0.)


def anchor_control(model, obs, edges, qrf_mass, gate, start):
    """Observable color gate; no held-out distances or labels enter the PDF."""
    centers = (edges[1:]+edges[:-1])/2
    likelihood,ix = kernel_loglike(model,{**obs,'band_mask':obs['band_mask']&USE_BANDS},
                                  centers,candidates=np.arange(start,len(model.weights)))
    binary = np.log([1-model.binary_reference_fraction,model.binary_reference_fraction])
    logp = logsumexp(likelihood+binary[:,None,None]+np.log(model.weights[ix])[None,:,None],
                    axis=(0,1))+norm.logpdf(centers,15.,5.)+np.log(np.diff(edges))
    anchor = np.exp(logp-logsumexp(logp))
    return (1-gate)*qrf_mass+gate*anchor, anchor


def window(edges, mass, distance):
    mu = 10+5*np.log10(distance)
    cdf = np.interp([mu-.5,mu+.5],edges,np.r_[0,np.cumsum(mass)])
    return float(np.diff(cdf)[0])


def component_diagnostics(model, obs, row, kind, distance):
    mu = 10+5*np.log10(distance)
    likelihood,_ = kernel_loglike(model,{**obs,'band_mask':obs['band_mask']&USE_BANDS},
                                  np.array([mu]))
    state = int(row['best_original_component'])
    if state<0:
        return []
    binary = int(np.argmax(likelihood[:,state,0]))
    brightness = model.binary_brightness_factor if binary else 1.
    brightness *= (1./distance**2)
    trans = obs['transmission'][USE_BANDS]
    pred = model.means[state,:6][USE_BANDS]*brightness*trans
    intrinsic_sigma = np.sqrt(model.covariances[state,:6,:6].diagonal()[USE_BANDS])*brightness*trans
    measurement = obs['flux_error'][USE_BANDS]
    value = obs['flux'][USE_BANDS]
    results = []
    for j,band in enumerate(BANDS):
        omitted = USE_BANDS.copy()
        omitted[np.flatnonzero(USE_BANDS)[j]] = False
        reduced,_ = kernel_loglike(model,{**obs,'band_mask':obs['band_mask']&omitted},
                                   np.array([mu]),candidates=[state])
        results.append(dict(ls_id=int(row['ls_id']),TYPE='PSF',branch=kind,band=band,
            compared_distance_kpc=float(distance),component=state,binary_state=binary,
            measured_flux=float(value[j]),predicted_flux=float(pred[j]),
            measurement_sigma=float(measurement[j]),component_sigma=float(intrinsic_sigma[j]),
            normalized_marginal_residual=float((value[j]-pred[j])/np.hypot(intrinsic_sigma[j],measurement[j])),
            component_weight=float(model.weights[state]),
            weighted_class_prior=float(model.weights[state]*model.classes[state,LABEL[row['cohort']]]),
            unweighted_log_likelihood=float(likelihood[binary,state,0]),
            conditional_band_log_likelihood=float(likelihood[binary,state,0]-reduced[binary,0,0])))
    return results


def main():
    require_compute()
    import csv
    OUT.mkdir(parents=True,exist_ok=True)
    marker = OUT/'complete.json'
    if marker.exists():
        raise FileExistsError(marker)
    anchors,test,_ = choose()
    all_ids = np.asarray([r['ls_id'] for r in anchors+test])
    photo = read_columns(read_cache(PHOTO,'photometry.npz'),
        ('ls_id','TYPE','flux','flux_ivar','band_mask','transmission','EBV'))
    j = unique_join(all_ids,photo['ls_id'])
    if np.any(j<0) or not np.all(photo['TYPE'][j]=='PSF'):
        raise ValueError('The frozen branch tests require verified PSFs')
    obs = [observation(photo,i) for i in j]
    with np.errstate(divide='ignore',invalid='ignore'):
        mag = 22.5-2.5*np.log10(photo['flux'][j][:,USE_BANDS]/photo['transmission'][j][:,USE_BANDS])
        err = (2.5/np.log(10))/(photo['flux'][j][:,USE_BANDS]*np.sqrt(photo['flux_ivar'][j][:,USE_BANDS]))
    if not np.isfinite(mag).all() or not np.isfinite(err).all():
        raise ValueError('These grz branch controls must have measured optical colors')
    inf = read_frozen()['models']['nominal']
    model = inference_model(joblib.load(verified(ROOT/inf['path'],inf['sha256'])))
    patched = patch(model,obs[:len(anchors)],anchors)
    n = len(anchors)
    archived = production_pdf(all_ids[n:])
    with read_cache(V38,'cases.csv').open() as stream:
        cases = {int(r['ls_id']):r for r in csv.DictReader(stream)}
    edges = np.linspace(-10,40,501)
    output, stats, bands = [],[],[]
    for i,(r,x) in enumerate(zip(test,obs[n:])):
        truth, pred = r['reference_kpc'],r['qrf_median_kpc']
        case = cases[r['ls_id']]
        for kind,distance in (('reference',truth),('production_median',pred)):
            bands.extend(component_diagnostics(model,x,case,kind,distance))
        sep = np.linalg.norm(colors(mag[:n])-colors(mag[n+i:n+i+1]),axis=1).min()
        gate = .20*np.exp(-.5*(sep/.12)**2)
        original = histogram_on(edges,*archived[i])
        if not np.isclose(original.sum(),1.,atol=1e-8):
            raise ValueError('Common output grid truncates frozen QRF mass')
        candidate,anchor = anchor_control(patched,x,edges,original,gate,len(model.weights))
        np.testing.assert_allclose([original.sum(),candidate.sum(),anchor.sum()],1.,atol=1e-8)
        output.append(np.stack([original,candidate,anchor]))
        stats.append(dict(**r,color_distance_to_anchor=float(sep),mixture_weight=float(gate),
                          original_reference_mass=window(edges,original,truth),
                          candidate_reference_mass=window(edges,candidate,truth),
                          original_branch_mass=window(edges,original,pred),
                          candidate_branch_mass=window(edges,candidate,pred)))
    csv_rows(OUT/'cases.csv',stats)
    csv_rows(OUT/'band_residuals.csv',bands)
    atomic_npz(OUT/'distance_pdf_control.npz',ls_id=all_ids[n:],TYPE=photo['TYPE'][j[n:]],
               mu_edges=edges,mu_mass=np.stack(output),pdf_names=['resampled_qrf','candidate','anchor_only'],
               frozen_mu_edges=np.stack([a for a,_ in archived]),frozen_mu_mass=np.stack([p for _,p in archived]))
    paths = (Path(__file__),V38/'complete.json',PHOTO/'complete.json',FREEZE,
             PROJECTION/'complete.json',DELIVERY/'complete.json')
    save_json(marker,dict(inputs={str(p.relative_to(ROOT)):sha(p) for p in paths},
              outputs={p.name:sha(p) for p in OUT.iterdir() if p!=marker},
              test=len(test),anchor=n,job=os.getenv('SLURM_JOB_ID'),
              candidate_state='development comparison; no change to production PDF'))


if __name__=='__main__':
    main()
