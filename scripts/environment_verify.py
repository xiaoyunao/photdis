#!/usr/bin/env python3
"""Check histogram scores, catalogue conservation and the 3D forecast replay."""
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr

import environment_dust as dust
from environment_pilot import C, CONFIG, OUT
from environment_results import histogram_scores
from photdis.stage1 import require_compute
from v2_1_near import ROOT, load, save_json, sha
from v3_3_sed import verified
from v3_4_near_forecast import prepare
from v3_9_dust import MU
from v3_10_figures import check_templates


def numeric():
    e = np.array([[-1.7, -.3, .8, 3.1], [1., 1., 2., 2.]])
    w = np.array([[.3,.55,.15],[0.,1.,0.]])
    y, sigma = np.array([.12,1.25]), np.array([.23,.100])
    log, crps = histogram_scores(e,w,y,sigma)
    errors = []
    for i,(edges,mass,target,err) in enumerate(zip(e,w,y,sigma)):
        widths=np.diff(edges)
        k=widths>0
        density=mass[k]/widths[k]
        center=edges[:-1][k]
        def cdf(x):
            return np.sum(mass[k]*np.clip((x-center)/widths[k],0,1))
        actual_crps = quad(lambda x: (cdf(x)-(x>=target))**2, edges[0], edges[-1],
                           points=sorted(set(np.r_[edges,target])))[0]
        actual_log = np.log(np.sum(density*(ndtr((edges[1:][k]-target)/err)
                                          -ndtr((center-target)/err))))
        errors.append([abs(actual_log-log[i]),abs(actual_crps-crps[i])])
    assert np.max(errors)<1e-10
    np.testing.assert_allclose(dust.modulus_quantiles(w,e)[1],[1.025,1.16,1.5,1.84,1.975])
    return errors


def main():
    require_compute()
    report = {'analytic_vs_quadrature_score_errors':numeric()}
    stages = ('census','forecast','dust','apogee','results','figures')
    receipts={}
    for name in stages:
        p=OUT/name/'complete.json'
        m=json.loads(p.read_text())
        assert m['config_sha256']==sha(CONFIG)
        for filename,digest in m['outputs'].items():
            verified(p.parent/filename,digest)
        receipts[name]=sha(p)
    t=load(OUT/'forecast/roster.npz')
    q=load(OUT/'forecast/pdf.npz')
    d=load(OUT/'dust/forecasts.npz')
    p=load(OUT/'census/qualified.npz')
    a=load(OUT/'apogee/references.npz')
    assert len(np.unique(t['ls_id']))==len(t['ls_id'])
    assert (t['TYPE']=='PSF').all() and (q['TYPE']=='PSF').all() and (d['TYPE']=='PSF').all()
    assert not (t['qrf_fit_overlap']&(t['gaia_evaluation']|t['specdis_evaluation']|t['spdist_evaluation'])).any()
    assert not (t['qrf_fit_overlap']&a['astro_evaluation']).any()
    np.testing.assert_array_equal(t['ls_id'][q['rows']],q['ls_id'])
    np.testing.assert_array_equal(t['ls_id'][d['rows']],d['ls_id'])
    quant=dust.modulus_quantiles(q['mass'],q['mu_edges'])
    with np.errstate(invalid='ignore'):
        err=abs(quant[:,2]-(10+5*np.log10(q['distance'][:,2])))
    assert err.max()<2e-5,err.max()
    model=prepare()[0]['qrf']
    converged=(d['iteration_state']==5)
    ii=np.flatnonzero(converged)[::max(1,int(converged.sum()/8))][:8]
    assert len(ii)
    j=d['rows'][ii]
    k=np.searchsorted(q['rows'],j)
    lo=np.searchsorted(MU,d['mu_min'][ii])
    hi=np.searchsorted(MU,d['mu_max'][ii])
    mag=t['mag_grz'][j]
    obs=mag-2.5*np.log10(p['transmission'][j][:,[0,1,3]])
    reddened=p['EBV']>.02
    extinction=-2.5*np.log10(p['transmission'][reddened][:,[0,1,3]])/p['EBV'][reddened,None]
    state=(model,mag,t['error_grz'][j],p['identity'][j],d['ebv_profile'][ii],lo,hi,obs)
    dust.STATE=(*state,d['reference_mu'][ii]+3,q['mu_edges'][k],q['mass'][k])
    replay=dust.compute(np.arange(len(ii)))
    for key in ('iteration_edges','iteration_mass','iteration_state','iteration_steps'):
        np.testing.assert_array_equal(replay[key],d[key][ii])
    sky=ROOT/'figures/sky_sequence_20261119/selected_channels/rgb_selected_fixed_gain.png'
    assert sha(sky)=='411b0086e9d4c6f9f208805f02f064e2c24bb117d61293083b9bb51dc7faaed5'
    report.update(receipts=receipts,precise_catalog_n=len(t['ls_id']),pdf_n=len(q['rows']),
        gaia_prediction_n=int(t['gaia_evaluation'].sum()),bayestar_converged=int(converged.sum()),
        supplemental_astro_reference_n=int(a['astro_evaluation'].sum()),
        median_original_grz_extinction_per_SFD_EBV=np.median(extinction,axis=0),
        frozen_median_replay_max_mag=float(err.max()),dust_reference_independence_replay_n=len(ii),
        origin_production_reference=C['baseline'],sky_sha256=sha(sky),style=check_templates())
    save_json(OUT/'validation.json',report)
    print('All bound stage artifacts, quantile replay and reference independence checks passed',flush=True)


if __name__=='__main__':
    main()
