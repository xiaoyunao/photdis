#!/usr/bin/env python3
"""Transfer real faint-end measurement errors onto held-out native templates."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

import numpy as np

from photdis.reference_v1_4 import unique_join
from photdis.stage1 import galactic, require_compute
from photdis.v3.data import stable_rank
from photdis.v3.metrics import distance_summary
from photdis.v3_3 import frozen_qrf
from v2_1_near import atomic_npz, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_4_near_forecast import prepare
from v3_6_distance_evidence import OUT as EVIDENCE
from v3_6_reference_photometry import OUT as PHOTO
from v3_6_vac_distances import read_columns
from v3_8_candidate import read_cache
from v3_9_catalog import OUT, thin
from v3_9_references import join, subset

ROOT = OUT/'noise_control'


def faint_donors(source):
    base = OUT/'south/photo'/Path(source['filename']).stem
    marker = base/'complete.json'
    inputs = json.loads(marker.read_text())['outputs']
    parts = []
    keys = ('ls_id','TYPE','r','g0','r0','error_grz','row_state')
    for name,h in inputs.items():
        if not name.endswith('_south_faint.npz'):
            continue
        t = read_columns(verified(base/name,h),keys)
        if not np.all(t['TYPE']=='PSF'):
            raise ValueError('Faint measurement noise must come from verified PSF rows')
        for tier in (6,7):
            use = (t['row_state']==tier)&np.isfinite(t['error_grz']).all(axis=1)
            j,_ = thin(t,use,16,3940+tier)
            if len(j):
                parts.append(subset(t,j))
    return parts, marker


def donor_frame():
    # Draw from the photometric parent so a faint source need not have an
    # astrometric or spectroscopic identification to supply its measured error.
    inventory = OUT/'south/inventory.json'
    sample = []
    for group in json.loads(inventory.read_text())['groups']:
        selected = []
        for source in group:
            ra0,lo,ra1,hi = source['bounds_deg']
            ra = (ra0+.5*((ra1-ra0)%360))%360
            _,lat = galactic(np.array([ra]),np.array([(lo+hi)/2]))
            if abs(lat[0])>32:
                selected.append(source)
        order = sorted(selected,key=lambda s:s['filename'])
        if order:
            sample.extend(order[i] for i in np.unique(np.linspace(0,len(order)-1,3,dtype=int)))
    with ThreadPoolExecutor(max_workers=8) as pool:
        chunks = list(pool.map(faint_donors,sample))
    return join([p for parts,_ in chunks for p in parts]), [inventory,*[marker for _,marker in chunks]]


def main():
    require_compute()
    prior = read_columns(read_cache(EVIDENCE,'Gaia.npz'),('ls_id','TYPE','high_quality','qrf_training',
                    'reference_kpc','observed_r','consensus_class','dereddened_g_r'))
    phot = read_columns(read_cache(PHOTO,'photometry.npz'),('ls_id','TYPE','magnitude','magnitude_error'))
    j = unique_join(prior['ls_id'],phot['ls_id'])
    good = (j>=0)&(prior['TYPE']=='PSF')&prior['high_quality']&~prior['qrf_training']
    good &= (16<=prior['observed_r'])&(prior['observed_r']<18)&(prior['consensus_class']==0)
    i = np.flatnonzero(good)
    z = j[i]
    mag = phot['magnitude'][z][:,[0,1,3]]
    err = phot['magnitude_error'][z][:,[0,1,3]]
    use = np.isfinite(mag).all(axis=1)&np.isfinite(err).all(axis=1)&(err<.1).all(axis=1)
    i,mag,err = i[use],mag[use],err[use]
    if not np.all(phot['TYPE'][j[i]]=='PSF'):
        raise ValueError('Faint-noise templates must be native PSFs')
    donors, source_paths = donor_frame()
    if not donors:
        raise ValueError('Native sweep sampling found no qualifying faint noise donors')
    original,_,path = prepare()
    qrf = original['qrf']
    rng = np.random.default_rng(3937)
    summary,parts = [],[]
    for a,b in ((20,21),(21,22),(22,23)):
        for tier in (6,7):
            for low,high in ((-.5,.5),(.5,.8),(.8,1.1),(1.1,1.4),(1.4,2.)):
                template = np.flatnonzero((prior['dereddened_g_r'][i]>=low)&(prior['dereddened_g_r'][i]<high))
                color = donors['g0']-donors['r0']
                good = ((donors['r']>=a)&(donors['r']<b)&(donors['row_state']==tier)&
                        (color>=low)&(color<high))
                k = np.flatnonzero(good)
                if not len(k) or not len(template):
                    continue
                n = min(len(k),len(template),256)
                t = template[np.argsort(stable_rank(prior['ls_id'][i[template]],3938))[:n]]
                r = k[np.argsort(stable_rank(donors['ls_id'][k],3939))[:n]]
                ids = prior['ls_id'][i[t]]
                sigma = donors['error_grz'][r]
                attenuate = donors['r'][r]-prior['observed_r'][i[t]]
                reference = prior['reference_kpc'][i[t]]*10**(attenuate/5.)
                expected = mag[t]+attenuate[:,None]
                # Perturb in linear flux, then pass the measured magnitudes and
                # their propagated noise through the same frozen predictor.
                snr = (2.5/np.log(10))/sigma
                factor = 1+rng.normal(size=sigma.shape)/snr
                keep = (factor>0).all(axis=1)
                if not keep.any():
                    continue
                noisy = expected[keep]-2.5*np.log10(factor[keep])
                noisy_err = sigma[keep]/factor[keep]
                q0 = frozen_qrf(qrf,expected[keep],err[t][keep],ids[keep])
                q1 = frozen_qrf(qrf,noisy,noisy_err,ids[keep])
                kept = q0['supported']&q1['supported']
                dm = 10+5*np.log10(reference[keep])
                for model,q in (('without_added_faint_noise',q0),('with_measured_faint_noise',q1)):
                    summary.append(dict(model=model,r_min=a,r_max=b,color_min=low,color_max=high,
                        observed_error_tier=tier,simulations=n,common_support=int(kept.sum()),
                        **distance_summary(10+5*np.log10(q['distance'][kept]),dm[kept])))
                parts.append(dict(template_ls_id=ids[keep],donor_ls_id=donors['ls_id'][r][keep],
                    TYPE=np.full(keep.sum(),'PSF'),r_min=np.full(keep.sum(),a),
                    donor_error_tier=np.full(keep.sum(),tier),reference_kpc=reference[keep],
                    original_kpc=q0['distance'],perturbed_kpc=q1['distance'],support=kept))
    if not summary:
        raise ValueError('No actual faint photometric-error donor or whole template met the pilot bounds')
    ROOT.mkdir(parents=True,exist_ok=True)
    csv_rows(ROOT/'noise_cells.csv',summary)
    atomic_npz(ROOT/'simulations.npz',**join(parts))
    save_json(ROOT/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),
        interpretation='Same-template perturbation conditional on the measured faint-error donors',
        inputs={str(x):sha(x) for x in (EVIDENCE/'complete.json',PHOTO/'complete.json',*source_paths,*path)},
        outputs={p.name:sha(p) for p in ROOT.iterdir() if p.name!='complete.json'}))


if __name__=='__main__':
    main()
