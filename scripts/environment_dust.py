#!/usr/bin/env python3
"""Finite-distance Bayestar input sensitivity with the same frozen QRF."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path

from astropy.coordinates import SkyCoord
import numpy as np

from environment_pilot import C, OUT, finish, sample
from photdis.baseline_v0 import Q
from photdis.baseline_v1 import support
from photdis.stage1 import require_compute
from photdis.v3_3 import frozen_qrf
from v2_1_near import atomic_npz, load, sha
from v3_4_near_forecast import prepare
from v3_9_dust import MU, dust_profiles
from v3_10_dust import BAYESTAR_TO_QUOTED_EBV, DECam_PER_BAYESTAR

STATE = None


def modulus_quantiles(mass, edges):
    """Inverse CDF on each star's own modulus bins, including clipped bins."""
    cumulative = np.cumsum(mass, axis=1)
    cumulative /= cumulative[:, -1, None]
    idx = (cumulative[:, :, None] >= Q).argmax(axis=1)
    row = np.arange(len(mass))[:, None]
    lower = np.where(idx>0, cumulative[row,idx-1], 0.)
    upper = cumulative[row,idx]
    fraction = np.divide(Q-lower,upper-lower,out=np.zeros_like(lower),where=upper>lower)
    return edges[row,idx] + np.clip(fraction,0,1)*(edges[row,idx+1]-edges[row,idx])


def restrict(edges, mass, lo, hi):
    width = np.diff(edges, axis=1)
    confined = np.clip(edges, lo[:, None], hi[:, None])
    retained = mass * (np.diff(confined, axis=1) / width)
    total = retained.sum(axis=1)
    np.divide(retained, total[:, None], out=retained, where=total[:, None]>0)
    return confined, retained, total


def compute(j):
    model, mag, err, ids, profiles, left, right, obs, ref, base_edges, base_mass = STATE
    lo, hi = MU[left[j]], MU[right[j]]
    base_edge, base_weight, base_probability = restrict(base_edges[j],base_mass[j],lo,hi)
    shape = base_weight.shape
    n = len(j)
    outputs = dict(base_edges=base_edge,base_mass=base_weight,base_probability=base_probability,
                   iteration_edges=np.full(base_edge.shape,np.nan),iteration_mass=np.full(shape,np.nan),
                   iteration_state=np.zeros(n,'i1'),iteration_steps=np.zeros(n,'i1'),
                   iteration_probability=np.full(n,np.nan),iteration_ebv=np.full(n,np.nan),
                   oracle_center=np.full(n,np.nan),oracle_state=np.zeros(n,'i1'),
                   oracle_ebv=np.full(n,np.nan),mu_min=lo,mu_max=hi)
    valid = right[j]>left[j]
    threshold = C['dust']['minimum_histogram_mass']
    active = valid & (base_probability>=threshold)
    outputs['iteration_state'][valid] = 1
    track = np.full(n,np.nan)
    track[active] = modulus_quantiles(base_weight[active],base_edge[active])[:,2]
    for step in range(C['dust']['max_steps']):
        k = np.flatnonzero(active)
        if not len(k):
            break
        reddening = np.array([np.interp(y,MU,profiles[j[i]]) for i,y in zip(k,track[k])])
        corrected = obs[j[k]]-reddening[:,None]*DECam_PER_BAYESTAR
        supported = support(model,np.pad(corrected,((0,0),(0,2))))
        outputs['iteration_state'][k] = 2
        outputs['iteration_steps'][k] = step+1
        active[k[~supported]] = False
        k = k[supported]
        if not len(k):
            continue
        q = frozen_qrf(model,corrected[supported],err[j[k]],ids[j[k]])
        edges, weights, probability = restrict(q['mu_edges'],q['mass'],lo[k],hi[k])
        outputs['iteration_probability'][k] = probability
        enough = probability>=threshold
        outputs['iteration_state'][k] = 3
        active[k[~enough]] = False
        k = k[enough]
        new = modulus_quantiles(weights[enough],edges[enough])[:,2]
        converged = abs(new-track[k])<=C['dust']['tolerance_mag']
        done = k[converged]
        outputs['iteration_state'][k] = 4
        outputs['iteration_state'][done] = 5
        outputs['iteration_edges'][done] = edges[enough][converged]
        outputs['iteration_mass'][done] = weights[enough][converged]
        outputs['iteration_ebv'][done] = reddening[supported][enough][converged]
        active[done] = False
        track[k] += C['dust']['damping']*(new-track[k])
    # This arm sees the reference distance and is only a scale sensitivity.
    eligible = valid & (ref[j]>=lo) & (ref[j]<=hi)
    k = np.flatnonzero(eligible)
    if len(k):
        reddening = np.array([np.interp(ref[j[i]],MU,profiles[j[i]]) for i in k])
        corrected = obs[j[k]]-reddening[:,None]*DECam_PER_BAYESTAR
        supported = support(model,np.pad(corrected,((0,0),(0,2))))
        outputs['oracle_state'][k] = 1
        k = k[supported]
        q = frozen_qrf(model,corrected[supported],err[j[k]],ids[j[k]]) if len(k) else None
        if len(k):
            edges, weights, probability = restrict(q['mu_edges'],q['mass'],lo[k],hi[k])
            enough = probability>=threshold
            outputs['oracle_state'][k] = 2
            k = k[enough]
            outputs['oracle_center'][k] = modulus_quantiles(weights[enough],edges[enough])[:,2]
            outputs['oracle_ebv'][k] = reddening[supported][enough]
            outputs['oracle_state'][k] = 3
    return outputs


def main():
    global STATE
    require_compute()
    dest = OUT/'dust'
    dest.mkdir(parents=True, exist_ok=True)
    t = load(OUT/'forecast/roster.npz')
    pdf = load(OUT/'forecast/pdf.npz')
    photo = load(OUT/'census/qualified.npz')
    use = [sample(t['ls_id'],(t['field']==name)&t['gaia_evaluation'],
                  C['bayestar_iteration_per_field'],20) for name,_,_ in C['fields']]
    j = np.concatenate(use)
    assert len(j) and (t['TYPE'][j]=='PSF').all()
    order = np.searchsorted(pdf['rows'],j)
    np.testing.assert_array_equal(pdf['rows'][order],j)
    coords = SkyCoord(t['RA'][j], t['DEC'][j], unit='deg', frame='icrs')
    path = Path(C['dust']['map'])
    digest = sha(path)
    assert digest == C['dust']['map_sha256'], digest
    maps, reliable, flags = dust_profiles(coords,path,'bayestar19')
    good = reliable & np.isfinite(maps).all(axis=1) & (maps>=0).all(axis=1)
    left = np.argmax(good,axis=1)
    right = len(MU)-1-np.argmax(good[:,::-1],axis=1)
    interval = (np.arange(len(MU))[None,:]>=left[:,None]) & (np.arange(len(MU))[None,:]<=right[:,None])
    contiguous = np.all(good==interval,axis=1)&(good.sum(axis=1)>=3)
    left = np.where(contiguous,left,0)
    right = np.where(contiguous,right,0)
    profile = np.median(maps,axis=1)/BAYESTAR_TO_QUOTED_EBV
    sfd = t['mag_grz'][j]
    observed = sfd - 2.5*np.log10(photo['transmission'][j][:,[0,1,3]])
    mu = 10+5*np.log10(t['gaia_distance_kpc'][j])
    models, _, paths = prepare()
    STATE = (models['qrf'], sfd, t['error_grz'][j], photo['identity'][j],
             profile, left, right, observed, mu, pdf['mu_edges'][order],pdf['mass'][order])
    chunks = np.array_split(np.arange(len(j)),max(1,int(np.ceil(len(j)/48))))
    with ProcessPoolExecutor(max_workers=16,mp_context=multiprocessing.get_context('fork')) as pool:
        pieces = list(pool.map(compute,chunks))
    result = {k:np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
    result.update(rows=j,TYPE=t['TYPE'][j],ls_id=t['ls_id'][j],map_reliable=good,
                  ebv_profile=profile,map_flags=flags,reference_mu=mu)
    atomic_npz(dest/'forecasts.npz',**result)
    finish(dest,{'forecast':sha(OUT/'forecast/complete.json'),'map':digest,
                 'script':sha(Path(__file__))})
    print('Bayestar',len(j),'query rows; iteration states',np.bincount(result['iteration_state']),
          'known distance states',np.bincount(result['oracle_state']),flush=True)


if __name__=='__main__':
    main()
