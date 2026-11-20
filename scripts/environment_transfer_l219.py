#!/usr/bin/env python3
"""Instrument the identical frozen L219 cohort and its archived iteration."""
import csv
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

from astropy.coordinates import SkyCoord
import numpy as np
from scipy.spatial import cKDTree

import environment_dust as dust
import environment_pilot as pilot
import environment_results as stats
from environment_transfer import OLD, TABLES, configure
from photdis.stage1 import legacy_ids, require_compute
from v2_1_near import atomic_npz, csv_rows, load, sha

STATE = None
STATUS = ('Map unavailable', 'Low initial mass', 'Colors unsupported',
          'Low updated mass', 'Step limit', 'Converged')


def trace(j):
    model, photo, old, observed = STATE
    lo, hi = old['mu_min'][j], old['mu_max'][j]
    previous = np.full(len(j), np.nan)
    state = (hi > lo).astype('i1')
    use = (state == 1) & (old['base_probability'][j] >= dust.C['dust']['minimum_histogram_mass'])
    previous[use] = dust.modulus_quantiles(old['base_mass'][j[use]], old['base_edges'][j[use]])[:,2]
    cfg = dust.C['dust']
    records = np.full((len(j), cfg['max_steps'], 6), np.nan)
    edges_out = np.full_like(old['iteration_edges'][j], np.nan)
    mass_out = np.full_like(old['iteration_mass'][j], np.nan)
    steps = np.zeros(len(j), 'i1')
    for step in range(cfg['max_steps']):
        k = np.flatnonzero(use)
        if not len(k):
            break
        ebv = np.array([np.interp(previous[i], dust.MU, old['ebv_profile'][j[i]]) for i in k])
        records[k,step,:2] = np.column_stack((previous[k], ebv))
        corrected = observed[j[k]] - ebv[:,None]*dust.DECam_PER_BAYESTAR
        supported = pilot.support(model, np.pad(corrected, ((0,0),(0,2))))
        state[k], steps[k], use[k[~supported]] = 2, step+1, False
        k = k[supported]
        if not len(k):
            continue
        q = dust.frozen_qrf(model, corrected[supported], photo['error_grz'][j[k]], photo['identity'][j[k]])
        edges, mass, prob = dust.restrict(q['mu_edges'], q['mass'], lo[k], hi[k])
        enough = prob >= cfg['minimum_histogram_mass']
        records[k,step,3] = prob
        state[k], use[k[~enough]] = 3, False
        k = k[enough]
        quantiles = dust.modulus_quantiles(mass[enough], edges[enough])
        nxt = quantiles[:,2]
        records[k,step,2] = nxt
        records[k,step,4] = quantiles[:,0]
        records[k,step,5] = quantiles[:,4]
        conv = abs(nxt-previous[k]) <= cfg['tolerance_mag']
        done = k[conv]
        state[k], state[done], use[done] = 4, 5, False
        edges_out[done] = edges[enough][conv]
        mass_out[done] = mass[enough][conv]
        previous[k] += cfg['damping']*(nxt-previous[k])
    np.testing.assert_array_equal(state, old['iteration_state'][j])
    np.testing.assert_array_equal(steps, old['iteration_steps'][j])
    np.testing.assert_allclose(edges_out, old['iteration_edges'][j], rtol=0, atol=1e-11)
    np.testing.assert_allclose(mass_out, old['iteration_mass'][j], rtol=0, atol=1e-11)
    return records


def raw_quality(photo):
    names = [f'{k}_{b}' for k in ('FRACFLUX','FRACMASKED','FRACIN') for b in 'GRZ']
    raw_set = pilot.settings()['south']['sweep_root']
    with (OLD/'census/raw_admission.csv').open() as file:
        paths = [row['sweep'] for row in csv.DictReader(file) if row['field']=='L219']
    from pathlib import Path
    from fitsio import read
    positions, ids = [], []
    qualities = {k:np.full(len(photo['ls_id']), np.nan) for k in names}
    matches = np.zeros(len(photo['ls_id']), 'i1')
    for basename in paths:
        raw = read(str(Path(raw_set)/basename),
                   columns=['RA','DEC','RELEASE','BRICKID','OBJID','TYPE',*names])
        lid = legacy_ids(raw)
        good = np.isfinite(raw['RA']+raw['DEC'])
        positions.append(np.column_stack((raw['RA'][good],raw['DEC'][good])))
        ids.append(lid[good])
        keys, rix, j = np.intersect1d(lid, photo['ls_id'], return_indices=True)
        assert (np.char.strip(raw['TYPE'][rix].astype(str))=='PSF').all()
        matches[j] += 1
        for name in names:
            qualities[name][j] = raw[name][rix]
    np.testing.assert_array_equal(matches, 1)
    gain = np.concatenate(positions)
    lid = np.concatenate(ids)

    def xyz(ra,dec):
        r,d = np.deg2rad(ra), np.deg2rad(dec)
        return np.column_stack((np.cos(d)*np.cos(r),np.cos(d)*np.sin(r),np.sin(d)))

    tree = cKDTree(xyz(*gain.T))
    distance, ix = tree.query(xyz(photo['RA'], photo['DEC']), k=8)
    nearest = np.min(np.where(lid[ix]!=photo['ls_id'][:,None], distance, np.inf),axis=1)
    qualities['nearest_detection_arcsec'] = np.rad2deg(2*np.arcsin(nearest/2))*3600
    qualities['detection_neighbors_10arcsec'] = tree.query_ball_point(
        xyz(photo['RA'], photo['DEC']), np.deg2rad(10/3600), return_length=True)-1
    return qualities


def main():
    global STATE
    require_compute()
    configure()
    TABLES.mkdir(parents=True, exist_ok=True)
    all_dust = load(OLD/'dust/forecasts.npz')
    parent = load(OLD/'census/qualified.npz')
    roster = load(OLD/'forecast/roster.npz')
    k = roster['field'][all_dust['rows']]=='L219'
    d = {key:val[k] for key,val in all_dust.items()}
    i = d['rows']
    p = {key:val[i] for key,val in parent.items()}
    assert len(i)==500 and (d['TYPE']=='PSF').all()
    np.testing.assert_array_equal(p['ls_id'],d['ls_id'])
    assert np.count_nonzero(d['iteration_state']==5)==198
    assert (p['TYPE']=='PSF').all()
    models,_,_ = pilot.prepare()
    observed = p['mag_grz'] - 2.5*np.log10(p['transmission'][:,[0,1,3]])
    STATE = models['qrf'],p,d,observed
    with ProcessPoolExecutor(max_workers=10, mp_context=multiprocessing.get_context('fork')) as ex:
        tracks = np.concatenate(list(ex.map(trace,np.array_split(np.arange(len(i)),15))))
    ref = d['reference_mu']
    known = (ref>=d['mu_min']) & (ref<=d['mu_max']) & (d['mu_min']<d['mu_max'])
    known_ebv = np.full(len(i), np.nan)
    for j in np.flatnonzero(known):
        known_ebv[j] = np.interp(ref[j], dust.MU, d['ebv_profile'][j])
    gaia = pilot.join_gaia(p['source_id'],pilot.GAIA)
    g = gaia['phot_g_mean_mag']
    color = gaia['bp_rp']
    quality = raw_quality(p)
    pdf = load(OLD/'forecast/pdf.npz')
    at = np.searchsorted(pdf['rows'],i)
    np.testing.assert_array_equal(pdf['ls_id'][at],p['ls_id'])
    full = dust.modulus_quantiles(pdf['mass'][at],pdf['mu_edges'][at])[:,2]-ref
    conv = d['iteration_state']==5
    resid = np.full(len(i), np.nan)
    resid[conv] = dust.modulus_quantiles(d['iteration_mass'][conv],d['iteration_edges'][conv])[:,2]-ref[conv]
    severe = (abs(resid)>1) & known
    last = np.maximum(d['iteration_steps']-1,0)
    j = np.arange(len(i))
    diff = tracks[:,:,2]-tracks[:,:,0]
    alternating = ((diff[:,1:]*diff[:,:-1]<0)&(abs(diff[:,1:])>.02)&(abs(diff[:,:-1])>.02)).sum(axis=1)
    jumps = np.max(np.where(np.isfinite(diff),abs(diff),0),axis=1)
    at_edge = np.minimum(tracks[j,last,2]-d['mu_min'],d['mu_max']-tracks[j,last,2]) <= 0.1
    data = dict(ls_id=p['ls_id'], TYPE=p['TYPE'], state=d['iteration_state'], steps=d['iteration_steps'],
                obs_r=p['r'], obs_gr=observed[:,0]-observed[:,1], gaia_bp_rp=color,
                gaia_absolute_g_uncorrected=g-ref, proxy=p['gaia_proxy'],
                error_g=p['error_grz'][:,0],error_r=p['error_grz'][:,1],error_z=p['error_grz'][:,2],
                sfd=p['EBV'], bayestar_at_gaia=known_ebv, bayestar_converged=d['iteration_ebv'],
                bayestar_last=tracks[j,last,1], sfd_residual=full, bayestar_residual=resid,
                oracle_residual=d['oracle_center']-ref, reference_in_map=known, reference_mu=ref,
                map_min=d['mu_min'],map_max=d['mu_max'],severe=severe, max_jump=jumps,
                reversals=alternating, near_map_edge=at_edge, **quality)
    rows = [dict(zip(data,v)) for v in zip(*data.values())]
    csv_rows(TABLES/'l219_stars.csv',rows)
    summary = []
    groups = {'all':np.ones(len(i),bool),'conditional_severe':severe,
              'conditional_nonsevere':known&conv&~severe}
    groups.update({name:d['iteration_state']==n for n,name in enumerate(STATUS)})
    groups.update({s:p['gaia_proxy']==s for s in np.unique(p['gaia_proxy'])})
    for name, mask in groups.items():
        row = dict(group=name,n=int(mask.sum()),converged=int((mask&conv).sum()),
                   actually_scored=int((mask&conv&known).sum()),severe=int((mask&severe).sum()),
                   reversal_2plus=int((mask&(alternating>=2)).sum()),
                   jump_half_mag=int((mask&(jumps>=.5)).sum()),near_map_edge=int((mask&at_edge).sum()))
        for key in ('sfd','bayestar_at_gaia','bayestar_last','sfd_residual','bayestar_residual','obs_gr',
                    'obs_r','error_g','error_r','error_z','nearest_detection_arcsec',
                    'detection_neighbors_10arcsec',*quality.keys()):
            valid = mask & np.isfinite(data[key])
            row['median_'+key] = float(np.median(data[key][valid])) if valid.any() else ''
        summary.append(row)
    csv_rows(TABLES/'l219_groups.csv',summary)
    curves = []
    for state, name in enumerate(STATUS):
        m = d['iteration_state']==state
        for k, mu in enumerate(dust.MU):
            vals = d['ebv_profile'][m&d['map_reliable'][:,k],k]
            if len(vals):
                q = np.quantile(vals,[.16,.5,.84])
                curves.append(dict(state=state,group=name,mu=mu,lo=q[0],center=q[1],hi=q[2],n=len(vals)))
    csv_rows(TABLES/'l219_profiles.csv',curves)
    for key in ('iteration_state','iteration_edges'):
        assert len(d[key])==500
    atomic_npz(TABLES/'l219_traces.npz', traces=tracks, ls_id=d['ls_id'], TYPE=d['TYPE'],
               columns=['input_mu','ebv','updated_mu','retained_mass','q025','q975'])
    print('L219 replay matched all 500 archived states, steps and converged PDFs.',flush=True)


if __name__=='__main__':
    main()
