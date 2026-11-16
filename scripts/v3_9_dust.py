#!/usr/bin/env python3
"""Compare fixed SFD and sampled 3D reddening on a native DECam pilot."""
import argparse
import json
import os
from pathlib import Path

from astropy.coordinates import SkyCoord
from astropy import units as u
import joblib
import numpy as np
from scipy.special import logsumexp, ndtr
from scipy.stats import norm

from photdis.gaia import derive_gaia_labels, join_gaia
from photdis.stage1 import galactic, require_compute, tile_bounds
from photdis.v3.data import stable_rank
from photdis.v3.model import inference_model
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_0_native import read_native
from v3_3_catalog import FREEZE, read_frozen
from v3_3_sed import verified
from v3_4_near_validate import GAIA
from v3_5_dr11_production import CUTS
from v3_6_gaia_census import distances
from v3_9_catalog import scan_chunk, settings, thin
from v3_9_references import join, subset

OUT = ROOT/'data/v3_9/dust'
GRZ = [0,1,3]
# Preserve the native filter-law shape while setting the color normalization
# to E(B-V) = 0.98 E(g-r), the same convention as the Bayestar conversion.
SFD = np.array([3.214,2.165,1.211])
NATIVE = SFD/(.98*(SFD[0]-SFD[1]))
MU_EDGES = np.arange(3.,22.01,.1)
MU = (MU_EDGES[1:]+MU_EDGES[:-1])/2
FIELDS = ((211.,-19.5),(213.8,-12.5),(198.,-8.),(170.,-10.),(37.,13.))


def photometry(centers=FIELDS):
    cfg = settings()['south'].copy()
    cfg['maximum_grz_error_mag'] = .1
    cfg['domains'] = {'dust_pilot':dict(r_min=14.,r_max=18.5,abs_b_max=25.,ebv_min=.08,ebv_max=1.25)}
    wanted = SkyCoord(*np.asarray(centers).T, unit='deg',frame='galactic').icrs
    parts = []
    for path in sorted(Path(cfg['sweep_root']).glob('sweep-*.fits')):
        ra0,dec0,ra1,dec1 = tile_bounds(path)
        r = wanted.ra.deg
        d = wanted.dec.deg
        fields = np.flatnonzero(((r-ra0)%360<(ra1-ra0)%360)&(d>=dec0)&(d<dec1))
        if not len(fields):
            continue
        raw = read_native(path)
        for k in fields:
            chunks = []
            for i in range(0,len(raw),150_000):
                for p,_ in scan_chunk(raw[i:i+150_000],(ra0,dec0,ra1,dec1),cfg,i):
                    if not p:
                        continue
                    within = np.hypot(((p['RA']-r[k]+180)%360-180)*np.cos(np.deg2rad(d[k])),p['DEC']-d[k])<.9
                    use = within&(p['row_state']==7)
                    chunks.append(subset(p,use))
            chunk = join(chunks)
            if not chunk:
                continue
            crowding = np.digitize(chunk['FRACFLUX_R'],[.02,.10,.25])
            red = np.digitize(chunk['EBV'],[.15,.35,.6])
            j = []
            for cell in np.unique(crowding*4+red):
                j.extend(thin(chunk,crowding*4+red==cell,3,3910)[0])
            j = np.asarray(j,dtype=int)
            j = j[np.argsort(stable_rank(chunk['ls_id'][j],3911),kind='stable')[:160]]
            if len(j):
                parts.append(dict(**subset(chunk,j),field=np.full(len(j),k,'i1')))
    return join(parts)


def dust_profiles(coords, path, kind):
    if kind=='bayestar19':
        from dustmaps.bayestar import BayestarQuery
        dust = BayestarQuery(map_fname=str(path),version='bayestar2019')
        factor = .883
    else:
        from dustmaps.decaps import DECaPSQuery
        dust = DECaPSQuery(map_fname=str(path),mean_only=False)
        factor = 1.
    sample,flags = dust(coords,mode='samples',return_flags=True)
    distance_moduli = np.asarray(dust.distmods)
    if sample.ndim!=3 or sample.shape[-1]!=len(distance_moduli):
        raise ValueError('Unexpected dust profile layout')
    interp = np.array([[np.interp(MU,distance_moduli,s,left=np.nan,right=np.nan) for s in row]
                      for row in sample])*factor
    reliable = flags['converged'][:,None]
    reliable = reliable & (MU>=flags['min_reliable_distmod'][:,None])
    reliable &= MU<=flags['max_reliable_distmod'][:,None]
    if 'infilled' in flags.dtype.names:
        reliable &= ~flags['infilled'][:,None]
    return interp,reliable,flags


def marginal_log_density(model, flux, error, transmission):
    """Gaussian raw-flux likelihood, marginalized over binary and dust draws."""
    means = model.means[:,GRZ]
    cov = model.covariances[:,GRZ][:,:,GRZ]
    weights = np.asarray(model.weights)
    pos = weights>0
    means,cov,weights = means[pos],cov[pos],weights[pos]
    dust_draws = []
    for trans in transmission:
        by_binary = []
        for brightness in (1., model.binary_brightness_factor):
            a = (10**(-.4*(MU-10))*brightness)[:,None]*trans/error[None,:]
            variance = cov[None,:,:,:]*a[:,None,:,None]*a[:,None,None,:]+np.eye(3)
            chol = np.linalg.cholesky(variance)
            residual = flux[None,None,:]/error[None,None,:]-means[None,:,:]*a[:,None,:]
            z = np.linalg.solve(chol,residual[...,None])[...,0]
            ll = -np.sum(np.log(np.diagonal(chol,axis1=-2,axis2=-1)),axis=-1)-.5*(z*z).sum(axis=-1)
            by_binary.append(logsumexp(ll+np.log(weights)[None,:],axis=1))
        binary = model.binary_reference_fraction
        dust_draws.append(logsumexp(np.stack(by_binary)+np.log([1-binary,binary])[:,None],axis=0))
    return logsumexp(dust_draws,axis=0)-np.log(len(dust_draws))+norm.logpdf(MU,15,5)


def normalize(logp, mask):
    p = np.where(mask,logp,-np.inf)
    return np.exp(p-logsumexp(p))


def matched_reference_summary(photo, gaia_state, reference, lo, hi, quality, pdf):
    compared = np.isfinite(pdf[:,2]).all(axis=1)
    if not np.allclose(pdf[:,0].sum(axis=1),1.,atol=1e-10,rtol=0):
        raise ValueError('Invalid unrestricted SFD histogram')
    if not np.allclose(pdf[compared,1:].sum(axis=2),1.,atol=1e-10,rtol=0):
        raise ValueError('Invalid common-support dust histogram')
    if np.any(pdf[compared,1:]*~quality[compared,None,:]):
        raise ValueError('Conditioned PDF has mass outside the reliable dust range')
    with np.errstate(divide='ignore',invalid='ignore'):
        mu = 10+5*np.log10(reference)
        sigma = 2.5*np.log10(hi/lo)
    j = np.searchsorted(MU_EDGES,mu)-1
    supported = (j>=0)&(j<len(MU))
    covered = np.zeros(len(j),bool)
    covered[supported] = quality[np.flatnonzero(supported),j[supported]]
    qualified = (gaia_state==6)&np.isfinite(sigma)&(sigma>0)
    same_star = qualified&covered&compared
    fields = [(-1,np.ones(len(j),bool)),*((i,photo['field']==i) for i in np.unique(photo['field']))]
    crowding = [('all',np.ones(len(j),bool)),('lower',photo['FRACFLUX_R']<.1),
                ('higher',photo['FRACFLUX_R']>=.1)]
    rows = []
    for field,region in fields:
        for label,crowd in crowding:
            base = region&crowd
            use = base&same_star
            row = dict(field=int(field),crowding=label,psf_rows=int(base.sum()),
                       map_qualified=int((base&compared).sum()),
                       gaia_qualified=int((base&qualified).sum()),common_support=int(use.sum()))
            if use.any():
                mask = quality[use]
                density = np.diff(ndtr((MU_EDGES[None,:]-mu[use,None])/sigma[use,None]),axis=1)
                kernel = density/np.diff(MU_EDGES)
                likelihood = np.sum(pdf[use,1:]*kernel[:,None,:],axis=-1)
                score = np.log(np.maximum(likelihood,np.finfo(float).tiny))
                row.update(mean_common_support_score_gain=float(np.mean(score[:,1]-score[:,0])),
                           median_sfd_mass_in_map_range=float(np.median(np.sum(pdf[use,0]*mask,axis=1))))
                for i,name in ((1,'sfd'),(2,'dust_3d')):
                    middle = np.argmax(np.cumsum(pdf[use,i],axis=1)>=.5,axis=1)
                    residual = MU[middle]-mu[use]
                    row.update({f'{name}_median_mu_residual':float(np.median(residual)),
                                f'{name}_nmad_mu_residual':float(1.4826*np.median(abs(residual-np.median(residual))))})
            rows.append(row)
    return rows


def run(path, kind):
    p = photometry()
    if not p or not np.all(p['TYPE']=='PSF'):
        raise ValueError('Dust test did not find native DECam PSFs in its registered fields')
    coords = SkyCoord(p['RA']*u.deg,p['DEC']*u.deg,frame='icrs')
    sample,good,flags = dust_profiles(coords,path,kind)
    quality = good & np.isfinite(sample).all(axis=1) & (sample>=0).all(axis=1)
    entry = read_frozen()['models']['nominal']
    model = inference_model(joblib.load(verified(ROOT/entry['path'],entry['sha256'])))
    gaia = join_gaia(p['source_id'],GAIA)
    gaia_state,refs,lo,hi = distances(p,p['source_id']>0,derive_gaia_labels(gaia),
                                      gaia,json.loads(CUTS.read_text()))
    fits, rows = [], []
    for k,ls_id in enumerate(p['ls_id']):
        f,err = p['flux'][k,GRZ],p['flux_error'][k,GRZ]
        sfd = np.broadcast_to(p['transmission'][k,GRZ],(1,len(MU),3))
        sfd_ll = marginal_log_density(model,f,err,sfd)
        baseline = normalize(sfd_ll,np.ones(len(MU),bool))
        qualified = quality[k]
        if qualified.sum()>=3:
            reddening = np.where(qualified[None,:],sample[k],0.)
            t = 10**(-.4*reddening[:,:,None]*NATIVE[None,None,:])
            ll = marginal_log_density(model,f,err,t)
            shared = normalize(sfd_ll,qualified)
            proposed = normalize(ll,qualified)
        else:
            shared = proposed = np.full(len(MU),np.nan)
        fits.append(np.stack([baseline,shared,proposed]))
        medians = []
        for q in (baseline,shared,proposed):
            m = MU[np.searchsorted(np.cumsum(q),.5)] if np.isfinite(q).all() else np.nan
            medians.append(10**((m-10)/5))
        l,b = galactic(p['RA'][k:k+1],p['DEC'][k:k+1])
        rows.append(dict(ls_id=int(ls_id),TYPE='PSF',field=int(p['field'][k]),lon=float(l[0]),lat=float(b[0]),
                         sfd_ebv=float(p['EBV'][k]),fracflux_r=float(p['FRACFLUX_R'][k]),
                         seeing_r=float(p['PSFSIZE_R'][k]),reliable_mu_bins=int(qualified.sum()),
                         baseline_mass_in_reliable_range=float(baseline[qualified].sum()),
                         sfd_kpc=medians[0],sfd_same_grid_kpc=medians[1],dust_3d_kpc=medians[2],
                         gaia_reference_kpc=float(refs[k]),gaia_state=int(gaia_state[k])))
    out = OUT/kind
    out.mkdir(parents=True,exist_ok=True)
    csv_rows(out/'sightlines.csv',rows)
    pdf = np.stack(fits)
    csv_rows(out/'matched_gaia.csv',matched_reference_summary(p,gaia_state,refs,lo,hi,quality,pdf))
    atomic_npz(out/'distance_pdfs.npz',ls_id=p['ls_id'],TYPE=p['TYPE'],mu_edges=MU_EDGES,
               pdf=pdf,map_reliable=quality,ebv_samples=sample,reference_kpc=refs,
               gaia_reference_lo=lo,gaia_reference_hi=hi)
    save_json(out/'complete.json',dict(rows=len(rows),valid=int((quality.sum(axis=1)>=3).sum()),
        dust_kind=kind,map_sha256=sha(path),coefficients=NATIVE,nominal_sha256=entry['sha256'],
        inference_source_sha256=sha(Path(__file__)),frozen_nominal_manifest_sha256=sha(FREEZE),
        job=os.getenv('SLURM_JOB_ID'),outputs={x.name:sha(x) for x in out.iterdir() if x.name!='complete.json'}))
    print(kind,len(rows),'pilot sources',flush=True)


if __name__=='__main__':
    require_compute()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind',choices=('bayestar19','decaps'))
    parser.add_argument('path',type=Path)
    a = parser.parse_args()
    run(a.path,a.kind)
