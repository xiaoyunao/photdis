#!/usr/bin/env python3
"""Test the corrected dust scale in a few native DECam fields inside ten degrees."""
import json
import os
from pathlib import Path

from astropy import units as u
from astropy.coordinates import SkyCoord
import joblib
import numpy as np

from photdis.gaia import derive_gaia_labels, join_gaia
from photdis.stage1 import galactic, require_compute
from photdis.v3.model import inference_model
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_3_catalog import read_frozen
from v3_3_sed import verified
from v3_4_near_validate import GAIA
from v3_5_dr11_production import CUTS
from v3_6_gaia_census import distances
from v3_9_dust import (GRZ, MU, MU_EDGES, dust_profiles, marginal_log_density,
                      matched_reference_summary, normalize, photometry)
from v3_9_references import matched, raw_catalogs, subset
from v3_10_dust import BAYESTAR_TO_QUOTED_EBV, DECam_PER_BAYESTAR

DEST = ROOT/'data/v3_10/dust/low_latitude'
MAP = Path('/pscratch/sd/y/yunao/dustmaps/bayestar2019_v3_9.h5')
FIELDS = tuple((l,b) for l in (198.,205.,212.,219.,226.,233.,240.) for b in (-9.,-6.))
FIELDS += ((165.,-8.),(171.,-7.),(33.,8.),(37.,8.),(41.,8.),(45.,8.))


def preselect(p, usable):
    candidates = []
    for field,(l,b) in enumerate(FIELDS):
        use = p['field']==field
        k = use&usable
        candidates.append(dict(field=field,longitude_deg=l,latitude_deg=b,psf_rows=int(use.sum()),
            map_qualified=int(k.sum()),median_sfd=float(np.median(p['EBV'][k])) if k.any() else None,
            higher_crowding=int((k&(p['FRACFLUX_R']>=.1)).sum())))
    qualified = [a for a in candidates if a['map_qualified']>=8]
    if not qualified:
        return candidates,[]
    first = min(qualified,key=lambda x:x['median_sfd'])
    last = max(qualified,key=lambda x:x['median_sfd'])
    crowded = max(qualified,key=lambda x:x['higher_crowding']/x['map_qualified'])
    return candidates,sorted({item['field'] for item in (first,last,crowded)})


def main():
    require_compute()
    p = photometry(FIELDS)
    _,latitude = galactic(p['RA'],p['DEC'])
    p = subset(p,np.abs(latitude)<10)
    if not len(p['ls_id']) or not np.all(p['TYPE']=='PSF'):
        raise ValueError('No available native DR11 PSFs inside the pilot latitude')
    coords = SkyCoord(p['RA']*u.deg,p['DEC']*u.deg,frame='icrs')
    samples,valid,_ = dust_profiles(coords,MAP,'bayestar19')
    samples /= BAYESTAR_TO_QUOTED_EBV
    quality = valid&np.isfinite(samples).all(axis=1)&(samples>=0).all(axis=1)
    candidates,fields = preselect(p,quality.sum(axis=1)>=3)
    DEST.mkdir(parents=True,exist_ok=True)
    csv_rows(DEST/'candidate_fields.csv',candidates)
    if not fields:
        raise ValueError('No registered fields have enough reliable Bayestar19 profiles')
    # On overlapping candidate pointings, retain each native source once.
    use = np.flatnonzero(np.isin(p['field'],fields))
    _,ix = np.unique(p['ls_id'][use],return_index=True)
    use = use[np.sort(ix)]
    samples,quality,p = samples[use],quality[use],subset(p,use)
    entry = read_frozen()['models']['nominal']
    model = inference_model(joblib.load(verified(ROOT/entry['path'],entry['sha256'])))
    bins = np.full((len(use),3,len(MU)),np.nan)
    for i,(f,e,t,k,s) in enumerate(zip(p['flux'][:,GRZ],p['flux_error'][:,GRZ],p['transmission'][:,GRZ],quality,samples)):
        sfd = marginal_log_density(model,f,e,np.broadcast_to(t,(1,len(MU),3)))
        bins[i,0] = normalize(sfd,np.ones(len(MU),bool))
        if k.sum()<3:
            continue
        reddening = np.where(k[None,:],s,0.)
        trans = 10**(-.4*reddening[:,:,None]*DECam_PER_BAYESTAR)
        bins[i,1] = normalize(sfd,k)
        bins[i,2] = normalize(marginal_log_density(model,f,e,trans),k)
    gaia = join_gaia(p['source_id'],GAIA)
    state,d,lo,hi = distances(p,p['source_id']>0,derive_gaia_labels(gaia),gaia,json.loads(CUTS.read_text()))
    catalog,inputs = raw_catalogs()
    outputs = dict(gaia_reference_kpc=d)
    stats = [dict(source='Gaia',**row) for row in matched_reference_summary(p,state,d,lo,hi,quality,bins)]
    for source in ('specdis','spdist'):
        c = catalog[source]
        j = matched(p,c)
        defined = j>=0
        center,error = np.full((2,len(j)),np.nan)
        good = defined.copy()
        good[defined] = c['historical_quality'][j[defined]]
        center[good],error[good] = c['distance_kpc'][j[good]],c['sigma_mu'][j[good]]
        meta = matched_reference_summary(p,np.where(good,6,0),center,center*10**(-error/5),
                                          center*10**(error/5),quality,bins)
        stats.extend(dict(source=source,**row) for row in meta)
        outputs[source+'_reference_kpc'] = center
    csv_rows(DEST/'matched_distances.csv',stats)
    atomic_npz(DEST/'distance_pdfs.npz',ls_id=p['ls_id'],TYPE=p['TYPE'],RA=p['RA'],DEC=p['DEC'],
               field=p['field'],mu_edges=MU_EDGES,pdf=bins,map_reliable=quality,map_samples=samples,**outputs)
    save_json(DEST/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,
        map_sha256=sha(MAP),fields=fields,rows=len(use),
        outputs={child.name:sha(child) for child in DEST.iterdir() if child.name!='complete.json'}))


if __name__=='__main__':
    main()
