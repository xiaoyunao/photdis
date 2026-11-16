#!/usr/bin/env python3
"""Compare identical-label and production QRFs on verified two-sided sky matches."""

import json
import os

import numpy as np

from photdis.reference_v1_4 import unique_join
from photdis.stage1 import require_compute
from photdis.v3.metrics import distance_summary
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_6_vac_distances import read_columns
from v3_9_catalog import OUT as EXPANSION
from v3_9_delivery import DEST as RELEASE
from v3_10_seam import DEST as V310

DEST = ROOT/'data/v3_11/seam'
MAGS = ((-np.inf,np.inf),(-np.inf,18.),(18.,20.),(20.,21.),(21.,22.),(22.,23.))
COLORS = ((-np.inf,np.inf),(-np.inf,.5),(.5,.8),(.8,1.4),(1.4,np.inf))


def read_product(marker, name, columns, inputs):
    receipt = json.loads(marker.read_text())
    inputs[str(marker.relative_to(ROOT))] = sha(marker)
    return read_columns(verified(marker.parent/name,receipt['outputs'][name]),columns)


def run():
    inputs = {}
    pilot = read_product(EXPANSION/'validation/overlap/complete.json','paired.npz',
        ('TYPE','north_ls_id','south_ls_id','source_id','reference_kpc','class_label','shared_support',
         'north_q','south_q','south_mag'),inputs)
    pairs = read_product(RELEASE/'seam/complete.json','paired_sources.npz',
                         ('TYPE','north_ls_id','south_ls_id'),inputs)
    production = {}
    for side in ('north','south'):
        p = read_product(V310/f'{side}.json',f'{side}.npz',
                         ('TYPE','ls_id','r','gr','state','qrf_quantiles_kpc'),inputs)
        np.testing.assert_array_equal(p['ls_id'],pairs[f'{side}_ls_id'])
        np.testing.assert_array_equal(p['TYPE'],pairs['TYPE'])
        production[side] = p
    if not np.all(pilot['TYPE']=='PSF') or not np.all(pairs['TYPE']=='PSF'):
        raise ValueError('Expected a native PSF in both seam measurements')
    index = unique_join(pilot['north_ls_id'],pairs['north_ls_id'])
    matched = index>=0
    matched[matched] &= pairs['south_ls_id'][index[matched]]==pilot['south_ls_id'][matched]
    i = np.flatnonzero(matched)
    j = index[i]
    s,n = production['south'],production['north']
    common = pilot['shared_support'][i] & (s['state'][j]==3) & (n['state'][j]==3)
    i,j = i[common],j[common]
    if not len(i):
        raise ValueError('No exact two-sided native pairs have the four forecasts')
    quantiles = np.stack([pilot['north_q'][i],pilot['south_q'][i],
                          n['qrf_quantiles_kpc'][j],s['qrf_quantiles_kpc'][j]])
    if not np.isfinite(quantiles).all() or (quantiles<=0).any():
        raise ValueError('Paired forecasts must be finite and positive')
    reference = pilot['reference_kpc'][i]
    mag,color = s['r'][j],s['gr'][j]
    mu = 10+5*np.log10(quantiles)
    ratio_native = mu[0,:,2]-mu[1,:,2]
    ratio_production = mu[2,:,2]-mu[3,:,2]
    rows = []
    for r0,r1 in MAGS:
        for c0,c1 in COLORS:
            k = (mag>=r0)&(mag<r1)&(color>=c0)&(color<c1)
            if not k.any():
                continue
            meta = dict(south_r_min=r0,south_r_max=r1,south_gr_min=c0,south_gr_max=c1)
            for side,q in zip(('matched_north','matched_south','production_north','production_south'),mu):
                rows.append(dict(**meta,view=side,**distance_summary(q[k],10+5*np.log10(reference[k]))))
            for name,vector in (('matched_north_minus_south',ratio_native),
                                ('production_north_minus_south',ratio_production),
                                ('production_minus_matched_seam',ratio_production-ratio_native)):
                med = np.median(vector[k])
                rows.append(dict(**meta,view=name,n=int(k.sum()),median_mu_residual=float(med),
                                 median_relative=float(np.expm1(np.log(10)*med/5))))
    DEST.mkdir(parents=True,exist_ok=True)
    csv_rows(DEST/'paired_scales.csv',rows)
    atomic_npz(DEST/'same_star.npz',TYPE=pilot['TYPE'][i],ls_id=pilot['north_ls_id'][i],
               south_ls_id=pilot['south_ls_id'][i],source_id=pilot['source_id'][i],
               distance_kpc=reference,south_r=mag,south_gr=color,quantiles=quantiles)
    save_json(DEST/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,paired=len(i),
        pilot_total=len(index),exact_matches=int(matched.sum()),
        outputs={f:sha(DEST/f) for f in ('paired_scales.csv','same_star.npz')}))


if __name__=='__main__':
    require_compute()
    run()
