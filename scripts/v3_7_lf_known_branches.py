#!/usr/bin/env python3
"""Evaluate SED branch probabilities for confirmed distance-conflict cases."""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path

import joblib
import numpy as np

from photdis.stage1 import galactic, require_compute
from photdis.v3.model import inference_model
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_3_catalog import FREEZE, read_frozen
from v3_3_sed import verified
from v3_6_blue_replay import OUT as BLUE
from v3_6_distance_evidence import OUT as EVIDENCE
from v3_6_reference_photometry import OUT as PHOTO
from v3_6_vac_distances import DISTANCE_EDGES, read_columns
import v3_6_lf_sensitivity as lf
from v3_6_lf_priors import empirical_weights, giant_luminosity_weights
from v3_7_lf_odds import reweight
from v3_7_weak_gaia import PROTOCOL

OUT = ROOT/'data/v3_7/lf_known_branches'
CASES = ('SpecDis','SPDist','DR20_MINESweeper','DR20_APO')


def examples(evidence,blue,excluded):
    p = read_columns(verified(BLUE/'blue_objects.npz',blue['outputs']['blue_objects.npz']),
                     ('ls_id','TYPE','sample','reference_kpc','qrf_quantiles_kpc','strict'))
    excluded['blue_non_psf'] = int(((p['TYPE']!='')&(p['TYPE']!='PSF')).sum())
    excluded['blue_unverified_type'] = int((p['TYPE']=='').sum())
    for i in np.flatnonzero(p['TYPE']=='PSF'):
        cohort = p['sample'][i]
        yield dict(ls_id=int(p['ls_id'][i]),TYPE='PSF',source='Gaia',cohort=cohort,
                   reference_kpc=p['reference_kpc'][i],strict=bool(p['strict'][i]),
                   qrf_median_kpc=p['qrf_quantiles_kpc'][i,2])
    keys = ('ls_id','TYPE','primary','strict','high_quality','qrf_training','consensus_class',
            'reference_kpc','reference_lo_kpc','qrf_quantiles_kpc')
    for source in CASES:
        name = f'{source}.npz'
        if name not in evidence['outputs']:
            continue
        a = read_columns(verified(EVIDENCE/name,evidence['outputs'][name]),keys)
        excluded[f'{source}_non_psf'] = int(((a['TYPE']!='')&(a['TYPE']!='PSF')).sum())
        excluded[f'{source}_unverified_type'] = int((a['TYPE']=='').sum())
        d50 = a['qrf_quantiles_kpc'][:,2]
        use = ((a['TYPE']=='PSF')&a['primary']&a['high_quality']&~a['qrf_training']
               &(a['consensus_class']==1)&(a['reference_lo_kpc']>10)&(d50<10))
        for i in np.flatnonzero(use):
            yield dict(ls_id=int(a['ls_id'][i]),TYPE='PSF',source=source,cohort='far_giant',
                       reference_kpc=a['reference_kpc'][i],strict=bool(a['strict'][i]),qrf_median_kpc=d50[i])


def evaluate(photo,model,l,b):
    base = empirical_weights(model)
    rgb_disk,disk_support = giant_luminosity_weights(model,.27,-.72,.22)
    rgb_halo,halo_support = giant_luminosity_weights(model,.32,-1.46,.30)
    faint = (model.means[:,0]>0)&(model.means[:,0]<10**((12.5-8.)/2.5))
    other = base[:,2]*np.where(faint,2.,1.)
    other /= other.sum()
    lf._STATE = photo,model,base,rgb_disk,rgb_halo,other,model.weights@model.classes,l,b
    with ProcessPoolExecutor(max_workers=int(os.getenv('SLURM_CPUS_PER_TASK','1')),
                             mp_context=mp.get_context('fork')) as pool:
        joint,curve = zip(*pool.map(lf.forecast,range(len(l)),chunksize=12))
    lf._STATE = None
    return np.stack(joint),np.stack(curve),dict(rgb_disk=disk_support,rgb_halo=halo_support)


def metrics(records,ids,joint,factors):
    for record in records:
        i = np.searchsorted(ids,record['ls_id'])
        j = np.searchsorted(DISTANCE_EDGES,record['reference_kpc'],side='right')-1
        for s,scenario in enumerate(lf.NAMES):
            for factor in factors:
                a = reweight(joint[i:i+1,s],factor)[0]
                correct = a[:,j].sum()
                branch = (a[1,3:].sum() if record['cohort']=='far_giant'
                          else a[2,0] if record['cohort']=='faint_blue_near' else None)
                yield dict(**record,scenario=scenario,giant_odds_factor=factor,
                           correct_reference_shell_probability=correct,correct_joint_branch_probability=branch)


def main():
    require_compute()
    marker = OUT/'complete.json'
    if marker.exists():
        raise FileExistsError(marker)
    inputs = [EVIDENCE/'complete.json',BLUE/'complete.json',PHOTO/'complete.json',FREEZE,PROTOCOL,
              Path(__file__).resolve(),Path(lf.__file__).resolve()]
    ev,blue,p = (json.loads(x.read_text()) for x in inputs[:3])
    factors = json.loads(PROTOCOL.read_text())['giant_odds_factors']
    excluded = {}
    records = list(examples(ev,blue,excluded))
    ids = np.unique([r['ls_id'] for r in records])
    path = verified(PHOTO/'photometry.npz',p['outputs']['photometry.npz'])
    with np.load(path,allow_pickle=False) as z:
        order = np.argsort(z['ls_id'])
        lookup = z['ls_id'][order]
        position = order[np.searchsorted(lookup,ids)]
        np.testing.assert_array_equal(z['ls_id'][position],ids)
        if not (z['TYPE'][position]=='PSF').all():
            raise ValueError('Every case requires native PSF after photometry join')
        a = {k:z[k][position] for k in ('catalog_row','flux','flux_ivar','transmission','band_mask','EBV')}
    sort = np.argsort(a['catalog_row'])
    ra,dec = np.full(len(ids),np.nan),np.full(len(ids),np.nan)
    for source in ('Gaia',*CASES):
        name = f'{source}.npz'
        if name not in ev['outputs']:
            continue
        q = read_columns(verified(EVIDENCE/name,ev['outputs'][name]),('catalog_row','RA','DEC'))
        positions,ii,jj = np.intersect1d(a['catalog_row'][sort],q['catalog_row'],return_indices=True)
        if len(positions):
            r = sort[ii]
            ra[r],dec[r] = q['RA'][jj],q['DEC'][jj]
    if not (np.isfinite(ra).all() and np.isfinite(dec).all()):
        raise ValueError('Missing sky position')
    with np.errstate(divide='ignore',invalid='ignore'):
        a['flux_error'] = 1./np.sqrt(a.pop('flux_ivar'))
    a.update(photometric_system=np.full(len(ids),'DECam_DR11'),RA=ra,DEC=dec)
    entry = read_frozen()['models']['nominal']
    model_file = verified(ROOT/entry['path'],entry['sha256'])
    inputs.append(model_file)
    model = inference_model(joblib.load(model_file))
    joint,curve,support = evaluate(a,model,*galactic(ra,dec))
    np.testing.assert_allclose(joint.sum(axis=(2,3)),1.,rtol=0,atol=1e-12)
    OUT.mkdir(parents=True,exist_ok=True)
    npz,csv = OUT/'forecast.npz',OUT/'case_probabilities.csv'
    atomic_npz(npz,ls_id=ids,TYPE=np.full(len(ids),'PSF'),scenario=np.asarray(lf.NAMES),
               class_shell=joint,marginal_log_density=curve,grid_mu=lf.GRID)
    csv_rows(csv,metrics(records,ids,joint,factors))
    save_json(marker,dict(inputs={str(x.relative_to(ROOT)):sha(x) for x in inputs},
        unique_cases=len(ids),source_cases=len(records),excluded_morphology=excluded,
        support=support,outputs={p.name:sha(p) for p in (npz,csv)}))


if __name__=='__main__':
    main()
