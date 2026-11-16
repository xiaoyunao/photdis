#!/usr/bin/env python3
"""Validate expansion products and build the fixed-label north/south comparison."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
import os
from pathlib import Path

import numpy as np

from photdis.baseline_v0 import HistogramQRF
from photdis.reference_v1_4 import unique_join
from photdis.stage1 import require_compute
from photdis.v3.data import stable_rank
from photdis.v3.metrics import distance_summary
from photdis.v3.model import inference_model
from photdis.v3_3 import frozen_qrf
from v2_1_near import atomic_npz, csv_rows, load, save_json, sha
from v3_3_sed import verified
import v3_4_near_forecast as forecast
from v3_4_near_reference_support import separation
from v3_6_distance_evidence import OUT as EVIDENCE
from v3_6_reference_photometry import OUT as SOUTH_PHOTO
from v3_6_vac_distances import read_columns
from v3_6_weak_gaia import histogram, score
from v3_8_candidate import read_cache
from v3_9_catalog import OUT, confirm
from v3_9_references import choose, join, subset

REGIONS = ('south_faint','south_low','north')
SUMMARY = ('ls_id','TYPE','source_id','RA','DEC','region','r','g0','r0','z0','EBV',
           'error_grz','PSFDEPTH_R','PSFSIZE_R','FRACFLUX_R','band_state','row_state',
           'north_primary_region','galactic_latitude')
REFV = ('ls_id','TYPE','source_id','RA','DEC','region','r','g0','r0','error_grz','row_state',
        'reference_kpc','reference_lo_kpc','reference_hi_kpc','class_label','spectype','gaia_state',
        'specdis_kpc','spdist_kpc','specdis_quality','spdist_quality','specdis_sigma_mu','spdist_sigma_mu',
        'signed_parallax_mas','parallax_error_mas','reference_sampling_weight')


def summarize_sweep(source, *, side):
    photo = OUT/side/'photo'/Path(source['filename']).stem
    output = OUT/side/'predict'/photo.name
    p = json.loads((photo/'complete.json').read_text())
    q = json.loads((output/'complete.json').read_text())
    if q['binding']['photo']!=sha(photo/'complete.json'):
        raise ValueError('Prediction and parent receipts refer to different inputs')
    projections = []
    for name,digest in p['outputs'].items():
        if name=='parent_state.npz':
            continue
        t = read_columns(verified(photo/name,digest),SUMMARY)
        state = load(verified(output/('state_'+name),q['outputs']['state_'+name]))
        np.testing.assert_array_equal(state['ls_id'],t['ls_id'])
        d = np.full(len(t['ls_id']),np.nan,'f4')
        ids = np.flatnonzero(state['state']==3)
        mass = np.zeros((len(d),6),'f4')
        de = np.full(len(d),np.nan)
        cl = np.full(len(d),np.nan)
        if name in q['outputs']:
            pred = read_columns(verified(output/name,q['outputs'][name]),
                                ('photo_row','ls_id','TYPE','qrf_quantiles_kpc','qrf_bins','dwarf_score'))
            np.testing.assert_array_equal(pred['photo_row'],ids)
            np.testing.assert_array_equal(pred['ls_id'],t['ls_id'][ids])
            np.testing.assert_array_equal(pred['TYPE'],t['TYPE'][ids])
            d[ids] = pred['qrf_quantiles_kpc'][:,2]
            mass[ids] = pred['qrf_bins']
            de[ids] = pred['qrf_quantiles_kpc'][:,3]-pred['qrf_quantiles_kpc'][:,1]
            cl[ids] = pred['dwarf_score']
        elif len(ids):
            raise ValueError('Missing prediction for plotted rows')
        owner = t['north_primary_region']==(side=='north')
        if not np.all(t['TYPE']=='PSF'):
            raise ValueError('Unexpected morphology in native catalog projection')
        t.update(qrf_median_kpc=d,d68_width=de,dwarf_score=cl,state=state['state'],
                 primary_survey_owner=owner,distance_mass=mass)
        projections.append(t)
    return projections


def area_rows(p,shard):
    if not p:
        return []
    output = []
    for region in REGIONS:
        r = p['region']==region
        if not r.any():
            continue
        for a,b in ((-np.inf,20),(20,21),(21,22),(22,23)):
            for owner in (False,True):
                k = r&(p['r']>=a)&(p['r']<b)&(p['primary_survey_owner']==owner)
                if not k.any():
                    continue
                main = k&(p['row_state']==7)&(p['state']==3)
                control = k&(p['row_state']==6)&(p['state']==3)
                mass = p['distance_mass'][main].sum(axis=0,dtype=float)
                if abs(mass.sum()-main.sum())>max(1e-5,main.sum()*1e-6):
                    raise ValueError('Catalog shell mass does not conserve stars')
                output.append(dict(region=region,shard=shard,min_r=a,max_r=b,primary_owner=owner,
                                   native_psf_rows=int(k.sum()),precise_rows=int((k&(p['row_state']==7)).sum()),
                                   precise_pdf_count=int(main.sum()),noise_control_count=int(control.sum()),
                                   **{f'qrf_mass_shell_{j}':float(m) for j,m in enumerate(mass)}))
    return output


def nominal_setup(side):
    if side=='north':
        from v3_9_catalog import from_north
        models,_ = from_north()
    else:
        models,_,_ = forecast.prepare()
    model = models.get('nominal',models.get('sed'))
    return inference_model(model) if side=='south' else model


def nominal_control(t,model,j):
    if not len(j):
        return np.empty(0),np.empty(0,bool)
    prior = model.weights@model.classes
    from v3_3_catalog import joint_summary
    from photdis.v3_3 import class_log_likelihood
    use = []
    for i in j:
        row = dict(flux=t['flux'][i],flux_error=t['flux_error'][i],transmission=t['transmission'][i],
                   band_mask=t['band_mask'][i],EBV=t['EBV'][i],
                   photometric_system='BASS_MzLS_DR11' if t['region'][i]=='north' else 'DECam_DR11')
        mu = forecast.MU
        result = joint_summary(mu,class_log_likelihood(model,row,mu),prior)
        use.append((result['qnear'],result['numerical_flag']))
    value,flags = np.asarray(use).T
    return value,flags.astype(bool)


def reference_sweep(stem, *, side, t):
    indices = np.flatnonzero(t['sweep']==stem)
    if not len(indices):
        return {}
    t = subset(t,indices)
    out = OUT/side/'predict'/stem
    receipt = json.loads((out/'complete.json').read_text())
    forecasts = np.full((len(indices),5),np.nan)
    score_raw = np.full(len(indices),np.nan)
    weak = np.full((len(indices),10),np.nan)
    y,e = t['signed_parallax_mas'],t['parallax_error_mas']
    for name,h in receipt['outputs'].items():
        if name.startswith('state_'):
            continue
        with np.load(verified(out/name,h),allow_pickle=False) as z:
            lookup = unique_join(t['ls_id'],z['ls_id'])
            ix = np.flatnonzero(lookup>=0)
            j = lookup[ix]
            if not len(j):
                continue
            forecasts[ix] = z['qrf_quantiles_kpc'][j]
            score_raw[ix] = z['dwarf_score'][j]
            evaluate = ix[t['gaia_state'][ix]==4]
            if len(evaluate):
                ids = choose(t['ls_id'],np.isin(np.arange(len(indices)),evaluate),128,3930)
                edges = z['qrf_mu_edges'][lookup[ids]]
                mass = z['qrf_mu_mass'][lookup[ids]]
                for k,ed,ma in zip(ids,edges,mass):
                    wide = np.linspace(0.,25.,251)
                    weak[k] = np.r_[score(*histogram(ed,ma),y[k],e[k]),
                                    score(*histogram(wide,np.diff(wide)/25),y[k],e[k])]
    return dict(**{k:t[k] for k in REFV},qrf_quantiles_kpc=forecasts,dwarf_score=score_raw,
                signed_gaia_observation_metrics=weak)


def group(side,shard):
    inventory = OUT/side/'inventory.json'
    sources = json.loads(inventory.read_text())['groups'][shard]
    path = OUT/side/'reference'/f'{shard:02d}.json'
    depend = json.loads(path.read_text())
    dest = OUT/'validation'/side
    dest.mkdir(parents=True,exist_ok=True)
    marker = dest/f'{shard:02d}.json'
    receipt_hashes = [sha(OUT/side/'predict'/Path(s['filename']).stem/'complete.json') for s in sources]
    binding = dict(inventory=sha(inventory),references=sha(path),predictions=receipt_hashes,
                   script=sha(Path(__file__)))
    if confirm(marker,binding):
        return
    cpus = int(os.getenv('SLURM_CPUS_PER_TASK','4'))
    with ProcessPoolExecutor(max_workers=min(cpus,8)) as pool:
        p = join([r for a in pool.map(partial(summarize_sweep,side=side),sources) for r in a])
    table = dest/f'{shard:02d}_count.csv'
    csv_rows(table,area_rows(p,shard))
    outputs = {table.name:sha(table)}
    if p:
        precise = (p['state']==3)&(p['row_state']==7)
        np.testing.assert_equal(len(np.unique(p['ls_id'])),len(p['ls_id']))
        compact = dest/f'{shard:02d}_sky.npz'
        atomic_npz(compact,**{k:p[k][precise] for k in ('ls_id','TYPE','RA','DEC','region','r','g0','r0','z0',
                      'EBV','qrf_median_kpc','primary_survey_owner','PSFDEPTH_R','PSFSIZE_R','FRACFLUX_R',
                      'galactic_latitude')})
        outputs[compact.name] = sha(compact)
    name = f'{shard:02d}.npz'
    if name in depend['outputs']:
        t = load(verified(path.parent/name,depend['outputs'][name]))
        ref = join([r for s in sources
                    if (r:=reference_sweep(Path(s['filename']).stem,side=side,t=t))])
        if ref:
            finite = np.isfinite(ref['qrf_quantiles_kpc'][:,2])
            j = unique_join(ref['ls_id'],t['ls_id'])
            if np.any(j<0):
                raise ValueError('Reference support lost a native row')
            value = np.full(len(j),np.nan)
            flag = np.zeros(len(j),bool)
            chosen = []
            for e in (-np.inf,20,21,22):
                upper = 20 if e==-np.inf else e+1
                use = (ref['r']>=e)&(ref['r']<upper)&finite
                chosen.extend(choose(ref['ls_id'],use,32,3931))
                for kind in ('STAR','GALAXY','QSO'):
                    chosen.extend(choose(ref['ls_id'],use&(ref['spectype']==kind),24,3932))
            c = np.unique(chosen).astype(int)
            value[c],flag[c] = nominal_control(t,nominal_setup(side),j[c])
            target = dest/f'{shard:02d}_reference.npz'
            atomic_npz(target,**ref,nominal_qnear=value,nominal_evaluated=np.isfinite(value),nominal_numerical_flag=flag)
            outputs[target.name] = sha(target)
    save_json(marker,dict(binding=binding,outputs=outputs,job=os.getenv('SLURM_JOB_ID')))


def same_label():
    out = OUT/'validation/overlap'
    out.mkdir(parents=True,exist_ok=True)
    south = read_columns(read_cache(EVIDENCE,'Gaia.npz'),
        ('ls_id','source_id','RA','DEC','TYPE','reference_kpc','high_quality','qrf_training','consensus_class'))
    south = subset(south,(south['TYPE']=='PSF')&south['high_quality']&~south['qrf_training']&(south['source_id']>0))
    photo = read_columns(read_cache(SOUTH_PHOTO,'photometry.npz'),('ls_id','TYPE','magnitude','magnitude_error'))
    reference = OUT/'north/reference'
    northern = join([read_columns(path,('ls_id','source_id','RA','DEC','TYPE','mag_grz','error_grz','row_state'))
                     for path in sorted(reference.glob('[0-9][0-9].npz'))])
    northern = subset(northern,(northern['row_state']==7)&(northern['TYPE']=='PSF'))
    n = unique_join(northern['source_id'],south['source_id'])
    dup = unique_join(northern['source_id'],northern['source_id'])>=0
    i = np.flatnonzero((n>=0)&dup)
    near = separation(northern['RA'][i],northern['DEC'][i],south['RA'][n[i]],south['DEC'][n[i]])
    i = i[near]
    s = n[i]
    z = unique_join(south['ls_id'][s],photo['ls_id'])
    if np.any(z<0):
        raise ValueError('Missing matched southern native photometry')
    if not np.all(photo['TYPE'][z]=='PSF'):
        raise ValueError('Same-label model comparison requires native PSF on both sides')
    grz = [0,1,3]
    mag,err = photo['magnitude'][z][:,grz],photo['magnitude_error'][z][:,grz]
    keep = np.isfinite(mag).all(axis=1)&np.isfinite(err).all(axis=1)&(err<.1).all(axis=1)
    n,s,mag,err = i[keep],s[keep],mag[keep],err[keep]
    ids = northern['source_id'][n]
    dec = northern['DEC'][n]
    north_phot = northern['mag_grz'][n],northern['error_grz'][n]
    d, cls = south['reference_kpc'][s],south['consensus_class'][s]
    block = (np.floor(northern['RA'][n]/5)+3*np.floor(northern['DEC'][n]/5)).astype(int)%4
    train = choose(ids,block<3,18000,3933)
    test = choose(ids,block==3,6000,3934)
    if min(len(train),len(test))<128:
        raise ValueError('Too few native pairs for identical-label fit')
    q = []
    mu = 10+5*np.log10(d)
    for m,e in ((mag,err),north_phot):
        model = HistogramQRF(2)
        model.forest.n_jobs = min(12,int(os.getenv('SLURM_CPUS_PER_TASK','1')))
        model.fit(m[train],m[train,0]-mu[train],cls[train])
        model.forest.n_jobs = 1
        q.append(join([frozen_qrf(model,m[j],e[j],ids[j]) for j in np.array_split(test,8)]))
    shared = q[0]['supported']&q[1]['supported']
    comparison = []
    for a,b in ((0,2),(2,5),(5,10),(10,np.inf)):
        for c in (-2,-1,0,1,2):
            mask = shared&(d[test]>=a)&(d[test]<b)
            if c!=-2:
                mask &= cls[test]==c
            for label,result in zip(('DECam same labels','BASS/MzLS same labels'),q):
                comparison.append(dict(model=label,class_label=c,low_kpc=a,high_kpc=b,
                    **distance_summary(10+5*np.log10(result['distance'][mask]),mu[test][mask])))
    csv_rows(out/'distances.csv',comparison)
    atomic_npz(out/'paired.npz',north_ls_id=northern['ls_id'][n[test]],south_ls_id=south['ls_id'][s[test]],
               source_id=ids[test],TYPE=northern['TYPE'][n[test]],class_label=cls[test],reference_kpc=d[test],
               north_mag=north_phot[0][test],south_mag=mag[test],DEC=dec[test],shared_support=shared,
               north_q=q[1]['distance'],south_q=q[0]['distance'])
    summary = []
    for low,high in ((31.,32.375),(32.375,34.0)):
        use = shared&(dec[test]>=low)&(dec[test]<high)
        diff = q[1]['distance'][use,2]/q[0]['distance'][use,2]-1
        summary.append(dict(dec_min=low,dec_max=high,shared_pairs=int(use.sum()),
                            median_north_to_south_shift=float(np.median(diff)) if use.any() else None))
    csv_rows(out/'seam.csv',summary)
    save_json(out/'complete.json',dict(train=len(train),test=len(test),shared=int(shared.sum()),
               job=os.getenv('SLURM_JOB_ID'),outputs={p.name:sha(p) for p in out.iterdir() if p.name!='complete.json'}))


def main():
    require_compute()
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('stage',choices=('group','overlap'))
    a.add_argument('side',choices=('north','south'))
    a.add_argument('--shard',type=int,default=0)
    args = a.parse_args()
    if args.stage=='overlap':
        same_label()
    else:
        group(args.side,args.shard)


if __name__=='__main__':
    main()
