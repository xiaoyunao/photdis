#!/usr/bin/env python3
"""Reference association against expansion sweeps and the northern native fit."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from photdis.baseline_v0 import HistogramQRF
from photdis.gaia import NPIX, SOURCE_ID_HEALPIX_FACTOR, derive_gaia_labels, join_gaia
from photdis.reference_v1_4 import unique_join
from photdis.stage1 import require_compute
from photdis.v3.data import stable_rank
from photdis.v3.metrics import distance_summary
from photdis.v3_3 import frozen_qrf
from v2_1_near import ROOT, atomic_npz, csv_rows, load, save_json, sha
from v3_3_sed import verified
from v3_4_near_reference_support import separation
from v3_4_near_validate import GAIA
from v3_5_dr11_production import CUTS
from v3_6_gaia_census import distances
from v3_6_vac_distances import OUT as VAC, read_columns
from v3_8_north import native_sed
from v3_9_catalog import OUT, confirm, settings, subset, thin

OLDER = ROOT/'data/baseline_v1_4/references.npz'
REFERENCE_PHOTOMETRY = ('ls_id','source_id','identity','TYPE','RA','DEC','EBV',
    'mag_grz','error_grz','flux','flux_error','transmission','band_mask','band_state',
    'region','row_state','r','g0','r0','FRACFLUX_R','PSFSIZE_R','PSFDEPTH_R')
VAC_FIELDS = ('source_id','reference_ra','reference_dec','distance_kpc','sigma_mu',
              'historical_quality','usable_uncertainty')


def raw_catalogs():
    """Start with raw VAC rows, independent of the legacy release association."""
    inputs = {'historical_labels':sha(OLDER)}
    catalogs = {}
    for kind in ('specdis','spdist'):
        path = VAC/kind/'complete.json'
        mark = json.loads(path.read_text())
        catalogs[kind] = read_columns(verified(path.parent/'vac.npz',mark['outputs']['vac.npz']),VAC_FIELDS)
        inputs[kind] = sha(path)
    fields = ('source_id','ls_id','RA','DEC','class_eligible','class_label','rvs_rr_spectype',
              'split')
    catalogs['spectra'] = read_columns(OLDER,fields)
    for catalog in catalogs.values():
        rows = np.flatnonzero(catalog['source_id']>0)
        catalog['_gaia'] = sorted_key(catalog['source_id'],rows)
    spectra = catalogs['spectra']
    spectra['_legacy'] = sorted_key(spectra['ls_id'],np.flatnonzero(spectra['ls_id']>0))
    return catalogs, inputs


def sorted_key(values, rows):
    ix = rows[np.argsort(values[rows],kind='stable')]
    return values[ix], ix


def lookup(values, index):
    keys, rows = index
    low = np.searchsorted(keys,values,'left')
    high = np.searchsorted(keys,values,'right')
    result = np.full(len(values),-1,'i8')
    good = high-low==1
    result[good] = rows[low[good]]
    return result


def matched(photo, catalog, *, legacy=False):
    """Require a unique key and agreement at the native source position."""
    row = lookup(photo['source_id'],catalog['_gaia'])
    if legacy:
        legacy_row = lookup(photo['ls_id'],catalog['_legacy'])
        conflict = (row>=0)&(legacy_row>=0)&(row!=legacy_row)
        row = np.where(conflict,-1,np.where(row>=0,row,legacy_row))
    idx = np.flatnonzero(row>=0)
    rak, deck = ('RA','DEC') if legacy else ('reference_ra','reference_dec')
    close = separation(photo['RA'][idx],photo['DEC'][idx],catalog[rak][row[idx]],catalog[deck][row[idx]])
    row[idx[~close]] = -1
    return row


def extract(source, *, side, catalogs, cap):
    parent = OUT/side/'photo'/Path(source['filename']).stem
    receipt = json.loads((parent/'complete.json').read_text())
    rows = []
    for name, digest in receipt['outputs'].items():
        if name=='parent_state.npz':
            continue
        p = read_columns(verified(parent/name,digest),REFERENCE_PHOTOMETRY)
        if not (p['TYPE']=='PSF').all():
            raise ValueError('Reference sample requires verified native PSF')
        reference = {}
        for kind, catalog in catalogs.items():
            idx = matched(p,catalog,legacy=(kind=='spectra'))
            present = idx>=0
            reference[f'{kind}_matched'] = present
            if kind=='spectra':
                cls = np.full(len(idx),-1,'i1')
                eligible = present.copy()
                eligible[present] = catalog['class_eligible'][idx[present]]
                cls[eligible] = catalog['class_label'][idx[eligible]]
                label = np.full(len(idx),'',dtype='U8')
                label[present] = catalog['rvs_rr_spectype'][idx[present]].astype('U8')
                reference.update(class_label=cls, spectype=label)
            else:
                d, s = np.full(len(idx),np.nan), np.full(len(idx),np.nan)
                d[present] = catalog['distance_kpc'][idx[present]]
                s[present] = catalog['sigma_mu'][idx[present]]
                quality = present.copy()
                quality[present] = catalog['historical_quality'][idx[present]]
                reference.update({f'{kind}_kpc':d, f'{kind}_sigma_mu':s, f'{kind}_quality':quality})
        with_spec = np.logical_or.reduce([reference[f'{k}_matched'] for k in catalogs])
        clean = p['row_state']>=6
        sample, weight = thin(p,clean&(p['source_id']>0)&~with_spec,cap,3902)
        select = np.flatnonzero((clean&with_spec)|np.isin(np.arange(len(clean)),sample))
        w = np.ones(len(clean))
        w[sample] = weight
        p.update(reference,reference_sampling_weight=w,
                 sweep=np.full(len(clean),parent.name,dtype='U24'))
        rows.append(subset(p,select))
    return rows


def join(source_rows):
    if not source_rows:
        return {}
    return {k:np.concatenate([x[k] for x in source_rows]) for k in source_rows[0]}


def group(side, shard):
    inventory_path = OUT/side/'inventory.json'
    entries = json.loads(inventory_path.read_text())['groups'][shard]
    cfg = settings()
    catalogs, ingredients = raw_catalogs()
    binding = dict(inventory=sha(inventory_path), script=sha(Path(__file__)), **ingredients)
    path = OUT/side/'reference'/f'{shard:02d}.json'
    if confirm(path,binding):
        return
    workers = min(8,int(os.getenv('SLURM_CPUS_PER_TASK','1')))
    process = partial(extract,side=side,catalogs=catalogs,
                      cap=cfg['north_reference_per_stratum_per_sweep'])
    with ThreadPoolExecutor(max_workers=workers) as pool:
        p = join([r for rows in pool.map(process,entries) for r in rows if len(r['ls_id'])])
    path.parent.mkdir(parents=True,exist_ok=True)
    name = f'{shard:02d}.npz'
    checks = {}
    if p:
        ids = p['source_id']
        valid = (ids>0)&(ids<NPIX*SOURCE_ID_HEALPIX_FACTOR)
        gaia = join_gaia(np.where(valid,ids,0),GAIA)
        flags = derive_gaia_labels(gaia)
        state,center,lo,hi = distances(p,valid,flags,gaia,json.loads(CUTS.read_text()))
        atomic_npz(path.parent/name,**p,gaia_state=state,reference_kpc=center,
                   reference_lo_kpc=lo,reference_hi_kpc=hi,
                   signed_parallax_mas=flags['p_corr'],parallax_error_mas=gaia['parallax_error'])
        checks[name] = sha(path.parent/name)
    save_json(path, dict(binding=binding, outputs=checks, rows=len(p.get('ls_id',[])),
                         job=os.getenv('SLURM_JOB_ID')))
    print(side, 'reference group', shard, len(p.get('ls_id',[])), flush=True)


def choose(ids, use, cap, salt):
    j = np.flatnonzero(use)
    return j[np.argsort(stable_rank(ids[j],salt),kind='stable')[:cap]]


def predictor_features(mag):
    return np.column_stack((mag[:,0]-mag[:,1], mag[:,1]-mag[:,2], mag[:,1]))


def northern_fit():
    root = OUT/'north/reference'
    sources = sorted(root.glob('[0-9][0-9].json'))
    expected = json.loads((OUT/'north/inventory.json').read_text())
    if len(sources)!=len(expected['groups']):
        raise ValueError('The full northern reference inventory is required')
    paths = [verified(root/name,digest) for path in sources
             for name,digest in json.loads(path.read_text())['outputs'].items()]
    out = OUT/'north/native_model'
    marker = out/'complete.json'
    binding = dict(receipts={p.name:sha(p) for p in sources},script=sha(Path(__file__)))
    if confirm(marker,binding):
        return
    p = join([load(path) for path in paths])
    if not np.all(p['TYPE']=='PSF'):
        raise ValueError('Northern labels must use native PSF photometry')
    good = unique_join(p['source_id'],p['source_id'])>=0
    good &= (p['source_id']>0)&(p['row_state']==7)
    role = (np.floor(p['RA']/10).astype(int)+3*np.floor((p['DEC']+90)/10).astype(int))%5
    d = p['reference_kpc']
    mag,err = p['mag_grz'],p['error_grz']
    cls,ids = p['class_label'],p['source_id']
    stellar = ~np.isin(p['spectype'],['QSO','GALAXY'])
    range_ok = np.isfinite(d)&(d>=.015)&(d<25)
    if not (good&stellar&range_ok).any():
        raise ValueError('No usable northern geometric reference stars')
    gaia_fit = choose(ids,good&stellar&range_ok&(role<3),61440,3903)
    spec_fit_ok = good&stellar&~range_ok&(role<3)&(cls==0)&p['specdis_quality']
    spec_fit_ok &= (p['specdis_kpc']>=.015)&(p['specdis_kpc']<25)
    spec_fit = choose(ids,spec_fit_ok,4096,3908)
    train = np.r_[gaia_fit,spec_fit]
    cal = choose(ids,good&stellar&range_ok&(role==3),8000,3904)
    test = choose(ids,good&stellar&range_ok&(role==4),12000,3905)
    if min(len(train),len(cal),len(test))<128:
        raise ValueError('Northern native reference split is too small')
    qrf = HistogramQRF(2)
    qrf.forest.n_jobs = min(16,int(os.getenv('SLURM_CPUS_PER_TASK','1')))
    fit_d = np.r_[d[gaia_fit],p['specdis_kpc'][spec_fit]]
    qrf.fit(mag[train],mag[train,0]-10-5*np.log10(fit_d),cls[train])
    qrf.forest.n_jobs = 1
    forecasts = []
    ix = np.r_[cal,test]
    for i in range(0,len(ix),1024):
        j = ix[i:i+1024]
        q = frozen_qrf(qrf,mag[j],err[j],ids[j])
        forecasts.append(dict(quantiles=q['distance'],supported=q['supported']))
    q = join(forecasts)
    cal_support = q['supported'][:len(cal)]
    if cal_support.sum()<128:
        raise ValueError('Northern zero-point calibration has too little in-support evidence')
    with np.errstate(invalid='ignore',divide='ignore'):
        delta = 5*np.log10(d[ix]/q['quantiles'][:,2])
    offset = np.median(delta[:len(cal)][cal_support])
    use = q['supported'][len(cal):]
    selected = test[use]
    estimates = q['quantiles'][len(cal):]*10**(offset/5)
    x = predictor_features(mag)
    spectroscopic = good&stellar&(cls>=0)&(role<3)
    learner = None
    if np.unique(cls[spectroscopic]).size>=2:
        learner = HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=12,random_state=3906)
        learner.fit(x[spectroscopic],cls[spectroscopic])
    sed_train = choose(ids,good&stellar&range_ok&(role<3),24000,3907)
    sed = native_sed({'flux':p['flux'],'trans':p['transmission'],'mag':mag},d,cls,sed_train)
    out.mkdir(parents=True,exist_ok=True)
    joblib.dump(dict(qrf=qrf,mu_offset=offset,classifier=learner,sed=sed),out/'model.joblib')
    atomic_npz(out/'validation.npz',**subset(p,test),qrf_quantiles_kpc=estimates,qrf_supported=use)
    fit_rows = np.r_[train,cal,test,np.flatnonzero(spectroscopic)]
    mode = np.repeat(['gaia_fit','specdis_fit','calibration','test','class_fit'],
                     [len(gaia_fit),len(spec_fit),len(cal),len(test),spectroscopic.sum()])
    atomic_npz(out/'roles.npz',ls_id=p['ls_id'][fit_rows],TYPE=p['TYPE'][fit_rows],
               source_id=ids[fit_rows],role=mode)
    metrics = []
    for a,b in zip((0,2,5,10),(2,5,10,np.inf)):
        j = (d[selected]>=a)&(d[selected]<b)
        metrics.append(dict(low_kpc=a,high_kpc=b,**distance_summary(
            10+5*np.log10(estimates[use][j]),10+5*np.log10(d[selected][j]))))
    csv_rows(out/'distance.csv',metrics)
    save_json(marker,dict(binding=binding,outputs={p.name:sha(p) for p in out.iterdir() if p.name!='complete.json'},
                         n_train=len(train),n_calibration=len(cal),n_test=len(test),
                         gaia_fit=len(gaia_fit),specdis_fit=len(spec_fit),
                         mu_offset=float(offset),n_spectroscopic_classifier=int(spectroscopic.sum()),
                         job=os.getenv('SLURM_JOB_ID')))
    print('Northern fit',len(train),len(cal),len(test),'calibration offset',offset,flush=True)


def main():
    require_compute()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('join','north_fit'))
    parser.add_argument('side',choices=('north','south'))
    parser.add_argument('--shard',type=int,default=0)
    a = parser.parse_args()
    if a.stage=='north_fit':
        northern_fit()
    else:
        group(a.side,a.shard)


if __name__=='__main__':
    main()
