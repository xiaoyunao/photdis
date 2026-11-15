#!/usr/bin/env python3
"""Summarize distance error by observed and reference-source properties (F04)."""
import json
import os
from pathlib import Path

import numpy as np

from photdis.stage1 import require_compute
from photdis.v3.data import BANDS, SIGNIFICANT, WEAK
from photdis.v3.metrics import distance_summary
from v2_1_near import ROOT, csv_rows, load, save_json, sha
from v3_3_sed import verified
from v3_6_distance_evidence import CLASSES, OUT as EVIDENCE
from v3_6_reference_photometry import OUT as PHOTO
from v3_6_vac_distances import DISTANCE_EDGES


OUT = ROOT/'data/v3_6/condition_errors'
EDGES = {
    'reference_kpc':DISTANCE_EDGES,
    'observed_r':(-np.inf,16,18,20,21,22,23,np.inf),
    'dereddened_g_r':(-np.inf,.2,.5,.8,1.1,1.4,np.inf),
    'dereddened_r_z':(-np.inf,0,.25,.5,.75,1.,np.inf),
    'maximum_grz_error_mag':(0,.02,.05,.1,.2,np.inf),
    'EBV':(0,.025,.05,.075,.1,np.inf),
    'spectroscopic_feh':(-np.inf,-2.,-1.5,-1.,-.5,0,.5,np.inf),
    'spectroscopic_logg':(-np.inf,1.,2.,3.,4.,4.5,np.inf),
    'spectroscopic_teff':(0,3500.,4500.,5500.,6500.,8000.,np.inf),
}


def partitions(a, photo):
    g,r,z = (BANDS.index(x) for x in ('G','R','Z'))
    magnitudes = photo['magnitude']
    failed = np.full(len(a['ls_id']),np.nan)
    values = {k:a[k] for k in ('reference_kpc','observed_r','dereddened_g_r')}
    values.update(dereddened_r_z=magnitudes[:,r]-magnitudes[:,z],
                  maximum_grz_error_mag=np.max(photo['magnitude_error'][:,(g,r,z)],axis=1),
                  EBV=photo['EBV'])
    quality = a.get('rvs_parameter_quality',False)
    for key in ('feh','logg','teff'):
        value = np.where(quality,a.get(f'rvs_{key}',failed),np.nan)
        valid_key = f'{key}_valid'
        if valid_key in a:
            external = np.where(a[valid_key],a[key],np.nan)
            value = np.where(np.isfinite(external),external,value)
        values[f'spectroscopic_{key}'] = value
    for key,x in values.items():
        lohi = EDGES[key]
        i = np.searchsorted(lohi,x,side='right')
        i[~np.isfinite(x)|(i==len(lohi))] = 0
        labels = ('unmeasured',*(f'[{lo:g}, {hi:g})' for lo,hi in zip(lohi[:-1],lohi[1:])))
        yield key,labels,i
    for band in ('W1','W2'):
        x = photo['band_state'][:,BANDS.index(band)]
        i = np.where(x==SIGNIFICANT,2,np.where(x==WEAK,1,0))
        yield band,('missing_or_unusable','legal_weak_or_signed','significant'),i


def summarize(source, a, photo):
    d = a['reference_kpc']
    with np.errstate(divide='ignore',invalid='ignore'):
        target = 10+5*np.log10(d)
        prediction = 10+5*np.log10(a['qrf_quantiles_kpc'])
    fields = list(partitions(a,photo))
    all_rows = np.ones(len(d),bool)
    history = ~a['qrf_training'] & ~a['recorded_history']
    for cut,selection in (('supported',all_rows),('primary',a['primary']),('strict',a['strict'])):
        for label in ('all',*CLASSES,'unknown'):
            cl = (all_rows if label=='all' else a['consensus_class']==CLASSES.index(label)
                  if label in CLASSES else a['consensus_class']<0)
            for quality,qual in (('available',all_rows),('high_quality',a['high_quality'])):
                for scope,scope_mask in (('all',all_rows),('outside_training',~a['qrf_training']),
                                         ('outside_recorded_history',history)):
                    select = np.flatnonzero(cl&qual&scope_mask&selection&np.isfinite(target))
                    for key,labels,bins in fields:
                        active = bins[select]
                        for i,bin_label in enumerate(labels):
                            take = select[active==i]
                            stats = distance_summary(prediction[take],target[take])
                            err = a['reference_error_agreement'][take]
                            valid = np.isfinite(err)
                            count = valid.sum(axis=0)
                            mean = np.divide(np.nansum(err,axis=0),count,
                                             out=np.full(2,np.nan),where=count>0)
                            yield dict(source=source,selection=cut,reference_class=label,
                                       reference_quality=quality,history=scope,property=key,
                                       bin=bin_label,bin_index=i,finite_reference_rows=len(take),
                                       reference_error_n68=count[0],reference_error_n95=count[1],
                                       error_integrated_agreement68=mean[0],error_integrated_agreement95=mean[1],
                                       **stats)


def main():
    require_compute()
    p = PHOTO/'complete.json'
    e = EVIDENCE/'complete.json'
    photo, evidence = [json.loads(x.read_text()) for x in (p,e)]
    if photo['inputs'][str(e.relative_to(ROOT))]!=sha(e):
        raise ValueError('Reference photometry and distance evidence differ')
    phot = load(verified(PHOTO/'photometry.npz',photo['outputs']['photometry.npz']))
    if not (phot['TYPE']=='PSF').all():
        raise ValueError('Conditional errors require native PSF')
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'complete.json').exists():
        raise FileExistsError(OUT/'complete.json')
    table = OUT/'conditions.csv'

    def rows():
        for source in evidence['rows']:
            path = EVIDENCE/f'{source}.npz'
            a = load(verified(path,evidence['outputs'][path.name]))
            order = np.searchsorted(phot['catalog_row'],a['catalog_row'])
            np.testing.assert_array_equal(phot['ls_id'][order],a['ls_id'])
            np.testing.assert_array_equal(phot['catalog_row'][order],a['catalog_row'])
            np.testing.assert_array_equal(phot['band_state'][order],a['band_state'])
            yield from summarize(source,a,{k:phot[k][order] for k in
                                          ('magnitude','magnitude_error','EBV','band_state')})
            print(source,len(order),flush=True)

    csv_rows(table,rows())
    save_json(OUT/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),
        inputs={str(x.relative_to(ROOT)):sha(x) for x in (p,e,Path(__file__))},
        interpretation=('Unweighted, source-conditioned reference comparisons. '
                        'Intervals describe agreement with reference centers or their available errors. '
                        'Selection and training history are shown for each cell.'),
        outputs={table.name:sha(table)}))


if __name__=='__main__':
    main()
