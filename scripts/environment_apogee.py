#!/usr/bin/env python3
"""Recover a fixed supplementary APOGEE window where DESI VACs have gaps."""
import json
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path

import numpy as np

import environment_pilot as pilot
from environment_pilot import C, OUT, finish, sample
from photdis.stage1 import require_compute
from photdis.v3.controls import apply_temperature, probe_features
from v2_1_near import atomic_npz, load, sha
from v3_4_near_forecast import prepare
from v3_6_external_reference import apogee_inventory
from v3_9_references import matched, sorted_key, subset
from v3_10_selection import DEST as CLASSIFIERS
import joblib


def main():
    require_compute()
    dest=OUT/'apogee'
    dest.mkdir(parents=True,exist_ok=True)
    inputs={}
    source=apogee_inventory(inputs)
    good=np.flatnonzero(source['source_id']>0)
    cat=dict(_gaia=sorted_key(source['source_id'],good),
             reference_ra=source['RA'],reference_dec=source['DEC'])
    t=load(OUT/'forecast/roster.npz')
    q=load(OUT/'forecast/pdf.npz')
    ix=matched(t,cat)
    present=(ix>=0)&t['unique_G3']
    label=np.full(len(ix),-1,'i1')
    label[present]=source['class_label'][ix[present]]
    precise=present.copy()
    precise[present]=source['distance_precise'][ix[present]]
    d=np.full(len(ix),np.nan)
    err=d.copy()
    d[precise]=source['distance_kpc'][ix[precise]]
    err[precise]=(5/np.log(10))*source['distance_error_kpc'][ix[precise]]/d[precise]
    allowed=precise&t['support']&~t['qrf_fit_overlap']
    rows=np.concatenate([sample(t['ls_id'],allowed&(t['field']==name),
                                C['reference_per_field_per_provider'],25)
                         for name,_,_ in C['fields']])
    rows=np.sort(rows)
    eligible=np.isin(np.arange(len(ix)),rows)
    atomic_npz(dest/'references.npz',ls_id=t['ls_id'],TYPE=t['TYPE'],
               astro_distance_kpc=d,astro_sigma_mu=err,astro_evaluation=eligible,
               astro_classification=label,astro_matched=present)
    need=rows[~np.isin(rows,q['rows'])]
    if len(need):
        photo=load(OUT/'census/qualified.npz')
        models,_,paths=prepare()
        pilot.MODEL=models['qrf']
        pilot.PREDICT_INPUT=photo
        chunks=np.array_split(need,max(1,int(np.ceil(len(need)/192))))
        with ProcessPoolExecutor(max_workers=min(len(chunks),8),
                                  mp_context=multiprocessing.get_context('fork')) as pool:
            results=list(pool.map(pilot.evaluate_batch,chunks))
        p={key:np.concatenate([z[key] for z in results])
           for key in ('supported','mu_edges','mass','distance')}
        assert p['supported'].all() and (t['TYPE'][need]=='PSF').all()
        bounds=10+5*np.log10([2.,10.])
        widths=np.diff(p['mu_edges'],axis=1)
        cdf=[np.sum(np.clip((mu-p['mu_edges'][:,:-1])/widths,0,1)*p['mass'],axis=1) for mu in bounds]
        classifier=joblib.load(CLASSIFIERS/'south/model.joblib')
        score=apply_temperature(classifier['model'].predict_proba(probe_features(subset(photo,need),'grz')),
                                classifier['temperature'])[:,0]
        atomic_npz(dest/'pdf.npz',**p,rows=need,TYPE=t['TYPE'][need],ls_id=t['ls_id'][need],
                   dwarf_score=score,mass_2_10=cdf[1]-cdf[0])
    finish(dest,dict(inputs,forecast=sha(OUT/'forecast/complete.json'),script=sha(Path(__file__))))
    print('APOGEE matches',int(present.sum()),'qualified AstroNN',int(precise.sum()),
          'nonfit supported evaluated',len(rows),'new predictions',len(need),flush=True)


if __name__=='__main__':
    main()
