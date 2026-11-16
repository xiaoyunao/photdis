#!/usr/bin/env python3
"""Re-fit the existing photometry classifiers on wider class coverage."""
import argparse
import json
import os

import joblib
import numpy as np
from sklearn.base import clone

from photdis.reference_v1_4 import unique_join
from photdis.stage1 import require_compute
from photdis.v3.controls import apply_temperature, calibrate_temperature, probe_features
from photdis.v3.metrics import distance_summary
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_4_near_forecast import prepare
from v3_6_vac_distances import read_columns
from v3_9_catalog import OUT as EXPANSION, from_north
from v3_9_references import join, predictor_features, subset
from v3_10_evidence import DEST as EVIDENCE

DEST = ROOT/'data/v3_10/selection'
ROUTES = {'north':('north',),'south':('south_low','south_faint')}
MAGS = ((-np.inf,20.),(20.,21.),(21.,22.),(22.,23.))
SHELLS = ((0.,2.),(2.,5.),(5.,10.),(10.,np.inf))
THRESHOLDS = (.50,.65,.80,.90)


def references(side):
    keys = ('TYPE','ls_id','source_id','r','sky_role','class_label','row_state','redrock_quality_assessed',
            'redrock_SPECTYPE','spectype','dwarf_score','qrf_quantiles_kpc','evaluation_row','gaia_state',
            'specdis_kpc','spdist_kpc','specdis_quality','spdist_quality','reference_kpc','nominal_qnear',
            'nominal_numerical_flag')
    xs,files = [],{}
    for region in ROUTES[side]:
        root = EVIDENCE/region
        marker = root/'complete.json'
        f = verified(root/'references.npz',json.loads(marker.read_text())['outputs']['references.npz'])
        d = read_columns(f,keys)
        keep = (d['row_state']==7)&((d['class_label']>=0)|np.isin(d['spectype'],('QSO','GALAXY'))|
                                  d['redrock_quality_assessed'])
        xs.append(dict(**subset(d,keep),region=np.full(int(keep.sum()),region,dtype='U11')))
        files[str(marker.relative_to(ROOT))] = sha(marker)
    t = join(xs)
    if not np.all(t['TYPE']=='PSF') or len(np.unique(t['ls_id']))!=len(t['ls_id']):
        raise ValueError('Selection inputs violate native PSF identities')
    return t,files


def features(side, t, inputs):
    rows = EXPANSION/side/'reference'
    columns = ('mag_grz',) if side=='north' else ('flux','flux_error','transmission','band_mask','RA','DEC','EBV')
    x = None
    covered = np.zeros(len(t['ls_id']),bool)
    for receipt in sorted(rows.glob('[0-9][0-9].json')):
        marker = json.loads(receipt.read_text())
        for name,digest in marker['outputs'].items():
            p = read_columns(verified(receipt.parent/name,digest),('ls_id','TYPE',*columns))
            j = unique_join(p['ls_id'],t['ls_id'])
            keep = np.flatnonzero(j>=0)
            j = j[keep]
            if not len(j):
                continue
            if covered[j].any() or not np.all(p['TYPE'][keep]=='PSF'):
                raise ValueError('Overlapping native reference shards in selection')
            v = subset(p,keep)
            X = predictor_features(v['mag_grz']) if side=='north' else probe_features(v,'grz')
            if x is None:
                x = np.full((len(covered),X.shape[1]),np.nan)
            x[j] = X
            covered[j] = True
        inputs[str(receipt.relative_to(ROOT))] = sha(receipt)
    if not covered.all():
        raise ValueError('Missing native photometry for labeled rows')
    return x


def fit(side, x, t, inputs):
    if side=='north':
        frozen,digest = from_north()
        base = frozen['classifier']
        inputs['frozen_north_model_sha256'] = digest
    else:
        _,probe,artifacts = prepare()
        base = probe['model']
        inputs.update({str(f.relative_to(ROOT)):sha(f) for f in artifacts})
    if base is None:
        raise ValueError('Cannot compare with a missing frozen classifier')
    ids = np.where(t['source_id']>0,t['source_id'],-t['ls_id'])
    label = t['class_label']
    role = t['sky_role']
    known = (label>=0)&~t['redrock_quality_assessed'] & ~np.isin(t['spectype'],('QSO','GALAXY'))
    known &= unique_join(ids,ids)>=0
    train = known&(role<3)
    cal = known&(role==3)
    test = (role==4)&t['evaluation_row']
    model = clone(base)
    np.testing.assert_array_equal(np.unique(label[train]),[0,1,2])
    model.fit(x[train],label[train])
    temp = calibrate_temperature(model.predict_proba(x[cal]),label[cal])
    score = apply_temperature(model.predict_proba(x),temp)[:,0]
    return dict(model=model,temperature=temp),score,train,cal,test


def evaluate(t, new, eligible):
    quantiles = t['qrf_quantiles_kpc']
    in_support = np.isfinite(quantiles).all(axis=1)&(quantiles>0).all(axis=1)
    masks = [(k,t['class_label']==i) for i,k in enumerate(('dwarf','giant','other_star'))]
    for k in ('QSO','GALAXY'):
        masks.append((k+'_qualified',t['redrock_quality_assessed']&(t['redrock_SPECTYPE'].astype('U')==k)))
        masks.append((k+'_raw',t['spectype']==k))
    rows,distances = [],[]
    for region in np.unique(t['region']):
        for lo,hi in MAGS:
            p = eligible&(t['region']==region)&(t['r']>=lo)&(t['r']<hi)
            for label,mask in masks:
                for method,score in (('published',t['dwarf_score']),('expanded_labels',new)):
                    # Use the same supported rows for both classifiers at every threshold.
                    k = p&mask&in_support&np.isfinite(t['dwarf_score'])&np.isfinite(new)
                    for cut in THRESHOLDS:
                        s = k&(score>=cut)
                        q = k&np.isfinite(t['nominal_qnear'])&~t['nominal_numerical_flag']
                        row = dict(region=region,r_min=lo,r_max=hi,class_label=label,method=method,
                                   threshold=cut,eligible=int((p&mask).sum()),supported=int(k.sum()),
                                   score_pass=int(s.sum()),sed_common=int(q.sum()),
                                   score_pass_on_sed_common=int((s&q).sum()),
                                   both_gates=int((s&q&(t['nominal_qnear']>=.90)).sum()))
                        rows.append(row)
                        if cut not in (.65,.80) or label.endswith(('_raw','_qualified')):
                            continue
                        for src,center,good in (
                            ('Gaia',t['reference_kpc'],t['gaia_state']==6),
                            ('SpecDis',t['specdis_kpc'],t['specdis_quality']),
                            ('SPDist',t['spdist_kpc'],t['spdist_quality'])):
                            for d0,d1 in SHELLS:
                                keep = good&(center>=d0)&(center<d1)&(center>0)
                                for stage,base in (('supported',k),('score',s)):
                                    j = base&keep
                                    dm = distance_summary(10+5*np.log10(quantiles[j]),10+5*np.log10(center[j]))
                                    distances.append(dict(region=region,r_min=lo,r_max=hi,class_label=label,
                                        method=method,threshold=cut,stage=stage,source=src,d_min=d0,d_max=d1,**dm))
    return rows,distances


def run(side):
    t,inputs = references(side)
    x = features(side,t,inputs)
    model,p,train,cal,test = fit(side,x,t,inputs)
    rows,distances = evaluate(t,p,test)
    d = DEST/side
    d.mkdir(parents=True,exist_ok=True)
    joblib.dump(model,d/'model.joblib')
    csv_rows(d/'class_retention.csv',rows)
    csv_rows(d/'heldout_distances.csv',distances)
    atomic_npz(d/'scores.npz',ls_id=t['ls_id'],TYPE=t['TYPE'],score=p,score_original=t['dwarf_score'],
               training=train,calibration=cal,held_out=test)
    save_json(d/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,train=int(train.sum()),
                                   calibration=int(cal.sum()),test=int(test.sum()),
                                   outputs={a.name:sha(a) for a in d.iterdir() if a.name!='complete.json'}))


if __name__=='__main__':
    require_compute()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('side',choices=tuple(ROUTES))
    run(parser.parse_args().side)
