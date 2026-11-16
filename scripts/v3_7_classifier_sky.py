#!/usr/bin/env python3
"""Measure sky-feature effects while preserving classifier training identities."""
import json
from pathlib import Path

import healpy as hp
import joblib
import numpy as np
from sklearn.base import clone

from photdis.stage1 import require_compute
from photdis.v3.controls import apply_temperature, calibrate_temperature, probe_features, probability_metrics
from v2_1_near import ROOT, csv_rows, save_json, sha
from v3_3_sed import BASE, MANIFESTS, verified
from v3_6_distance_evidence import OUT as EVIDENCE, SHELLS
from v3_6_reference_photometry import OUT as PHOTO
from v3_6_vac_distances import DISTANCE_EDGES, read_columns
from v3_7_reference_comparison import SOURCES, load_source

OUT = ROOT/'data/v3_7/classifier_sky'
FEATURES = ('flux','flux_error','transmission','band_mask','RA','DEC','EBV')


def without_sky(x):
    # The grz control places sin(RA), cos(RA) and sin(Dec) in columns 1–3.
    return np.column_stack((x[:,0],x[:,4:]))


def fit(controls):
    original = MANIFESTS/'controls/complete.json'
    record = json.loads(original.read_text())
    train = read_columns(verified(BASE/'controls/training.npz',record['outputs']['training.npz']),
                        ('identity','TYPE','rows','class_label'))
    probe = joblib.load(verified(BASE/'controls/probes.joblib',record['outputs']['probes.joblib']))['grz']
    reference = verified(BASE/'references.npz',record['inputs'][str((BASE/'references.npz').relative_to(ROOT))])
    ref = read_columns(reference,('identity','TYPE','role','class_label',*FEATURES))
    row = train['rows']
    np.testing.assert_array_equal(ref['identity'][row],train['identity'])
    if not (ref['TYPE'][row]=='PSF').all() or not (train['TYPE']=='PSF').all():
        raise ValueError('The matched classifier training sample must retain native PSFs')
    y = ref['class_label']
    idx = row[train['class_label']>=0]
    expect = y>=0
    np.testing.assert_array_equal(y[idx],train['class_label'][train['class_label']>=0])
    hyper = (ref['role']==3)&expect
    dev = (ref['role']==4)&expect
    if not (ref['TYPE'][hyper|dev]=='PSF').all():
        raise ValueError('Calibration and development require native PSFs')
    if len(idx)!=record['classifier_training_sources'] or int(hyper.sum())!=record['temperature_calibration_sources']:
        raise ValueError('Original classifier identity counts disagree')
    if np.intersect1d(ref['identity'][idx],ref['identity'][hyper|dev]).size:
        raise ValueError('Training identities appear in classifier evaluation')
    fields = lambda selection: {k:ref[k][selection] for k in FEATURES}
    a,b,c = (probe_features(fields(s),'grz') for s in (idx,hyper,dev))
    model = clone(probe['model']).fit(without_sky(a),y[idx])
    temperature = calibrate_temperature(model.predict_proba(without_sky(b)),y[hyper])
    paired = []
    for name,clf,t,x in (('published_grz',probe['model'],probe['temperature'],c),
                         ('grz_without_sky',model,temperature,without_sky(c))):
        q = clf.predict_proba(x)
        for label,p in (('raw',q),('temperature_calibrated',apply_temperature(q,t))):
            paired.append(dict(model=name,probabilities=label,temperature=t,**probability_metrics(p,y[dev])))
    controls.extend([original,reference])
    return probe,model,temperature,paired


def selection_comparison(source,receipt,photo,probe,model,temp,excluded):
    a,excluded[source] = load_source(source,receipt)
    name = f'{source}.npz'
    coords = read_columns(verified(EVIDENCE/name,receipt['outputs'][name]),('catalog_row','RA','DEC'))
    order = np.argsort(coords['catalog_row'])
    j = order[np.searchsorted(coords['catalog_row'][order],a['catalog_row'])]
    np.testing.assert_array_equal(coords['catalog_row'][j],a['catalog_row'])
    ra,dec = coords['RA'][j],coords['DEC'][j]
    i = np.searchsorted(photo['catalog_row'],a['catalog_row'])
    np.testing.assert_array_equal(photo['ls_id'][i],a['ls_id'])
    if not (photo['TYPE'][i]=='PSF').all():
        raise ValueError('Unverified native PSF in classifier replay')
    with np.errstate(divide='ignore',invalid='ignore'):
        error = 1./np.sqrt(photo['flux_ivar'][i])
    t = {k:photo[k][i] for k in ('flux','transmission','band_mask','EBV')}
    t.update(flux_error=error,RA=ra,DEC=dec)
    x = probe_features(t,'grz')
    published = apply_temperature(probe['model'].predict_proba(x),probe['temperature'])[:,0]
    candidate = apply_temperature(model.predict_proba(without_sky(x)),temp)[:,0]
    score = a['dwarf_selection_score']
    same = abs(published-score)<=1e-8
    sky = hp.ang2pix(4,ra,dec,nest=True,lonlat=True)
    shell = np.searchsorted(DISTANCE_EDGES,a['reference_kpc'],side='right')-1
    if not np.isfinite(candidate).all():
        raise ValueError('No-sky predictions must have finite class probabilities')
    # Compare the two fits calculated on the same bytes and expose any
    # difference from archived forecast scores rather than mixing denominators.
    for label,gate in (('dwarf',a['consensus_class']==0),('giant',a['consensus_class']==1),
                       ('other',a['consensus_class']==2),('unknown',a['consensus_class']<0)):
        for j,distance in enumerate(SHELLS):
            use = gate&(shell==j)
            use = np.flatnonzero(use)
            h = sky[use]
            for cell in (-1,*np.unique(h)):
                k = use if cell==-1 else use[h==cell]
                for limit in (.65,.8,.95):
                    yield dict(source=source,reference_class=label,reference_shell=distance,
                               healpix_nside4=int(cell),dwarf_threshold=limit,n=len(k),
                               archived_score_agreement=int(same[k].sum()),stored=int((score[k]>=limit).sum()),
                               published_replayed=int((published[k]>=limit).sum()),
                               without_sky=int((candidate[k]>=limit).sum()))


def main():
    require_compute()
    marker = OUT/'complete.json'
    if marker.exists():
        raise FileExistsError(marker)
    inputs = [EVIDENCE/'complete.json',PHOTO/'complete.json',Path(__file__).resolve()]
    evidence,meta = (json.loads(p.read_text()) for p in inputs[:2])
    photo = read_columns(verified(PHOTO/'photometry.npz',meta['outputs']['photometry.npz']),
                        ('catalog_row','ls_id','TYPE','flux','flux_ivar','transmission','band_mask','EBV'))
    probe,model,temp,summary = fit(inputs)
    OUT.mkdir(parents=True,exist_ok=True)
    csv_rows(OUT/'development.csv',summary)
    excluded = {}
    csv_rows(OUT/'retention.csv',(row for source in SOURCES for row in
                                 selection_comparison(source,evidence,photo,probe,model,temp,excluded)))
    joblib.dump(dict(model=model,temperature=temp),OUT/'diagnostic_classifier.joblib')
    save_json(marker,dict(inputs={str(p.relative_to(ROOT)):sha(p) for p in inputs},
        excluded_morphology=excluded,training_type='native PSF only',
        comparison_scope='classifier-only retention, before nominal SED admission',
        outputs={p.name:sha(p) for p in OUT.iterdir() if p!=marker}))


if __name__=='__main__':
    main()
