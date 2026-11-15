#!/usr/bin/env python3
"""Explain same-frame Gaia attrition and test the cuts on archived external Gaia predictions."""
import json
import os
from pathlib import Path

import numpy as np

from photdis.gaia import derive_gaia_labels, join_gaia
from photdis.stage1 import require_compute
from photdis.v3.controls import apply_temperature, probe_features
from photdis.v3.likelihood import normalize_joint
from photdis.v3.metrics import distance_summary
from v2_1_near import ROOT, csv_rows, load, save_json, sha
from v3_3_prospective import FORECAST as OLD_FORECAST, SCORES as OLD_SCORES
from v3_3_prospective_inputs import OUT as OLD_INPUTS
from v3_3_sed import BASE, MANIFESTS, verified
from v3_4_near_reference_support import separation
from v3_4_near_validate import FREEZE, GAIA, OUT as EVIDENCE, geometry, load_closed, tile_inventory
from v3_4_near_validation_inputs import CONFIG
import joblib

OUT = EVIDENCE.parent/'followup'


def gaia_attrition(cfg, predicted, index, hashes):
    freeze = json.loads(FREEZE.read_text())
    receipt = json.loads((EVIDENCE/'complete.json').read_text())
    verified(FREEZE, receipt['freeze_sha256'])
    hashes[str(FREEZE.relative_to(ROOT))] = receipt['freeze_sha256']
    s = index['source_id']
    if any(freeze['gaia_tiles'].get(k) != m for k,m in tile_inventory(s).items()):
        raise ValueError('Gaia archive no longer agrees with the frozen outcome read')
    gaia = join_gaia(s, GAIA)
    labels = derive_gaia_labels(gaia)
    position = gaia['gaia_match'] & separation(index['RA'], index['DEC'], gaia['ra'], gaia['dec'])
    conditions = dict(match=gaia['gaia_match'], position=position, finite=labels['finite_parallax'],
        valid_solution=labels['astrometric_solution_valid'], before_duplicate=labels['astro_quality_before_duplicate'],
        quality=labels['astro_quality'], zero_point=labels['zpt_valid'],
        qualified=position & labels['astro_quality'] & labels['finite_parallax'] & labels['zpt_valid'])
    evidence = load(verified(EVIDENCE/'fresh_gaia.npz', receipt['outputs']['fresh_gaia.npz']))
    np.testing.assert_array_equal(conditions['qualified'], evidence['signed_eligible'])
    rows = index['prediction_row']
    return [dict(selection=k, n=int(m.sum()), **{key:int((m & v).sum()) for key,v in conditions.items()})
            for k,m in [('envelope', np.ones(len(s),bool)),
                        *[(k,predicted[k][rows]) for k in ('primary','strict')]]]


def historical_payload(path, view, hashes):
    marker = path/'complete.json'
    rec = json.loads(marker.read_text())
    hashes[str(marker.relative_to(ROOT))] = sha(marker)
    f = path/(view+'.npz')
    hashes[str(f.relative_to(ROOT))] = rec['outputs'][f.name]
    return load(verified(f, rec['outputs'][f.name]))


def score_archived_cuts(cfg, hashes):
    marker = OLD_INPUTS/'complete.json'
    input_digest = json.loads(marker.read_text())['sha256']
    photo_path = verified(OLD_INPUTS/'photometry.npz', input_digest)
    hashes[str(photo_path.relative_to(ROOT))] = input_digest
    p = load(photo_path)
    assert (p['TYPE']=='PSF').all()
    nominal, qrf = [historical_payload(OLD_FORECAST,k,hashes) for k in ('nominal','qrf')]
    scores = {k:historical_payload(OLD_SCORES,k,hashes) for k in ('nominal','qrf')}
    for table in (nominal,qrf,*scores.values()):
        np.testing.assert_array_equal(p['identity'],table['identity'])
        np.testing.assert_array_equal(p['TYPE'],table['TYPE'])
    control = MANIFESTS/'controls/complete.json'
    probe_path = verified(BASE/'controls/probes.joblib', json.loads(control.read_text())['outputs']['probes.joblib'])
    hashes[str(probe_path.relative_to(ROOT))] = sha(probe_path)
    probe = joblib.load(probe_path)['grz']
    dwarf = apply_temperature(probe['model'].predict_proba(probe_features(p,'grz')),probe['temperature'])[:,0]
    near_mu = 10+5*np.log10(cfg['near_kpc'])
    near = np.asarray([normalize_joint(nominal['mu'], curve[None,:]).cdf(near_mu).sum()
                       for curve in nominal['logp']])
    score = scores['qrf']
    np.testing.assert_array_equal(score['eligible'],scores['nominal']['eligible'])
    reference, _, upper = geometry(score['corrected_parallax'],score['parallax_error'],score['eligible'],cfg)
    cdf = np.cumsum(qrf['mass'],axis=1)
    qtile = np.asarray([np.interp([.025,.16,.5,.84,.975],np.r_[0.,a],b) for a,b in zip(cdf,qrf['edges'])])
    r = 22.5-2.5*np.log10(p['flux'][:,1])
    groups = {'all':r < 20,'r_16_18':(r>=16)&(r<18),'r_18_20':(r>=18)&(r<20)}
    groups.update({k:(r<20)&(p['validation_role']==k) for k in np.unique(p['validation_role'])})
    weak = score['corrected_parallax'] < 5*score['parallax_error']
    rows = []
    for k in ('primary','strict'):
        admitted = qrf['supported'] & (dwarf>=cfg[k]['minimum_dwarf_score']) & (near>=cfg[k]['minimum_near_probability'])
        for name, group in groups.items():
            use = admitted & group
            eligible = use & score['eligible']
            contrast = score['log_score']-scores['nominal']['log_score']
            rows.append(dict(selection=k,group=name,selected_n=int(use.sum()),signed_n=int(eligible.sum()),
                weak_n=int((eligible&weak).sum()),near_geometry=int((use & (upper<cfg['near_kpc'])).sum()),
                mean_qrf_minus_nominal=float(contrast[eligible].mean()) if eligible.any() else np.nan,
                weak_mean_qrf_minus_nominal=float(contrast[eligible&weak].mean()) if (eligible&weak).any() else np.nan,
                **distance_summary(qtile[use],10+5*np.log10(reference[use]))))
    return rows


def main():
    require_compute()
    cfg = json.loads(CONFIG.read_text())
    hashes = {str(p.relative_to(ROOT)):sha(p) for p in (CONFIG,Path(__file__))}
    predicted, data = load_closed(hashes)
    attrition = gaia_attrition(cfg,predicted,data['fresh_gaia'],hashes)
    archived = score_archived_cuts(cfg,hashes)
    OUT.mkdir(parents=True,exist_ok=False)
    csv_rows(OUT/'gaia_attrition.csv',attrition)
    csv_rows(OUT/'archived_external.csv',archived)
    save_json(OUT/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'), inputs=hashes,
        new_outcomes_read=False, all_PSF=True, outputs={f.name:sha(f) for f in sorted(OUT.iterdir())}))


if __name__ == '__main__':
    main()
