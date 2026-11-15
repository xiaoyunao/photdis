#!/usr/bin/env python3
"""Check whether existing PSF class references cover the actual application selection."""
import json
import os
from pathlib import Path

import numpy as np

from photdis.reference_v1_4 import unique_join
from photdis.stage1 import require_compute
from v2_1_near import ROOT, csv_rows, load, save_json, sha
from v3_3_sed import BASE, MANIFESTS, verified
from v3_4_near_reference_support import separation
from v3_4_near_validate import CONFIG, OUT as EVIDENCE, load_closed

OUT = EVIDENCE.parent/'label_recovery'


def main():
    require_compute()
    config = json.loads(CONFIG.read_text())
    hashes = {str(p.relative_to(ROOT)):sha(p) for p in (CONFIG,Path(__file__))}
    predicted, data = load_closed(hashes)
    app = data['application']
    marker = MANIFESTS/'cohorts.json'
    key = str((BASE/'references.npz').relative_to(ROOT))
    hashes[key] = json.loads(marker.read_text())['inputs'][key]
    hashes[str(marker.relative_to(ROOT))] = sha(marker)
    with np.load(verified(ROOT/key,hashes[key])) as z:
        use = np.flatnonzero((z['TYPE']=='PSF') & z['reference_stellar_eligible'] & (z['class_label']>=0))
        columns = {k:z[k][use] for k in ('identity','ls_id','RA','DEC','class_label','role','historical_fit')}
    native = np.where(app['gaia_unique'],app['source_id'],0)
    ix = unique_join(native, columns['identity'])
    legacy = unique_join(app['ls_id'],columns['ls_id'])
    conflict = (ix>=0) & (legacy>=0) & (ix!=legacy)
    ix = np.where(legacy>=0,legacy,ix)
    match = (ix>=0) & ~conflict & separation(app['RA'],app['DEC'],columns['RA'][ix],columns['DEC'][ix])
    label = np.where(match,columns['class_label'][ix],-1)
    independent_role = match & ~columns['historical_fit'][ix] & np.isin(columns['role'][ix],[3,4])
    evidence_receipt = EVIDENCE/'complete.json'
    rec = json.loads(evidence_receipt.read_text())
    hashes[str(evidence_receipt.relative_to(ROOT))] = sha(evidence_receipt)
    path = verified(EVIDENCE/'application.npz', rec['outputs']['application.npz'])
    hashes[str(path.relative_to(ROOT))] = rec['outputs']['application.npz']
    e = load(path)
    np.testing.assert_array_equal(app['identity'],e['identity'])
    p, err = e['p_corr'], e['parallax_error']
    p_limit = 1/config['near_kpc']
    delta = config['geometry_sigma_multiplier']*err
    # Post-opening sensitivity in the measurement space, with no weak-parallax inversion.
    parallax_near = e['signed_eligible'] & (p-delta > p_limit)
    parallax_far = e['signed_eligible'] & (p+delta < p_limit)
    r = app['observed_r']
    weights, rows = app['design_weight'], app['prediction_row']
    records = []
    for cut in ('primary','strict'):
        for name, group in [('all',np.ones(len(r),bool)), *[(f'r_{lo}_{hi}',(r>=lo)&(r<hi))
                                                           for lo,hi in [(14,16),(16,18),(18,20)]]]:
            selected = predicted[cut][rows] & group
            total = weights[selected].sum()
            records.append(dict(selection=cut, group=name, sample_n=int(selected.sum()),
                class_match=int((selected&match).sum()), conflict=int((selected&conflict).sum()),
                reference_dwarfs=int((selected&(label==0)).sum()), reference_giants=int((selected&(label==1)).sum()),
                reference_other=int((selected&(label==2)).sum()), fit_eligible_or_unknown_role=int(
                    (selected&match&~independent_role).sum()), opened_held_out=int((selected&independent_role).sum()),
                weighted_dwarf_lower=float(weights[selected&(label==0)].sum()/total),
                signed_interval_near=int((selected&parallax_near).sum()), signed_interval_far=int(
                    (selected&parallax_far).sum()), signed_interval_far_lower=float(weights[selected&parallax_far].sum()/total),
                signed_interval_far_upper=float(1-weights[selected&parallax_near].sum()/total)))
    OUT.mkdir(parents=True,exist_ok=False)
    csv_rows(OUT/'recovery.csv',records)
    save_json(OUT/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=hashes,all_PSF=True,new_outcomes_read=False,
        interpretation='Archived reference association and post-opening signed Gaussian interval sensitivity.',
        outputs={'recovery.csv':sha(OUT/'recovery.csv')}))


if __name__ == '__main__':
    main()
