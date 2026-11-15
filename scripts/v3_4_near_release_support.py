#!/usr/bin/env python3
"""Assess the proposed joint magnitude, color and W1 release support."""
import json
import os
from pathlib import Path

import numpy as np

from photdis.stage1 import require_compute
from photdis.v3.metrics import distance_summary
from v2_1_near import ROOT, csv_rows, load, save_json, sha
from v3_3_sed import verified
from v3_4_near_production import observable_domain
from v3_4_near_recover_labels import archived_labels
from v3_4_near_validate import CONFIG, OUT as EVIDENCE, load_closed

DOMAIN = ROOT/'configs/v3_4_near_release_domain.json'
OUT = EVIDENCE.parent/'release_support'


def main():
    require_compute()
    hashes = {str(p.relative_to(ROOT)):sha(p) for p in (CONFIG,DOMAIN,Path(__file__))}
    cfg, domain = [json.loads(p.read_text()) for p in (CONFIG,DOMAIN)]
    pred, data = load_closed(hashes)
    app = data['application']
    rows = app['prediction_row']
    w1 = data['photometry']['band_state'][rows,4]
    field, r, c, weight = (app[k] for k in ('field','observed_r','dereddened_g_r','design_weight'))
    joint = np.zeros(len(r),bool)
    for f in np.unique(field):
        m = field==f
        joint[m] = observable_domain(f,r[m],c[m],w1[m],domain)
    marker = EVIDENCE/'complete.json'
    receipt = json.loads(marker.read_text())
    names = ('application.npz', 'DR20_MINESweeper.npz','DR20_APO.npz','DR19_Astra.npz')
    read = {k:load(verified(EVIDENCE/k,receipt['outputs'][k])) for k in names}
    hashes[str(marker.relative_to(ROOT))] = sha(marker)
    e = read['application.npz']
    np.testing.assert_array_equal(app['identity'],e['identity'])
    p, err = e['p_corr'], e['parallax_error']
    delta, boundary = cfg['geometry_sigma_multiplier']*err, 1/cfg['near_kpc']
    near_signed = e['signed_eligible'] & (p-delta>boundary)
    far_signed = e['signed_eligible'] & (p+delta<boundary)
    weak = e['signed_eligible'] & (p<5*err)
    label, match, _, _ = archived_labels(app,hashes)
    giant = np.zeros(len(pred['identity']),bool)
    giant_far = giant.copy()
    for stream in ('DR20_MINESweeper','DR20_APO','DR19_Astra'):
        index, observed = data[stream], read[stream+'.npz']
        np.testing.assert_array_equal(index['identity'],observed['identity'])
        i = index['prediction_row']
        giant[i] |= observed['class_label']==1
        giant_far[i] |= ((observed['class_label']==1) & observed['distance_precise'] &
                         (observed['distance_kpc']-cfg['geometry_sigma_multiplier']*
                          observed['distance_error_kpc']>cfg['near_kpc']))
    records, radial = [], []
    groups = {'all':joint,'w1_2':joint&(w1==2),'w1_4':joint&(w1==4)}
    groups.update({f'field_{f:02d}':joint&(field==f) for f in np.unique(field)})
    for selection in ('primary','strict'):
        for name,mask in groups.items():
            m = mask & pred[selection][rows]
            total = weight[m].sum()
            if total == 0:
                continue
            geo = m & np.isfinite(e['distance_kpc'])
            signed = m & e['signed_eligible']
            contrast = e['qrf_log_score']-e['nominal_log_score']
            weighted_score = lambda use: float(np.average(contrast[use],weights=weight[use])) if use.any() else np.nan
            masses = weight[m]@pred['qrf_bins'][rows[m]]
            no_far = m & ~far_signed & ~giant_far[rows]
            clean_masses = weight[no_far]@pred['qrf_bins'][rows[no_far]]
            records.append(dict(selection=selection,group=name,sample_n=int(m.sum()),estimated_n=float(total),
                signed_n=int(signed.sum()),weak_signed_n=int((m&weak).sum()), signed_gain=weighted_score(signed),
                weak_signed_gain=weighted_score(m&weak), high_snr_n=int(geo.sum()),
                signed_near_n=int((m&near_signed).sum()),signed_far_n=int((m&far_signed).sum()),
                signed_far_lower=float(weight[m&far_signed].sum()/total),
                signed_far_upper=float(1-weight[m&near_signed].sum()/total),
                archived_class_n=int((m&match).sum()),archived_dwarf=int((m&(label==0)).sum()),
                archived_giant=int((m&(label==1)).sum()),spectrum_giant=int((m&giant[rows]).sum()),
                spectrum_far_giant=int((m&giant_far[rows]).sum()),
                **{f'geometry_{k}':v for k,v in distance_summary(
                   10+5*np.log10(pred['qrf_quantiles_kpc'][rows[geo]]),10+5*np.log10(e['distance_kpc'][geo])).items()}))
            if name=='all':
                radial.append(dict(selection=selection, bin_names=['0–2','2–5','5–10','10–20','20–50','50+'],
                    model_count=masses.tolist(),excluding_far_evidence=clean_masses.tolist()))
    OUT.mkdir(parents=True,exist_ok=False)
    csv_rows(OUT/'joint_support.csv',records)
    save_json(OUT/'radius_sensitivity.json',radial)
    save_json(OUT/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'), inputs=hashes, all_PSF=True,
        new_outcomes_read=False, outputs={f.name:sha(f) for f in sorted(OUT.iterdir())}))


if __name__ == '__main__':
    main()
