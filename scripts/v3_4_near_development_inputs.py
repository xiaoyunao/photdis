#!/usr/bin/env python3
"""Census of the existing multisky photometric envelope and opened reference support."""
import json
import os
from pathlib import Path

import numpy as np

from photdis.reference_v1_4 import unique_join
from photdis.stage1 import require_compute
from photdis.v3.controls import optical_magnitudes
from photdis.v3.data import SIGNIFICANT, WEAK, stable_rank
from v2_1_near import ROOT, atomic_npz, csv_rows, load, save_json, sha
from v3_3_sed import BASE, MANIFESTS, verified

CONFIG = ROOT/'configs/v3_4_near_development.json'
OUT = ROOT/'data/v3_4/near_development/inputs'


def cells(photo, cfg):
    kinds = np.char.strip(photo['TYPE'].astype(str))
    if not (kinds == 'PSF').all() or not (photo['photometric_system'] == 'DECam_DR11').all():
        raise ValueError('The registered envelope requires verified DECam DR11 PSFs')
    mag, _ = optical_magnitudes(photo)
    observed_r = 22.5-2.5*np.log10(photo['flux'][:, 1])
    color = mag[:, 0]-mag[:, 1]
    r = np.searchsorted(cfg['observed_r_edges'], observed_r, side='right')-1
    c = np.searchsorted(cfg['dereddened_g_r_edges'], color, side='right')-1
    w1 = photo['band_state'][:, 4]
    w = np.where(w1 == SIGNIFICANT, 2, np.where(w1 == WEAK, 1, 0))
    nr, nc = (len(cfg[k])-1 for k in ('observed_r_edges', 'dereddened_g_r_edges'))
    if not (np.isfinite(observed_r).all() and np.isfinite(color).all()):
        raise ValueError('Non-finite optical feature')
    if (r < 0).any() or (r >= nr).any() or (c < 0).any() or (c >= nc).any():
        raise ValueError('Histogram edges do not span the existing photometric envelope')
    return ((r*nc+c)*3+w), observed_r, color


def opened_matches(photo, reference_photo, evidence):
    # Exact agreement on both identities avoids silently importing a mismatched Gaia alias.
    ix = unique_join(photo['identity'], reference_photo['identity'])
    match = ix >= 0
    same = match.copy()
    same[match] = photo['ls_id'][match] == reference_photo['ls_id'][ix[match]]
    safe = np.where(same, ix, 0)
    if not np.isin(evidence['class_label'][safe[same]], (-1, 0, 1, 2)).all():
        raise ValueError('Unexpected reference class encoding')
    return ix, same, match & ~same


def split_field(photo, opened, evidence, cfg):
    group, r_mag, gr = cells(photo, cfg)
    ids = photo['identity']
    ix, same, alternate_owner = opened_matches(photo, opened, evidence)
    ref = np.where(same, ix, 0)
    kind = np.where(same, evidence['class_label'][ref], -1)
    specdist = same & (evidence['specdis_valid'][ref] | evidence['spdist_valid'][ref])
    gaia = same & evidence['parallax_eligible'][ref]
    precise = gaia & (evidence['p_corr'][ref] > 10*evidence['gaia_parallax_error'][ref])
    nc = len(cfg['dereddened_g_r_edges'])-1
    summary, parts = [], []
    for cell in np.unique(group):
        rows = np.flatnonzero(group == cell)
        rank = stable_rank(ids[rows], cfg['sample_salt'])
        sample = rows[np.argsort(rank, kind='stable')[:cfg['sample_cap_per_field_cell']]]
        ci, wi = divmod(int(cell), 3)
        mi, gi = divmod(ci, nc)
        reference_only = rows[same[rows] & ~np.isin(rows, sample)]
        k = np.r_[sample, reference_only]
        d = {key: a[k] for key, a in photo.items()}
        d.update(observed_r=r_mag[k], dereddened_g_r=gr[k], stratum=group[k],
                 application_sample=np.isin(k, sample),
                 application_weight=np.r_[np.full(len(sample), len(rows)/len(sample)),
                                          np.zeros(len(reference_only))],
                 opened_reference_row=np.where(same[k], ix[k], -1))
        parts.append(d)
        summary.append(dict(stratum=int(cell), r_lo=cfg['observed_r_edges'][mi],
            r_hi=cfg['observed_r_edges'][mi+1], gr_lo=cfg['dereddened_g_r_edges'][gi],
            gr_hi=cfg['dereddened_g_r_edges'][gi+1], W1=('unusable', 'weak', 'significant')[wi],
            parent=len(rows), sampled=len(sample), opened_match=int(same[rows].sum()),
            identity_with_different_legacy_owner=int(alternate_owner[rows].sum()),
            dwarf=int((kind[rows] == 0).sum()), giant=int((kind[rows] == 1).sum()),
            other_star=int((kind[rows] == 2).sum()), unknown_class=int((same[rows] & (kind[rows] < 0)).sum()),
            spectral_distance=int(specdist[rows].sum()), signed_gaia=int(gaia[rows].sum()),
            precise_gaia=int(precise[rows].sum())))
    return summary, parts


def main():
    require_compute()
    cfg = json.loads(CONFIG.read_text())
    catalog_marker = MANIFESTS/'catalog_summary/complete.json'
    cohort_marker = MANIFESTS/'cohorts.json'
    catalog_receipts = json.loads(catalog_marker.read_text())['inputs']
    cohort_outputs = json.loads(cohort_marker.read_text())['outputs']
    keys = ('cohorts/development.npz', 'evaluation/development.npz')
    opened, evidence = [load(verified(BASE/k, cohort_outputs[k])) for k in keys]
    np.testing.assert_array_equal(opened['identity'], evidence['identity'])
    for table in (opened, evidence):
        if not (table['TYPE'] == 'PSF').all() or len(np.unique(table['identity'])) != len(table['identity']):
            raise ValueError('Historical opened development roster differs from PSF contract')
    if not (evidence['role'] == 4).all():
        raise ValueError('Only already opened development data may enter the census')
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in (CONFIG, Path(__file__), catalog_marker, cohort_marker)}
    inputs.update({str((BASE/k).relative_to(ROOT)): cohort_outputs[k] for k in keys})
    totals, census, parts, owners = [], [], [], []
    for field in cfg['fields']:
        root = BASE/'catalog'/f'field_{field:02d}'
        marker = root/'prepared.json'
        key = str(marker.relative_to(ROOT))
        old = json.loads(verified(marker, catalog_receipts[key]).read_text())
        path = root/'inputs.npz'
        inputs[key] = catalog_receipts[key]
        inputs[str(path.relative_to(ROOT))] = old['outputs']['inputs.npz']
        t = load(verified(path, inputs[str(path.relative_to(ROOT))]))
        if len(t['identity']) != old['candidate'] or not (t['field'] == field).all():
            raise ValueError(f'Previous candidate receipt mismatch: {field}')
        owners.append(t['identity'])
        rows, sample = split_field(t, opened, evidence, cfg)
        for row in rows:
            row['field'] = field
        census.extend(rows)
        parts.extend(sample)
        totals.append(dict(field=field, original_parent=old['original_parent'],
            native_PSF_entry=old['native_PSF_entry'], missing_TYPE=old['missing_TYPE'], non_PSF=old['nonPSF'],
            candidate=len(t['identity']), opened_match=sum(r['opened_match'] for r in rows),
            identity_with_different_legacy_owner=sum(r['identity_with_different_legacy_owner'] for r in rows),
            sampled=sum(r['sampled'] for r in rows)))
    ids = np.concatenate(owners)
    if len(np.unique(ids)) != len(ids):
        raise ValueError('Application parent has duplicate cross-field identities')
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    if (len(np.unique(data['identity'])) != len(data['identity']) or not (data['TYPE'] == 'PSF').all()
            or not np.isclose(data['application_weight'].sum(), len(ids), rtol=0, atol=1e-5)):
        raise ValueError('Stratified sample does not reconcile to the photometric envelope')
    OUT.mkdir(parents=True, exist_ok=False)
    atomic_npz(OUT/'photometry.npz', **data)
    csv_rows(OUT/'coverage.csv', census)
    csv_rows(OUT/'fields.csv', totals)
    save_json(OUT/'complete.json', dict(job=os.getenv('SLURM_JOB_ID'), parent=len(ids), n=len(data['identity']),
        sampled=int(data['application_sample'].sum()), opened_matches=int((data['opened_reference_row'] >= 0).sum()),
        all_PSF=True, new_outcomes_read=False, inputs=inputs, outputs={p.name:sha(p) for p in sorted(OUT.iterdir())},
        scope='Existing 13-field DECam PSF envelope. Photometry-only parent and design sample, plus exact opened-development references.'))


if __name__ == '__main__':
    main()
