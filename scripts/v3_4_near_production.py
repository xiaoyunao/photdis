#!/usr/bin/env python3
"""Process exactly the frozen v3.4 release domain and reconcile every native PSF."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from photdis.stage1 import require_compute
from photdis.v3.controls import apply_temperature, probe_features
from v2_1_near import ROOT, atomic_npz, csv_rows, load, save_json, sha
from v3_3_sed import verified
from v3_4_near_development_inputs import CONFIG as DOMAIN, OUT as CENSUS, cells
from v3_4_near_forecast import forecast, prepare
from v3_4_near_validate import OUT as EVIDENCE
from v3_4_near_validation_inputs import CONFIG as WORKPOINT

RELEASE = ROOT/'configs/v3_4_near_release.json'
OUT = ROOT/'data/v3_4/near_catalog'


def check_release():
    release = json.loads(RELEASE.read_text())
    for p, digest in ((WORKPOINT, release['workpoint_sha256']),
                      (EVIDENCE/'complete.json', release['evidence_sha256'])):
        verified(p,digest)
    if not json.loads((EVIDENCE/'complete.json').read_text())['new_outcomes_read']:
        raise ValueError('Need verified external scores for the registered release')
    return release


def observable_domain(field, r, gr, w1, release):
    admitted = np.zeros(len(r), bool)
    for d in release['domains']:
        if field not in d['fields']:
            continue
        lo, hi = d['observed_r']
        lower, upper = d['dereddened_g_r']
        use = (r >= lo) & (r < hi) & (gr >= lower) & (gr < upper) & np.isin(w1,d['w1_states'])
        if (use & admitted).any():
            raise ValueError('Overlapping release domains')
        admitted |= use
    return admitted


def field_input(field, census):
    path = ROOT/f'data/v3_0/catalog/field_{field:02d}/inputs.npz'
    digest = census['inputs'][str(path.relative_to(ROOT))]
    photo = load(verified(path, digest))
    if not (photo['TYPE'] == 'PSF').all() or not (photo['field'] == field).all():
        raise ValueError('Expected verified native Legacy PSFs in the registered field')
    return path, digest, photo


def apply(field):
    release = check_release()
    cfg = json.loads(WORKPOINT.read_text())
    census_p = CENSUS/'complete.json'
    census = json.loads(census_p.read_text())
    path, digest, photo = field_input(field,census)
    stratum, r, gr = cells(photo,json.loads(DOMAIN.read_text()))
    w1 = photo['band_state'][:,4]
    in_domain = observable_domain(field,r,gr,w1,release)
    models, probe, sources = prepare()
    hashes = {str(p.relative_to(ROOT)):sha(p) for p in (RELEASE, WORKPOINT, DOMAIN, census_p, Path(__file__),
             EVIDENCE/'complete.json', ROOT/'scripts/v3_4_near_forecast.py', *sources)}
    hashes[str(path.relative_to(ROOT))] = digest
    dest = OUT/f'field_{field:02d}'
    dest.mkdir(parents=True, exist_ok=True)
    started = dest/'started.json'
    if (dest/'complete.json').exists():
        raise FileExistsError(f'Production already complete for field {field}')
    if started.exists():
        if json.loads(started.read_text()) != hashes:
            raise ValueError(f'Inputs changed on a production retry for field {field}')
    else:
        with started.open('x') as f:
            json.dump(hashes, f, indent=2, sort_keys=True)
    status = np.zeros(len(r),'u1')
    status[in_domain] = 1
    in_rows = np.flatnonzero(in_domain)
    steps = {'outside_domain':int((~in_domain).sum()), 'dwarf_score_low':0, 'outside_qrf_support':0,
             'nominal_near_score_low':0, 'selected':0, 'strict':0, 'numerical_flag_selected':0}
    near_mass = np.zeros(6)
    for lo in range(0,len(in_rows),4096):
        inds = in_rows[lo:lo+4096]
        chunk = {k:v[inds] for k,v in photo.items()}
        score = apply_temperature(probe['model'].predict_proba(probe_features(chunk,'grz')),
                                  probe['temperature'])[:,0]
        candidates = score >= cfg['primary']['minimum_dwarf_score']
        if not candidates.any():
            continue
        rows = inds[candidates]
        chunk = {k:v[candidates] for k,v in chunk.items()}
        output = forecast(chunk,cfg,models,probe,retain_curve=False)
        np.testing.assert_allclose(output['dwarf_selection_score'],score[candidates],atol=1e-12,rtol=0)
        supported = output['qrf_supported']
        selected = output['primary']
        status[rows] = 2
        status[rows[supported]] = 3
        status[rows[selected]] = 4
        steps['strict'] += int(output['strict'].sum())
        steps['numerical_flag_selected'] += int((selected & output['nominal_numerical_flag']).sum())
        near_mass += output['qrf_bins'][selected].sum(axis=0)
        if selected.any():
            chosen = rows[selected]
            atomic_npz(dest/f'part_{lo:06d}.npz', **{k:v[selected] for k,v in output.items()},
                field=photo['field'][chosen], RA=photo['RA'][chosen], DEC=photo['DEC'][chosen],
                parent_row=photo['parent_row'][chosen], observed_r=r[chosen], dereddened_g_r=gr[chosen],
                stratum=stratum[chosen], band_state=photo['band_state'][chosen],
                band_mask=photo['band_mask'][chosen], photometric_system=photo['photometric_system'][chosen])
    for code, name in enumerate(('outside_domain','dwarf_score_low','outside_qrf_support',
                                 'nominal_near_score_low','selected')):
        steps[name] = int((status==code).sum())
    atomic_npz(dest/'selection_audit.npz', identity=photo['identity'], ls_id=photo['ls_id'], status=status)
    outputs = {p.name:sha(p) for p in sorted(dest.glob('*.npz'))}
    # Reconcile the written partitions rather than trusting the in-memory counts.
    parts = [load(dest/p) for p in outputs if p.startswith('part_')]
    if parts:
        for k in ('identity','ls_id'):
            np.testing.assert_array_equal(np.concatenate([a[k] for a in parts]),photo[k][status==4])
        if any(not (a['TYPE']=='PSF').all() for a in parts):
            raise ValueError('Non-PSF rows exported')
    elif steps['selected']:
        raise ValueError('Selected rows have no partitions')
    np.testing.assert_allclose(near_mass.sum(),steps['selected'],atol=1e-7,rtol=0)
    save_json(dest/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'), field=field, input_n=len(r),
        all_PSF=True, inputs=hashes, outputs=outputs, steps=steps, expected_n_by_distance=near_mass.tolist()))


def collect():
    release = check_release()
    census = json.loads((CENSUS/'complete.json').read_text())
    summaries, count, identities, legacy, children = [], np.zeros(6), [], [], {}
    for field in json.loads(DOMAIN.read_text())['fields']:
        dest = OUT/f'field_{field:02d}'
        receipt = dest/'complete.json'
        record = json.loads(receipt.read_text())
        verified(RELEASE, record['inputs'][str(RELEASE.relative_to(ROOT))])
        verified(EVIDENCE/'complete.json', record['inputs'][str((EVIDENCE/'complete.json').relative_to(ROOT))])
        if record['field'] != field or not record['all_PSF']:
            raise ValueError(f'Unexpected production field {field}')
        for name,digest in record['outputs'].items():
            verified(dest/name,digest)
        audit = load(dest/'selection_audit.npz')
        if (not np.isin(audit['status'],np.arange(5)).all() or len(audit['status']) != record['input_n']
                or (audit['status']==4).sum() != record['steps']['selected']):
            raise ValueError('Production exclusion ledger does not reconcile')
        identities.append(audit['identity'])
        legacy.append(audit['ls_id'])
        summaries.append(dict(field=field, parent=record['input_n'], **record['steps']))
        count += record['expected_n_by_distance']
        children[str(receipt.relative_to(ROOT))] = sha(receipt)
    for key, arrays in [('identity',identities),('ls_id',legacy)]:
        ids = np.concatenate(arrays)
        if len(np.unique(ids)) != len(ids) or len(ids)!=census['parent']:
            raise ValueError(f'Duplicate or missing parent {key}')
    n = sum(s['selected'] for s in summaries)
    np.testing.assert_allclose(count.sum(),n,atol=1e-7,rtol=0)
    summary = OUT/'field_counts.csv'
    csv_rows(summary,summaries)
    save_json(OUT/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'), all_PSF=True, selected=n,
        parent=census['parent'], expected_n_by_distance=count.tolist(), choices=release,
        release_sha256=sha(RELEASE), child_receipts=children, field_counts_sha256=sha(summary)))


def main():
    require_compute()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('field',help='Registered field number or collect')
    args = parser.parse_args()
    if args.field == 'collect':
        collect()
    else:
        apply(int(args.field))


if __name__ == '__main__':
    main()
