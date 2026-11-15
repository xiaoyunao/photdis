#!/usr/bin/env python3
"""Reopen the scoped catalog and compare its selected sample with frozen predictions."""
import json
import os

import numpy as np

from photdis.stage1 import require_compute
from v2_1_near import ROOT, load, save_json, sha
from v3_3_sed import verified
from v3_4_near_forecast import OUT as FORECAST, qrf_bins
from v3_4_near_production import CENSUS, DOMAIN, OUT, check_release, field_input, observable_domain
from v3_4_near_development_inputs import cells
from v3_4_near_validation_inputs import CONFIG, OUT as INPUTS, receipt_payload


def main():
    require_compute()
    final = OUT/'complete.json'
    catalog = json.loads(final.read_text())
    release = check_release()
    cfg = json.loads(CONFIG.read_text())
    domain_cfg = json.loads(DOMAIN.read_text())
    census = json.loads((CENSUS/'complete.json').read_text())
    hashes = {}
    app = receipt_payload(INPUTS, 'application.npz', hashes)
    pred = receipt_payload(FORECAST, 'forecasts.npz', hashes)
    seen = 0
    counts = dict(selected=0, strict=0, numerical=0, eligible_domain=0)
    mass = np.zeros(6)
    strict_mass = np.zeros(6)
    for name,digest in catalog['child_receipts'].items():
        receipt = verified(ROOT/name,digest)
        record = json.loads(receipt.read_text())
        field = record['field']
        _, _, photo = field_input(field,census)
        g, r, color = cells(photo,domain_cfg)
        raw = load(verified(receipt.parent/'selection_audit.npz',record['outputs']['selection_audit.npz']))
        for key in ('identity','ls_id'):
            np.testing.assert_array_equal(raw[key],photo[key])
        expected_domain = observable_domain(field,r,color,photo['band_state'][:,4],release)
        np.testing.assert_array_equal(raw['status']>0,expected_domain)
        counts['eligible_domain'] += int(expected_domain.sum())
        p_order = np.argsort(photo['identity'])
        a = np.flatnonzero(app['field']==field)
        i = app['prediction_row'][a]
        from_photo = p_order[np.searchsorted(photo['identity'][p_order],app['identity'][a])]
        np.testing.assert_array_equal(photo['identity'][from_photo],app['identity'][a])
        np.testing.assert_array_equal(raw['status'][from_photo]==4,pred['primary'][i]&expected_domain[from_photo])
        seen += len(a)
        selected = []
        for file, h in record['outputs'].items():
            if not file.startswith('part_'):
                continue
            part = load(verified(receipt.parent/file,h))
            k = p_order[np.searchsorted(photo['identity'][p_order],part['identity'])]
            np.testing.assert_array_equal(part['identity'],photo['identity'][k])
            np.testing.assert_array_equal(part['ls_id'],photo['ls_id'][k])
            np.testing.assert_array_equal(part['stratum'],g[k])
            np.testing.assert_array_equal(part['observed_r'],r[k])
            np.testing.assert_array_equal(part['dereddened_g_r'],color[k])
            np.testing.assert_array_equal(part['band_state'],photo['band_state'][k])
            np.testing.assert_array_equal(part['band_mask'],photo['band_mask'][k])
            if (not expected_domain[k].all() or not (part['TYPE']=='PSF').all()
                    or not (part['photometric_system']=='DECam_DR11').all() or not part['qrf_supported'].all()):
                raise ValueError('Selected row lacks native PSF or observable support')
            for cut in ('primary','strict'):
                good = ((part['dwarf_selection_score']>=cfg[cut]['minimum_dwarf_score'])
                        & (part['nominal_qnear']>=cfg[cut]['minimum_near_probability']))
                np.testing.assert_array_equal(part[cut],good)
            if not part['primary'].all():
                raise ValueError('Non-primary row in selected partitions')
            edges, pdf, q = [part[k] for k in ('qrf_mu_edges','qrf_mu_mass','qrf_quantiles_kpc')]
            if (not np.isfinite(q).all() or (q<=0).any() or (np.diff(q,axis=1)<0).any()
                    or not np.isfinite(edges).all() or (np.diff(edges,axis=1)<=0).any() or (pdf<0).any()):
                raise ValueError('Invalid QRF posterior')
            np.testing.assert_allclose(pdf.sum(axis=1),1,atol=1e-12,rtol=0)
            np.testing.assert_allclose(part['qrf_bins'],qrf_bins(edges,pdf),atol=1e-12,rtol=0)
            np.testing.assert_allclose(part['qrf_qnear'],part['qrf_bins'][:,:3].sum(axis=1),atol=1e-12,rtol=0)
            _, out_rows, app_rows = np.intersect1d(part['identity'],app['identity'][a],return_indices=True)
            for key in ('qrf_qnear','nominal_qnear','qrf_quantiles_kpc','dwarf_selection_score'):
                np.testing.assert_allclose(part[key][out_rows],pred[key][i[app_rows]],atol=1e-12,rtol=0)
            np.testing.assert_array_equal(part['strict'][out_rows],pred['strict'][i[app_rows]])
            selected.append(part['identity'])
            counts['strict'] += int(part['strict'].sum())
            counts['numerical'] += int(part['nominal_numerical_flag'].sum())
            mass += part['qrf_bins'].sum(axis=0)
            strict_mass += part['qrf_bins'][part['strict']].sum(axis=0)
        ids = np.concatenate(selected) if selected else np.empty(0,dtype=photo['identity'].dtype)
        np.testing.assert_array_equal(ids,photo['identity'][raw['status']==4])
        counts['selected'] += len(ids)
    if seen != len(app['identity']) or counts['selected'] != catalog['selected']:
        raise ValueError('Catalog or overlap count failed')
    np.testing.assert_allclose(mass,catalog['expected_n_by_distance'],atol=1e-7,rtol=0)
    np.testing.assert_allclose(strict_mass.sum(),counts['strict'],atol=1e-7,rtol=0)
    path = OUT/'delivery_validated.json'
    if path.exists():
        raise FileExistsError('Catalog already validated')
    save_json(path,dict(job=os.getenv('SLURM_JOB_ID'), all_PSF=True, catalog_sha256=sha(final), inputs=hashes,
                        probability_draw_replayed=seen, counts=counts,
                        primary_mass=mass.tolist(), strict_mass=strict_mass.tolist()))


if __name__=='__main__':
    main()
