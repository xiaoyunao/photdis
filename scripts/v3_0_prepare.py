#!/usr/bin/env python3
"""Allocated reference inventory, prior to any v3 fitted model/outcome selection."""
import json
import os
from pathlib import Path
import numpy as np
from v2_1_near import ROOT, sha, save_json, atomic_npz, csv_rows
from photdis.stage1 import require_compute
from photdis.reference_v1_4 import unique_join
from photdis.v3.data import (BANDS, ROLES, observations, specdis_qualification,
                            primary_channel, assign_roles)

D = ROOT/'data/v3_0'
T = D/'tables'
OLD = ROOT/'data/baseline_v1_4'
REF_KEYS = ['identity', 'ls_id', 'source_id', 'targetid', 'source_row', 'RA', 'DEC',
    'EBV', 'cell', 'split', 'photometric_eligible', 'grz_clean', 'identity_conflict',
    'nonstellar_risk', 'redrock_nonstellar_evidence', 'class_label', 'class_eligible',
    'rvs_quality_reason', 'rvs_parameter_quality', 'rvs_feh', 'rvs_feh_err',
    'rvs_teff', 'rvs_logg', 'rvs_teff_err', 'rvs_logg_err', 'rvs_sn_r',
    'specdis_row', 'specdis_valid', 'spdist_row', 'spdist_valid', 'spdist_distance',
    'spdist_sigma_mu', 'parallax_eligible', 'p_corr', 'zpt_mas', 'zpt_valid',
    'gaia_parallax_error', 'anchor_astrometry', 'i_measurement_state',
    'zero_id_match', 'match_epoch_known', 'match_separation_arcsec',
    'specdis_source_only_join', 'spdist_source_only_join', 'RELEASE']


def projected(path, keys, rows=None):
    with np.load(path) as z:
        return {k: z[k] if rows is None else z[k][rows] for k in keys}


def gather(path, rows, keys):
    """Column-wise bounded-memory gather; -1 is never a valid array subscript."""
    result = {}; use = rows >= 0
    with np.load(path) as z:
        for key in keys:
            a = z[key]
            fill = '' if a.dtype.kind in 'SU' else np.nan if a.dtype.kind == 'f' else False if a.dtype.kind == 'b' else 0 if a.dtype.kind == 'u' else -1
            out = np.full((len(rows),)+a.shape[1:], fill, dtype=a.dtype)
            out[use] = a[rows[use]]
            result[key] = out
    return result


def main():
    require_compute(); T.mkdir(parents=True, exist_ok=True)
    paths = [Path(__file__), ROOT/'src/photdis/v3/data.py', ROOT/'docs/v3_0_protocol.md',
             OLD/'references.npz', OLD/'specdis_raw.npz', ROOT/'data/v2_0/native_report.npz',
             ROOT/'data/v2_3/training.npz', ROOT/'data/v2_4/prospective_identity_reservation.npz']
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    freeze = D/'preparation_freeze.json'
    if freeze.exists():
        assert json.loads(freeze.read_text()) == hashes, 'Explicit preparation migration required'
    else:
        save_json(freeze, hashes)
    if (D/'preparation.json').exists():
        record = json.loads((D/'preparation.json').read_text())
        assert sha(D/'references.npz') == record['output_sha256']
        print('Reused verified preparation', flush=True); return
    t = projected(OLD/'references.npz', REF_KEYS)
    rawpath = ROOT/'data/v2_0/native_report.npz'
    rawids = projected(rawpath, ['ls_id'])['ls_id']
    j = unique_join(t['ls_id'], rawids)
    assert (j >= 0).all(), 'All reference native rows must be independently recovered'
    types = projected(rawpath, ['TYPE'], j)['TYPE'].astype('U4')
    psf = np.char.strip(types) == 'PSF'
    # Only imaging/morphology/identity admission. Gaia residuals, labels, and
    # distance outcomes cannot remove an object from the reference parent.
    use = t['photometric_eligible'] & psf & ~t['identity_conflict']
    counts = dict(full_rows=len(j), native_missing=int((j < 0).sum()),
                  nonPSF=int((~psf & (types != '')).sum()), missing_TYPE=int((types == '').sum()),
                  photometry_failure=int((~t['photometric_eligible']).sum()),
                  identity_conflict=int(t['identity_conflict'].sum()), admitted=int(use.sum()))
    t = {k: v[use] for k, v in t.items()}; j = j[use]; types = types[use]
    rawkeys = ['TYPE', 'EBV'] + [p+'_'+b for b in BANDS for p in
                               ('FLUX', 'FLUX_IVAR', 'MW_TRANSMISSION', 'NOBS')]
    rawkeys += ['FRACFLUX_W1', 'FRACFLUX_W2', 'WISEMASK_W1', 'WISEMASK_W2']
    raw = projected(rawpath, rawkeys, j)
    obs = observations(raw, 'DECam_DR11', t['grz_clean'], t['i_measurement_state'])
    assert obs['psf'].all() and obs['band_mask'][:, [0, 1, 3]].all()
    t.update(obs); t['TYPE'] = types
    # Raw fields stay available for forensic checks and source-specific comparisons.
    t.update(raw)
    spec = gather(OLD/'specdis_raw.npz', t['specdis_row'],
                  ['dist', 'disterr', 'binary_flag', 'binary_possibility', 'mg_nn',
                   'mg_geo', 'parallax', 'parallax_err', 'parallax_zpc'])
    spectral_quality = ((t['rvs_quality_reason'].astype('i8') & ~(1 << 4)) == 0) & (t['source_row'] >= 0)
    good, sigma = specdis_qualification(spec['dist'], spec['disterr'], spectral_quality)
    t['specdis_old_flag1_valid'] = t.pop('specdis_valid')
    t['specdis_valid'] = good; t['specdis_sigma_mu'] = sigma
    for k, a in spec.items(): t['specdis_raw_'+k] = a
    t['specdis_distance'] = spec['dist']
    # Photometric parent retains nonstellar evidence; fitting eligibility does not.
    t['reference_stellar_eligible'] = ~t['redrock_nonstellar_evidence']
    t['primary_channel'] = primary_channel(good, t['spdist_valid'], t['parallax_eligible'])
    t['class_label'] = np.where(t['class_eligible'], t['class_label'], -1)
    fit = projected(ROOT/'data/v2_3/training.npz', ['identity'])['identity']
    reserve = projected(ROOT/'data/v2_4/prospective_identity_reservation.npz', ['identity'])['identity']
    t['historical_fit'] = np.isin(t['identity'], fit)
    t['role'] = assign_roles(t['split'], t['cell'], t['historical_fit'], np.isin(t['identity'], reserve))
    assert not (t['role'] == 5).any(), 'Reservation unexpectedly joined reference parent'
    assert not (t['historical_fit'] & np.isin(t['role'], [2, 3, 4])).any()
    assert len(np.unique(t['identity'])) == len(t['identity'])
    # Preserve the original full Gaia payload by source row, including negative
    # observations and source-quality flags. Zero-ID/VAC-only rows retain audited
    # assembled parallax plus explicit missing source payload, never bogus row -1.
    with np.load(OLD/'sources.npz') as z: gaia_keys = [k for k in z.files if k.startswith('gaia_')]
    gaia = gather(OLD/'sources.npz', t['source_row'], gaia_keys)
    for k, a in gaia.items(): t['source_'+k] = a
    t['gaia_payload_from_mws'] = t['source_row'] >= 0
    rows = []
    for role, name in enumerate(ROLES):
        m = t['role'] == role
        for channel, label in [(None, 'all'), (0, 'no_distance'), (1, 'SpecDis'), (2, 'SPDist'), (3, 'Gaia')]:
            u = m if channel is None else m & (t['primary_channel'] == channel)
            rows.append(dict(role=name, source=label, n=int(u.sum()),
                missing_class=int((u & (t['class_label'] < 0)).sum()),
                binary_flag_not1=int((u & good & (spec['binary_flag'] != 1)).sum()),
                signed_Gaia=int((u & t['parallax_eligible']).sum()),
                negative_Gaia=int((u & t['parallax_eligible'] & (t['p_corr'] < 0)).sum()),
                weak_W1=int((u & (t['band_state'][:, 4] == 3)).sum()),
                middle_FeH=int((u & t['rvs_parameter_quality'] & (t['rvs_feh'] >= -1.5) & (t['rvs_feh'] < -.5)).sum())))
    csv_rows(T/'reference_inventory.csv', rows)
    cellrows = [dict(cell=int(c), n=int((t['cell'] == c).sum()),
                     split=int(t['split'][np.flatnonzero(t['cell'] == c)[0]])) for c in np.unique(t['cell'])]
    csv_rows(T/'reference_sky_inventory.csv', cellrows)
    atomic_npz(D/'references.npz', **t)
    save_json(D/'preparation.json', dict(**counts, output_sha256=sha(D/'references.npz'),
        specdis_without_flag=int(good.sum()), specdis_flag_not1=int((good & (spec['binary_flag'] != 1)).sum()),
        primary_channels={str(k): int((t['primary_channel'] == k).sum()) for k in range(4)},
        native_source='v2_0 all-reference native_report, exact ls_id',
        primary_policy='SpecDis then SPDist then signed Gaia; alternatives not multiplied',
        prospective_outcomes_opened=False, job=os.getenv('SLURM_JOB_ID'),
        remaining='No new-sky/application sample yet; no fitted v3 model; source errors not calibrated.'))
    print(json.dumps(json.loads((D/'preparation.json').read_text()), indent=2), flush=True)


if __name__ == '__main__': main()
