#!/usr/bin/env python3
"""Photometry-only native inputs: existing parents and previously unused sky.

No Gaia outcome column, VAC, prediction or reference-quality cut is read here.
New-sky selection is checkpointed before any subsequent reference association.
"""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from v2_1_near import ROOT, sha, save_json, atomic_npz
from photdis.stage1 import require_compute, legacy_ids, galactic
from photdis.photometry import derive_photometry
from photdis.reference_v1_4 import unique_join
from photdis.v3.data import observations, stable_rank, BANDS
D = ROOT/'data/v3_0'
RAW = ['RELEASE', 'BRICKID', 'OBJID', 'TYPE', 'RA', 'DEC', 'EBV', 'REF_ID', 'REF_CAT', 'MASKBITS']
RAW += [p+'_'+b for b in BANDS for p in ('FLUX', 'FLUX_IVAR', 'MW_TRANSMISSION', 'NOBS', 'FRACFLUX')]
RAW += [p+'_'+b for b in ('G', 'R', 'I', 'Z') for p in ('FRACMASKED', 'FRACIN')]
RAW += ['WISEMASK_W1', 'WISEMASK_W2']
NEW_SKY = [(10., -45., 15., -40.), (350., -45., 355., -40.)]


def read_native(path):
    from astropy.io import fits
    import fitsio
    with fits.open(path, memmap=True) as hdus:
        table = hdus[1]; dtype = table.data.dtype; header = table.header
        assert dtype.itemsize == header['NAXIS1'] and not any(dtype[k].hasobject for k in dtype.names)
        names = [k for k in RAW if k in dtype.names]
        for key in names:
            column = list(dtype.names).index(key)+1
            if header.get(f'TSCAL{column}', 1) != 1 or header.get(f'TZERO{column}', 0) != 0:
                raise ValueError('Native FITS scaling must be decoded explicitly: '+key)
        n = header['NAXIS2']; offset = table.fileinfo()['datLoc']
    raw = np.fromfile(path, dtype=dtype, count=n, offset=offset)
    assert len(raw) == n
    rows = np.unique(np.linspace(0, n-1, 16, dtype=int))
    check = fitsio.read(str(path), columns=names, rows=rows)
    for key in names:
        left, right = raw[key][rows], check[key]
        if left.dtype.kind in 'SU':
            np.testing.assert_array_equal(np.char.strip(left.astype('U')), np.char.strip(right.astype('U')))
        else: np.testing.assert_allclose(left, right, rtol=0, atol=0, equal_nan=True)
    return raw


def native_payload(raw, system):
    phot = derive_photometry(raw, w2_snr_min=5)
    t = {k: raw[k].copy() for k in RAW if k in raw.dtype.names}
    i_state = np.zeros(len(raw), 'i1')
    if system == 'DECam_DR11' and 'FLUX_I' in t:
        quality = (t['FRACMASKED_I'] < .4) & (t['FRACIN_I'] > .7) & (t['FRACFLUX_I'] < .5)
        quality &= (t['MASKBITS'].astype('i8') & ((1 << 14) | (1 << 15) | (1 << 10))) == 0
        i_state[quality] = 3
    t.update(observations(t, system, phot['grz_clean'], i_state))
    t['ls_id'] = legacy_ids(raw)
    t['source_id'] = np.where(np.char.strip(t['REF_CAT'].astype('U')) == 'G3', t['REF_ID'], 0).astype('i8')
    t['identity'] = np.where(t['source_id'] > 0, t['source_id'], -t['ls_id'])
    t['TYPE'] = np.char.strip(t['TYPE'].astype('U4'))
    t['r'] = phot['r']; t['r0'] = phot['r0']; t['g0'] = phot['g0']; t['z0'] = phot['z0']
    t['optical_error'] = np.maximum.reduce([phot['sigma_g'], phot['sigma_r'], phot['sigma_z']])
    b = galactic(t['RA'], t['DEC'])[1]
    t['photometric_entry'] = t['psf'] & phot['grz_clean'] & (abs(b) > 30) & (t['EBV'] >= 0) & (t['EBV'] < .03)
    t['cell'] = (np.floor(t['RA']/10).astype('i8')*18+np.floor((t['DEC']+90)/10).astype('i8'))
    return t


def subset(t, rows): return {k: v[rows] for k, v in t.items()}


def existing(field):
    destination = D/'application_inputs'; destination.mkdir(parents=True, exist_ok=True)
    marker = destination/f'field_{field:02}.json'; output = marker.with_suffix('.npz')
    if marker.exists():
        record = json.loads(marker.read_text()); assert sha(output) == record['sha256']; return
    f = json.loads((ROOT/'data/baseline_v0/prepared.json').read_text())['fields'][field]
    task = json.loads((ROOT/f['config']).read_text()); cfg = json.loads((ROOT/task['base_config']).read_text())
    parent = ROOT/f'data/baseline_v0/parent_{field:02}.npz'
    with np.load(parent) as z: ids, identities = z['ls_id'], z['identity']
    pieces = []; seen = np.zeros(len(ids), bool); sources = {}
    for sweep in task['sweep_files']:
        path = Path(task.get('sweep_root', cfg['sweep_root']))/sweep
        raw = read_native(path); native = legacy_ids(raw); j = unique_join(native, ids)
        found = j >= 0
        assert not seen[j[found]].any(), 'Duplicate native parent owner'
        seen[j[found]] = True
        t = native_payload(raw[found], 'BASS_MzLS_DR9' if field == 13 else 'DECam_DR11')
        # Existing catalog's exact identity ownership is retained, even where the
        # original survey reference catalog used a different astrometric release.
        t['identity'] = identities[j[found]]; t['parent_row'] = j[found]
        pieces.append(t)
        stat = path.stat(); sources[str(path)] = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns, independently_checked_rows=16)
        print('NATIVE FIELD', field, sweep, int(seen.sum()), len(ids), flush=True)
    if not seen.all(): raise RuntimeError('Original parent native identities not all recovered')
    allrows = {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
    eligible = allrows['photometric_entry']; t = subset(allrows, eligible)
    t['field'] = np.full(len(t['identity']), field, 'i2')
    assert (t['TYPE'] == 'PSF').all() and len(np.unique(t['identity'])) == len(t['identity'])
    atomic_npz(output, **t)
    save_json(marker, dict(field=field, original_parent=len(ids), admitted=len(t['identity']),
        nonPSF=int((allrows['TYPE'] != 'PSF').sum()), missing_TYPE=int((allrows['TYPE'] == '').sum()),
        photometric_or_morphology_excluded=int((~eligible).sum()), sha256=sha(output),
        parent_sha256=sha(parent), native_sources=sources, reference_outcomes_read=False,
        program_sha256=sha(Path(__file__)), job=os.getenv('SLURM_JOB_ID')))


def new_sky(field):
    destination = D/'new_sky_photometry'; destination.mkdir(parents=True, exist_ok=True)
    marker = destination/f'field_{field:02}.json'; output = marker.with_suffix('.npz')
    if marker.exists():
        record = json.loads(marker.read_text()); assert sha(output) == record['sha256']; return
    bounds = NEW_SKY[field]; rlo, dlo, rhi, dhi = bounds
    # Conservative exclusion against every recorded field config, not merely
    # the fourteen production fields. Chosen cells also lack v1.4 references.
    for p in sorted((ROOT/'configs').glob('field*.json')):
        old = json.loads(p.read_text()).get('bounds_deg')
        if old and max(old[0], rlo) < min(old[2], rhi) and max(old[1], dlo) < min(old[3], dhi):
            raise RuntimeError('New sky overlaps historical config '+str(p))
    path = Path('/global/cfs/projectdirs/cosmo/data/legacysurvey/dr11/south/sweep/11.0')/f'sweep-{int(rlo):03}m{abs(int(dlo)):03}-{int(rhi):03}m{abs(int(dhi)):03}.fits'
    raw = read_native(path)
    t = native_payload(raw, 'DECam_DR11'); del raw
    eligible = t['photometric_entry'] & (t['RA'] >= rlo) & (t['RA'] < rhi) & (t['DEC'] >= dlo) & (t['DEC'] < dhi)
    with np.load(ROOT/'data/baseline_v1_4/references.npz') as z:
        oldcells = np.unique(z['cell']); oldids = z['identity']
    assert not np.intersect1d(t['cell'][eligible], oldcells).size, 'New-sky block previously examined in reference pool'
    eligible &= ~np.isin(t['identity'], oldids)
    # Duplicate positive astrometric IDs are excluded before random ranking.
    ids, counts = np.unique(t['identity'][eligible], return_counts=True)
    eligible &= ~np.isin(t['identity'], ids[counts > 1])
    selections = []; strata = []; sampling_weight = np.zeros(len(eligible))
    for lo, hi in [(0, 18), (18, 20), (20, 22), (22, np.inf)]:
        rows = np.flatnonzero(eligible & (t['r'] >= lo) & (t['r'] < hi))
        selected = rows[np.argsort(stable_rank(t['identity'][rows], 301301), kind='stable')[:256]]
        if len(selected): sampling_weight[selected] = len(rows)/len(selected)
        selections.extend(selected.tolist()); strata.append(dict(r_low=lo, r_high=None if np.isinf(hi) else hi, parent=len(rows), selected=len(selected)))
    rows = np.array(selections, dtype='i8'); t['sample_weight'] = sampling_weight; t = subset(t, rows)
    t['field'] = np.full(len(rows), 100+field, 'i2')
    atomic_npz(output, **t)
    save_json(marker, dict(field=100+field, bounds_deg=bounds, n=len(rows), strata=strata,
        sha256=sha(output), native_path=str(path), native_sha256=sha(path),
        historical_reference_identity_overlap=0, historical_reference_cell_overlap=0,
        all_field_config_overlap=0, reference_outcomes_read=False, predictions_generated=False,
        program_sha256=sha(Path(__file__)), job=os.getenv('SLURM_JOB_ID')))


def main():
    require_compute(); p = argparse.ArgumentParser(); p.add_argument('mode', choices=['application', 'new_sky']); p.add_argument('field', type=int); args = p.parse_args()
    (existing if args.mode == 'application' else new_sky)(args.field)


if __name__ == '__main__': main()
