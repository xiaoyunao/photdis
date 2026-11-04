#!/usr/bin/env python3
"""Allocated full MWS/VAC inventory and resumable native DR11 photometry scan.

No residuals/model fitting here. Every source row and raw distance VAC survives;
identity/quality qualification and new-sky scoring follow the frozen protocol.
"""
from __future__ import annotations
import concurrent.futures
import hashlib
import json
import multiprocessing
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts'), str(ROOT / '.deps')]
import fitsio
import numpy as np
from photdis.reference_v1_4 import spatial_split, optional_photometry
from photdis.photometry import derive_photometry
from photdis.spectroscopy import DESI_COLUMNS, FIBERMAP_COLUMNS
from photdis.stage1 import require_compute, write_json, legacy_ids
from giant_vac_inventory import SPEC, SP

CFG = json.loads((ROOT / 'configs/baseline_v1_4.json').read_text())
OUT = ROOT / CFG['output_dir']
GAIA = ['SOURCE_ID', 'RA', 'DEC', 'REF_EPOCH', 'PMRA', 'PMDEC', 'PARALLAX',
        'PARALLAX_ERROR', 'RUWE', 'ASTROMETRIC_PARAMS_SOLVED', 'NU_EFF_USED_IN_ASTROMETRY',
        'PSEUDOCOLOUR', 'ECL_LAT', 'PHOT_G_MEAN_MAG', 'BP_RP', 'DUPLICATED_SOURCE',
        'VISIBILITY_PERIODS_USED', 'IPD_FRAC_MULTI_PEAK', 'IPD_GOF_HARMONIC_AMPLITUDE',
        'IN_QSO_CANDIDATES', 'IN_GALAXY_CANDIDATES', 'NON_SINGLE_STAR']
RAW_COLUMNS = ['RELEASE', 'BRICKID', 'OBJID', 'RA', 'DEC', 'REF_ID', 'REF_CAT',
               'REF_EPOCH', 'PMRA', 'PMDEC', 'TYPE', 'MASKBITS', 'FITBITS', 'EBV']
for b in ['G', 'R', 'I', 'Z', 'W1', 'W2']:
    RAW_COLUMNS += [f'{p}_{b}' for p in ['FLUX', 'FLUX_IVAR', 'MW_TRANSMISSION', 'NOBS', 'FRACFLUX']]
for b in ['G', 'R', 'I', 'Z']:
    RAW_COLUMNS += [f'{p}_{b}' for p in ['FRACMASKED', 'FRACIN', 'ANYMASK', 'ALLMASK']]
RAW_COLUMNS += ['WISEMASK_W1', 'WISEMASK_W2']


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_npz(path, **arrays):
    temp = path.with_suffix('.tmp.npz')
    np.savez(temp, **arrays)
    temp.replace(path)


def member(values, sorted_unique):
    if not len(sorted_unique):
        return np.zeros(len(values), bool)
    ix = np.searchsorted(sorted_unique, values)
    return (ix < len(sorted_unique)) & (sorted_unique[np.minimum(ix, len(sorted_unique) - 1)] == values)


def build_sources():
    """Retain ALL MWS rows, including PRIMARY and unqualified/unknown evidence."""
    path = OUT / 'sources.npz'
    if path.exists() and (OUT / 'sources.json').exists():
        if sha(path) != json.loads((OUT / 'sources.json').read_text())['source_sha256']:
            raise RuntimeError('Source checkpoint hash mismatch')
        return
    with fitsio.FITS(CFG['mws_path']) as f:
        rv = f['RVTAB'].read(columns=list(dict.fromkeys(DESI_COLUMNS + ['TARGET_RA', 'TARGET_DEC'])))
        flags = [c for c in f['FIBERMAP'].get_colnames() if c.endswith('_TARGET')]
        fib = f['FIBERMAP'].read(columns=list(dict.fromkeys(FIBERMAP_COLUMNS + flags + ['REF_EPOCH', 'PMRA', 'PMDEC', 'EBV'])))
        ga = f['GAIA'].read(columns=GAIA)
    assert len(rv) == len(fib) == len(ga)
    assert np.array_equal(rv['TARGETID'], fib['TARGETID'])
    sid = ga['SOURCE_ID'].astype('i8')
    sid[(sid <= 0) | (sid == 999999)] = 0
    identity = np.where(sid > 0, sid, -rv['TARGETID'])
    ra, dec = fib['TARGET_RA'].copy(), fib['TARGET_DEC'].copy()
    good = (sid > 0) & np.isfinite(ga['RA']) & np.isfinite(ga['DEC'])
    ra[good], dec[good] = ga['RA'][good], ga['DEC'][good]
    legacy = json.loads((OUT / 'freeze.json').read_text())['legacy_bounds']
    cell, split = spatial_split(ra, dec, legacy, CFG['seed'])
    # All repeats of one identity inherit the first row's geometry, except any
    # spatial conflict/legacy occurrence quarantines the entire identity.
    _, first, inverse = np.unique(identity, return_index=True, return_inverse=True)
    conflict = cell != cell[first][inverse]
    bad_ids = np.unique(inverse[conflict | (split == 3)])
    split = split[first][inverse]
    split[np.isin(inverse, bad_ids)] = 3
    source = dict(identity=identity, source_id=sid, ra=ra, dec=dec,
                  cell=cell[first][inverse], split=split,
                  original_row=np.arange(len(rv), dtype='i8'))
    for prefix, data in [('rvs', rv), ('target', fib), ('gaia', ga)]:
        source.update({prefix + '_' + k.lower(): data[k] for k in data.dtype.names})
    atomic_npz(path, **source)
    write_json(OUT / 'sources.json', dict(rows=len(rv), identities=len(first),
        primary=int(rv['PRIMARY'].sum()), spatial_conflict_rows=int(conflict.sum()),
        unique_split_counts={str(s):int((split[first] == s).sum()) for s in range(4)},
        targeting_columns=flags, source_sha256=sha(path)))
    print('Sources frozen:', len(rv), 'rows;', len(first), 'identities', flush=True)


def build_vacs():
    for kind, path, ext in [('specdis', SPEC, 'SPECDIS'), ('spdist', SP, 'SPDIST')]:
        output = OUT / (kind + '_raw.npz')
        marker = OUT / (kind + '_raw.json')
        if output.exists() and marker.exists():
            if sha(output) != json.loads(marker.read_text())['sha256']:
                raise RuntimeError('VAC checkpoint hash mismatch: ' + kind)
            continue
        raw = fitsio.read(str(path), ext=ext)
        atomic_npz(output, **{k.lower():raw[k] for k in raw.dtype.names})
        write_json(OUT / (kind + '_raw.json'), dict(path=str(path), rows=len(raw),
            columns=list(raw.dtype.names), sha256=sha(output)))
        print('Full VAC retained:', kind, len(raw), flush=True)


def scan(path):
    destination = OUT / 'photometry' / (Path(path).stem + '.npz')
    marker = destination.with_suffix('.json')
    if marker.exists():
        saved = json.loads(marker.read_text())
        if not destination.exists() or sha(destination) != saved['sha256']:
            raise RuntimeError('Invalid completed checkpoint: ' + str(destination))
        return saved
    parts = []
    with fitsio.FITS(str(path)) as f:
        n = f[1].get_nrows()
        for start in range(0, n, 200000):
            rows = np.arange(start, min(start + 200000, n))
            keys = f[1].read(columns=['REF_ID', 'REF_CAT', 'RELEASE', 'BRICKID', 'OBJID'], rows=rows)
            lsid = legacy_ids(keys)
            gaia = np.char.strip(keys['REF_CAT'].astype('U')) == 'G3'
            keep = (gaia & member(keys['REF_ID'], SOURCE_IDS)) | member(lsid, PARENT_LSIDS)
            # Positive parent identities are Gaia source IDs, included separately
            # from DESI so this remains the original probability sample.
            keep |= gaia & member(keys['REF_ID'], PARENT_IDS)
            if keep.any():
                raw = f[1].read(columns=RAW_COLUMNS, rows=rows[keep])
                parts.append(raw)
    if parts:
        raw = np.concatenate(parts)
        arrays = {k:raw[k] for k in raw.dtype.names}
        arrays.update(derive_photometry(raw, w2_snr_min=5))
        arrays.update(optional_photometry(raw))
    else:
        arrays = {'REF_ID':np.empty(0, 'i8')}
    atomic_npz(destination, **arrays)
    result = dict(sweep=str(path), scanned_rows=n, selected_rows=len(arrays['REF_ID']),
                  sha256=sha(destination))
    write_json(marker, result)
    print(Path(path).name, n, '->', result['selected_rows'], flush=True)
    return result


def main():
    global SOURCE_IDS, PARENT_IDS, PARENT_LSIDS
    require_compute()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'photometry').mkdir(exist_ok=True)
    frozen_files = ['configs/baseline_v1_4.json', 'docs/baseline_v1_4_protocol.md',
                    'src/photdis/reference_v1_4.py', 'scripts/baseline_v1_4_extract.py']
    hashes = {p:sha(ROOT / p) for p in frozen_files}
    legacy = sorted({tuple(json.loads(p.read_text())['bounds_deg'])
                     for p in (ROOT / 'configs').glob('field_*.json')
                     if 'bounds_deg' in json.loads(p.read_text())})
    freeze = OUT / 'freeze.json'
    if freeze.exists():
        old = json.loads(freeze.read_text())
        if old['hashes'] != hashes or old['legacy_bounds'] != [list(b) for b in legacy]:
            raise RuntimeError('Protocol/code/legacy geometry changed; refusing mixed checkpoint resume')
    else:
        write_json(freeze, dict(hashes=hashes, legacy_bounds=legacy,
            git_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
            hostname=socket.gethostname(), note='Frozen before reading any model scores'))
    build_sources()
    build_vacs()
    with np.load(OUT / 'sources.npz') as s:
        SOURCE_IDS = np.unique(s['source_id'][s['source_id'] > 0])
    # Include VAC-only Gaia IDs in the photometry inventory. The merge stage must
    # report exact pair coverage and handle source-only/ambiguous VAC entries.
    for kind in ['specdis', 'spdist']:
        with np.load(OUT / (kind + '_raw.npz')) as v:
            sid = v['source_id']
            SOURCE_IDS = np.union1d(SOURCE_IDS, sid[(sid > 0) & (sid != 999999)])
    with np.load(CFG['parent_sample']) as p:
        PARENT_IDS = np.sort(p['identity'])
        assert len(PARENT_IDS) == 26000
        legacy = []
        for field in np.unique(p['field']):
            mask = p['field'] == field
            raw = fitsio.read(str(ROOT / f'data/baseline_v0/catalog_{field:02}.fits'),
                             columns=['identity', 'ls_id'], rows=p['parent_row'][mask])
            assert np.array_equal(raw['identity'], p['identity'][mask])
            legacy.append(raw['ls_id'])
        PARENT_LSIDS = np.unique(np.concatenate(legacy))
        assert len(PARENT_LSIDS) == 26000
    paths = sorted(Path(CFG['sweep_root']).glob('sweep-*.fits'))
    if not paths:
        raise RuntimeError('No native south sweeps found')
    # Scan every south sweep for exact G3 IDs, including VAC-only sources and
    # high-proper-motion/boundary objects. No sky/temperature ranking is applied.
    write_json(OUT / 'scan_plan.json', dict(sweeps=[str(p) for p in paths],
        source_ids=len(SOURCE_IDS), fixed_parent_ids=len(PARENT_IDS),
        match='exact G3 REF_ID; ALL fixed parent rows also by original packed Legacy ID',
        limitation='MWS/VAC rows without positive Gaia ID retained, optical positional join pending'))
    with concurrent.futures.ProcessPoolExecutor(max_workers=CFG['scan_workers'],
            mp_context=multiprocessing.get_context('fork')) as pool:
        results = list(pool.map(scan, paths))
    write_json(OUT / 'extraction_complete.json', dict(sweeps=len(results),
        scanned_rows=sum(r['scanned_rows'] for r in results),
        selected_records=sum(r['selected_rows'] for r in results),
        source_sha256=sha(OUT / 'sources.npz'),
        status='raw_extraction_complete_not_scientific_validation',
        next='audit identity/coverage, resolve unmatched sources, assemble independent eligibility, then frozen scores'))


if __name__ == '__main__':
    main()
