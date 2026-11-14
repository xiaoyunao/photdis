#!/usr/bin/env python3
"""Open exact new DR19 outcomes after verifying the complete prediction freeze."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import fitsio
import numpy as np

from photdis.gaia import SOURCE_ID_HEALPIX_FACTOR, derive_gaia_labels, join_gaia
from photdis.stage1 import require_compute
from photdis.v3_1.freeze import verify_freeze
from v2_1_near import ROOT, atomic_npz, csv_rows, load, save_json, sha
from v3_2_reference_freeze import ROLE, aligned, require_native
from v3_2_reference_inventory import ARCHIVES

ID_COLS = ('sdss_id', 'gaia_dr3_source_id')
DIST_COLS = ('nn_dist', 'nn_dist_err', 'nn_dist_model_err', 'flag_bad', 'flag_warn')
STAR_COLS = ('teff', 'logg', 'e_logg', 'fe_h', 'e_fe_h', 'fe_h_flags', 'snr', 'flag_bad')


def projected(path, index, rows, columns):
    order = np.argsort(rows)
    with fitsio.FITS(str(path)) as handle:
        names = {n.lower(): n for n in handle[2].get_colnames()}
        result = handle[2].read(rows=rows[order], columns=[names[n] for n in (*ID_COLS, *columns)])
    restore = np.argsort(order)
    result = {name: result[names[name]][restore] for name in (*ID_COLS, *columns)}
    for key, expected in [('sdss_id', index['sdss_id']), ('gaia_dr3_source_id', index['identity'])]:
        if result[key].dtype.kind not in 'iu':
            raise ValueError('Reference ID must remain integer')
        np.testing.assert_array_equal(result[key], expected)
    return result


def channels(dist, star):
    raw = dist['nn_dist'].astype(float)
    err, model_err = [dist[k].astype(float) for k in ('nn_dist_err', 'nn_dist_model_err')]
    sigma = np.hypot(err, model_err)
    available = ((raw > 0) & np.isfinite(raw) & (err > 0) & (model_err >= 0)
                 & np.isfinite(sigma) & ~dist['flag_bad'].astype(bool) & ~dist['flag_warn'].astype(bool))
    precise = available & (sigma < .2 * raw)
    teff, g, eg, feh, efeh = [star[k].astype(float) for k in ('teff', 'logg', 'e_logg', 'fe_h', 'e_fe_h')]
    quality = (~star['flag_bad'].astype(bool) & np.isfinite(star['snr']) & (star['snr'] >= 50)
               & np.isfinite(teff) & (teff > 0) & (teff < 7000)
               & np.isfinite(g) & (g > -9000) & np.isfinite(eg) & (eg > 0))
    label = np.full(len(raw), -1, dtype='i1')
    label[quality & (g-eg > 4.2)] = 0
    label[quality & (g+eg < 3.2)] = 1
    feh_valid = quality & (star['fe_h_flags'] == 0) & np.isfinite(feh) & (feh > -9000) & np.isfinite(efeh) & (efeh > 0)
    return dict(distance_kpc=np.where(available, raw/1000, np.nan),
                distance_error_kpc=np.where(available, sigma/1000, np.nan),
                distance_available=available, distance_precise=precise, class_label=label,
                feh=feh, feh_error=efeh, feh_valid=feh_valid)


def coverage(evidence):
    label, d = evidence['class_label'], evidence['distance_kpc']
    groups = {'all': np.ones(len(d), bool), 'dwarf': label == 0, 'giant': label == 1,
              'metal_poor_giant': (label == 1) & evidence['feh_valid'] & (evidence['feh'] < -1.5)}
    rows = []
    edges = (0, 2, 5, 10, 20, 50, np.inf)
    for name, group in groups.items():
        for lo, hi in zip(edges[:-1], edges[1:]):
            good = group & evidence['distance_precise'] & (d >= lo) & (d < hi)
            rows.append(dict(group=name, lo_kpc=lo, hi_kpc=hi, n=int(good.sum()),
                             blocks=len(np.unique(evidence['cell'][good])),
                             signed_gaia=int((good & evidence['parallax_eligible']).sum())))
    return rows


def main():
    require_compute()
    root = ROOT/'data/v3_2'
    out = root/'final_validation_dr19'
    freeze = out/'freeze.json'
    digest = verify_freeze(ROOT, freeze)
    receipt = json.loads(freeze.read_text())
    if receipt['cohort'] != ROLE or not receipt['all_PSF']:
        raise ValueError('Wrong outcome cohort')
    native = root/'reference_native'
    base = root/'reference_sources/dr19_astra_0.6.0'
    native_paths = [native/name for name in ('photometry.npz', 'reference_index.npz')]
    required = native_paths + [base/name for name in ARCHIVES.values()] + [
        Path(__file__).resolve(), ROOT/'scripts/v3_2_reference_freeze.py', ROOT/'src/photdis/gaia.py']
    required += [ROOT/receipt[key] for key in ('decision', 'forecast', 'evaluator')]
    if any(str(p.relative_to(ROOT)) not in receipt['hashes'] for p in required):
        raise ValueError('Incomplete freeze inventory')
    photo, index = map(load, native_paths)
    require_native(photo)
    aligned(photo, index)
    if len(photo['identity']) != receipt['n']:
        raise ValueError('Cohort size diverged from freeze')
    complete = out/'evidence_complete.json'
    if complete.exists():
        saved = json.loads(complete.read_text())
        if saved['freeze_sha256'] != digest:
            raise ValueError('Evidence binds a different freeze')
        for key, expected in saved['outputs'].items():
            if sha(out/key) != expected:
                raise ValueError('Evidence output changed after release')
        return
    gaia_root = Path('/global/cfs/cdirs/cosmo/data/gaia/dr3/healpix')
    tile_paths = [gaia_root/f'healpix-{pixel:05d}.fits'
                  for pixel in np.unique(photo['identity'] // SOURCE_ID_HEALPIX_FACTOR)]
    def tile_stats():
        records = {}
        for path in tile_paths:
            stat = path.stat() if path.exists() else None
            records[str(path)] = None if stat is None else [stat.st_size, stat.st_mtime_ns]
        return records
    sources = dict(freeze_sha256=digest, gaia_tiles=tile_stats())
    started = out/'started.json'
    # Same-hash interrupted runs may resume. Exclusive creation prevents
    # replacing the outcome-opening receipt with a different freeze.
    if started.exists():
        if json.loads(started.read_text())['sources'] != sources:
            raise ValueError('Interrupted evidence run has different frozen sources')
    else:
        with started.open('x') as stream:
            json.dump(dict(sources=sources, utc=datetime.now(timezone.utc).isoformat(),
                           job=os.getenv('SLURM_JOB_ID')), stream, indent=2)
    dist = projected(base/ARCHIVES['distance'], index, index['distance_row'], DIST_COLS)
    star = projected(base/ARCHIVES['stellar'], index, index['stellar_row'], STAR_COLS)
    gaia = join_gaia(photo['identity'], gaia_root)
    label = derive_gaia_labels(gaia)
    if tile_stats() != sources['gaia_tiles']:
        raise ValueError('Gaia archive changed during evidence join')
    evidence = {k: photo[k] for k in ('identity', 'ls_id', 'TYPE', 'cell')}
    evidence.update(role=np.full(receipt['n'], ROLE), **channels(dist, star),
                    p_corr=label['p_corr'], gaia_parallax_error=gaia['parallax_error'],
                    parallax_eligible=label['astro_quality'] & label['zpt_valid'] & label['finite_parallax'])
    atomic_npz(out/'evidence.npz', **evidence)
    csv_rows(out/'coverage.csv', coverage(evidence))
    save_json(complete, dict(freeze_sha256=digest, n=receipt['n'], all_PSF=True, outcomes_opened=True,
                            signed_gaia=int(evidence['parallax_eligible'].sum()),
                            precise_teacher=int(evidence['distance_precise'].sum()), job=os.getenv('SLURM_JOB_ID'),
                            outputs={name: sha(out/name) for name in ('evidence.npz', 'coverage.csv')}))


if __name__ == '__main__':
    main()
