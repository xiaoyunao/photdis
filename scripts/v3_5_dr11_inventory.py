#!/usr/bin/env python3
"""Inventory native DR11 tiles and select distributed pilot candidates."""
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from photdis.stage1 import eligible_tiles, require_compute, tile_bounds
from v2_1_near import ROOT, save_json, sha

CONFIG = ROOT/'configs/v3_5_dr11.json'
OUT = ROOT/'data/v3_5'


def describe(path, *, sweep):
    import fitsio

    stat = path.stat()
    row = dict(filename=path.name, size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    if sweep:
        row['bounds_deg'] = tile_bounds(path)
    try:
        with fitsio.FITS(str(path)) as f:
            header = f[1].read_header()
            row.update(n_rows=f[1].get_nrows(), columns=f[1].get_colnames(),
                       density=header.get('DENSITY'), extname=header.get('EXTNAME'))
    except (OSError, RuntimeError, ValueError, IndexError) as exc:
        row.update(error=f'{type(exc).__name__}: {exc}')
    return row


def candidate_pilot(sweeps):
    # Two adjacent low-extinction southern blocks have already been used in
    # model studies. Add different parts of the survey for the transfer check.
    preferred = ('sweep-010m045-015m040.fits', 'sweep-015m045-020m040.fits',
                 'sweep-175p020-180p025.fits', 'sweep-350m045-355m040.fits')
    usable = {row['filename']: row for row in sweeps if 'error' not in row}
    picked = [name for name in preferred if name in usable]
    if len(picked) < 3 and usable:
        names = list(usable)
        picked = list(dict.fromkeys([*picked, *[names[i] for i in
                                    np.linspace(0, len(names)-1, 4, dtype=int)]]))
    return picked


def main():
    require_compute()
    cfg = json.loads(CONFIG.read_text())
    sweep_root, random_root = (Path(cfg[key]) for key in ('sweep_root', 'random_root'))
    sweeps = sorted(sweep_root.glob('sweep-*.fits'))
    _, paths = eligible_tiles(dict(sweep_root=str(sweep_root),
                                  sky={'abs_b_min':cfg['minimum_abs_galactic_latitude']}))
    eligible = {Path(p).name for p in paths}
    randoms = sorted(p for p in random_root.iterdir()
                     if p.is_file() and (p.name.endswith('.fits') or p.name.endswith('.fits.gz')))
    if not sweeps or not randoms:
        raise ValueError('The sweep and native random directories must both be populated')
    nworkers = int(os.getenv('SLURM_CPUS_PER_TASK', '4'))
    with ThreadPoolExecutor(max_workers=nworkers) as pool:
        headers = list(pool.map(lambda p: describe(p, sweep=True), sweeps))
        # The random directory metadata inventory covers all files. A small
        # set of headers suffices to establish the schema before a real scan.
        random_headers = list(pool.map(lambda p: describe(p, sweep=False), randoms[:8]))
    for row in headers:
        row['possibly_high_latitude'] = row['filename'] in eligible
    result = dict(job=os.getenv('SLURM_JOB_ID'), config_sha256=sha(CONFIG), sweep_root=str(sweep_root),
                  sweeps=headers, sweep_n=len(sweeps), possibly_high_latitude_n=len(eligible),
                  random_root=str(random_root), random_n=len(randoms),
                  random_files=[dict(filename=p.name,size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
                                for p in randoms], random_header_examples=random_headers,
                  pilot=candidate_pilot([r for r in headers if r['possibly_high_latitude']]))
    if not result['pilot']:
        raise ValueError('No usable high-latitude pilot sweeps')
    target = OUT/'inventory'/'complete.json'
    if target.exists():
        raise FileExistsError(target)
    save_json(target, result)
    print('DR11 sweep files, possible high latitude, random files, pilot',
          len(sweeps), len(eligible), len(randoms), result['pilot'])


if __name__ == '__main__':
    main()
