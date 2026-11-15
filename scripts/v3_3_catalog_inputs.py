#!/usr/bin/env python3
"""Extract complete photometry-only windows from the audited DR11 parent."""
import json
import os
from pathlib import Path

import numpy as np

from photdis.stage1 import require_compute
from photdis.v3.data import MISSING, WEAK, SIGNIFICANT
from v2_1_near import ROOT, atomic_npz, save_json, sha
from v3_3_sed import BASE, MANIFESTS, verified

CONFIG = ROOT/'configs/v3_3_catalog.json'
OUTPUT = ROOT/'data/v3_3/catalog/inputs'
COLUMNS = ('identity', 'ls_id', 'TYPE', 'RA', 'DEC', 'field', 'parent_row',
           'photometric_system', 'flux', 'flux_error', 'transmission', 'EBV', 'band_mask', 'band_state')


def select_window(table, footprint, observed_r):
    ra_lo, ra_hi = footprint['ra_deg']
    dec_lo, dec_hi = footprint['dec_deg']
    r_lo, r_hi = observed_r
    ra, dec = table['RA'], table['DEC']
    window = (ra >= ra_lo) & (ra < ra_hi) & (dec >= dec_lo) & (dec < dec_hi)
    types = np.char.strip(table['TYPE'].astype(str))
    psf = types == 'PSF'
    r_flux = table['flux'][:, 1]
    valid = np.isfinite(r_flux) & (r_flux > 0)
    r_mag = np.full(len(ra), np.inf)
    r_mag[valid] = 22.5 - 2.5*np.log10(r_flux[valid])
    selected = window & psf & (r_mag >= r_lo) & (r_mag < r_hi)
    if not (table['photometric_system'][selected] == 'DECam_DR11').all():
        raise ValueError('Expected native DECam photometry in the selected window')
    data = {k: table[k][selected] for k in COLUMNS}
    data['TYPE'] = types[selected]
    data['observed_r'] = r_mag[selected]
    if not np.all(data['field'] == footprint['field']) or len(np.unique(data['identity'])) != selected.sum():
        raise ValueError('Selected catalog identities do not reconcile to their field')
    wise = data['band_state'][:, 4]
    counts = dict(parent=int(window.sum()), admitted=int(selected.sum()),
                  missing_TYPE=int((window & (types == '')).sum()),
                  non_PSF=int((window & ~psf & (types != '')).sum()),
                  outside_magnitude=int((window & psf & ~selected).sum()),
                  weak_w1=int((wise == WEAK).sum()), significant_w1=int((wise == SIGNIFICANT).sum()),
                  missing_w1=int((wise == MISSING).sum()))
    return data, counts


def main():
    require_compute()
    config = json.loads(CONFIG.read_text())
    frozen = MANIFESTS/'catalog_summary/complete.json'
    receipts = json.loads(frozen.read_text())['inputs']
    dependencies = {str(p.relative_to(ROOT)): sha(p) for p in (frozen, CONFIG, Path(__file__).resolve())}
    chunks, census = [], {}
    for footprint in config['footprints']:
        name = f'field_{footprint["field"]:02d}'
        marker = BASE/'catalog'/name/'prepared.json'
        path = marker.with_name('inputs.npz')
        key = str(marker.relative_to(ROOT))
        prior = json.loads(verified(marker, receipts[key]).read_text())
        digest = prior['outputs']['inputs.npz']
        with np.load(verified(path, digest)) as source:
            table = {k: source[k] for k in COLUMNS}
        if len(table['identity']) != prior['candidate']:
            raise ValueError(f'Parent census differs for {name}')
        data, counts = select_window(table, footprint, config['observed_r'])
        if not counts['admitted']:
            raise ValueError(f'Empty footprint: {name}')
        chunks.append((name, data))
        census[name] = counts
        dependencies[key] = receipts[key]
        dependencies[str(path.relative_to(ROOT))] = digest
    ids = np.concatenate([data['identity'] for _, data in chunks])
    if len(np.unique(ids)) != len(ids):
        raise ValueError('Duplicate catalog identities across windows')
    OUTPUT.mkdir(parents=True, exist_ok=False)
    for name, data in chunks:
        atomic_npz(OUTPUT/f'{name}.npz', **data)
    save_json(OUTPUT/'complete.json', dict(job=os.getenv('SLURM_JOB_ID'), n=len(ids), all_PSF=True,
        census=census, inputs=dependencies, outputs={p.name: sha(p) for p in sorted(OUTPUT.iterdir())},
        scope='All admitted rows in the half-open footprints and magnitude envelope, before any new model prediction.'))
    print(json.dumps(census, indent=2), flush=True)


if __name__ == '__main__':
    main()
