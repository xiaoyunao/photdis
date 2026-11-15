#!/usr/bin/env python3
"""Read native photometry once per reference PSF for the F04 and F05 analyses."""
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fitsio
import numpy as np

from photdis.stage1 import legacy_ids, require_compute
from photdis.v3.data import BANDS, SIGNIFICANT
from v2_1_near import ROOT, atomic_npz, load, save_json, sha
from v3_3_sed import verified
from v3_5_dr11_inventory import CONFIG, OUT as V35
from v3_5_dr11_production import FREEZE, PRODUCTION
from v3_5_dr11_spatial import DELIVERY
from v3_5_dr11_validate import join_parts, take_rows
from v3_6_distance_evidence import OUT as REFERENCES
from v3_6_reference_index import OUT as INDEX, workers
from v3_6_vac_distances import read_columns


OUT = ROOT/'data/v3_6/reference_photometry'
STORED = ('band_state', 'band_mask', 'EBV')


def native_magnitudes(raw, state, mask):
    flux, inverse_variance, transmission = (np.column_stack([
        raw[f'{base}_{band}'] for band in BANDS]).astype('f8')
        for base in ('FLUX','FLUX_IVAR','MW_TRANSMISSION'))
    valid = mask & (flux>0) & np.isfinite(flux) & (transmission>0) & np.isfinite(transmission)
    # WISE flux measurements remain in the audit even when a logarithmic
    # magnitude would give a misleading color.
    valid[:,4:] &= state[:,4:]==SIGNIFICANT
    with np.errstate(divide='ignore',invalid='ignore'):
        magnitude = np.where(valid,22.5-2.5*np.log10(flux/transmission),np.nan)
        error = np.where(valid & (inverse_variance>0) & np.isfinite(inverse_variance),
                         (2.5/np.log(10))/(flux*np.sqrt(inverse_variance)),np.nan)
    return dict(magnitude=magnitude, magnitude_error=error, flux=flux,
                flux_ivar=inverse_variance, transmission=transmission, usable_magnitude=valid)


def photometry_for_sweep(name, tile_info, digest, root, offsets, expected):
    folder = PRODUCTION/Path(name).stem
    marker = json.loads(verified(folder/'complete.json',digest).read_text())
    offset = 0
    chunks = []
    for filename, checksum in sorted(marker['outputs'].items()):
        if not filename.startswith('part_'):
            continue
        part = folder/filename
        with np.load(part,allow_pickle=False) as z:
            ids = z['ls_id']
            a,b = np.searchsorted(offsets,(offset,offset+len(ids)))
            if b>a:
                verified(part,checksum)
                relative = offsets[a:b]-offset
                np.testing.assert_array_equal(ids[relative],expected[a:b])
                chunks.append({k:z[k][relative] for k in ('parent_row',*STORED)})
            offset += len(ids)
    forecast = join_parts(chunks)
    selected = forecast.pop('parent_row')
    path = root/name
    stat = path.stat()
    if (stat.st_size,stat.st_mtime_ns)!=(tile_info['size'],tile_info['mtime_ns']):
        raise ValueError(f'Native sweep differs from inventory: {path}')
    order = np.argsort(selected)
    if len(np.unique(selected))!=len(expected):
        raise ValueError(f'Repeated or missing native row in {name}')
    keys = ('RELEASE','BRICKID','OBJID','TYPE')
    columns = [f'{base}_{band}' for base in ('FLUX','FLUX_IVAR','MW_TRANSMISSION') for band in BANDS]
    with fitsio.FITS(str(path)) as handle:
        table = handle[1]
        available = set(table.get_colnames())
        raw = table.read(columns=[*keys,*[k for k in columns if k in available]],
                         rows=selected[order])[np.argsort(order)]
    np.testing.assert_array_equal(legacy_ids(raw),expected)
    if not (np.char.strip(raw['TYPE'].astype('U'))=='PSF').all():
        raise ValueError(f'Unverified native TYPE: {path}')
    raw = {k:raw[k] if k in available else np.full(len(raw),np.nan) for k in columns}
    return dict(**forecast,**native_magnitudes(raw,forecast['band_state'],forecast['band_mask']),
                TYPE=np.full(len(expected),'PSF'),ls_id=expected)


def main():
    require_compute()
    ref_marker = REFERENCES/'complete.json'
    index_marker = INDEX/'complete.json'
    delivery_marker = DELIVERY/'complete.json'
    references, index, delivery = [json.loads(p.read_text()) for p in (ref_marker,index_marker,delivery_marker)]
    if references['inputs'][str(index_marker.relative_to(ROOT))]!=sha(index_marker):
        raise ValueError('Distance evidence uses a different identity index')
    cat = load(verified(INDEX/'catalog_rows.npz',index['outputs']['catalog_rows.npz']))
    rows = np.unique(np.concatenate([
        read_columns(verified(REFERENCES/f'{name}.npz',references['outputs'][f'{name}.npz']),
                     ('catalog_row',))['catalog_row'] for name in references['rows']]))
    if not len(rows) or rows[0]<0 or rows[-1]>=len(cat['ls_id']):
        raise ValueError('Reference union falls outside the supported catalog')
    freeze = json.loads(verified(FREEZE,delivery['production_freeze_sha256']).read_text())
    inventory = V35/'inventory/complete.json'
    inv = json.loads(verified(inventory,freeze['inputs'][str(inventory.relative_to(ROOT))]).read_text())
    tiles = {t['filename']:t for t in inv['sweeps']}
    root = Path(json.loads(verified(CONFIG,freeze['inputs'][str(CONFIG.relative_to(ROOT))]).read_text())['sweep_root'])
    with verified(DELIVERY/'sweeps.csv',delivery['outputs']['sweeps.csv']).open() as stream:
        checks = {t['sweep']:t['receipt_sha256'] for t in csv.DictReader(stream)}
    bounds = np.searchsorted(rows,cat['sweep_offset'])
    names = cat['sweep']
    groups = [k for k in range(len(names)) if bounds[k]<bounds[k+1]]

    def read(k):
        name = names[k]
        use = rows[bounds[k]:bounds[k+1]]
        return photometry_for_sweep(name,tiles[name],checks[name],root,
                                    use-cat['sweep_offset'][k],cat['ls_id'][use])

    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'complete.json').exists():
        raise FileExistsError(OUT/'complete.json')
    with ThreadPoolExecutor(max_workers=workers()) as pool:
        result = join_parts(list(pool.map(read,groups)))
    np.testing.assert_array_equal(result['ls_id'],cat['ls_id'][rows])
    output = OUT/'photometry.npz'
    atomic_npz(output,catalog_row=rows,**result)
    save_json(OUT/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),bands=BANDS,rows=len(rows),
        touched_sweeps=len(groups),inputs={str(p.relative_to(ROOT)):sha(p) for p in
        (Path(__file__),ref_marker,index_marker,delivery_marker,FREEZE,inventory,CONFIG)},
        outputs={output.name:sha(output)}))
    print(len(rows),len(groups),flush=True)


if __name__=='__main__':
    main()
