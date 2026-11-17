#!/usr/bin/env python3
"""Count sources in equal-area sky cells and bind each core row to an original PDF."""
import argparse
import json
import os

import healpy as hp
import numpy as np

from photdis.stage1 import require_compute
from v2_1_near import ROOT, atomic_npz, save_json, sha
from v3_3_sed import verified
from v3_6_vac_distances import read_columns
from v3_8_direct_figures import SHELLS
from v3_9_unified import owner_rows
from final_catalogue_sky import selected_ids
from final_science_area import run as random_area
from final_science_catalog import indices, REGIONS

OUT = ROOT/'data/v3_final'
NSIDE = 128
NPIX = hp.nside2npix(NSIDE)
SHARDS = 24
CORE_FIELDS = ('ls_id', 'partition_id', 'pdf_row', 'photo_row', 'row_state',
               'near_numerical_flag', 'w1_state', 'selection_score')


def pixels(ra, dec):
    return hp.ang2pix(NSIDE, ra, dec, nest=True, lonlat=True)


def accumulate(counts, pix, distances):
    np.add.at(counts[0], pix, 1)
    for i, (lo, hi) in enumerate(SHELLS, 1):
        np.add.at(counts[i], pix[(distances >= lo) & (distances < hi)], 1)


def scan_part(part, index, overrides, core, counts):
    frozen = part['region'] == 'south_v35'
    region = REGIONS.index(part['region'])
    extra = ('qrf_quantiles_kpc', 'band_state', 'nominal_numerical_flag',
             'dwarf_selection_score') if frozen else ('north_primary_region', 'row_state')
    photo = read_columns(verified(ROOT/part['photo_file'], part['photo_sha256']),
                         ('TYPE', 'ls_id', 'RA', 'DEC', *extra))
    owner, _ = owner_rows(part['side'], photo, overrides, frozen=frozen)
    assert len(owner) == int(part['native_rows']) and owner.sum() == int(part['owned_rows'])
    pix = pixels(photo['RA'], photo['DEC'])
    np.add.at(counts[region, 0, 0], pix[owner], 1)
    if not part['pdf_file']:
        assert int(part['owned_pdf_rows']) == 0
        return None
    if frozen:
        photo_row = np.arange(len(owner))
        q = photo['qrf_quantiles_kpc']
    else:
        pdf = read_columns(verified(ROOT/part['pdf_file'], part['pdf_sha256']),
                           ('TYPE', 'ls_id', 'photo_row', 'qrf_quantiles_kpc'))
        photo_row = pdf['photo_row']
        np.testing.assert_array_equal(photo['ls_id'][photo_row], pdf['ls_id'])
        np.testing.assert_array_equal(photo['TYPE'][photo_row], pdf['TYPE'])
        q = pdf['qrf_quantiles_kpc']
    keep = owner[photo_row]
    assert keep.sum() == int(part['owned_pdf_rows']) and len(q) == int(part['native_pdf_rows'])
    pdf_rows = np.flatnonzero(keep)
    j = photo_row[pdf_rows]
    d = q[pdf_rows, 2]
    assert np.all(np.isfinite(d) & (d > 0))
    accumulate(counts[region, 1], pix[j], d)
    ids = photo['ls_id'][j]
    k = np.searchsorted(core, ids)
    selected = k < len(core)
    selected[selected] = core[k[selected]] == ids[selected]
    accumulate(counts[region, 2], pix[j[selected]], d[selected])
    pdf_rows, j = pdf_rows[selected], j[selected]
    if not len(j):
        return None
    n = len(j)
    return dict(ls_id=photo['ls_id'][j], partition_id=np.full(n, index, 'i4'),
                pdf_row=pdf_rows.astype('i4'), photo_row=j.astype('i4'),
                row_state=np.full(n, 7, 'u1') if frozen else photo['row_state'][j],
                near_numerical_flag=photo['nominal_numerical_flag'][j] if frozen else np.zeros(n, bool),
                w1_state=photo['band_state'][j, 4].astype('i1') if frozen else np.full(n, -1, 'i1'),
                selection_score=photo['dwarf_selection_score'][j].astype('f4') if frozen else np.full(n, np.nan, 'f4'))


def scan(group):
    parts, overrides, inputs = indices()
    core, inputs['science'] = selected_ids()
    where = np.array_split(np.arange(len(parts)), SHARDS)[group]
    counts = np.zeros((len(REGIONS), 3, 1+len(SHELLS), NPIX), 'i8')
    links = []
    for i in where:
        row = scan_part(parts[i], i, overrides, core, counts)
        if row is not None:
            links.append(row)
    np.testing.assert_array_equal(counts[:, 1:, 1:4].sum(axis=2), counts[:, 1:, 4])
    dest = OUT/'scan'
    dest.mkdir(parents=True, exist_ok=True)
    p = dest/f'{group:02}.npz'
    atomic_npz(p, counts=counts, **{k:np.concatenate([r[k] for r in links]) if links
                                  else np.empty(0, 'i8') for k in CORE_FIELDS})
    save_json(p.with_suffix('.json'), dict(job=os.getenv('SLURM_JOB_ID'), parts=len(where), inputs=inputs,
              totals=counts[:, :, 0].sum(axis=-1).tolist(), outputs={p.name:sha(p)}))


if __name__ == '__main__':
    require_compute()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=('scan','north','south'))
    a = parser.parse_args()
    if a.task == 'scan':
        scan(int(os.environ['SLURM_ARRAY_TASK_ID']))
    else:
        random_area(a.task, nside=NSIDE, destination=OUT/'area')
