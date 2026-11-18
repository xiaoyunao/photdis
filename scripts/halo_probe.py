#!/usr/bin/env python3
"""Inspect source schemas and actual cached admission inventories on a CPU node."""
from zipfile import ZipFile
from numpy.lib.format import read_magic, _read_array_header
from photdis.stage1 import require_compute
from halo_common import *
from final_science_catalog import indices
from v3_5_dr11_production import PRODUCTION, FREEZE
from v3_9_catalog import OUT as V39
from v3_17_sample import INPUT


def schema(path):
    result = {}
    with ZipFile(path) as z:
        for name in z.namelist():
            with z.open(name) as f:
                version = read_magic(f)
                shape, _, dtype = _read_array_header(f, version)
                result[name.removesuffix('.npy')] = dict(shape=shape, dtype=str(dtype))
    return result


if __name__ == '__main__':
    require_compute()
    parts, overrides, inputs = indices()
    result = dict(parts=len(parts), schemas={}, domains={}, native_override_rows=len(overrides['ls_id']))
    for region in ('south_v35', 'south_faint', 'south_low', 'north'):
        p = next(x for x in parts if x['region']==region)
        result['schemas'][region] = schema(ROOT / p['photo_file'])
        result['domains'][region] = p
    for side in ('south','north'):
        path = ROOT / f'data/v3_10/seam/{side}.npz'
        result['schemas']['seam_'+side] = schema(path)
    result['schemas']['old_noise'] = schema(ROOT / 'data/science_validation_20261118/noise/individual_noise.npz')
    for name in ('M5','M13','M30','M92'):
        result['schemas'][name] = schema(ROOT / f'data/v3_14/audit/{name}/native_join.npz')
    from astropy.io import fits
    p = INPUT / 'v4_Figure3_dotter_isochrone.fits'
    with fits.open(p) as h:
        result['dotter'] = dict(columns=h[1].columns.names, header=dict(h[1].header), rows=len(h[1].data))
    # Persist private paths and schemas, without reading reference outcomes.
    d = folder('probe')
    save_json(d/'metadata.json', result)
    complete(d, inputs)
    print(json.dumps(result))
