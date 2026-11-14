#!/usr/bin/env python3
"""Compare the repeat-00 entry cache with the direct pilot and its users."""
import json
import os
from pathlib import Path

import numpy as np

from photdis.stage1 import require_compute
from v2_1_near import ROOT, load, save_json, sha
import v3_2_radial_recover as radial
from v3_2_tilt_inputs import verified


def main():
    require_compute()
    root = ROOT/'data/v3_2'
    design_path = root/'radial_design/complete.json'
    design = json.loads(design_path.read_text())
    model_sha = design['inputs'][str(radial.MODEL_PATH.relative_to(ROOT))]
    radial._GRID = radial.entry_grid()
    verified(radial.MODEL_PATH, model_sha)
    sources = {str(p.relative_to(ROOT)): sha(p) for p in
               (design_path, radial.MODEL_PATH, Path(__file__).resolve())}
    rows = []
    for catalog in (0, 128):
        photo_path = design_path.parent/f'photometry_{catalog:03}.npz'
        photo = load(verified(photo_path, design['outputs'][photo_path.name]))
        radial.verify_psf(photo)
        entry, cache_marker = radial.cached_entry(catalog, photo, design_path, model_sha)
        folder = root/'radial_recovery'/f'catalog_{catalog:03}'
        marker = folder/'complete.json'
        receipt = json.loads(marker.read_text())
        if (receipt['catalog'] != catalog or not receipt['all_PSF']
                or receipt['inputs'][str(photo_path.relative_to(ROOT))] != sha(photo_path)):
            raise ValueError('Recovery receipt does not bind the pilot PSF cohort')
        curves = load(verified(folder/'curves.npz', receipt['outputs']['curves.npz']))
        for key in ('identity', 'TYPE'):
            np.testing.assert_array_equal(curves[key], photo[key])
        np.testing.assert_array_equal(curves['mu'], radial._GRID)
        np.testing.assert_array_equal(curves['entry'], entry)
        sources.update({str(p.relative_to(ROOT)): sha(p)
                        for p in (photo_path, marker, cache_marker)})
        rows.append(dict(catalog=catalog, n=len(photo['identity']), array_equal=True,
                         cached_in_recovery=str(cache_marker.relative_to(ROOT)) in receipt['inputs']))
    if rows[0]['cached_in_recovery']:
        raise ValueError('Catalog 000 must be the independently integrated pilot')
    if not rows[1]['cached_in_recovery']:
        raise ValueError('Catalog 128 must use the frozen repeat cache')
    output = root/'radial_entry_verification'
    output.mkdir(exist_ok=False)
    save_json(output/'complete.json', dict(inputs=sources, checks=rows, all_PSF=True,
              passed=True, job=os.getenv('SLURM_JOB_ID'),
              scope='Exact array comparison on the original direct and cached pilot contexts.'))


if __name__ == '__main__':
    main()
