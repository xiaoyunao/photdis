#!/usr/bin/env python3
"""Cache the two frozen entry quadratures for one radial design repeat."""
import argparse
import json
import os

import numpy as np

from photdis.stage1 import require_compute
from photdis.v3.selection import effective_errors
from v2_1_near import ROOT, atomic_npz, load, save_json, sha
from v3_2_radial_checkpoint import compute_rows
from v3_2_radial_recover import (
    CONTEXT_KEYS, catalog_rows, entry_curve, entry_grid, entry_prior,
    entry_signature, load_entry_model, verify_psf,
)
from v3_2_tilt_inputs import verified

_MODEL = _GRID = _PRIOR = _ERROR = _PHOTO = None


def compute_row(row):
    obs = {key: _PHOTO[key][row] for key in ('transmission', 'EBV')}
    return entry_curve(_MODEL, obs, _ERROR[row], _GRID, _PRIOR)


def main():
    global _MODEL, _GRID, _PRIOR, _ERROR, _PHOTO

    require_compute()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('repeat', type=int)
    parser.add_argument('--workers', type=int, choices=range(1, 33), default=16)
    args = parser.parse_args()
    root = ROOT / 'data/v3_2/radial_design'
    design_path = root / 'complete.json'
    record = json.loads(design_path.read_text())
    rows = [r for r in catalog_rows(design_path, record) if int(r['repeat']) == args.repeat]
    if not rows:
        raise ValueError('Repeat outside the frozen design')
    catalog = int(rows[0]['catalog'])
    photo_path = root / f'photometry_{catalog:03}.npz'
    _PHOTO = load(verified(photo_path, record['outputs'][photo_path.name]))
    verify_psf(_PHOTO)
    _MODEL, model_sha = load_entry_model(record)
    _PRIOR, _ = entry_prior(_MODEL)
    _GRID = entry_grid()
    _ERROR = effective_errors(_PHOTO)
    output = ROOT / 'data/v3_2/radial_entry' / f'repeat_{args.repeat:02}'
    if output.exists():
        raise FileExistsError(output)
    inputs = dict(repeat=args.repeat, source_catalog=catalog, n_contexts=len(_PHOTO['TYPE']),
                  all_PSF=True, design_sha256=sha(design_path), model_sha256=model_sha,
                  photo_sha256=sha(photo_path), entry_code_sha256=entry_signature())
    checkpoint = output.parent/'row_checkpoints'/output.name
    results = compute_rows(checkpoint, inputs,
                           _PHOTO['context_identity'], compute_row, workers=args.workers, keys=('entry',))
    entry = np.stack([value[0] for value in results])
    output.mkdir(parents=True, exist_ok=False)
    curves = output / 'curves.npz'
    atomic_npz(curves, mu=_GRID, entry=entry, **{key: _PHOTO[key] for key in CONTEXT_KEYS})
    save_json(output / 'complete.json', dict(
        **inputs, outputs={curves.name: sha(curves)}, job=os.getenv('SLURM_JOB_ID'),
        row_checkpoint_sha256=sha(checkpoint/'inputs.json')))


if __name__ == '__main__':
    main()
