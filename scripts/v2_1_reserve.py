#!/usr/bin/env python3
"""Reserve actually unused identities/sky before any candidate fitting."""
import json
import numpy as np
from v2_1_near import ROOT, DEST, load, subset, save_json, sha, atomic_npz
from photdis.stage1 import require_compute
from photdis.probability_v2_1 import psf_admission


def tiles(ra, dec):
    return np.floor(ra).astype(int)*180 + np.clip(np.floor(dec+90).astype(int), 0, 179)


def main():
    require_compute()
    t = load(ROOT/'data/v2_0/population.npz')
    psf, audit = psf_admission(t['TYPE'])
    used = []
    inputs = {}
    paths = sorted((ROOT/'data/v2_0').glob('training_*.npz'))
    paths += sorted((ROOT/'data/baseline_v1_4/experiments').glob('*_identities.npz'))
    paths += sorted((ROOT/'data/baseline_v1_4/experiments').glob('*_legacy_training.npz'))
    for path in paths:
        with np.load(path) as z:
            for key in z.files:
                if 'identity' in key:
                    used.append(z[key])
        inputs[str(path.relative_to(ROOT))] = sha(path)
    used = np.unique(np.concatenate(used))
    # All previously inspected cal/test roles are development, even rows outside
    # the distance-selected view. Quarantined split3 is never offered as holdout.
    occupied = np.isin(t['identity'], used) | np.isin(t['split'], [1, 2]) | t['historical_identity']
    candidate = (psf & t['photometric_eligible'] & (t['split'] == 0) & ~occupied)
    cell = tiles(t['RA'], t['DEC'])
    occupied_tiles = np.unique(cell[occupied])
    # Conservative one-degree neighboring-tile buffer; no candidate residuals,
    # class values, distance predictions or qnear enter this rule.
    forbidden = []
    for dra in (-1, 0, 1):
        for ddec in (-1, 0, 1):
            dec = occupied_tiles % 180 + ddec
            ok = (dec >= 0) & (dec < 180)
            forbidden.append(((occupied_tiles[ok]//180+dra) % 360)*180+dec[ok])
    candidate &= ~np.isin(cell, np.unique(np.concatenate(forbidden)))
    reserved = subset(t, candidate)
    assert (reserved['TYPE'] == 'PSF').all()
    assert not np.intersect1d(reserved['identity'], used).size
    path = DEST/'future_validation_reservation.npz'
    if path.exists():
        old = load(path)
        assert np.array_equal(old['identity'], reserved['identity']), 'Reservation cannot silently change'
    else:
        atomic_npz(path, **reserved)
    previous = DEST/'final_validation_reservation.json'
    archive = DEST/'initial_validation_availability_note.json'
    if previous.exists() and not archive.exists():
        archive.write_text(previous.read_text())
    save_json(previous, dict(reserved_identities=len(reserved['identity']),
        status='reserved_pending_evidence_and_system_audit' if len(reserved['identity']) else 'external_new_sky_acquisition_required',
        native_psf_verified=True, training_and_evaluation_identity_disjoint=True,
        one_degree_tiles=np.unique(cell[candidate]), exclusion_buffer='Occupied one-degree tiles and their eight neighbors',
        inputs=inputs, reservation_sha256=sha(path), morphology=audit,
        remaining=['Verify system/cluster membership with independent membership catalogs before final scientific use',
                   'Acquire adequate independent reference labels/observations where absent',
                   'Keep reserved identities and every future noise replica out of model training/calibration',
                   'No current candidate/final performance has been inspected on this reservation'],
        note='Spatial buffering reduces spatial leakage but cannot certify unprovided system membership; no final-validation claim until that audit passes. Existing development test stays development.'))
    print('RESERVATION', len(reserved['identity']), 'identities in', len(np.unique(cell[candidate])), 'isolated tiles', flush=True)


if __name__ == '__main__':
    main()
