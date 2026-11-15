#!/usr/bin/env python3
"""Inventory class-conditional training support in the existing reference split."""
import json
import os
from pathlib import Path

import numpy as np

from photdis.audit import rvs_metallicity
from photdis.stage1 import require_compute
from photdis.v3.data import ROLES, WEAK, SIGNIFICANT
from v2_1_near import ROOT, csv_rows, save_json, sha


def inventory(ref):
    types = np.char.strip(ref['TYPE'].astype(str))
    psf = types == 'PSF'
    roles = (1, 3, 4)
    active = np.isin(ref['role'], roles)
    if len(np.unique(ref['identity'][active])) != active.sum():
        raise ValueError('Duplicate identities across fitting and opened roles')
    eligible = psf & active & ref['reference_stellar_eligible'] & (ref['primary_channel'] > 0)
    if not np.isin(ref['class_label'][eligible], (-1, 0, 1, 2)).all():
        raise ValueError('Unexpected reference class label')
    if not np.isin(ref['primary_channel'][eligible], (1, 2, 3)).all():
        raise ValueError('Unexpected distance reference channel')
    feh, valid = rvs_metallicity(ref['rvs_feh'], ref['rvs_teff'], ref['rvs_logg'])
    measured_feh = valid & ref['rvs_parameter_quality'] & np.isfinite(feh)
    counts, exclusions = [], []
    for role in roles:
        raw = ref['role'] == role
        admitted = raw & eligible
        exclusions.append(dict(role=ROLES[role], parent=int(raw.sum()),
            missing_type=int((raw & (types == '')).sum()),
            non_psf=int((raw & ~psf & (types != '')).sum()),
            nonstellar=int((raw & psf & ~ref['reference_stellar_eligible']).sum()),
            no_distance=int((raw & psf & ref['reference_stellar_eligible'] & (ref['primary_channel'] <= 0)).sum()),
            admitted=int(admitted.sum())))
        for label in (-1, 0, 1, 2):
            for channel, source in ((1, 'SpecDis'), (2, 'SPDist'), (3, 'Gaia')):
                use = admitted & (ref['class_label'] == label) & (ref['primary_channel'] == channel)
                counts.append(dict(role=ROLES[role], class_label=label, source=source,
                    n=int(use.sum()), sky_blocks=len(np.unique(ref['cell'][use])),
                    corrected_feh=int((use & measured_feh).sum()),
                    metal_poor=int((use & measured_feh & (feh < -1.5)).sum()),
                    weak_w1=int((use & (ref['band_state'][:, 4] == WEAK)).sum()),
                    significant_w1=int((use & (ref['band_state'][:, 4] == SIGNIFICANT)).sum())))
        if sum(row['n'] for row in counts if row['role'] == ROLES[role]) != admitted.sum():
            raise ValueError(f'Unreconciled reference census for {ROLES[role]}')
    return counts, exclusions


def main():
    require_compute()
    source = ROOT/'data/v3_0/references.npz'
    marker = ROOT/'manifests/v3_0/preparation.json'
    source_sha = sha(source)
    if source_sha != json.loads(marker.read_text())['output_sha256']:
        raise ValueError('Reference parent differs from the frozen preparation')
    columns = ('identity', 'TYPE', 'role', 'reference_stellar_eligible', 'primary_channel',
               'class_label', 'cell', 'rvs_feh', 'rvs_teff', 'rvs_logg',
               'rvs_parameter_quality', 'band_state')
    with np.load(source) as data:
        counts, exclusions = inventory({key: data[key] for key in columns})
    out = ROOT/'data/v3_3/reference_support'
    out.mkdir(parents=True, exist_ok=False)
    csv_rows(out/'by_class_source.csv', counts)
    csv_rows(out/'admission.csv', exclusions)
    inputs = {str(path.relative_to(ROOT)): sha(path) for path in (marker, Path(__file__).resolve())}
    inputs[str(source.relative_to(ROOT))] = source_sha
    save_json(out/'complete.json', dict(job=os.getenv('SLURM_JOB_ID'), inputs=inputs,
        outputs={name: sha(out/name) for name in ('by_class_source.csv', 'admission.csv')},
        scope='Frozen training, hyperparameter and seen-development support. No reserved outcomes read.'))


if __name__ == '__main__':
    main()
