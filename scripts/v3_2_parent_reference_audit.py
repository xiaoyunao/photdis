#!/usr/bin/env python3
"""Audit reference identity coverage of the frozen v3.1 application frame."""
import json
import os
from pathlib import Path

import numpy as np

from photdis.stage1 import require_compute
from v2_1_near import ROOT, csv_rows, load, save_json, sha
from v3_2_reference_freeze import require_native
from v3_2_tilt_inputs import verified


def main():
    require_compute()
    highq = ROOT/'data/v3_1/highq_counts/complete.json'
    archive = ROOT/'manifests/v3_1/highq_counts/complete.json'
    verified(highq, sha(archive))
    frame = json.loads(highq.read_text())
    selection_path = verified(highq.with_name('selection.npz'), frame['outputs']['selection.npz'])
    parent = load(selection_path)
    ids, selected = parent['identity'], parent['selected']
    if (frame['all_PSF'] is not True or not (parent['TYPE'] == 'PSF').all()
            or ids.dtype.kind not in 'iu' or ids.shape != (3540,) or len(np.unique(ids)) != len(ids)
            or selected.dtype.kind != 'b' or selected.shape != ids.shape or selected.sum() != frame['n']
            or frame['parent_n'] != len(ids) or frame['n'] != 3003):
        raise ValueError('Frozen high-q frame cannot be reproduced')
    np.testing.assert_array_equal(selected, parent['qnear'] > .95)

    sources = [archive, highq, selection_path, Path(__file__).resolve(),
               ROOT/'scripts/v3_2_reference_freeze.py']
    rows = []
    for role, native_dir, field in (('DR19_Astra', 'reference_native', 'admitted'),
                                    ('DR20_LCO', 'minesweeper_native', 'admitted'),
                                    ('HST_common', 'common_scale_inventory', 'admitted_members')):
        local = ROOT/f'manifests/v3_2/{native_dir}/complete.json'
        native = verified(ROOT/f'data/v3_2/{native_dir}/complete.json', sha(local))
        sources.extend([local, native])
        receipt = json.loads(native.read_text())
        n = receipt[field]
        if receipt['all_admitted_PSF'] is not True:
            raise ValueError(f'Unverified native PSF cohort: {role}')
        other = np.array([], dtype=ids.dtype)
        if n:
            photo = verified(native.with_name('photometry.npz'), receipt['outputs']['photometry.npz'])
            sources.append(photo)
            data = load(photo)
            require_native(data)
            other = data['identity']
            if len(other) != n:
                raise ValueError(f'Native population differs from receipt: {role}')
        overlap = np.isin(ids, other)
        rows.append(dict(cohort=role, parent_n=len(ids), high_q_n=int(selected.sum()),
                         admitted_reference_n=n, parent_overlap=int(overlap.sum()),
                         high_q_overlap=int((overlap & selected).sum())))
    out = ROOT/'data/v3_2/parent_reference_support'
    out.mkdir(exist_ok=False)
    csv_rows(out/'overlap.csv', rows)
    save_json(out/'complete.json', dict(
        job=os.getenv('SLURM_JOB_ID'), inputs={str(p.relative_to(ROOT)): sha(p) for p in sources},
        outputs={'overlap.csv': sha(out/'overlap.csv')}, all_PSF=True, new_outcomes_read=False,
        scope='Exact identity intersections for the previously frozen v3.1 frame and three admitted '
              'v3.2 validation cohorts. Targeted inclusion is not a probability sample.'))


if __name__ == '__main__':
    main()
