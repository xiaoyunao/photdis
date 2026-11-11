#!/usr/bin/env python3
"""Audit prospective evidence availability without altering admission or outcomes."""
import json
import os
import numpy as np
from v2_1_near import ROOT, load, sha, save_json, csv_rows
from photdis.stage1 import require_compute
from v3_0_validation_inputs import check_freeze
D = ROOT/'data/v3_0'


def main():
    require_compute(); method = check_freeze(); rows = []; inputs = {}
    old = load(ROOT/'data/v2_4/prospective_identity_reservation.npz')
    for name in ('reservation', 'new_sky', 'independent_scale'):
        folder = D/'validation'/name
        t, e, g = [load(folder/x) for x in ('photometry.npz', 'evidence.npz', 'gaia_payload.npz')]
        for x in ('photometry.npz', 'evidence.npz', 'gaia_payload.npz', 'evidence.json'):
            inputs[str((folder/x).relative_to(ROOT))] = sha(folder/x)
        np.testing.assert_array_equal(t['identity'], e['identity'])
        np.testing.assert_array_equal(t['identity'], g['identity'])
        assert (t['TYPE'] == 'PSF').all()
        if name == 'reservation':
            np.testing.assert_array_equal(t['identity'], old['identity'])
            np.testing.assert_array_equal(t['ls_id'], old['ls_id'])
        reasons, counts = np.unique(e['no_reference_reason'], return_counts=True)
        row = dict(cohort=name, n=len(t['identity']), positive_native_Gaia_ID=int((t['source_id']>0).sum()),
            positive_identity=int((t['identity']>0).sum()), exact_Gaia_matches=int(g['gaia_match'].sum()),
            missing_Gaia_tiles=int(g['gaia_file_missing'].sum()),
            signed_Gaia_eligible=int(e['parallax_eligible'].sum()), negative_Gaia=int((e['parallax_eligible']&(e['p_corr']<0)).sum()),
            specdis_valid=int(e['specdis_valid'].sum()), spdist_valid=int(e['spdist_valid'].sum()),
            precise_Gaia=int((e['parallax_eligible']&(e['p_corr']>10*e['gaia_parallax_error'])).sum()),
            reliable_class=int((e['class_label']>=0).sum()),
            no_reference_reasons=json.dumps(dict(zip(reasons,map(int,counts))), sort_keys=True))
        rows.append(row)
    dest = D/'validation/audit'; dest.mkdir(exist_ok=True)
    csv_rows(dest/'availability.csv', rows)
    save_json(dest/'complete.json', dict(method_freeze=method, inputs=inputs, rows=rows, job=os.getenv('SLURM_JOB_ID'),
        outputs={'availability.csv':sha(dest/'availability.csv')},
        reservation_selection='Original reservation explicitly excluded all identities in baseline_v0, v1_3, v1_4 reference inventories and the 26000 application sample; no replacement after seeing outcomes.',
        scope='Zero reference availability is lack of validation evidence, never successful distance validation. Native Gaia IDs and exact original reservation ownership checked independently.'))
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == '__main__': main()
