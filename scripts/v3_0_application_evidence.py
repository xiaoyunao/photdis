#!/usr/bin/env python3
"""Associate already-used application sample after photometric predictions.

This is development, not the unopened3250 or new-sky validation. Native source
ownership and original signed Gaia evidence are preserved, including nonstellar
risk controls and fitting-identity strata.
"""
import json
import os
import joblib
import numpy as np
from v2_1_near import ROOT, load, sha, save_json, atomic_npz
from photdis.stage1 import require_compute
from photdis.reference_v1_4 import unique_join
from photdis.v3.data import primary_channel
from v3_0_cohorts import EVIDENCE_KEYS
D = ROOT/'data/v3_0'


def main():
    require_compute()
    paths = [D/'cohorts/application.npz', D/'references.npz', D/'sed_fit.joblib',
        ROOT/'data/v2_2/application_inputs.npz', ROOT/'scripts/v3_0_application_evidence.py']
    paths += [ROOT/'data/baseline_v0'/f'parent_{field:02}.npz' for field in range(13)]
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    marker = D/'evaluation/application.json'; output = marker.with_suffix('.npz')
    if marker.exists():
        old = json.loads(marker.read_text()); assert old['inputs'] == inputs
        assert sha(output) == old['sha256']; return
    app = load(paths[0]); ref = load(paths[1]); model = joblib.load(paths[2]); old = load(paths[3])
    np.testing.assert_array_equal(app['identity'], old['identity'])
    assert (app['TYPE'] == 'PSF').all()
    index = unique_join(app['identity'], ref['identity']); found = index >= 0
    t = {}
    for key in EVIDENCE_KEYS:
        x = ref[key]; fill = False if x.dtype.kind == 'b' else np.nan if x.dtype.kind == 'f' else '' if x.dtype.kind in 'SU' else 0 if x.dtype.kind == 'u' else -1
        value = np.full((len(index),)+x.shape[1:], fill, dtype=x.dtype)
        value[found] = x[index[found]]; t[key] = value
    t['identity'] = app['identity']; t['TYPE'] = app['TYPE']
    t['reference_match'] = found; t['current_SED_fit'] = np.isin(app['identity'], model.training_identity)
    t['parallax_eligible'][:] = False; t['p_corr'][:] = np.nan; t['gaia_parallax_error'][:] = np.nan
    for field in np.unique(old['field']):
        use = old['field'] == field; rows = old['parent_row'][use]
        with np.load(ROOT/'data/baseline_v0'/f'parent_{field:02}.npz') as z:
            np.testing.assert_array_equal(z['identity'][rows], app['identity'][use])
            p = z['p_corr'][rows]; error = z['gaia_parallax_error'][rows]
            t['p_corr'][use] = p; t['gaia_parallax_error'][use] = error
            t['parallax_eligible'][use] = z['astro_quality'][rows] & z['zpt_valid'][rows] & np.isfinite(p) & np.isfinite(error) & (error > 0)
            t['nonstellar_risk'][use] = z['nonstellar_risk'][rows]
    t['primary_channel'] = primary_channel(t['specdis_valid'], t['spdist_valid'], t['parallax_eligible'])
    atomic_npz(output, **t)
    save_json(marker, dict(inputs=inputs, sha256=sha(output), n=len(index),
        matched_reference=int(found.sum()), current_SED_fit=int(t['current_SED_fit'].sum()),
        Gaia=int(t['parallax_eligible'].sum()), negative_Gaia=int((t['parallax_eligible'] & (t['p_corr'] < 0)).sum()),
        scope='Already-seen original26000photometric application sample, not prospective; no reference-based admission.', job=os.getenv('SLURM_JOB_ID')))
    print('APPLICATION EVIDENCE', len(index), int(found.sum()), int(t['parallax_eligible'].sum()), flush=True)


if __name__ == '__main__': main()
