#!/usr/bin/env python3
"""Freeze exact finalcandidate eligibility before external outcome opening."""
import json
import os
import joblib
import numpy as np
from v2_1_near import ROOT,load,sha,save_json,atomic_npz,csv_rows
from photdis.stage1 import require_compute
from photdis.v3_1.admission import candidate_admission


def main():
    require_compute(); root=ROOT/'data/v3_1'; out=root/'validation_admission'; out.mkdir(exist_ok=True)
    final=root/'final_validation'
    if any((final/cohort/'started.json').exists() for cohort in ['field','common']):
        raise RuntimeError('Candidate admission must be registered before final outcomes')
    roles=root/'target_native_pilot/roles.npz'; source=ROOT/'data/v3_0/controls/qrf_grz.joblib'
    paths=[source,roles,root/'target_native_pilot/complete.json',root/'validation_roles/complete.json',
        ROOT/'data/v3_0/cohorts/application.npz',root/'validation_roles/field_photometry.npz',root/'validation_roles/common_photometry.npz',
        ROOT/'scripts/v3_1_validation_admission.py',ROOT/'src/photdis/v3_1/admission.py',ROOT/'src/photdis/baseline_v1.py',
        ROOT/'src/photdis/v3/controls.py',ROOT/'docs/v3_1_validation_admission_protocol.md']
    inputs={str(path.relative_to(ROOT)):sha(path) for path in paths}; freeze=out/'inputs.json'; marker=out/'complete.json'
    if freeze.exists():assert json.loads(freeze.read_text())==inputs
    else:save_json(freeze,inputs)
    if marker.exists():
        done=json.loads(marker.read_text()); assert done['inputs']==inputs
        for name,digest in done['outputs'].items():assert sha(out/name)==digest
        print('VERIFIED CANDIDATE ADMISSION',done['cohorts'],flush=True); return
    pilot=json.loads(paths[2].read_text()); vr=json.loads(paths[3].read_text())
    assert sha(source)==pilot['inputs'][str(source.relative_to(ROOT))]
    assert sha(roles)==pilot['outputs'][roles.name]
    model=joblib.load(source); model.nn.n_jobs=1; original=load(roles); rows=[]; outputs=[]
    for cohort,path in zip(['application','field','common'],paths[4:7]):
        if cohort=='application':assert sha(path)==pilot['inputs'][str(path.relative_to(ROOT))]
        else:assert sha(path)==vr['outputs'][path.name]
        t=load(path); assert (t['TYPE']=='PSF').all(); result=candidate_admission(t,model)
        if cohort=='application':
            np.testing.assert_array_equal(result['identity'],original['identity'])
            np.testing.assert_array_equal(result['candidate'],original['candidate'])
            assert result['candidate'].sum()==3540
        name=cohort+'.npz'; atomic_npz(out/name,**result); outputs.append(name)
        rows.append(dict(cohort=cohort,n=len(t['identity']),candidate=int(result['candidate'].sum()),
            outside=int((~result['candidate']).sum()),nonPSF_excluded=0,missing_TYPE_excluded=0,
            nonnative=int((~result['native_psf']).sum()),not_bright=int((~result['bright']).sum()),
            optical_precision_failed=int((~result['precise']).sum()),qrf_support_failed=int((~result['qrf_support']).sum()),
            reason_counts_overlap=True))
    csv_rows(out/'summary.csv',rows); outputs.append('summary.csv')
    save_json(marker,dict(inputs=inputs,outputs={name:sha(out/name) for name in outputs},cohorts=rows,
        job=os.getenv('SLURM_JOB_ID'),all_PSF=True,photometry_only=True,references_read=False,original_candidate_reproduced=True,
        scope='Exact3540-cohort native observed-r/error/QRF support gate. All frozen validation rows retained and separately tagged. QRF support is a selection proxy, not demonstrated SED or absolute-distance calibration.'))


if __name__=='__main__':main()
