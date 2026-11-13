#!/usr/bin/env python3
"""Separate historical parent inventory from actual PSF scientific eligibility."""
import os
import numpy as np
from v2_1_near import ROOT,sha,save_json,csv_rows
from v3_1_inventory import projected
from photdis.stage1 import require_compute
from photdis.reference_v1_4 import unique_join


def main():
    require_compute(); old=ROOT/'data/baseline_v1_4/references.npz'
    native=ROOT/'data/v2_0/native_report.npz'; current=ROOT/'data/v3_0/references.npz'
    fitted=ROOT/'data/v2_3/training.npz'
    t=projected(old,['identity','ls_id','cell','split','photometric_eligible','identity_conflict','class_label','class_eligible',
        'specdis_valid','spdist_valid','parallax_eligible'])
    n=projected(native,['ls_id','TYPE']); j=unique_join(t['ls_id'],n['ls_id']); found=j>=0
    types=np.full(len(j),'','U8'); types[found]=np.char.strip(n['TYPE'][j[found]].astype('U'))
    psf=found & (types=='PSF'); validphoto=t['photometric_eligible'] & psf
    eligible=validphoto & ~t['identity_conflict']
    now=projected(current,['identity','cell','role']); fit=projected(fitted,['identity'])
    in_v3=np.isin(t['identity'],now['identity']); in_fit=np.isin(t['identity'],fit['identity'])
    masks=dict(historical_parent=np.ones(len(j),bool),native_PSF=psf,
        native_PSF_photometric=validphoto,native_PSF_photometric_unique=eligible,
        eligible_outside_v3_parent=eligible & ~in_v3,
        eligible_outside_v3_sky=eligible & ~np.isin(t['cell'],now['cell']),
        eligible_not_v2_3_fit=eligible & ~in_fit)
    rows=[]
    for name,mask in masks.items():
        rows.append(dict(stage=name,n=int(mask.sum()),identities=len(np.unique(t['identity'][mask])),
            sky_cells=len(np.unique(t['cell'][mask])),missing_TYPE=int((mask & (types=='')).sum()),
            nonPSF=int((mask & (types!='') & ~psf).sum()),
            class_labels=int((mask & t['class_eligible'] & (t['class_label']>=0)).sum()),
            evolved=int((mask & t['class_eligible'] & (t['class_label']==1)).sum()),
            specdis_historical_flag=int((mask & t['specdis_valid']).sum()),spdist=int((mask & t['spdist_valid']).sum()),
            signed_gaia=int((mask & t['parallax_eligible']).sum()),v2_3_fit=int((mask & in_fit).sum())))
    roles=[]
    for role in np.unique(now['role']):
        roles.append(dict(v3_role=int(role),n=int((now['role']==role).sum()),
            v2_3_fit=int(np.isin(now['identity'][now['role']==role],fit['identity']).sum())))
    out=ROOT/'data/v3_1/reference_role_audit';out.mkdir(parents=True,exist_ok=True)
    csv_rows(out/'eligibility.csv',rows);csv_rows(out/'roles.csv',roles)
    save_json(out/'complete.json',dict(inputs={str(p.relative_to(ROOT)):sha(p) for p in [old,native,current,fitted,ROOT/'scripts/v3_1_reference_roles.py']},
        outputs={p.name:sha(p) for p in out.glob('*.csv')},job=os.getenv('SLURM_JOB_ID'),native_missing=int((~found).sum()),
        admitted_missing_or_nonPSF=int((eligible & ~psf).sum()),
        scope='Historical parent inclusion is not fitted/scored use. Native morphology/photometry eligibility precedes prior-role exclusions. v2.3 fitting and all v3 parent roles separately counted. No new final-test assignment, reference outcome comparison or model fit.'))
    print('V3.1 REFERENCE ROLE AUDIT COMPLETE',flush=True)


if __name__=='__main__':main()
