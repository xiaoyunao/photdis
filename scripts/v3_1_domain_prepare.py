#!/usr/bin/env python3
"""Photometry-only fixed use-domain grid, before external outcome association."""
import json
import os
import numpy as np
from v2_1_near import ROOT,load,sha,save_json,atomic_npz,csv_rows
from photdis.stage1 import require_compute
from photdis.v3_1.use_domains import observable_cells,cell_descriptors,N_CELLS


def main():
    require_compute(); root=ROOT/'data/v3_1'; out=root/'domain_grid'; out.mkdir(exist_ok=True)
    assert not any((root/f'final_validation/{c}/started.json').exists() for c in ['field','common'])
    admission=root/'validation_admission'; done=json.loads((admission/'complete.json').read_text())
    assert done['original_candidate_reproduced'] and done['photometry_only'] and not done['references_read']
    photos={'application':ROOT/'data/v3_0/cohorts/application.npz',
        'field':root/'validation_roles/field_photometry.npz','common':root/'validation_roles/common_photometry.npz'}
    paths=[admission/'complete.json',ROOT/'scripts/v3_1_domain_prepare.py',ROOT/'src/photdis/v3_1/use_domains.py',
        ROOT/'src/photdis/v3/controls.py',ROOT/'docs/v3_1_use_domain_protocol.md']+list(photos.values())
    paths += [admission/(c+'.npz') for c in photos]
    inputs={str(p.relative_to(ROOT)):sha(p) for p in paths}; freeze=out/'inputs.json'
    if freeze.exists():assert json.loads(freeze.read_text())==inputs
    else:save_json(freeze,inputs)
    marker=out/'complete.json'
    if marker.exists():
        record=json.loads(marker.read_text()); assert record['inputs']==inputs
        for name,digest in record['outputs'].items():assert sha(out/name)==digest
        return
    rows=cell_descriptors(); outputs=[]; inventory=[]
    for cohort,path in photos.items():
        assert sha(path)==done['inputs'][str(path.relative_to(ROOT))]
        ap=admission/(cohort+'.npz'); assert sha(ap)==done['outputs'][ap.name]
        table=load(path); values=observable_cells(table,load(ap)); assert (table['TYPE']=='PSF').all()
        if 'role' in table:values['role']=table['role']
        name=cohort+'.npz'; atomic_npz(out/name,**values); outputs.append(name)
        count=np.bincount(values['cell'],minlength=N_CELLS); candidate=np.bincount(values['cell'][values['candidate']],minlength=N_CELLS)
        for i,row in enumerate(rows):row.update({cohort+'_n':int(count[i]),cohort+'_candidate':int(candidate[i])})
        inventory.append(dict(cohort=cohort,n=len(values['identity']),candidate=int(values['candidate'].sum()),
            nonPSF_excluded=0,missing_TYPE_excluded=0))
    csv_rows(out/'cells.csv',rows); outputs.append('cells.csv')
    save_json(marker,dict(inputs=inputs,outputs={name:sha(out/name) for name in outputs},inventory=inventory,cells=N_CELLS,
        all_PSF=True,references_read=False,job=os.getenv('SLURM_JOB_ID'),
        scope='Fixed observable cells and exactcandidate tags only. No external score, accuracy claim, density or calibration certificate.'))


if __name__=='__main__':main()
