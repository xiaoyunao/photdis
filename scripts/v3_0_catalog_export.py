#!/usr/bin/env python3
"""Publish discoverable compact candidate tables without discarding joint PDFs."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from v2_1_near import ROOT,load,sha,save_json,atomic_npz,csv_rows
from photdis.stage1 import require_compute
from v3_0_validation_inputs import check_freeze
D=ROOT/'data/v3_0'


def main():
    require_compute();p=argparse.ArgumentParser()
    p.add_argument('--field',type=int,default=int(os.getenv('SLURM_ARRAY_TASK_ID',-1)));a=p.parse_args()
    assert 0<=a.field<14
    method=check_freeze();planpath=D/'catalog/transfer_plan.json';plan=json.loads(planpath.read_text())
    assert plan['method_freeze']==method and len(plan['responses'])==1
    response=plan['responses'][0];base=D/'catalog'/f'field_{a.field:02}'
    prepared=json.loads((base/'prepared.json').read_text());source=load(base/'inputs.npz')
    assert sha(base/'inputs.npz')==prepared['outputs']['inputs.npz'] and (source['TYPE']=='PSF').all()
    assert len(source['identity'])==prepared['candidate'] and prepared['candidate']>0
    folder=base/'population_transfer'/f'refit_{response["draw"]:02}_s{response["draws"]}'
    complete=json.loads((folder/'complete.json').read_text());assert complete['integration_passed']
    assert complete['inputs']['method_freeze']==method
    inputs={str(x.relative_to(ROOT)):sha(x) for x in [base/'prepared.json',folder/'complete.json',planpath,Path(__file__)]}
    output=base/'candidate_v3_0.npz';marker=base/'candidate_v3_0.json'
    if marker.exists():
        old=json.loads(marker.read_text());assert old['inputs']==inputs and old['table_sha256']==sha(output)
        print('VERIFIED EXISTING EXPORT',a.field,len(source['identity']));return
    pieces=[];index=[];seen=np.zeros(len(source['identity']),bool);order=np.argsort(source['identity'])
    for path in sorted((folder/'chunks').glob('chunk_*.npz')):
        if path.stem.endswith('_counts'):continue
        metadata=json.loads(path.with_suffix('.json').read_text());assert sha(path)==metadata['outputs'][path.name]
        star=load(path);rows=order[np.searchsorted(source['identity'][order],star['identity'])]
        np.testing.assert_array_equal(source['identity'][rows],star['identity']);assert not seen[rows].any();seen[rows]=True
        assert (star['TYPE']=='PSF').all()
        keys=['identity','ls_id','TYPE','field','photometric_system','band_mask','mu','modes','mode_mass',
              'minor_mode_mass','qnear','probability','supported','extrapolation','multimodal','prior_sensitive',
              'reference_mu','reference_qnear','prior_delta_mu50','prior_delta_qnear',
              'reference_numerical_converged','negative_intrinsic_support','discrepancy_probability','binary_probability']
        part={k:star[k] for k in keys}
        part.update(RA=source['RA'][rows],DEC=source['DEC'][rows],band_state=source['band_state'][rows],
            class_probability=star['probability'].sum(axis=2),distance_probability=star['probability'].sum(axis=1),
            final_half_delta=np.max(abs(star['half_a']-star['half_b']),axis=(1,2))+2*plan['discarded_response_mass'])
        part['numerically_converged']=part['reference_numerical_converged']&(part['final_half_delta']<.005)
        part['class_ambiguous']=part['class_probability'].max(axis=1)<.8
        assert np.isfinite(part['mu']).all() and np.isfinite(part['probability']).all()
        np.testing.assert_allclose(part['probability'].sum(axis=(1,2)),1,atol=1e-7)
        pieces.append(part)
        joint=ROOT/metadata['source_joint'];assert sha(joint)==metadata['source_joint_sha256']
        index.append(dict(chunk=str(path.relative_to(ROOT)),chunk_sha256=sha(path),rows=len(rows),
            reference_joint=str(joint.relative_to(ROOT)),reference_joint_sha256=sha(joint),
            shared_parameters=str((folder/'shared_parameters.npz').relative_to(ROOT)),
            counts=str((path.parent/(path.stem+'_counts.npz')).relative_to(ROOT))))
    assert seen.all()
    table={k:np.concatenate([part[k] for part in pieces]) for k in pieces[0]}
    assert len(np.unique(table['identity']))==len(source['identity'])
    atomic_npz(output,**table);csv_rows(base/'candidate_v3_0_chunks.csv',index)
    save_json(marker,dict(inputs=inputs,method_freeze=method,rows=len(source['identity']),field=a.field,
        table_sha256=sha(output),chunk_index_sha256=sha(base/'candidate_v3_0_chunks.csv'),
        shared_parameters_sha256=sha(folder/'shared_parameters.npz'),
        numerical_failed=int((~table['numerically_converged']).sum()),
        near95=int((table['qnear']>.95).sum()),multimodal=int(table['multimodal'].sum()),
        response=plan['responses'][0],discarded_response_mass=plan['discarded_response_mass'],
        columns={k:dict(dtype=str(v.dtype),shape=list(v.shape)) for k,v in table.items()},
        mu_quantiles=[.025,.16,.5,.84,.975],distance_units='mu=5log10(d/kpc)+10',
        class_order=['dwarf','evolved','other'],distance_edges_kpc=[0,2,5,10,20,50,'infinity'],
        flag_definitions=dict(class_ambiguous='max class probability <0.8; descriptive, not a calibrated hard classifier',
            prior_sensitive='abs(target qnear - same-response reference qnear)>0.05',
            multimodal='at least two distance modes with mass>0.05',
            numerical_failed='reference quadrature failed or final joint half-chain difference >=0.005; retained, not silently filtered'),
        scope='Research candidate only; modular target-sample transfer, not full-catalog refit or deployment. Finite-response support and physical calibration remain limited. Shared parameter indices are global; chunk PDFs and reweightable joint coefficients remain linked.',
        job=os.getenv('SLURM_JOB_ID')))
    print('CANDIDATE EXPORT',a.field,len(source['identity']),'numeric failed',int((~table['numerically_converged']).sum()),flush=True)


if __name__=='__main__':main()
