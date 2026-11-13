#!/usr/bin/env python3
"""Allocated all-kernel native likelihood curves; truth never enters inference."""
import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
import joblib
import numpy as np
from scipy.special import logsumexp
from v2_1_near import ROOT,load,sha,save_json,atomic_npz
from photdis.stage1 import require_compute
from photdis.v3.model import inference_model
from photdis.v3.inference import kernel_loglike
from photdis.v3.compression import DISTANCE_MU_BOUNDARIES
from photdis.v3.selection import effective_errors,gaussian_component_entry
from photdis.v3_1.radial_selection import RadialSelectionTable
from v3_0_response import observation

_MODEL=_PHOT=_GRID=None


def one_object(i):
    likelihood,_=kernel_loglike(_MODEL,observation(_PHOT,i),_GRID)
    binary=np.array([1-_MODEL.binary_reference_fraction,_MODEL.binary_reference_fraction])
    lk=logsumexp(likelihood+np.log(binary)[:,None,None],axis=0)
    prior=_MODEL.weights[:,None]*_MODEL.classes;prior/=prior.sum(axis=0)[None]
    return np.array([logsumexp(lk+np.log(prior[:,c])[:,None],axis=0) for c in range(3)])


def main():
    global _MODEL,_PHOT,_GRID
    require_compute();p=argparse.ArgumentParser();p.add_argument('--catalog',type=int,required=True)
    p.add_argument('--step',type=float,default=.02);p.add_argument('--workers',type=int,default=32)
    p.add_argument('--audit-selection',action='store_true');a=p.parse_args()
    assert a.step in [.02,.01]
    assert 0<=a.catalog<176
    root=ROOT/'data/v3_1';source=root/'radial_catalogs/photometry.npz';model_path=ROOT/'data/v3_0/sed_fit.joblib'
    select_path=root/'radial_selection/q256_seed311701/table.npz';done=json.loads(source.with_name('complete.json').read_text())
    selected=json.loads(select_path.with_name('complete.json').read_text());assert selected['passed'] and sha(select_path)==selected['outputs']['table.npz']
    assert sha(source)==done['outputs'][source.name]
    paths=[source,model_path,select_path,ROOT/'docs/v3_1_radial_protocol.md',ROOT/'scripts/v3_1_radial_likelihood.py',
        ROOT/'src/photdis/v3/fast.py',ROOT/'src/photdis/v3/inference.py',ROOT/'src/photdis/v3_1/radial_selection.py']
    inputs={str(p.relative_to(ROOT)):sha(p) for p in paths};inputs.update(catalog=a.catalog,step=a.step,audit_selection=a.audit_selection)
    out=root/'radial_likelihood'/f'catalog_{a.catalog:03}_step{a.step:g}';out.mkdir(parents=True,exist_ok=True);marker=out/'complete.json'
    if marker.exists():
        record=json.loads(marker.read_text());assert record['inputs']==inputs
        for name,digest in record['outputs'].items():assert sha(out/name)==digest
        return
    # Catalog blocks were frozen at128 objects. No truth file is read here.
    source_table=load(source);indices=np.arange(a.catalog*128,(a.catalog+1)*128)
    assert indices[-1]<len(source_table['identity'])
    _PHOT={k:v[indices] for k,v in source_table.items()};assert (_PHOT['TYPE']=='PSF').all()
    assert len(np.unique(_PHOT['field']))==1
    _MODEL=inference_model(joblib.load(model_path));assert _MODEL.converged
    _GRID=np.linspace(-15,45,int(round(60/a.step))+1)
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('fork')) as pool:
        logs=np.array(list(pool.map(one_object,range(128))))
    errors=effective_errors(_PHOT);table=RadialSelectionTable(**load(select_path));entry,audit=table.evaluate(errors,_GRID)
    if not audit['inside_profile_hull'].all():raise RuntimeError('Uncertified radial selection profile; keep failure')
    direct=[]
    if a.audit_selection:
        prior=_MODEL.weights[:,None]*_MODEL.classes;prior/=prior.sum(axis=0)[None]
        grid=np.linspace(0,25,251)
        for i in [0,31,63,95,127]:
            raw=gaussian_component_entry(_MODEL,errors[i],grid,draws=512,seed=311702)
            exact=prior.T@((1-_MODEL.binary_reference_fraction)*raw[0]+_MODEL.binary_reference_fraction*raw[1])
            interp,_=table.evaluate(errors[i:i+1],grid)
            direct.append(dict(row=i,max_absolute_delta=float(abs(exact-interp[0]).max())))
    atomic_npz(out/'curves.npz',identity=_PHOT['identity'],TYPE=_PHOT['TYPE'],field=_PHOT['field'],mu=_GRID,
        log_likelihood=logs,entry=entry,class_frequency=_MODEL.weights@_MODEL.classes,
        nearest_profile_distance=audit['nearest_profile_distance'])
    save_json(marker,dict(inputs=inputs,outputs={'curves.npz':sha(out/'curves.npz')},job=os.getenv('SLURM_JOB_ID'),
        n=128,grid=len(_GRID),all_PSF=True,all_profiles_inside=True,direct_selection_audit=direct,
        direct_selection_audit_performed=a.audit_selection,selection_passed=all(x['max_absolute_delta']<.003 for x in direct),
        scope='Fixed nominal SED/binary/zero within-class tilt; no truth values or new real validation outcomes read. Native class-conditional log likelihood retains all kernels.'))
    print('V3.1 RADIAL LIKELIHOOD COMPLETE',a.catalog,a.step,direct,flush=True)


if __name__=='__main__':main()
