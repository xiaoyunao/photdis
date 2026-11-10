#!/usr/bin/env python3
"""Single fixed fit, model freeze, and resumable full-PDF evaluation."""
import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
import joblib
import numpy as np
from v2_3_prepare import ROOT, DEST, TABLES
from v2_1_near import load, subset, save_json, sha, atomic_npz
from baseline_v1_4_experiments import arrays
from photdis.stage1 import require_compute
from photdis.baseline_v0 import HistogramQRF, distance_quantiles
from photdis.baseline_v1 import integrate, quantiles_interpolated
from photdis.baseline_v1_2 import full_parallax_log_density
from photdis.baseline_parallax import parallax_check
from photdis.density_v2_3 import ErrorDeconvolvedMixture, observation_covariance
from photdis.probability_v2_1 import near_mass
from photdis.selection_v2_2 import canonical_probability

_T = _MODEL = _CONTROL = _NAME = _ROLE = _HASH = None


def freeze_inputs():
    paths=['data/v2_3/training.npz','data/v2_3/reserved_inputs.npz','docs/v2_3_protocol.md',
           'src/photdis/density_v2_3.py','scripts/v2_3_run.py','src/photdis/baseline_v0.py',
           'src/photdis/baseline_v1.py','src/photdis/baseline_v1_2.py','src/photdis/baseline_parallax.py']
    return {p:sha(ROOT/p) for p in paths}


def fit():
    t=load(DEST/'training.npz');assert (t['TYPE']=='PSF').all()
    m,e,ids=arrays(t,np.arange(len(t['identity'])))
    frozen=freeze_inputs();path=DEST/'fit_inputs.json'
    if path.exists(): assert json.loads(path.read_text())==frozen
    else: save_json(path,frozen)
    control_path=DEST/'control.joblib';candidate_path=DEST/'candidate.joblib'
    if not control_path.exists():
        model=HistogramQRF(2);model.forest.n_jobs=16
        print('FIT QRF',len(ids),flush=True);model.fit(m,t['Mg'],t['class_label']);model.forest.n_jobs=1
        joblib.dump(model,control_path.with_suffix('.tmp'));os.replace(control_path.with_suffix('.tmp'),control_path)
    if not candidate_path.exists():
        y=np.column_stack((m[:,0]-m[:,1],m[:,1]-m[:,2],t['Mg']))
        s=observation_covariance(e,t['sigma_mu'])
        model=ErrorDeconvolvedMixture()
        model.fit(y,s,t['class_label'],callback=lambda i,ll:print('XD',i,ll,flush=True))
        joblib.dump(model,candidate_path.with_suffix('.tmp'));os.replace(candidate_path.with_suffix('.tmp'),candidate_path)
    model=joblib.load(candidate_path)
    save_json(DEST/'fit_convergence.json',dict(converged=model.converged,iterations=len(model.history),history=model.history,
        components=model.components,unlabeled_train=int((t['class_label']<0).sum())))
    if not model.converged: raise RuntimeError('Candidate not converged: reserve stays closed; no automatic retuning')
    assert freeze_inputs()==frozen
    models={n:sha(DEST/(n+'.joblib')) for n in ('control','candidate')}
    save_json(DEST/'models_frozen.json',dict(inputs=frozen,models=models,job=os.getenv('SLURM_JOB_ID'),
        reserved_outcomes_opened=False,roles='All reserved identities evaluation only; no calibration'))
    print('MODELS FROZEN',models,flush=True)


def support(t,m):
    x=_CONTROL.feature_matrix(m)
    dis=_CONTROL.nn.kneighbors((x-_CONTROL.mean)/_CONTROL.scale,n_neighbors=10)[0][:,-1]
    return (dis<=_CONTROL.radius)&np.all((x>=_CONTROL.lo)&(x<=_CONTROL.hi),axis=1)


def chunk(rows):
    path=DEST/'chunks'/f'{_ROLE}_{_NAME}_{rows[0]:07}.npz';marker=path.with_suffix('.json')
    if marker.exists():
        a=json.loads(marker.read_text());assert a['model']==_HASH and sha(path)==a['sha256']
        assert np.array_equal(load(path)['identity'],_T['identity'][rows])
        return str(path)
    m,e,ids=arrays(_T,rows)
    if _NAME=='control':
        p=integrate(_MODEL,m,e,ids,draws=128)['wide'];edges=_MODEL.edges
        p['qnear']=near_mass(p['pdf'],edges,m[:,0]);extra={}
    else:
        edges=np.arange(-30,40.0001,.1);p=_MODEL.predict(m,e,edges)
        p['mu']=distance_quantiles(m[:,0],quantiles_interpolated(p['pdf'],edges))[0]
        extra={k:p[k] for k in ('mixture_weight','mixture_mean','mixture_sigma','escaped_mass')}
    q=canonical_probability(p['qnear'])
    supported=support(_T,m)
    scope=(q>.95)&(_T['r'][rows]<20)&(_T['optical_error'][rows]<.1)&supported
    ga=_T['parallax_eligible'][rows] if 'parallax_eligible' in _T else np.zeros(len(rows),bool)
    log=np.full(len(rows),np.nan);pit=log.copy()
    if ga.any():
        args=(p['pdf'][ga],edges,m[ga,0],_T['p_corr'][rows][ga],_T['gaia_parallax_error'][rows][ga])
        log[ga]=full_parallax_log_density(*args,order=16)
        pit[ga]=parallax_check(*args,np.full(ga.sum(),-np.inf),np.full(ga.sum(),np.inf))['pit']
    assert np.isfinite(p['pdf']).all() and np.all(p['pdf']>=0)
    np.testing.assert_allclose(p['pdf'].sum(axis=1),1,atol=2e-6)
    atomic_npz(path,identity=ids,TYPE=_T['TYPE'][rows],rows=rows,mu=p['mu'],pdf=p['pdf'].astype('f8' if _NAME=='candidate' else 'f4'),
        qnear=q,weights=p['weights'],supported=supported,near_scope=scope,gaia_log=log,gaia_pit=pit,**extra)
    save_json(marker,dict(sha256=sha(path),model=_HASH))
    return str(path)


def evaluate(role):
    global _T,_MODEL,_CONTROL,_NAME,_ROLE,_HASH
    freeze=json.loads((DEST/'models_frozen.json').read_text())
    assert freeze['inputs']==freeze_inputs()
    for n,h in freeze['models'].items(): assert sha(DEST/(n+'.joblib'))==h
    input_path={'development':ROOT/'data/v2_1/development.npz','reserved':DEST/'reserved_inputs.npz',
                'application':ROOT/'data/v2_2/application_inputs.npz'}[role]
    _T=load(input_path);_ROLE=role
    assert (_T['TYPE']=='PSF').all()
    ids=load(DEST/'training.npz')['identity']
    if role!='application': assert not np.intersect1d(ids,_T['identity']).size
    if role=='reserved':
        save_json(DEST/'reserved_opened.json',dict(models=freeze['models'],input_sha256=sha(input_path),
            role='Same-sky new identities, predictions opened only after both models frozen',job=os.getenv('SLURM_JOB_ID')))
    _CONTROL=joblib.load(DEST/'control.joblib');(DEST/'chunks').mkdir(exist_ok=True)
    for name in ('control','candidate'):
        _NAME=name;_HASH=freeze['models'][name];_MODEL=_CONTROL if name=='control' else joblib.load(DEST/'candidate.joblib')
        edges=_MODEL.edges if name=='control' else np.arange(-30,40.0001,.1)
        atomic_npz(DEST/(name+'_grid.npz'),edges=edges)
        tasks=[np.arange(i,min(i+256,len(_T['identity']))) for i in range(0,len(_T['identity']),256)]
        with concurrent.futures.ProcessPoolExecutor(max_workers=16,mp_context=mp.get_context('fork')) as pool:
            paths=[]
            for i,path in enumerate(pool.map(chunk,tasks),1):
                paths.append(path)
                if i%32==0:print('PREDICT',role,name,i,len(tasks),flush=True)
        keys=['identity','TYPE','mu','qnear','weights','supported','near_scope','gaia_log','gaia_pit']
        parts=[load(p) for p in paths]
        p={k:np.concatenate([a[k] for a in parts]) for k in keys}
        assert np.array_equal(p['identity'],_T['identity'])
        atomic_npz(DEST/f'{role}_{name}.npz',**p)
        save_json(DEST/f'{role}_{name}_complete.json',dict(input_sha256=sha(input_path),model=_HASH,
            chunks={str(x):sha(x) for x in paths},rows=len(p['identity']),job=os.getenv('SLURM_JOB_ID')))
    print('EVALUATE COMPLETE',role,flush=True)


if __name__=='__main__':
    require_compute();parser=argparse.ArgumentParser();parser.add_argument('task',choices=['fit','development','reserved','application']);a=parser.parse_args()
    if a.task=='fit':fit()
    else:evaluate(a.task)
