#!/usr/bin/env python3
"""Full and prior-withdrawn moment-Gaussian structure likelihood sidecars."""
import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
import joblib
import numpy as np
from scipy.special import logsumexp,softmax
from scipy.stats import norm
from v2_1_near import ROOT,load,sha,save_json,atomic_npz,csv_rows
from v3_0_response import observation
from v3_1_validation_predict import class_loglike
from photdis.stage1 import require_compute
from photdis.v3.model import inference_model
from photdis.v3.systematics import calibration_vectors
from photdis.v3.inference import DISTANCE_BASIS_MEAN,DISTANCE_BASIS_SIGMA
from photdis.v3.likelihood import trapezoid_weights
from photdis.v3_1.gaussian_response import class_gaussian_logresponse
from photdis.v3_1.structure import structure_quadrature,withdrawn_gaussian_logresponse

_TABLE=_MODEL=_CAL=_FREQUENCY=_OUT=_INPUTS=_COMMON=None


def one(row):
    identity=int(_TABLE['identity'][row]);marker=_OUT/f'object_{identity}.json';payload=marker.with_suffix('.npz')
    if marker.exists():
        r=json.loads(marker.read_text());assert r['inputs']==_INPUTS and sha(payload)==r['sha256'];return r
    means=np.r_[DISTANCE_BASIS_MEAN,_COMMON];sigmas=np.r_[DISTANCE_BASIS_SIGMA,np.full(len(_COMMON),.02)]
    arrays={};values=[]
    for name,number,step in [('coarse',3501,.004),('fine',7001,.002)]:
        regional=np.linspace(_COMMON.min()-.2,_COMMON.max()+.2,int(np.ceil((_COMMON.max()-_COMMON.min()+.4)/step))+1)
        grid=np.unique(np.r_[np.linspace(-20,50,number),regional]);logw=np.log(trapezoid_weights(grid))
        curve=class_loglike(_MODEL,observation(_TABLE,row),grid,_CAL[row])
        reference=curve+norm.logpdf(grid,15,5)[None]+logw[None]
        reference_z=logsumexp(reference,axis=1);mass=np.exp(reference-reference_z[:,None])
        center=mass@grid;variance=np.sum(mass*(grid[None]-center[:,None])**2,axis=1)
        assert (variance>0).all()
        full=class_gaussian_logresponse(curve,grid,means,sigmas,np.zeros((1,3,2)))[0]
        baseline=withdrawn_gaussian_logresponse(reference_z,center,np.sqrt(variance),means,sigmas)
        arrays.update({name+'_grid':grid,name+'_class_loglike':curve,name+'_reference_logevidence':reference_z,
            name+'_reference_mean':center,name+'_reference_sigma':np.sqrt(variance),
            name+'_full_logresponse':full,name+'_gaussian_logresponse':baseline})
        values.append((full,baseline))
    checks=[]
    for index,branch in enumerate(['full','gaussian']):
        coarse,fine=values[0][index],values[1][index];relevant=np.maximum(coarse,fine)>max(coarse.max(),fine.max())-30
        delta=float(abs(coarse-fine)[relevant].max())
        probability=[softmax(v+np.log(_FREQUENCY)[:,None],axis=0) for v in (coarse,fine)]
        pd=float(abs(probability[0]-probability[1]).max())
        checks.append(dict(branch=branch,maximum_relevant_logresponse_delta=delta,maximum_class_probability_delta=pd,passed=delta<.02 and pd<.003))
    atomic_npz(payload,identity=np.array([identity]),TYPE=np.array(['PSF']),row=np.array([row]),
        radial_means=means,radial_sigmas=sigmas,common_mu=_COMMON,class_frequency=_FREQUENCY,**arrays)
    r=dict(inputs=_INPUTS,identity=identity,row=row,TYPE='PSF',checks=checks,passed=all(c['passed'] for c in checks),sha256=sha(payload),job=os.getenv('SLURM_JOB_ID'))
    save_json(marker,r);return r


def main():
    global _TABLE,_MODEL,_CAL,_FREQUENCY,_OUT,_INPUTS,_COMMON
    require_compute();p=argparse.ArgumentParser();p.add_argument('--draw',type=int,default=int(os.getenv('SLURM_ARRAY_TASK_ID',-1)));a=p.parse_args();assert 0<=a.draw<32
    root=ROOT/'data/v3_1/structure';old=ROOT/'data/v3_0';model_dir=old/f'uncertainty_refit/draw_{a.draw:02}'
    paths=[root/'complete.json',root/'photometry.npz',model_dir/'complete.json',model_dir/'model.joblib',old/'sed_fit.joblib',old/'uncertainty/shared_draws.npz',
        ROOT/'scripts/v3_1_structure_native.py',ROOT/'scripts/v3_1_validation_predict.py',ROOT/'docs/v3_1_structure_numerical_protocol.md',
        ROOT/'docs/v3_1_structure_protocol.md']
    paths += [ROOT/'src/photdis/v3_1'/n for n in ['structure.py','gaussian_response.py','gaussian_entry.py']]
    _INPUTS={str(p.relative_to(ROOT)):sha(p) for p in paths};_INPUTS['response_draw']=a.draw
    prep=json.loads(paths[0].read_text());assert prep['all_PSF'] and sha(paths[1])==prep['outputs']['photometry.npz']
    md=json.loads(paths[2].read_text());assert md['converged'] and sha(paths[3])==md['sha256']
    _TABLE=load(paths[1]);assert len(_TABLE['identity'])==512 and (_TABLE['TYPE']=='PSF').all()
    _MODEL=inference_model(joblib.load(paths[3]));nominal=inference_model(joblib.load(paths[4]));_FREQUENCY=nominal.weights@nominal.classes
    _CAL,_=calibration_vectors(_TABLE,load(paths[5]),a.draw)
    _COMMON=np.unique(np.concatenate([structure_quadrature(n,64)[0] for n in (16,32,64)]))
    _OUT=root/f'native/draw_{a.draw:02}';_OUT.mkdir(parents=True,exist_ok=True);freeze=_OUT/'inputs.json'
    if freeze.exists():assert json.loads(freeze.read_text())==_INPUTS
    else:save_json(freeze,_INPUTS)
    with concurrent.futures.ProcessPoolExecutor(max_workers=32,mp_context=mp.get_context('fork')) as pool:records=list(pool.map(one,range(512)))
    csv_rows(_OUT/'checks.csv',[dict(identity=r['identity'],row=r['row'],TYPE='PSF',**c) for r in records for c in r['checks']])
    save_json(_OUT/'complete.json',dict(inputs=_INPUTS,objects={f'object_{r["identity"]}.json':sha(_OUT/f'object_{r["identity"]}.json') for r in records},
        outputs={'checks.csv':sha(_OUT/'checks.csv')},response_draw=a.draw,n=512,passed=all(r['passed'] for r in records),
        failures=sum(not r['passed'] for r in records),all_PSF=True,job=os.getenv('SLURM_JOB_ID'),
        scope='Native and analytically prior-withdrawn Gaussian moment baseline. Preparation only; all32response states, entry/count/evidence/shared-distance transfers remain required.'))


if __name__=='__main__':main()
