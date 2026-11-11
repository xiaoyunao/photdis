#!/usr/bin/env python3
"""Bounded development sensitivity of selection to per-object dust covariance."""
import copy
import concurrent.futures
import multiprocessing as mp
import os
import joblib
import numpy as np
from v2_1_near import ROOT,load,sha,save_json,atomic_npz
from photdis.stage1 import require_compute
from photdis.v3.model import inference_model
from photdis.v3.data import stable_rank
from photdis.v3.selection import gaussian_component_entry,effective_errors
from photdis.v3.inference import distance_basis
from photdis.v3.likelihood import trapezoid_weights
D=ROOT/'data/v3_0'
_MODEL=None


def compare(item):
    error,coefficients,sigma=item;mu=np.arange(-15.,45.0001,.05)
    dust=copy.copy(_MODEL);dust.covariances=_MODEL.covariances.copy()
    perturb=.4*np.log(10)*sigma*_MODEL.means[:,:6]*coefficients[None]
    dust.covariances[:,:6,:6]+=perturb[:,:,None]*perturb[:,None,:]
    values=[]
    for model in (_MODEL,dust):
        response=gaussian_component_entry(model,error,mu,draws=1024)
        curve=(1-model.binary_reference_fraction)*response[0]+model.binary_reference_fraction*response[1]
        values.append((curve*trapezoid_weights(mu))@distance_basis(mu).T)
    return values


def main():
    global _MODEL
    require_compute();model_path=D/'sed_fit.joblib';_MODEL=inference_model(joblib.load(model_path))
    t=load(D/'cohorts/application.npz');error=effective_errors(t)
    ix=np.unique(np.r_[np.argsort(stable_rank(t['identity'],309101))[:48],error.argmax(axis=0),error.argmin(axis=0),np.argmax(t['EBV'])])
    assert (t['TYPE'][ix]=='PSF').all()
    coefficients=np.zeros((len(ix),6));positive=t['EBV'][ix]>0
    coefficients[positive]=-2.5*np.log10(t['transmission'][ix][positive])/t['EBV'][ix][positive,None]
    coefficients=np.where(np.isfinite(coefficients),coefficients,0)
    # Actual selected objects and an envelope stress at E(B-V)=.03.
    args=[(error[i],c,.1*float(t['EBV'][i])) for i,c in zip(ix,coefficients)]
    args += [(error[i],c,.003) for i,c in zip(ix,coefficients)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=32,mp_context=mp.get_context('fork')) as pool:
        result=np.array(list(pool.map(compare,args)))
    delta=abs(result[:,1]-result[:,0]);out=D/'dust_selection_audit';out.mkdir(exist_ok=True)
    atomic_npz(out/'comparison.npz',identity=np.tile(t['identity'][ix],2),TYPE=np.full(2*len(ix),'PSF'),
        nominal=result[:,0],dust=result[:,1],sigma_ebv=np.array([a[2] for a in args]),effective_errors=np.array([a[0] for a in args]))
    save_json(out/'complete.json',dict(n=len(args),actual_maximum=float(delta[:len(ix)].max()),
        envelope_maximum=float(delta[len(ix):].max()),maximum_relative_where_alpha_above_point01=float((delta/np.maximum(result[:,0],.01)).max()),
        passed=bool(delta.max()<.005),scope='Finite photometry-only development sensitivity, same Sobol quadrature; not a uniform proof over all future inputs.',
        sha256=sha(out/'comparison.npz'),model_sha256=sha(model_path),source_sha256=sha(D/'cohorts/application.npz'),job=os.getenv('SLURM_JOB_ID')))
    print('DUST SELECTION AUDIT',float(delta.max()),flush=True)


if __name__=='__main__':main()
