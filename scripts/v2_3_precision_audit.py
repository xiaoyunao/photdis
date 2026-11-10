#!/usr/bin/env python3
"""Measure histogram compression error without changing fitted models."""
import numpy as np
from scipy.special import ndtr
from v2_3_prepare import ROOT,DEST
from v2_1_near import load,save_json,sha
from photdis.stage1 import require_compute


def main():
    require_compute();result={};steps=[.1,.05,.02,.01,.005,.002]
    for role,source in [('development',ROOT/'data/v2_1/development.npz'),('reserved',DEST/'reserved_inputs.npz'),('application',ROOT/'data/v2_2/application_inputs.npz')]:
        t=load(source);maxerr=np.zeros(len(steps));sigmin=np.inf;observed=0
        for path in sorted((DEST/'chunks').glob(role+'_candidate_*.npz')):
            p=load(path);w,m,s=(p[k] for k in ['mixture_weight','mixture_mean','mixture_sigma']);target=t['g0'][p['rows']]-15
            exact=np.sum(w*ndtr((m-target[:,None])/s),axis=1)
            assert np.max(abs(exact-p['qnear']))<1e-12
            for j,h in enumerate(steps):
                left=np.floor((target+30)/h)*h-30;f=(target-left)/h
                approximation=np.sum(w*((1-f[:,None])*ndtr((m-left[:,None])/s)+f[:,None]*ndtr((m-left[:,None]-h)/s)),axis=1)
                maxerr[j]=max(maxerr[j],np.max(abs(exact-approximation)))
            sigmin=min(sigmin,float(s.min()));observed+=len(w)
        result[role]=dict(n=observed,min_component_sigma=sigmin,step_max_q_error={str(h):float(e) for h,e in zip(steps,maxerr)})
    save_json(DEST/'precision_audit.json',dict(results=result,model_sha256=sha(DEST/'candidate.joblib')))
    print(result,flush=True)


if __name__=='__main__':main()
