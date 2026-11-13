#!/usr/bin/env python3
"""Numerical and held-out-calibration guards for the registered correction."""
import concurrent.futures
import json
import multiprocessing as mp
import os
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm
from v2_1_near import ROOT, load, sha, save_json, atomic_npz, csv_rows
import v3_0_evaluate as evaluator
from photdis.stage1 import require_compute
from photdis.v3.likelihood import trapezoid_weights, ContinuousPosterior
from photdis.v3.metrics import observed_parallax, normal_reference_score
from photdis.v3_1.luminosity_log import log_luminosity_transform
from photdis.v3_1.radial_products import bin_integrals
from photdis.v3_1.comparison import paired_mean

_SOURCE=_OUT=_INPUTS=_PREPARED=_EVIDENCE=_PARAMETERS=_VIEWS=None


def predict(values,grid,frequency,parameters,truncate=8.):
    draws = []; w = trapezoid_weights(grid)
    for parameter in parameters:
        transformed = np.array([log_luminosity_transform(values[c],grid,*parameter[c],truncate=truncate) for c in range(3)])
        joint = transformed+np.log(frequency)[:,None]+norm.logpdf(grid,15,5)[None]
        joint -= logsumexp(joint+np.log(w)[None]); draws.append(joint)
    logjoint = logsumexp(np.array(draws),axis=0)-np.log(len(draws)); lp = logsumexp(logjoint,axis=0)
    probability = bin_integrals(np.exp(logjoint),grid)
    q = ContinuousPosterior(grid,np.exp(lp)[None],0.,{}).quantiles()
    return dict(logpdf=lp,mu=q,class_distance=probability,
        draw_probability=np.array([bin_integrals(np.exp(x),grid) for x in draws]))


def scores(row,grid,prediction):
    lp = prediction['logpdf']; finite = np.isfinite(lp)
    indices = np.flatnonzero(finite); assert len(indices)>2 and np.all(np.diff(indices)==1)
    # Exact zeros from zero-extended likelihood interpolation occur only at
    # endpoints. Never floor log density, which would fabricate distant tails.
    g,lp = grid[finite],lp[finite]; result = {}
    if _EVIDENCE['parallax_eligible'][row]:
        result['gaia'] = observed_parallax(g,lp,_EVIDENCE['p_corr'][row],_EVIDENCE['gaia_parallax_error'][row])
    references = {}
    for name,(mu,error,valid,fraction) in _VIEWS.items():
        if not valid[row]: continue
        source = {1:'specdis',2:'spdist'}.get(int(_EVIDENCE['primary_channel'][row])) if name=='primary' else name
        broad = np.hypot(_EVIDENCE[source+'_sigma_mu'][row],2.) if source in ('specdis','spdist') else 2.
        references[name] = dict(mu=float(mu[row]),error=float(error[row]),logscore=normal_reference_score(g,lp,mu[row],error[row],
            broad_fraction=fraction[row],broad_error=broad))
    result['references'] = references
    result['fit_channel_logscore'] = result['gaia']['log_score'] if _EVIDENCE['fit_channel'][row]==0 else references['specdis' if _EVIDENCE['fit_channel'][row]==1 else 'spdist']['logscore']
    result.update(mu=prediction['mu'].tolist(),class_distance=prediction['class_distance'].tolist(),
        qnear=float(prediction['class_distance'][:,:3].sum()))
    return result


def one(row):
    identity = int(_EVIDENCE['identity'][row]); marker = _OUT/f'object_{identity}.json'; pdf = marker.with_suffix('.npz')
    if marker.exists():
        record = json.loads(marker.read_text()); assert record['inputs']==_INPUTS and sha(pdf)==record['sha256']; return record
    old = _SOURCE/marker.name; source_record = json.loads(old.read_text())
    assert sha(old)==_PREPARED['objects'][old.name] and sha(old.with_suffix('.npz'))==source_record['sha256']
    value = load(old.with_suffix('.npz')); assert value['TYPE'][0]=='PSF'
    variants = dict(baseline=np.zeros((1,3,2)),point=_PARAMETERS['point'][None],bootstrap=_PARAMETERS['bootstrap'])
    arrays = {}; results = {}; checks = []
    for name,parameters in variants.items():
        coarse = predict(value['coarse_class_loglike'],value['coarse_grid'],value['class_frequency'],parameters)
        results[name] = scores(row,value['coarse_grid'],coarse)
        arrays.update({name+'_logpdf':coarse['logpdf'],name+'_draw_probability':coarse['draw_probability']})
        if _EVIDENCE['quadrature_audit'][row]:
            fine = predict(value['fine_class_loglike'],value['fine_grid'],value['class_frequency'],parameters)
            fs = scores(row,value['fine_grid'],fine)
            # A proper-score guard also checks tails that carry very little mass.
            longer = predict(value['fine_class_loglike'],value['fine_grid'],value['class_frequency'],parameters,truncate=12.)
            ls = scores(row,value['fine_grid'],longer)
            delta = float(abs(fine['class_distance']-coarse['class_distance']).max())
            qdelta = float(abs(fine['mu']-coarse['mu']).max())
            score_delta = abs(fs['fit_channel_logscore']-results[name]['fit_channel_logscore'])
            tail_delta = abs(ls['fit_channel_logscore']-fs['fit_channel_logscore'])
            checks.append(dict(variant=name,joint_delta=delta,quantile_delta=qdelta,logscore_delta=score_delta,
                truncate8_to12_logscore_delta=tail_delta,passed=bool(delta<.003 and qdelta<.03 and score_delta<.02 and tail_delta<.02)))
            arrays.update({name+'_fine_logpdf':fine['logpdf'],name+'_fine12_logpdf':longer['logpdf']})
    arrays['coarse_grid'] = value['coarse_grid']
    if 'fine_grid' in value: arrays['fine_grid'] = value['fine_grid']
    atomic_npz(pdf,**arrays)
    record = dict(inputs=_INPUTS,sha256=sha(pdf),identity=identity,TYPE='PSF',partition=int(_EVIDENCE['partition'][row]),
        reference_class=int(_EVIDENCE['class_label'][row]),cell=int(_EVIDENCE['cell'][row]),
        design_weight=float(1/_EVIDENCE['inclusion_probability'][row]),results=results,numerical_checks=checks)
    save_json(marker,record); return record


def main():
    global _SOURCE,_OUT,_INPUTS,_PREPARED,_EVIDENCE,_PARAMETERS,_VIEWS
    require_compute(); _SOURCE = ROOT/'data/v3_1/luminosity_calibration'; fit = ROOT/'data/v3_1/luminosity_fit_refined'
    _OUT = ROOT/'data/v3_1/luminosity_guard'; _OUT.mkdir(exist_ok=True)
    paths = [_SOURCE/'complete.json',_SOURCE/'evidence.npz',fit/'complete.json',fit/'parameters.npz',ROOT/'data/v3_0/calibration.npz',
        ROOT/'scripts/v3_1_luminosity_guard.py',ROOT/'src/photdis/v3_1/luminosity_log.py',
        ROOT/'docs/v3_1_luminosity_correction_protocol.md',ROOT/'docs/v3_1_luminosity_numerics.md']
    _INPUTS = {str(p.relative_to(ROOT)):sha(p) for p in paths}; freeze = _OUT/'inputs.json'
    if freeze.exists(): assert json.loads(freeze.read_text())==_INPUTS
    else: save_json(freeze,_INPUTS)
    _PREPARED = json.loads(paths[0].read_text()); f = json.loads(paths[2].read_text())
    assert sha(paths[1])==_PREPARED['outputs'][paths[1].name] and sha(paths[3])==f['outputs'][paths[3].name]
    _PARAMETERS = load(paths[3]); _EVIDENCE = load(paths[1]); assert (_EVIDENCE['TYPE']=='PSF').all()
    _EVIDENCE['primary_channel'] = np.where(_EVIDENCE['specdis_valid'],1,np.where(_EVIDENCE['spdist_valid'],2,3))
    evaluator._CAL = load(paths[4]); _VIEWS = evaluator.evidence_views(_EVIDENCE)
    rows = np.flatnonzero((_EVIDENCE['partition']==1)|_EVIDENCE['quadrature_audit'])
    with concurrent.futures.ProcessPoolExecutor(max_workers=32,mp_context=mp.get_context('fork')) as pool:
        records = []
        for i,r in enumerate(pool.map(one,rows)):
            records.append(r)
            if i%64==0: print('LUMINOSITY GUARD',i,flush=True)
    comparisons = []; gates = []
    for variant in ('point','bootstrap'):
        for c in range(3):
            group = [r for r in records if r['partition']==1 and r['reference_class']==c]
            metrics = {'fit_channel_logscore':(-.02,True),'severe_fraction':(.01,False),
                'coverage68':(-.01,True),'coverage95':(-.01,True),'gaia_tail':(.01,False)}
            for metric,(limit,lower) in metrics.items():
                difference=[]; weights=[]; cells=[]
                for r in group:
                    a,b = r['results'][variant],r['results']['baseline']
                    if metric=='fit_channel_logscore': x,y=a[metric],b[metric]
                    elif metric=='gaia_tail':
                        if 'gaia' not in a: continue
                        x=float(a['gaia']['pit']<.025 or a['gaia']['pit']>.975); y=float(b['gaia']['pit']<.025 or b['gaia']['pit']>.975)
                    else:
                        if 'primary' not in a['references']: continue
                        mu=a['references']['primary']['mu']; qa,qb=a['mu'],b['mu']
                        if metric=='severe_fraction': x,y=float(abs(qa[2]-mu)>1),float(abs(qb[2]-mu)>1)
                        else:
                            lo,hi=(1,3) if metric=='coverage68' else (0,4)
                            x,y=float(qa[lo]<=mu<=qa[hi]),float(qb[lo]<=mu<=qb[hi])
                    difference.append(x-y); weights.append(r['design_weight']); cells.append(r['cell'])
                summary=paired_mean(difference,weights,cells)
                sufficient=summary['n']>=20 and summary['cells']>=2
                passed=sufficient and (summary['mean']>=limit if lower else summary['mean']<=limit)
                row=dict(variant=variant,reference_class=c,metric=metric,**summary,limit=limit,lower_bound=lower,
                    sufficient=sufficient,passed=bool(passed)); comparisons.append(row)
                if variant=='bootstrap': gates.append(row)
    checks=[dict(identity=r['identity'],**check) for r in records for check in r['numerical_checks']]
    csv_rows(_OUT/'comparisons.csv',comparisons); csv_rows(_OUT/'numerical.csv',checks)
    save_json(_OUT/'complete.json',dict(inputs=_INPUTS,outputs={name:sha(_OUT/name) for name in ['comparisons.csv','numerical.csv']},
        objects={f'object_{r["identity"]}.json':sha(_OUT/f'object_{r["identity"]}.json') for r in records},
        job=os.getenv('SLURM_JOB_ID'),objects_evaluated=len(records),numerical_checks=len(checks),
        numerical_passed=all(x['passed'] for x in checks),guard_passed=all(x['passed'] for x in gates),
        adopted=False,final_outcomes_opened=False,
        scope='Held-out calibration sky only. Main correction uses equal shared bootstrap cut predictions; point fit is separately reported. Labels only group scores, never condition predictions. Small sky counts and source ancestry remain limitations; reference-point agreement is not true coverage.'))


if __name__ == '__main__': main()
