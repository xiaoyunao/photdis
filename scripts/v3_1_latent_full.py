#!/usr/bin/env python3
"""Full registered development latent-sign audit, retaining unsupported cases."""
import concurrent.futures
import json
import multiprocessing as mp
import os
import joblib
import numpy as np
from v2_1_near import ROOT,load,sha,save_json,csv_rows
from v3_0_response import observation
from photdis.stage1 import require_compute
from photdis.v3.model import inference_model
from photdis.v3.systematics import calibration_vectors
from photdis.v3.likelihood import normalize_joint
from photdis.v3_1.diagnostics import control_log_density,condition_feh
from photdis.v3_1.latent_flux import negative_flux_support
from photdis.v3_1.comparison import paired_mean

_TABLE=_EV=_MODEL=_CAL=_BASE=_DONE=_OUT=_INPUTS=None


def one(name):
    destination=_OUT/name
    if destination.exists():
        record=json.loads(destination.read_text()); assert record['inputs']==_INPUTS
        return record['rows']
    marker=_BASE/name; assert sha(marker)==_DONE['objects'][name]
    record=json.loads(marker.read_text()); assert sha(marker.with_suffix('.npz'))==record['pdf_sha256']
    pdf=load(marker.with_suffix('.npz')); rows=[]
    for r in record['results']:
        if r['control'] not in ('response12','class_feh'):continue
        j=r['row']; assert int(_TABLE['identity'][j])==r['identity'] and _TABLE['TYPE'][j]=='PSF'
        o=observation(_TABLE,j); grid=pdf[r['control']+'_mu']
        row=dict(identity=r['identity'],TYPE='PSF',control=r['control'],reference_class=r['class_label'],cell=r['cell'],
            inclusion_probability=r['inclusion_probability'],kernel_proxy_max=max(r['kernel_negative_support_proxy_by_band']))
        if np.asarray(o.get('upper_limit',np.zeros(6,bool)))[o['band_mask']].any():
            row.update(supported=False,reason='Censored latent Gaussian requires truncated integration; not inferred from kernel proxy')
        else:
            control=dict(calibration_mag=_CAL[j]); conditioned=_MODEL
            if r['control']=='class_feh':
                control.update(reference_class=r['class_label'],reference_feh=r['reference_feh'],reference_feh_error=_EV['rvs_feh_err'][j])
                conditioned,_=condition_feh(_MODEL,r['reference_feh'],_EV['rvs_feh_err'][j])
            p=normalize_joint(grid,control_log_density(_MODEL,o,grid,**control))
            fidelity=float(np.max(abs(p.quantiles()-r['mu']))); assert fidelity<1e-9
            value=negative_flux_support(conditioned,o,p,calibration_mag=_CAL[j])
            assert 0<=value['any_negative_lower']<=value['any_negative_upper']<=1+1e-12
            row.update(supported=True,reason='',quantile_reproduction_delta=fidelity,
                any_negative_lower=value['any_negative_lower'],any_negative_upper=value['any_negative_upper'])
            row.update({f'posterior_negative_{band}':float(prob) for band,prob in zip(['G','R','I','Z','W1','W2'],value['probability_by_band'])})
        rows.append(row)
    save_json(destination,dict(inputs=_INPUTS,rows=rows,job=os.getenv('SLURM_JOB_ID'))); return rows


def main():
    global _TABLE,_EV,_MODEL,_CAL,_BASE,_DONE,_OUT,_INPUTS
    require_compute(); d=ROOT/'data/v3_0'; _BASE=ROOT/'data/v3_1/diagnostics/diagnostic'; _DONE=json.loads((_BASE/'complete.json').read_text())
    _OUT=_BASE/'latent_full'; _OUT.mkdir(exist_ok=True)
    paths=[_BASE/'complete.json',d/'cohorts/development.npz',d/'evaluation/development.npz',
        d/'uncertainty_refit/draw_12/model.joblib',d/'uncertainty/shared_draws.npz',ROOT/'scripts/v3_1_latent_full.py',
        ROOT/'src/photdis/v3_1/latent_flux.py',ROOT/'src/photdis/v3_1/diagnostics.py',ROOT/'src/photdis/v3_1/comparison.py']
    _INPUTS={str(p.relative_to(ROOT)):sha(p) for p in paths}; freeze=_OUT/'inputs.json'
    if freeze.exists(): assert json.loads(freeze.read_text())==_INPUTS
    else: save_json(freeze,_INPUTS)
    if (_OUT/'complete.json').exists():
        done=json.loads((_OUT/'complete.json').read_text()); assert done['inputs']==_INPUTS
        for name,digest in done['outputs'].items(): assert sha(_OUT/name)==digest
        return
    for path in paths[1:5]: assert sha(path)==_DONE['inputs'][str(path.relative_to(ROOT))]
    _TABLE=load(paths[1]); _EV=load(paths[2]); np.testing.assert_array_equal(_TABLE['identity'],_EV['identity'])
    _MODEL=inference_model(joblib.load(paths[3])); _CAL,_=calibration_vectors(_TABLE,load(paths[4]),12)
    rows=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=32,mp_context=mp.get_context('fork')) as pool:
        for i,rr in enumerate(pool.map(one,sorted(_DONE['objects']))):
            rows.extend(rr)
            if i%64==0: print('FULL LATENT SIGN',i,len(rows),flush=True)
    # Uniform columns also preserve explicit unsupported/censored denominators.
    keys=sorted(set().union(*(r.keys() for r in rows))); csv_rows(_OUT/'posterior_negative_flux.csv',[{k:r.get(k) for k in keys} for r in rows])
    summaries=[]
    for control in ('response12','class_feh'):
        for c in ('all',-1,0,1,2):
            original=[r for r in rows if r['control']==control and (c=='all' or r['reference_class']==c)]
            rr=[r for r in original if r['supported']]
            for metric in ['any_negative_lower','any_negative_upper']+[f'posterior_negative_{band}' for band in ['G','R','I','Z','W1','W2']]:
                summaries.append(dict(control=control,reference_class=c,metric=metric,eligible_views=len(original),unsupported=len(original)-len(rr),
                    **paired_mean([r[metric] for r in rr],[1/r['inclusion_probability'] for r in rr],[r['cell'] for r in rr])))
    csv_rows(_OUT/'summary.csv',summaries)
    save_json(_OUT/'complete.json',dict(inputs=_INPUTS,outputs={name:sha(_OUT/name) for name in ['posterior_negative_flux.csv','summary.csv']},
        objects={p.name:sha(p) for p in _OUT.glob('object_*.json')},job=os.getenv('SLURM_JOB_ID'),views=len(rows),
        unsupported=sum(not r['supported'] for r in rows),max_any_negative_upper=max(r['any_negative_upper'] for r in rows if r['supported']),
        max_quantile_reproduction_delta=max(r['quantile_reproduction_delta'] for r in rows if r['supported']),
        scope='Full registered seen-development cohort, same response12/class+FeH controls. Gaussian conditional under frozen mean-linearized dust; union-event bounds, not independent-band product. Unsupported censoring retained; observed negative flux is legal.'))


if __name__=='__main__':main()
