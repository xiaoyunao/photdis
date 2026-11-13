#!/usr/bin/env python3
"""Saved-grid mode responsibilities for every frozen main candidate identity."""
import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing as mp
import os
import time
import joblib
import numpy as np
from scipy.special import logsumexp
from v2_1_near import ROOT,load,sha,save_json,atomic_npz,csv_rows
from v3_0_response import observation
from photdis.stage1 import require_compute
from photdis.v3.model import inference_model
from photdis.v3.inference import kernel_loglike
from photdis.v3.systematics import calibration_vectors
from photdis.v3_1.gaussian_joint import interval_linear_weights
from photdis.v3_1.candidate_modes import population_pdf,basin_partition,basin_joint_mass,describe_basins

_TABLE=_MODEL=_CAL=_THETA=_PARAMETERS=_BASE=_CONDITIONAL=_BINARY=_PARENT=_OUT=_INPUTS=_ACCEPTED=None


def attribute(saved,boundaries):
    return basin_joint_mass(saved['native_binary_kernel_loglike'],saved['grid'],_CONDITIONAL,_BINARY,
        _PARAMETERS,_BASE['state_log_weights'],saved,boundaries)


def one(index):
    start=time.monotonic();identity=int(_BASE['identity'][index]);row=int(_BASE['rows'][index]);field=int(_BASE['field'][index])
    folder=_OUT/f'object_{identity}';folder.mkdir(exist_ok=True);marker=folder/'complete.json'
    if marker.exists():
        result=json.loads(marker.read_text());assert result['inputs']==_INPUTS
        for name,digest in result['outputs'].items():assert sha(folder/name)==digest
        return result
    theta=[d[field] for d in _THETA];previous=None;previous_record=None;attempts=[]
    for step in (.02,.01,.005,.0025):
        path=folder/f'step{step:g}.npz';record=path.with_suffix('.json')
        if record.exists():
            result=json.loads(record.read_text());assert result['inputs']==_INPUTS and sha(path)==result['sha256'];saved=load(path)
        else:
            grid=np.linspace(-20,50,round(70/step)+1)
            native,_=kernel_loglike(_MODEL,observation(_TABLE,row),grid,calibration_mag=_CAL[row])
            with np.errstate(divide='ignore'):
                class_native=np.array([logsumexp(native+np.log(_BINARY)[:,None,None]+np.log(_CONDITIONAL[:,c])[None,:,None],axis=(0,1)) for c in range(3)])
            pdf=population_pdf(class_native,grid,_PARAMETERS,theta,_BASE['state_log_weights'])
            saved=dict(grid=grid,native_binary_kernel_loglike=native,native_class_loglike=class_native,**pdf)
            peaks,boundaries=basin_partition(pdf['logpdf'],grid);mass=attribute(saved,boundaries)
            modes,pairs=describe_basins(peaks,boundaries,grid,mass,_PARENT)
            expected=(np.exp(pdf['class_logpdf'])@interval_linear_weights(grid,boundaries).T).T
            conservation=float(abs(mass.sum(axis=(1,2))-expected).max());total=float(mass.sum())
            original=float(abs(pdf['class_distance']-_BASE['class_distance'][index]).max())
            pd=qd=md=None;stable=False;coarse_mass=np.empty((0,2,len(_PARENT),3))
            if previous is not None:
                pd=float(abs(pdf['class_distance']-previous['class_distance']).max());qd=float(abs(pdf['mu']-previous['mu']).max())
                coarse_mass=attribute(previous,boundaries);md=float(abs(mass-coarse_mass).max())
                stable=sum(not r['weak_basin'] for r in modes)==previous_record['significant_modes']
            internal=abs(total-1)<1e-8 and conservation<1e-8 and abs(float(pdf['log_normalization']))<1e-8 and float(pdf['joint_delta'])<1e-8
            passed=internal and original<.003 and pd is not None and pd<.003 and qd<.03 and md<.003 and stable
            saved.update(identity=np.array([identity]),TYPE=np.array(['PSF']),state_ids=_BASE['state_ids'],state_log_weights=_BASE['state_log_weights'],
                correction_parameters=_PARAMETERS,conditional_kernel_weights=_CONDITIONAL,binary_prior=_BINARY,parent_component=_PARENT,
                mode_peaks=peaks,mode_boundaries=boundaries,mode_joint_mass=mass,previous_grid_mass_on_same_boundaries=coarse_mass)
            atomic_npz(path,**saved)
            result=dict(inputs=_INPUTS,sha256=sha(path),step=step,passed=bool(passed),internal_passed=bool(internal),
                total_mass=total,component_conservation_delta=conservation,original_probability_delta=original,
                probability_delta=pd,quantile_delta=qd,same_basin_joint_mass_delta=md,significant_mode_count_stable=bool(stable),
                significant_modes=sum(not r['weak_basin'] for r in modes),modes=modes,significant_mode_pairs=pairs,
                marginal_half_delta=float(pdf['marginal_half_delta']),joint_delta=float(pdf['joint_delta']),
                inherited_ensemble_accepted=_ACCEPTED,references_read=False,job=os.getenv('SLURM_JOB_ID'))
            save_json(record,result)
        attempts.append(dict(step=step,marker=record.name,marker_sha256=sha(record),passed=result['passed']))
        if result['passed']:break
        previous=saved;previous_record=result
    result=dict(inputs=_INPUTS,identity=identity,row=row,field=field,TYPE='PSF',fixed_main_highq=bool(_BASE['qnear'][index]>.95),
        attempts=attempts,selected_step=step,numerical_passed=result['passed'],significant_modes=result['significant_modes'],
        inherited_ensemble_accepted=_ACCEPTED,references_read=False,elapsed_seconds=time.monotonic()-start,
        outputs={p.name:sha(p) for p in folder.iterdir() if p.name!='complete.json'},job=os.getenv('SLURM_JOB_ID'))
    save_json(marker,result);print('CANDIDATE MODES',identity,step,result['numerical_passed'],round(result['elapsed_seconds'],2),flush=True);return result


def summarize(records,name):
    rows=[];modes=[];pairs=[]
    for r in records:
        folder=_OUT/f'object_{r["identity"]}';selected=r['attempts'][-1];path=folder/selected['marker']
        assert sha(path)==selected['marker_sha256'];value=json.loads(path.read_text())
        rows.append({**{k:r[k] for k in ['identity','row','field','TYPE','fixed_main_highq','selected_step','numerical_passed','significant_modes','inherited_ensemble_accepted']},
            **{k:value[k] for k in ['total_mass','component_conservation_delta','original_probability_delta','probability_delta','quantile_delta','same_basin_joint_mass_delta','marginal_half_delta']},
            'binary_probability':sum(m['mass']*m['binary_probability'] for m in value['modes']),
            **{key:any(pair[key] for pair in value['significant_mode_pairs']) for key in ['different_class','same_class_different_binary','same_class_different_parent']}})
        modes.extend(dict(identity=r['identity'],fixed_main_highq=r['fixed_main_highq'],numerical_passed=r['numerical_passed'],**m) for m in value['modes'])
        pairs.extend(dict(identity=r['identity'],fixed_main_highq=r['fixed_main_highq'],numerical_passed=r['numerical_passed'],**m) for m in value['significant_mode_pairs'])
    csv_rows(_OUT/f'{name}_summary.csv',rows);save_json(_OUT/f'{name}_modes.json',modes);save_json(_OUT/f'{name}_pairs.json',pairs)
    overview=[];histograms=[]
    for group,selected in [('all',rows),('fixed_main_highq',[r for r in rows if r['fixed_main_highq']])]:
        if not selected:continue
        overview.append(dict(group=group,n=len(selected),numerical_passed=sum(r['numerical_passed'] for r in selected),
            multimodal=sum(r['significant_modes']>=2 for r in selected),
            mean_binary_probability=float(np.mean([r['binary_probability'] for r in selected])),
            original_binary_prior=float(_BINARY[1]),inherited_ensemble_accepted=_ACCEPTED,
            **{key:sum(r[key] for r in selected) for key in ['different_class','same_class_different_binary','same_class_different_parent']}))
        for k in range(6):
            histograms.append(dict(group=group,metric='significant_modes',lower=k,upper=k+1 if k<5 else None,
                n=sum((r['significant_modes']==k if k<5 else r['significant_modes']>=5) for r in selected)))
        counts,edges=np.histogram([r['binary_probability'] for r in selected],bins=np.linspace(0,1+1e-8,11))
        for k,count in enumerate(counts):histograms.append(dict(group=group,metric='binary_probability',lower=float(edges[k]),upper=float(edges[k+1]),n=int(count)))
    csv_rows(_OUT/f'{name}_overview.csv',overview);csv_rows(_OUT/f'{name}_histograms.csv',histograms)
    save_json(_OUT/('complete.json' if name=='full' else name+'_complete.json'),dict(inputs=_INPUTS,n=len(records),all_PSF=True,
        fixed_main_highq=sum(r['fixed_main_highq'] for r in records),numerical_passed=all(r['numerical_passed'] for r in records),
        failed=sum(not r['numerical_passed'] for r in records),inherited_ensemble_accepted=_ACCEPTED,references_read=False,
        objects={f'object_{r["identity"]}/complete.json':sha(_OUT/f'object_{r["identity"]}/complete.json') for r in records},
        outputs={f'{name}_{suffix}':sha(_OUT/f'{name}_{suffix}') for suffix in ['summary.csv','modes.json','pairs.json','overview.csv','histograms.csv']},
        job=os.getenv('SLURM_JOB_ID'),scope='Exact main candidate mode responsibilities. All weak basins and numerical failures retained; no new fit or independent physical class/binary confirmation.'))


def main():
    global _TABLE,_MODEL,_CAL,_THETA,_PARAMETERS,_BASE,_CONDITIONAL,_BINARY,_PARENT,_OUT,_INPUTS,_ACCEPTED
    require_compute();p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');p.add_argument('--report',action='store_true')
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--workers',type=int,default=16);a=p.parse_args()
    assert 1<=a.workers<=32 and 0<=a.shard<a.shards and not(a.pilot and a.report)
    root=ROOT/'data/v3_1';old=ROOT/'data/v3_0';ensemble=root/'target_ensemble/candidate';view=ensemble/'bootstrap_evidence'
    paths=[ensemble/'complete.json',ensemble/'states.json',view/'complete.json',view/'base_probabilities.npz',old/'cohorts/application.npz',
        old/'uncertainty_refit/draw_12/model.joblib',old/'uncertainty/shared_draws.npz',ROOT/'scripts/v3_1_candidate_modes.py',
        ROOT/'docs/v3_1_candidate_modes_protocol.md',ROOT/'scripts/v3_0_response.py']
    paths += [ROOT/'src/photdis/v3_1'/n for n in ['candidate_modes.py','luminosity_adjoint.py','luminosity_log.py','target_prediction.py','gaussian_joint.py','radial_products.py']]
    paths += [ROOT/'src/photdis/v3'/n for n in ['model.py','inference.py','likelihood.py','systematics.py']]
    native_pilot=root/'target_native_pilot/complete.json';system=old/'shared_predictions/application_refit_0/draw_12/shared_system.npz'
    paths += [native_pilot,system]
    _INPUTS={str(path.relative_to(ROOT)):sha(path) for path in paths}
    source=json.loads(native_pilot.read_text())
    for path in [paths[4],paths[5],system]:assert sha(path)==source['inputs'][str(path.relative_to(ROOT))]
    done=json.loads(paths[0].read_text());assert sha(paths[1])==done['outputs']['states.json'] and sha(paths[2])==done['view_markers']['bootstrap_evidence/complete.json']
    states=json.loads(paths[1].read_text())['states'];vd=json.loads(paths[2].read_text());assert sha(paths[3])==vd['outputs']['base_probabilities.npz']
    assert len(states)==34 and all(s['passed'] for s in states);_ACCEPTED=vd['view']['accepted'];_BASE=load(paths[3]);_TABLE=load(paths[4])
    assert len(_BASE['identity'])==3540 and (_BASE['TYPE']=='PSF').all() and (_TABLE['TYPE'][_BASE['rows']]=='PSF').all()
    np.testing.assert_array_equal(_BASE['identity'],_TABLE['identity'][_BASE['rows']]);np.testing.assert_array_equal(_BASE['field'],_TABLE['field'][_BASE['rows']])
    np.testing.assert_array_equal(_BASE['state_ids'],np.arange(2,34));assert (_BASE['qnear']>.95).sum()==3003
    _MODEL=inference_model(joblib.load(paths[5]));_CAL=calibration_vectors(_TABLE,load(paths[6]),12)[0]
    recorded=load(system);np.testing.assert_array_equal(recorded['identity'],_TABLE['identity']);np.testing.assert_array_equal(recorded['calibration_mag'],_CAL)
    _CONDITIONAL=_MODEL.weights[:,None]*_MODEL.classes;_CONDITIONAL/=_CONDITIONAL.sum(axis=0)[None]
    _BINARY=np.array([1-_MODEL.binary_reference_fraction,_MODEL.binary_reference_fraction]);_PARENT=getattr(_MODEL,'parent_component',np.arange(len(_MODEL.weights)))
    _THETA=[]
    for index,state in enumerate(states):
        assert state['correction']==index
        pop=ROOT/state['population'];assert sha(pop)==state['population_sha256'];_INPUTS[str(pop.relative_to(ROOT))]=sha(pop)
        fit=ROOT/state['fit'];assert sha(fit)==state['fit_sha256'];fd=json.loads(fit.read_text())
        assert fd['inputs'][str(paths[5].relative_to(ROOT))]==sha(paths[5]);_INPUTS[str(fit.relative_to(ROOT))]=sha(fit)
        if index not in _BASE['state_ids']:continue
        d=load(pop);np.testing.assert_array_equal(d['identity'],_BASE['identity']);np.testing.assert_array_equal(d['rows'],_BASE['rows'])
        assert (d['TYPE']=='PSF').all();_THETA.append({int(field):d['theta'][:,j] for j,field in enumerate(d['fields'])})
    _PARAMETERS=np.array([states[i]['parameters'] for i in _BASE['state_ids']]);_OUT=root/'candidate_modes';_OUT.mkdir(exist_ok=True)
    freeze=_OUT/'inputs.json'
    if freeze.exists():assert json.loads(freeze.read_text())==_INPUTS
    else:save_json(freeze,_INPUTS)
    indices=list(range(len(_BASE['identity'])))
    if a.report:
        records=[]
        for identity in _BASE['identity']:
            folder=_OUT/f'object_{identity}';r=json.loads((folder/'complete.json').read_text());assert r['inputs']==_INPUTS and r['identity']==identity
            for name,digest in r['outputs'].items():assert sha(folder/name)==digest
            records.append(r)
        summarize(records,'full');return
    if a.pilot:indices=sorted(indices,key=lambda i:hashlib.sha256(f'321001:{int(_BASE["identity"][i])}'.encode()).digest())[:8]
    else:indices=indices[a.shard::a.shards]
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('fork')) as pool:records=list(pool.map(one,indices))
    summarize(records,'pilot' if a.pilot else f'shard_{a.shard:02}')


if __name__=='__main__':main()
