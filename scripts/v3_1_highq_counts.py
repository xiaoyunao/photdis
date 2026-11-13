#!/usr/bin/env python3
"""Shared counts for one fixed final-candidate high-q subset and fair controls."""
import concurrent.futures
import json
import multiprocessing as mp
import os
import numpy as np
from v2_1_near import ROOT,load,sha,save_json,csv_rows,atomic_npz
from photdis.stage1 import require_compute
from photdis.v3.population import SharedCounts
from photdis.v3_1.count_mixture import mix_counts,realized_count_intervals

_STATES=_SELECT=_IDENTITY=_ROWS=_JOINT=_META=_OUT=_INPUTS=None
VARIANTS=['base','entry_seed','entry_level','native_fine']


def one(index):
    state=_STATES[index];fit=ROOT/state['fit'];fd=json.loads(fit.read_text());assert sha(fit)==state['fit_sha256']
    population_path=ROOT/state['population'];assert sha(population_path)==state['population_sha256'];population=load(population_path)
    np.testing.assert_array_equal(population['identity'],_IDENTITY);np.testing.assert_array_equal(population['rows'],_ROWS)
    assert (population['TYPE']=='PSF').all();source=(ROOT/state['counts']).parent
    assert sha(source/'complete.json')==state['counts_sha256'];cd=json.loads((source/'complete.json').read_text())
    for name in ['numerical_weights.npz','per_star.npz']:assert sha(source/name)==cd['outputs'][name]
    num=load(source/'numerical_weights.npz');expected=load(source/'per_star.npz');draw_ids=population['draw_ids']
    np.testing.assert_array_equal(num['draw_ids'],draw_ids);np.testing.assert_array_equal(expected['identity'],_IDENTITY)
    path=_JOINT/f'correction_{index:02}.npz';assert sha(path)==_META['outputs'][path.name];joint=load(path)
    folder=_OUT/f'correction_{index:02}';folder.mkdir(exist_ok=True);marker=folder/'complete.json'
    if marker.exists():
        result=json.loads(marker.read_text());assert result['inputs']==_INPUTS
        for name,digest in result['outputs'].items():assert sha(folder/name)==digest
        return result
    metadata=load(_JOINT/'metadata.npz');np.testing.assert_array_equal(metadata['identity'][_ROWS],_IDENTITY)
    fine_lookup={int(r):j for j,r in enumerate(metadata['fine_rows'])};local=np.flatnonzero(_SELECT);global_rows=_ROWS[local]
    fine_mask=np.array([int(r) in fine_lookup for r in global_rows]);base=joint['coarse_joint_response'][global_rows];fine=base.copy()
    for k,row in enumerate(global_rows):
        if fine_mask[k]:fine[k]=joint['fine_joint_response'][fine_lookup[int(row)]]
    fields=expected['field'][local];theta=population['theta'];s=len(theta);accs={name:SharedCounts(draw_ids,18,num[name+'_weight'],seed=320331) for name in VARIANTS}
    means={name:np.empty((len(local),3,6)) for name in VARIANTS}
    for fi,field in enumerate(population['fields']):
        uses=np.flatnonzero(fields==field)
        for start in range(0,len(uses),32):
            use=uses[start:start+32];prob=np.einsum('scd,ncdb->sncb',theta[:,fi],base[use],optimize=True);prob/=prob.sum(axis=(2,3),keepdims=True)
            fp=prob.copy();audited=np.flatnonzero(fine_mask[use])
            if len(audited):
                f=np.einsum('scd,ncdb->sncb',theta[:,fi],fine[use[audited]],optimize=True);f/=f.sum(axis=(2,3),keepdims=True);fp[:,audited]=f
            for name,acc in accs.items():means[name][use]=acc.add((fp if name=='native_fine' else prob).reshape(s,len(use),18),draw_ids).reshape(len(use),3,6)
    checks=[]
    for name,acc in accs.items():
        r=acc.result();delta=float(abs(means[name]-expected[name+'_probability'][local]).max()) if len(local) else 0.
        assert delta<1e-10 and abs(r['mean'].sum()-len(local))<1e-7
        atomic_npz(folder/f'{name}_counts.npz',**r)
        atomic_npz(folder/f'{name}_probabilities.npz',identity=_IDENTITY[local],TYPE=population['TYPE'][local],class_distance=means[name])
        checks.append(dict(variant=name,reproduced_probability_delta=delta,marginal_half_delta=r['marginal_half_delta']))
    result=dict(inputs=_INPUTS,correction=index,n=len(local),all_PSF=True,checks=checks,outputs={p.name:sha(p) for p in folder.iterdir() if p.suffix=='.npz'},job=os.getenv('SLURM_JOB_ID'))
    save_json(marker,result);return result


def main():
    global _STATES,_SELECT,_IDENTITY,_ROWS,_JOINT,_META,_OUT,_INPUTS
    require_compute();root=ROOT/'data/v3_1';ensemble=root/'target_ensemble/candidate';_JOINT=root/'target_joint_matrices'
    _OUT=root/'highq_counts';_OUT.mkdir(exist_ok=True)
    paths=[ensemble/'complete.json',ensemble/'states.json',_JOINT/'complete.json',_JOINT/'metadata.npz',
        ROOT/'scripts/v3_1_highq_counts.py',ROOT/'src/photdis/v3_1/count_mixture.py',ROOT/'src/photdis/v3/population.py',ROOT/'docs/v3_1_highq_counts_protocol.md']
    _INPUTS={str(p.relative_to(ROOT)):sha(p) for p in paths};done=json.loads(paths[0].read_text());assert sha(paths[1])==done['outputs']['states.json']
    _STATES=json.loads(paths[1].read_text())['states'];assert len(_STATES)==34
    if not all(r['passed'] for r in _STATES):
        save_json(_OUT/'complete.json',dict(inputs=_INPUTS,available=False,passed=False,failed_states=[r['correction'] for r in _STATES if not r['passed']],references_read=False))
        return
    _META=json.loads(paths[2].read_text());assert _META['passed'] and sha(paths[3])==_META['outputs']['metadata.npz']
    main_marker=ensemble/'bootstrap_evidence/complete.json';assert sha(main_marker)==done['view_markers']['bootstrap_evidence/complete.json']
    main_done=json.loads(main_marker.read_text());main_path=main_marker.parent/'base_probabilities.npz';assert sha(main_path)==main_done['outputs'][main_path.name]
    data=load(main_path);_IDENTITY=data['identity'];_ROWS=data['rows'];_SELECT=data['qnear']>.95;assert (data['TYPE']=='PSF').all()
    _INPUTS[str(main_path.relative_to(ROOT))]=sha(main_path);freeze=_OUT/'inputs.json'
    if freeze.exists():assert json.loads(freeze.read_text())==_INPUTS
    else:save_json(freeze,_INPUTS)
    atomic_npz(_OUT/'selection.npz',identity=_IDENTITY,TYPE=data['TYPE'],rows=_ROWS,qnear=data['qnear'],selected=_SELECT)
    with concurrent.futures.ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('fork')) as pool:records=list(pool.map(one,range(34)))
    summaries=[];checks=[];views=[];leakages=[]
    for view in done['views']:
        label=view['view'];folder=_OUT/label;folder.mkdir(exist_ok=True);vm=ensemble/label/'complete.json'
        assert sha(vm)==done['view_markers'][f'{label}/complete.json'];vd=json.loads(vm.read_text());means={};results={};passed=view['accepted']
        for variant in VARIANTS:
            vp=vm.parent/f'{variant}_probabilities.npz';assert sha(vp)==vd['outputs'][vp.name];v=load(vp);indices=v['state_ids'];weight=v['state_weights']
            results[variant]=mix_counts([load(_OUT/f'correction_{i:02}/{variant}_counts.npz') for i in indices],weight,indices);r=results[variant]
            mean=np.einsum('s,sncd->ncd',weight,np.array([load(_OUT/f'correction_{i:02}/{variant}_probabilities.npz')['class_distance'] for i in indices]));means[variant]=mean
            np.testing.assert_allclose(mean,v['class_distance'][_SELECT],atol=1e-10,rtol=0)
            interval,half=realized_count_intervals(r);atomic_npz(folder/f'{variant}_counts.npz',**r,intervals95=interval,count_CDF_half_delta=np.array(half))
            pd=float(abs(mean-means['base']).max()) if len(mean) else 0.;cd=float(abs(r['mean']-results['base']['mean']).max())
            good=pd<.003 and cd<.5;passed &= good
            checks.append(dict(view=label,variant=variant,maximum_probability_delta=pd,maximum_count_mean_delta=cd,passed=bool(good)))
            for k,(lo,median,hi) in enumerate(interval):
                summaries.append(dict(view=label,variant=variant,class_index=k//6,distance_bin=k%6,n=int(r['n']),mean=float(r['mean'][k]),
                    lower95=int(lo),median=int(median),upper95=int(hi),variance=float(r['covariance'][k,k]),
                    conditional_variance=float(r['conditional_covariance'][k,k]),shared_variance=float(r['shared_covariance'][k,k]),count_CDF_half_delta=half))
            near=np.array([k%6<3 for k in range(18)]);far=~near
            countdraw=r['count_draws'];pair=np.c_[countdraw[:,near].sum(axis=1),countdraw[:,far].sum(axis=1)]
            pairresult={**r,'count_draws':pair};pi,ph=realized_count_intervals(pairresult);n=int(r['n'])
            leakages.append(dict(view=label,variant=variant,n=n,expected_far_count=float(r['mean'][far].sum()),
                model_expected_leakage=None if n==0 else float(r['mean'][far].sum()/n),far_count_lower95=int(pi[1,0]),far_count_upper95=int(pi[1,2]),
                far_count_variance=float(r['covariance'][np.ix_(far,far)].sum()),count_CDF_half_delta=ph,independently_measured=False))
        views.append(dict(view=label,n=int(_SELECT.sum()),accepted=bool(passed),inherited_ensemble_accepted=view['accepted']))
        save_json(folder/'complete.json',dict(inputs=_INPUTS,view=views[-1],outputs={p.name:sha(p) for p in folder.iterdir() if p.suffix=='.npz'}))
    csv_rows(_OUT/'counts.csv',summaries);csv_rows(_OUT/'checks.csv',checks);csv_rows(_OUT/'model_leakage.csv',leakages)
    save_json(_OUT/'complete.json',dict(inputs=_INPUTS,available=True,passed=all(v['accepted'] for v in views),views=views,n=int(_SELECT.sum()),parent_n=len(_SELECT),all_PSF=True,
        outputs={n:sha(_OUT/n) for n in ['selection.npz','counts.csv','checks.csv','model_leakage.csv']},
        states={f'correction_{i:02}/complete.json':sha(_OUT/f'correction_{i:02}/complete.json') for i in range(34)},
        view_markers={f'{v["view"]}/complete.json':sha(_OUT/v['view']/'complete.json') for v in views},references_read=False,job=os.getenv('SLURM_JOB_ID'),
        scope='Same final-candidate qnear>.95 identities for all controls and numerical variants. Full shared joint count covariance. Leakage is model-expected, not independent contamination. No fullcatalog expansion.'))


if __name__=='__main__':main()
