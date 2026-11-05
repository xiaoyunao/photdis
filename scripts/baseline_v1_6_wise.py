#!/usr/bin/env python3
"""Evaluate the frozen optical / signed W1 / signed W1+W2 template contrast."""
import concurrent.futures
import json
import multiprocessing as mp
import os
from pathlib import Path

import joblib
import numpy as np

from baseline_v1_4_experiments import OUT, ROOT, KEYS, groups, load, subset, truth, save_json, sha, atomic_npz
from baseline_v1_5_r0 import BASELINE, REFINED, SPECTRAL, metal_layers
from baseline_v1_6_wise_raw import DEST, states
from photdis.baseline_v0 import distance_quantiles
from photdis.baseline_v1 import quantiles_interpolated
from photdis.diagnostics_v1_4 import paired_stats, residual_stats
from photdis.gaia_numerics_v1_4 import refined_gaia
from photdis.stage1 import require_compute
from photdis.weak_wise import JointWISE

ARMS = ('optical','w1','w1w2')
_T = _RAW = _MODEL = None


def bands(raw,key):
    return np.column_stack([raw[f'{key}_{band}'] for band in ('W1','W2')])


def chunk(bounds):
    start,stop = bounds
    path = DEST/'chunks'/f'{start:06}.npz'
    marker = path.with_suffix('.json')
    if marker.exists():
        assert sha(path)==json.loads(marker.read_text())['sha256']
        return path
    q = slice(start,stop)
    t = subset(_T,q)
    raw = subset(_RAW,q)
    state = raw['state']
    valid = state>=3
    flux,ivar,trans = (bands(raw,k) for k in ('FLUX','FLUX_IVAR','MW_TRANSMISSION'))
    mags = np.column_stack([t[k] for k in ('g0','r0','z0')])
    errors = np.column_stack([t[k] for k in ('sigma_g','sigma_r','sigma_z')])
    pdf,ess = _MODEL.predict(mags,errors,t['identity'],flux,ivar,trans,valid)
    assert np.isfinite(pdf).all() and np.allclose(pdf.sum(axis=2),1)
    assert np.array_equal(pdf[~valid[:,0],0],pdf[~valid[:,0],1])
    assert np.array_equal(pdf[~valid[:,1],1],pdf[~valid[:,1],2])
    ga = t['parallax_eligible']
    mu,log_density,pit = [],[],[]
    for i in range(3):
        mu.append(distance_quantiles(t['g0'],quantiles_interpolated(pdf[:,i],_MODEL.edges))[0])
        if ga.any():
            result = refined_gaia(pdf[ga,i],_MODEL.edges,t['g0'][ga],t['p_corr'][ga],t['gaia_parallax_error'][ga])
            log_density.append(result['full_log_density'])
            pit.append(result['full_pit'])
        else:
            log_density.append(np.empty(0))
            pit.append(np.empty(0))
    atomic_npz(path,identity=t['identity'],mu=np.stack(mu,axis=1),ess=ess,pdf=pdf.astype('f4'),
               gaia_identity=t['identity'][ga],log_density=np.column_stack(log_density),pit=np.column_stack(pit))
    save_json(marker,dict(sha256=sha(path)))
    return path


def measurements(state):
    w1,w2 = state.T
    eligible = state>=3
    return dict(w1_weak=w1==3,w2_weak=w2==3,any_weak=(state==3).any(axis=1),
                w2_weak_given_w1=(w2==3)&eligible[:,0],
                w2_weak_without_w1=(w2==3)&~eligible[:,0],
                w2_without_w1=eligible[:,1]&~eligible[:,0],
                w1_high=w1==4,both_high=(state==4).all(axis=1),
                w2_high_given_w1=(w2==4)&eligible[:,0],
                w1_valid=eligible[:,0],w2_valid=eligible[:,1],
                both_valid=eligible.all(axis=1),either_valid=eligible.any(axis=1),
                either_high=(state==4).any(axis=1),neither_valid=~eligible.any(axis=1),
                w1_failed=w1==2,w2_failed=w2==2,w1_missing=w1<2,w2_missing=w2<2)


def coverage(parent):
    state = states(parent)
    assert np.array_equal(state,np.column_stack([parent[k+'_measurement_state'] for k in ('w1','w2')]))
    masks = measurements(state)
    for b,band in enumerate(('W1','W2')):
        masks[band.lower()+'_negative_valid'] = (state[:,b]>=3)&(parent[f'FLUX_{band}']<0)
    w = parent['sample_weight']
    return {name:dict(n=int(mask.sum()),fraction=np.average(mask,weights=w)) for name,mask in masks.items()}


def comparisons(t, raw, pred, frozen, old_gaia):
    groups_out = groups(t)
    metal = metal_layers(t)
    groups_out.update(metal)
    status = measurements(raw['state'])
    groups_out.update(status)
    magnitude = (t['r']>=18)&(t['r']<20)
    negative = (bands(raw,'FLUX')<0)&(raw['state']>=3)
    groups_out['negative_valid'] = negative.any(axis=1)
    groups_out['w1_negative_valid'] = negative[:,0]
    groups_out['w2_negative_valid'] = negative[:,1]
    primary = ['w1_weak','w2_weak_given_w1','w2_weak_without_w1','any_weak','both_high','neither_valid']
    for name in primary:
        mask = status[name]
        groups_out['r18_20:'+name] = magnitude&mask
        for k in ('class0','class1','class2','class-1','dwarf_corrected_feh_lt_m1p5'):
            groups_out[k+':'+name] = groups_out[k]&mask
            groups_out['r18_20:'+k+':'+name] = magnitude&groups_out[k]&mask
    ga = t['parallax_eligible']
    assert np.array_equal(pred['gaia_identity'],old_gaia['identity'])
    mu = np.concatenate([frozen['mu'][:,None,:],pred['mu']],axis=1)
    log = np.column_stack([old_gaia['full_log_density'],pred['log_density']])
    pit = np.column_stack([old_gaia['full_pit'],pred['pit']])
    refs = {kind:truth(t,kind) for kind in ('chosen','Gaia','specdis','spdist')}
    report = {}
    for split,label in [(1,'cal'),(2,'development_test')]:
        result = {}
        for name,mask in groups_out.items():
            take = mask&(t['split']==split)
            sel = take[ga]
            layer = dict(n=int(take.sum()),full_Gaia=dict(n=int(sel.sum()),negative=int((take&ga&(t['p_corr']<0)).sum())))
            for j,arm in enumerate(('expanded_A',*ARMS)):
                pp = pit[sel,j]
                layer['full_Gaia'][arm] = dict(mean_log_density=np.mean(log[sel,j]) if len(pp) else None,
                    pit_extreme=np.mean((pp<.05)|(pp>.95)) if len(pp) else None,
                    pit_histogram=np.histogram(pp,np.linspace(0,1,11))[0])
            for kind,ref in refs.items():
                a = ref[take]
                scores = {arm:residual_stats(mu[take,j],a) for j,arm in enumerate(('expanded_A',*ARMS))}
                for i,j,key in [(1,2,'w1_minus_optical'),(2,3,'w1w2_minus_w1'),(1,3,'w1w2_minus_optical')]:
                    scores[key] = paired_stats(mu[take,i,2]-a,mu[take,j,2]-a,t['cell'][take])
                layer[kind] = scores
            layer['median_effective_particles'] = np.median(pred['ess'][take],axis=0) if take.any() else None
            result[name] = layer
        report[label] = result
    return report


def main():
    global _T,_RAW,_MODEL
    require_compute()
    (DEST/'chunks').mkdir(parents=True,exist_ok=True)
    inputs = [DEST/'raw.npz', OUT/'references.npz', OUT/'diagnostics/parent.npz',
              OUT/'experiments/expanded_A_distance.joblib', BASELINE/'evaluation.npz', REFINED/'evaluation.npz']
    code = [Path(__file__),ROOT/'src/photdis/weak_wise.py',ROOT/'src/photdis/baseline_v1.py',
            ROOT/'src/photdis/gaia_numerics_v1_4.py',ROOT/'docs/baseline_v1_6_wise_protocol.md']
    hashes = {str(p.relative_to(ROOT)):sha(p) for p in inputs+code}
    marker = json.loads((DEST/'raw.json').read_text())
    assert sha(inputs[0])==marker['sha256'] and sha(inputs[1])==marker['reference_sha256']
    for folder in (BASELINE,REFINED):
        assert sha(folder/'evaluation.npz')==json.loads((folder/'scores.json').read_text())['evaluation_sha256']
    freeze = DEST/'fit_freeze.json'
    if freeze.exists():
        assert json.loads(freeze.read_text())==hashes
    else:
        save_json(freeze,hashes)
    result_path = DEST/'result.json'
    if result_path.exists():
        assert sha(DEST/'evaluation.npz')==json.loads(result_path.read_text())['evaluation_sha256']
        return
    raw = load(inputs[0])
    frozen,old_gaia = (load(p) for p in inputs[-2:])
    with np.load(inputs[1]) as z:
        keys = list(dict.fromkeys(KEYS+SPECTRAL))
        t = {k:z[k][raw['rows']] for k in keys}
    assert np.array_equal(raw['identity'],t['identity'])
    train = (t['split']==0)&(raw['state']==4).all(axis=1)
    use = np.isin(t['split'],[1,2])
    _T,_RAW = subset(t,use),subset(raw,use)
    assert np.array_equal(_T['identity'],frozen['identity'])
    assert np.array_equal(_T['identity'][_T['parallax_eligible']],old_gaia['identity'])
    assert not np.intersect1d(t['identity'][train],_T['identity']).size
    edges = joblib.load(inputs[3]).edges
    _MODEL = JointWISE(np.column_stack([t[k][train] for k in ('g0','r0','z0')]),
                       t['Mg'][train],np.column_stack([t[k][train] for k in ('w10','w20')]),edges)
    print('Templates',train.sum(),'evaluation',use.sum(),flush=True)
    jobs = [(j,min(j+192,use.sum())) for j in range(0,use.sum(),192)]
    files = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=24,mp_context=mp.get_context('fork')) as pool:
        for j,p in enumerate(pool.map(chunk,jobs),1):
            files.append(p)
            if j%40==0 or j==len(jobs):
                print(f'weak WISE chunks {j}/{len(jobs)}',flush=True)
    # Full PDFs remain in immutable chunks; compact summary arrays are joined once.
    chunks = []
    for p in files:
        with np.load(p) as z:
            chunks.append({k:z[k] for k in z.files if k!='pdf'})
    pred = {k:np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    assert np.array_equal(pred['identity'],_T['identity'])
    assert np.isfinite(pred['mu']).all() and (np.diff(pred['mu'],axis=2)>=0).all()
    assert np.isfinite(pred['log_density']).all() and ((pred['pit']>=0)&(pred['pit']<=1)).all()
    atomic_npz(DEST/'evaluation.npz',**pred,state=_RAW['state'],flux=bands(_RAW,'FLUX'),
               train_identity=t['identity'][train],edges=edges)
    score = comparisons(_T,_RAW,pred,frozen,old_gaia)
    parent = coverage(load(inputs[2]))
    flux = bands(_RAW,'FLUX')
    signed = {name:{k:[int(((flux[:,b]<0 if k=='negative' else flux[:,b]==0)&
              (_RAW['state'][:,b]>=3)&(_T['split']==split)).sum()) for b in range(2)] for k in ('negative','zero')}
              for split,name in [(1,'cal'),(2,'development_test')]}
    assert all(sha(ROOT/p)==h for p,h in hashes.items())
    save_json(result_path,dict(job_id=os.getenv('SLURM_JOB_ID'),n_template=int(train.sum()),n_evaluation=int(use.sum()),
              templates_by_class={str(k):int((train&(t['class_label']==k)).sum()) for k in (-1,0,1,2)},
              parent=parent,signed_valid_flux=signed,inputs=hashes,raw_sweeps=len(marker['sweep_hashes']),
              raw_zero_fallback=marker['zero_fallback'],evaluation_sha256=sha(DEST/'evaluation.npz'),scores=score))
    print('COMPLETE weak WISE',flush=True)


if __name__=='__main__':
    main()
