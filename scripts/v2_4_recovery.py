#!/usr/bin/env python3
"""Independent empirical-SED populations; no draws from the model under test."""
import concurrent.futures
import multiprocessing as mp
import json
import os
import joblib
import numpy as np
from scipy.spatial import cKDTree
from v2_1_near import ROOT,load,subset,sha,save_json,csv_rows,atomic_npz
from v2_4_audit import OLD,DEST,TABLES
from v2_4_population import joint_model,DISTANCE_EDGES
from baseline_v1_4_experiments import arrays
from photdis.stage1 import require_compute
from photdis.science_v2_4 import distance_bin_mass
from photdis.baseline_v1 import integrate
from photdis.audit import rvs_metallicity
_J=_D=None
POPULATIONS={'ordinary_dominated':(.8,.05,.15,.1),'giant_rich':(.5,.35,.15,.1),
    'metal_poor_rich':(.8,.05,.15,.6),'balanced':(1/3,1/3,1/3,.3)}


def design():
    t=load(ROOT/'data/v2_1/development.npz');train=load(OLD/'training.npz')['identity'];app=load(ROOT/'data/v2_2/application_inputs.npz')
    feh,valid=rvs_metallicity(t['rvs_feh'],t['rvs_teff'],t['rvs_logg'])
    quality=valid&t['rvs_parameter_quality']&np.isfinite(t['rvs_feh_err'])&(t['rvs_feh_err']>0)
    base=(t['split']==2)&(t['TYPE']=='PSF')&np.isfinite(t['ref_chosen'])&(t['optical_error']<.02)&~np.isin(t['identity'],train)
    masks=[base&(t['class_label']==0)&quality&(feh>=-.5),base&(t['class_label']==0)&quality&(feh< -1.5),base&(t['class_label']==1),base&(t['class_label']==2)]
    take=[];groups=[]
    for k,(u,limit) in enumerate(zip(masks,[64,64,48,48])):
        ix=np.flatnonzero(u)
        with np.errstate(over='ignore'):rank=t['identity'][ix].astype('u8')*np.uint64(11400714819323198485)
        ix=ix[np.argsort(rank,kind='stable')[:limit]];assert len(ix)>=10,'Insufficient independent SED templates'
        take.extend(ix);groups.extend([k]*len(ix))
    q=subset(t,np.array(take));m,e,ids=arrays(q,np.arange(len(take)));g=np.array(groups)
    assert not np.intersect1d(ids,train).size and len(np.unique(ids))==len(ids)
    tm,te,tids=arrays(app,np.arange(len(app['identity'])))
    finite=np.isfinite(te[:,:3]).all(axis=1)&(te[:,:3]>0).all(axis=1)&(app['TYPE']=='PSF')
    target=np.flatnonzero(finite);features=np.column_stack((app['r'],tm[:,0]-tm[:,1],tm[:,1]-tm[:,2]));scale=np.array([1,.3,.3]);tree=cKDTree(features[target]/scale)
    d=np.array([1,3.5,7.5,15,30]);mus=10+5*np.log10(d);records=[];mags=[];errors=[]
    rng=np.random.default_rng(20261111)
    for replica in (-1,0,1,2):
        for i in range(len(ids)):
            for b,mu in enumerate(mus):
                shift=mu-q['ref_chosen'][i];clean=m[i]+shift;observed_r=q['r'][i]+shift
                feature=np.array([observed_r,clean[0]-clean[1],clean[1]-clean[2]])
                dist,j=tree.query(feature/scale,k=16);choices=target[j];w=app['sample_weight'][choices];chosen=rng.choice(choices,p=w/w.sum())
                err=e[i].copy();err[:3]=np.maximum(te[chosen,:3],e[i,:3])
                observed=clean.copy() if replica==-1 else clean+rng.standard_normal(len(clean))*np.nan_to_num(err,nan=0)
                if replica==-1:err=e[i].copy()
                mags.append(observed);errors.append(err)
                records.append(dict(identity=ids[i],TYPE='PSF',template_index=i,group=g[i],class_label=q['class_label'][i],cell=q['cell'][i],distance_bin=b,
                    truth_mu=mu,teacher_sigma=q['sigma_mu'][i],replica=replica,r=observed_r+(observed[1]-clean[1]),target_identity=tids[chosen],
                    target_color_distance=float(np.linalg.norm((features[chosen]-feature)/scale)),label_source=q['label_source'][i]))
    out={k:np.asarray([r[k] for r in records]) for k in records[0]};out['mags']=np.asarray(mags);out['errors']=np.asarray(errors)
    atomic_npz(DEST/'recovery_design.npz',**out)
    save_json(DEST/'recovery_design.json',dict(templates=len(ids),rows=len(records),groups={str(k):int((g==k).sum()) for k in range(4)},
        train_overlap=0,source='held-out empirical SED estimates, not model posterior draws',teacher_scale_limit=True,
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'data/v2_1/development.npz',ROOT/'data/v2_2/application_inputs.npz',OLD/'training.npz']}))
    return out


def chunk(rows):
    path=DEST/'recovery_chunks'/f'{rows[0]:07}.npz';meta=path.with_suffix('.json')
    if meta.exists():assert sha(path)==json.loads(meta.read_text())['sha256'];return str(path)
    m=_D['mags'][rows];e=_D['errors'][rows];ids=_D['identity'][rows];p=_J.predict(m,e,ids)
    scalar=integrate(_J.model,m[:2],e[:2],ids[:2],128)['wide'];err=float(np.max(abs(p['pdf'][:2]-scalar['pdf'])));assert err<2e-6
    mass=distance_bin_mass(p['joint'],_J.model.edges,m[:,0],DISTANCE_EDGES)
    x=_J.model.feature_matrix(m);distance=_J.model.nn.kneighbors((x-_J.model.mean)/_J.model.scale,n_neighbors=10)[0][:,-1]
    support=(distance<=_J.model.radius)&np.all((x>=_J.model.lo)&(x<=_J.model.hi),axis=1)
    observed=(_D['r'][rows]<20)&(np.max(e[:,:3],axis=1)<.1)&support
    qnear=mass[:,:4,:3].sum(axis=(1,2));scope=observed&(qnear>.95)
    atomic_npz(path,identity=ids,TYPE=_D['TYPE'][rows],rows=rows,joint_mass=mass,mu=p['mu'],observable=observed,near_scope=scope,qnear=qnear)
    save_json(meta,dict(sha256=sha(path),n=len(rows),scalar_error=err));return str(path)


def summarize(d,p):
    nstar=1000.;truth_bin=np.searchsorted(DISTANCE_EDGES,10**((d['truth_mu']-10)/5),side='right')-1
    point=np.searchsorted(DISTANCE_EDGES,10**((p['mu'][:,2]-10)/5),side='right')-1
    total=p['joint_mass'][:,:4].sum(axis=1);records=[];intervals=[];sensitivity=[]
    rng=np.random.default_rng(20261111);templates=np.unique(d['template_index']);nt=len(templates)
    # Same source multiplier for all distance translations and noise replicas.
    multipliers=rng.exponential(size=(500,nt))
    for population,(dw,ev,ot,poor) in POPULATIONS.items():
        fractions=np.array([dw*(1-poor),dw*poor,ev,ot]);dp=np.array([.25,.30,.25,.12,.08])
        for condition,reps in [('clean',[-1]),('target_noise',[0,1,2])]:
            u=np.isin(d['replica'],reps);ix=np.flatnonzero(u);groups=d['group'][u];tb=truth_bin[u];idx=d['template_index'][u]
            counts=np.array([len(np.unique(d['template_index'][d['group']==k])) for k in range(4)])
            w=fractions[groups]/counts[groups]*dp[d['distance_bin'][u]]/len(reps);np.testing.assert_allclose(w.sum(),1)
            boot=multipliers[:,idx]*w;# normalize WITHIN each class/metal group to hold design composition fixed.
            for k in range(4):
                use=groups==k;boot[:,use]*=fractions[k]/boot[:,use].sum(axis=1)[:,None]
            for selection,selected in [('all',np.ones(len(ix),bool)),('observable_parent',p['observable'][u]),('near_scope',p['near_scope'][u])]:
                for channel in ['all','dwarf_reference','evolved_reference','other_reference','unlabeled_reference']:
                    c=None if channel=='all' else ['dwarf_reference','evolved_reference','other_reference','unlabeled_reference'].index(channel)
                    pred=total[u] if c is None else p['joint_mass'][u,c]
                    truthclass=np.ones(len(ix),bool) if c is None else d['class_label'][u]==c
                    for b,(lo,hi) in enumerate(zip(DISTANCE_EDGES[:-1],DISTANCE_EDGES[1:])):
                        prob=pred[:,b]*selected;event=(tb==b)&truthclass&selected
                        truth=nstar*(w@event);pdf=nstar*(w@prob);variance=nstar*(w@(prob*(1-prob)))
                        estimate=nstar*np.sum(w*((point[u]==b)&selected)) if c is None else None
                        # Intervals condition on the imposed empirical-SED population.
                        boottruth=nstar*(boot@event);bootpdf=nstar*(boot@prob);bootsd=np.sqrt(nstar*(boot@(prob*(1-prob))))
                        coverage=float(np.mean(abs(bootpdf-boottruth)<=1.95996398454*bootsd))
                        records.append(dict(population=population,condition=condition,selection=selection,channel=channel,low_kpc=lo,high_kpc=hi,
                            expected_truth=truth,pdf_count=pdf,point_count=estimate,bias=pdf-truth,conditional_sd=np.sqrt(variance),
                            conditional_interval_covers=bool(abs(pdf-truth)<=1.95996398454*np.sqrt(variance)),template_bootstrap_coverage=coverage,
                            effective_templates=nt,selected_weight=float(w@selected),population_size=nstar))
                        if c is None:
                            low,high=np.quantile(bootpdf-boottruth,[.025,.975]);intervals.append(dict(population=population,condition=condition,selection=selection,low_kpc=lo,high_kpc=hi,bias_low=low,bias_high=high))
                    for direction in (-1,1):
                        changed=np.searchsorted(DISTANCE_EDGES,10**((d['truth_mu'][u]+direction*d['teacher_sigma'][u]-10)/5),side='right')-1
                        for b in range(6):sensitivity.append(dict(population=population,condition=condition,selection=selection,channel=channel,teacher_sigma_shift=direction,
                            low_kpc=DISTANCE_EDGES[b],high_kpc=DISTANCE_EDGES[b+1],truth_count=nstar*np.sum(w*((changed==b)&truthclass&selected))))
    csv_rows(TABLES/'population_recovery.csv',records);csv_rows(TABLES/'population_recovery_uncertainty.csv',intervals);csv_rows(TABLES/'population_teacher_sensitivity.csv',sensitivity)
    save_json(DEST/'recovery_complete.json',dict(templates=nt,rows=len(d['identity']),population_scenarios=POPULATIONS,independent_model_fit_ids=True,
        intervals='Conditional Bernoulli predictive variance atN=1000;500identity-cluster Bayesian-bootstrap pseudo-populations hold group composition fixed. Coverage is a conditional stress diagnostic, not certified real-population calibration.',
        limitations='Empirical SEDs and chosen teacher distances share selection/scale errors; matched target noise does not supply new faint truth. +/-1sigma coherent teacher sensitivity retained.',job=os.getenv('SLURM_JOB_ID')))


def main():
    global _J,_D
    require_compute();TABLES.mkdir(parents=True,exist_ok=True);(DEST/'recovery_chunks').mkdir(exist_ok=True)
    freeze={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'docs/v2_4_protocol.md',ROOT/'scripts/v2_4_recovery.py',ROOT/'src/photdis/science_v2_4.py',OLD/'control.joblib']}
    marker=DEST/'recovery_freeze.json'
    if marker.exists():assert json.loads(marker.read_text())==freeze
    else:save_json(marker,freeze)
    _D=load(DEST/'recovery_design.npz') if (DEST/'recovery_design.npz').exists() else design();_J=joint_model()
    tasks=[np.arange(i,min(i+64,len(_D['identity']))) for i in range(0,len(_D['identity']),64)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=16,mp_context=mp.get_context('fork')) as pool:paths=list(pool.map(chunk,tasks))
    parts=[load(path) for path in paths];p={k:np.concatenate([v[k] for v in parts]) for k in parts[0]}
    assert np.array_equal(p['identity'],_D['identity']);np.testing.assert_allclose(p['joint_mass'][:,:4].sum(axis=(1,2)),1,atol=2e-6)
    atomic_npz(DEST/'recovery_summary.npz',**p);summarize(_D,p);print('RECOVERY COMPLETE',flush=True)


if __name__=='__main__':main()
