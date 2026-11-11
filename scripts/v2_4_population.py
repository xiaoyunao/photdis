#!/usr/bin/env python3
"""Observable parent admission, exact joint counts and unopened ID reservation."""
import argparse
import concurrent.futures
import multiprocessing as mp
import json
import os
import joblib
import numpy as np
from v2_1_near import ROOT,load,subset,sha,save_json,csv_rows,atomic_npz
from v2_4_audit import OLD,DEST,TABLES,pdf_rows
from baseline_v1_4_experiments import arrays
from photdis.stage1 import require_compute,legacy_ids
from photdis.reference_v1_4 import unique_join
from photdis.probability_v2_1 import JointReferenceForest,psf_admission
from photdis.science_v2_4 import distance_bin_mass
from photdis.selection_v2_2 import evidence_cells
from photdis.baseline_parallax import parallax_check
from photdis.baseline_v1_2 import full_parallax_log_density
from v2_3_prepare import qualify
from v2_3_diagnostics import references

DISTANCE_EDGES=np.array([0,2,5,10,20,50,np.inf])
CHANNELS=['dwarf_reference','evolved_reference','other_reference','unlabeled_reference','Gaia_label','SpecDis_label']
_J=_T=None


def observable(t,model):
    m,_,_=arrays(t,np.arange(len(t['identity'])));x=model.feature_matrix(m)
    distance=model.nn.kneighbors((x-model.mean)/model.scale,n_neighbors=10)[0][:,-1]
    return (t['r']<20)&(t['optical_error']<.1)&(distance<=model.radius)&np.all((x>=model.lo)&(x<=model.hi),axis=1)


def joint_model():
    train=load(OLD/'training.npz');assert (train['TYPE']=='PSF').all()
    model=joblib.load(OLD/'control.joblib');m,_,_=arrays(train,np.arange(len(train['identity'])))
    np.testing.assert_allclose((model.feature_matrix(m)-model.mean)/model.scale,model.nn._fit_X,atol=1e-12)
    return JointReferenceForest(model,m,train['Mg'],train['class_label'],train['label_source'])


def joint_chunk(rows):
    path=DEST/'application_joint_chunks'/f'{rows[0]:07}.npz';marker=path.with_suffix('.json')
    if marker.exists():assert sha(path)==json.loads(marker.read_text())['sha256'];return str(path)
    m,e,ids=arrays(_T,rows);p=_J.predict(m,e,ids)
    expected,edges=pdf_rows('application','control',rows)
    err=float(np.max(abs(expected-p['pdf'])));assert err<2e-6
    old=load(OLD/'application_control.npz');ce=float(np.max(abs(p['joint'][:,:4].sum(axis=2)-old['weights'][rows])));assert ce<2e-6
    mass=distance_bin_mass(p['joint'],edges,m[:,0],DISTANCE_EDGES)
    ga=_T['parallax_eligible'][rows];log=np.full(len(rows),np.nan);pit=log.copy()
    if ga.any():
        args=(p['pdf'][ga],edges,m[ga,0],_T['p_corr'][rows][ga],_T['gaia_parallax_error'][rows][ga])
        log[ga]=full_parallax_log_density(*args,order=32)
        pit[ga]=parallax_check(*args,np.full(ga.sum(),-np.inf),np.full(ga.sum(),np.inf),order=32)['pit']
    atomic_npz(path,identity=ids,TYPE=_T['TYPE'][rows],rows=rows,joint_mass=mass,class_mass=p['joint'].sum(axis=2),gaia_log=log,gaia_pit=pit)
    save_json(marker,dict(sha256=sha(path),pdf_max_error=err,class_max_error=ce,n=len(rows)))
    return str(path)


def application():
    global _J,_T
    app=load(ROOT/'data/v2_2/application_inputs.npz');p=load(OLD/'application_control.npz');assert (app['TYPE']=='PSF').all()
    model=joblib.load(OLD/'control.joblib');base=observable(app,model)
    assert np.array_equal(base&(p['qnear']>.95),p['near_scope'])
    atomic_npz(DEST/'observable_target.npz',identity=app['identity'],TYPE=app['TYPE'],observable=base,weight=app['sample_weight'],field=app['field'])
    # Admission above is frozen before observing any reference content.
    n=len(base);ref=load(ROOT/'data/baseline_v1_4/references.npz');j=unique_join(app['identity'],ref['identity']);found=j>=0
    q=qualify(subset(ref,j[found]));del ref
    app['parallax_eligible']=np.zeros(n,bool);app['p_corr']=np.full(n,np.nan);app['gaia_parallax_error']=np.full(n,np.nan)
    app['known_nonstellar_risk']=np.zeros(n,bool)
    # Signed Gaia from original photometric-parent owner is also retained for
    # rows outside the spectroscopic reference table. No SNR admission cut.
    for field in np.unique(app['field']):
        use=app['field']==field;rr=app['parent_row'][use]
        with np.load(ROOT/f'data/baseline_v0/parent_{field:02}.npz') as z:
            assert np.array_equal(z['identity'][rr],app['identity'][use])
            for key in ('p_corr','gaia_parallax_error','nonstellar_risk'):
                app['known_nonstellar_risk' if key=='nonstellar_risk' else key][use]=z[key][rr]
            app['parallax_eligible'][use]=z['astro_quality'][rr]&z['zpt_valid'][rr]&np.isfinite(z['p_corr'][rr])&np.isfinite(z['gaia_parallax_error'][rr])&(z['gaia_parallax_error'][rr]>0)
    # Keep risk-flagged Gaia as a separate control, never silently omit it.
    rows=np.flatnonzero(base);_T=app;_J=joint_model();assert np.array_equal(_J.source_values,[1,2])
    (DEST/'application_joint_chunks').mkdir(exist_ok=True)
    tasks=[rows[i:i+64] for i in range(0,len(rows),64)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=16,mp_context=mp.get_context('fork')) as pool:
        paths=list(pool.map(joint_chunk,tasks))
    parts=[load(path) for path in paths];out={k:np.concatenate([a[k] for a in parts]) for k in parts[0]}
    assert np.array_equal(out['identity'],app['identity'][rows])
    atomic_npz(DEST/'application_joint_summary.npz',**out)
    np.testing.assert_allclose(out['joint_mass'][:,:4].sum(axis=(1,2)),1,atol=2e-6)
    np.testing.assert_allclose(out['joint_mass'][:,4:].sum(axis=(1,2)),1,atol=2e-6)
    dev=load(ROOT/'data/v2_1/development.npz');cell=evidence_cells(app);dc=evidence_cells(dev)
    support_count=np.bincount(dc[np.isfinite(dev['ref_chosen'])],minlength=10000)[cell]
    direct=np.zeros(n,bool);direct[found]=np.isfinite(q['ref_chosen'])|q['spdist_valid']
    records=[];coverage=[];gaia=[];groups={'all':np.ones(len(rows),bool),'no_W1':~app['B'][rows],'W1':app['B'][rows]}
    for field in np.unique(app['field']):groups['field_'+str(field)]=app['field'][rows]==field
    for key,val,edges in [('r',app['r'][rows],[-np.inf,18,19,20]),('gr',app['g0'][rows]-app['r0'][rows],[-np.inf,.3,.8,1.3,np.inf]),('error',app['optical_error'][rows],[0,.01,.03,.1])]:
        for lo,hi in zip(edges[:-1],edges[1:]):groups[f'{key}_{lo}_{hi}']=(val>=lo)&(val<hi)
    w=app['sample_weight'][rows]
    for selection,choose in [('observable_parent',np.ones(len(rows),bool)),('near_scope',p['near_scope'][rows])]:
        for group,g in groups.items():
            use=choose&g;rr=rows[use];ww=w[use];mass=out['joint_mass'][use]
            near=mass[:,:4,:3].sum(axis=(1,2));cm=out['class_mass'][use]
            for c,ch in enumerate(CHANNELS):
                for b,(lo,hi) in enumerate(zip(DISTANCE_EDGES[:-1],DISTANCE_EDGES[1:])):
                    value=mass[:,c,b]
                    records.append(dict(selection=selection,group=group,channel=ch,low_kpc=lo,high_kpc=hi,n=len(rr),
                        expected_count=float(ww@value),conditional_weighted_count_sd=float(np.sqrt(max(0,np.sum(ww**2*value*(1-value)))))))
                records.append(dict(selection=selection,group=group,channel=ch,low_kpc=0,high_kpc=10,n=len(rr),expected_count=float(ww@mass[:,c,:3].sum(axis=1)),
                    product_marginals_count=float(ww@(cm[:,c]*near))))
            coverage.append(dict(selection=selection,group=group,n=len(rr),weight=float(ww.sum()),direct_distance_weight=float(ww@direct[rr]),
                Gaia_observed_weight=float(ww@app['parallax_eligible'][rr]),none_weight=float(ww@(support_count[rr]==0)),
                sparse_weight=float(ww@((support_count[rr]>0)&(support_count[rr]<20))),dense_weight=float(ww@(support_count[rr]>=20)),
                unlabeled_near_count=float(ww@mass[:,3,:3].sum(axis=1))))
            for risk in (False,True):
                u=use&app['parallax_eligible'][rows]&(app['known_nonstellar_risk'][rows]==risk);v=out['gaia_pit'][u]
                gaia.append(dict(selection=selection,group=group,risk_flag=risk,n=int(u.sum()),negative=int((app['p_corr'][rows[u]]<0).sum()),
                    log_mean=float(np.average(out['gaia_log'][u],weights=w[u])) if u.any() else None,low=float(np.average(v<.05,weights=w[u])) if u.any() else None,high=float(np.average(v>.95,weights=w[u])) if u.any() else None))
    csv_rows(TABLES/'application_joint_counts.csv',records);csv_rows(TABLES/'application_target_coverage.csv',coverage);csv_rows(TABLES/'application_gaia.csv',gaia)
    save_json(DEST/'application_complete.json',dict(sample=26000,observable_n=len(rows),observable_weight=float(w.sum()),near_n=int(p['near_scope'].sum()),
        no_reference_selection=True,model=sha(OLD/'control.joblib'),joint_leaf_errors=_J.reconstruction_max_errors,
        max_pdf_error=max(json.loads(x.with_suffix('.json').read_text())['pdf_max_error'] for x in paths),
        interpretation='Weighted reference-contribution expectations, not verified physical population counts. Unlabeled is missing training evidence. Observable cells measure support, not representativeness.',job=os.getenv('SLURM_JOB_ID')))
    print('APPLICATION JOINT COMPLETE',len(rows),flush=True)


def reserve():
    import fitsio
    used=[]
    for path in [ROOT/'data/baseline_v0/references.npz',ROOT/'data/baseline_v1_3/references.npz',ROOT/'data/baseline_v1_4/references.npz',ROOT/'data/v2_2/application_inputs.npz']:
        with np.load(path) as z:used.append(z['identity'])
    used=np.unique(np.concatenate(used));model=joblib.load(OLD/'control.joblib')
    fields=json.loads((ROOT/'data/baseline_v0/prepared.json').read_text())['fields'];pieces=[];counts=[];native_sources={}
    for f in fields[:13]:
        field=f['index'];path=ROOT/f'data/baseline_v1/catalog_{field:02}.fits'
        cols=['identity','ls_id','r','g0','r0','z0','w10','w20','sigma_g','sigma_r','sigma_z','sigma_w1','sigma_w2','optical_error','ra','dec','ebv','B','C']
        with fitsio.FITS(str(path)) as h:a=h[1].read(columns=cols)
        t={k:a[k] for k in cols};u=(t['r']<20)&(t['optical_error']<.1)&~np.isin(t['identity'],used)
        ix=np.flatnonzero(u)
        with np.errstate(over='ignore'):rank=t['identity'][ix].astype('u8')*np.uint64(11400714819323198485)
        # Bounded candidates chosen on photometry/identity only; no outcome read.
        ix=ix[np.argsort(rank,kind='stable')[:2500]];t=subset(t,ix);u=observable(t,model)
        types=np.full(len(ix),'','U4');task=json.loads((ROOT/f['config']).read_text());cfg=json.loads((ROOT/task['base_config']).read_text())
        sweep_root=task.get('sweep_root',cfg['sweep_root'])
        for sweep in task['sweep_files']:
            source=os.path.join(sweep_root,sweep)
            with fitsio.FITS(source) as h:a=h[1].read(columns=['RELEASE','BRICKID','OBJID','TYPE'])
            ids=legacy_ids(a);j=unique_join(t['ls_id'],ids);found=j>=0;types[found]=np.char.strip(a['TYPE'][j[found]].astype('U'))
            stat=os.stat(source);native_sources[source]=dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns)
        good,audit=psf_admission(types);take=np.flatnonzero(good&u)[:250];t['TYPE']=types;t['field']=np.full(len(ix),field)
        # No model prediction/reference outcomes are stored or inspected.
        pieces.append(subset(t,take));counts.append(dict(field=field,candidates=len(ix),observable=int(u.sum()),reserved=len(take),**audit))
        print('RESERVE FIELD',field,len(take),flush=True)
    out={k:np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
    assert len(np.unique(out['identity']))==len(out['identity']) and not np.intersect1d(used,out['identity']).size and (out['TYPE']=='PSF').all()
    atomic_npz(DEST/'prospective_identity_reservation.npz',**out);csv_rows(TABLES/'reservation_counts.csv',counts)
    save_json(DEST/'reservation.json',dict(n=len(out['identity']),sha256=sha(DEST/'prospective_identity_reservation.npz'),native_sources=native_sources,
        reference_outcomes_opened=False,predictions_generated=False,known_historical_overlap=0,
        limitations='Same existing application footprints; not new sky. Future system-membership audit and independent reference acquisition required before final-validation claim.',job=os.getenv('SLURM_JOB_ID')))


if __name__=='__main__':
    require_compute();TABLES.mkdir(parents=True,exist_ok=True)
    parser=argparse.ArgumentParser();parser.add_argument('task',choices=['application','reserve']);a=parser.parse_args()
    freeze={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'docs/v2_4_protocol.md',ROOT/'scripts/v2_4_population.py',ROOT/'src/photdis/science_v2_4.py',OLD/'control.joblib',OLD/'training.npz']}
    marker=DEST/'population_freeze.json'
    if marker.exists():assert json.loads(marker.read_text())==freeze
    else:save_json(marker,freeze)
    (application if a.task=='application' else reserve)()
