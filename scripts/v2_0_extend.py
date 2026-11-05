#!/usr/bin/env python3
"""Add old application and later experimental views to the unified report."""
import json
import numpy as np
import fitsio

from v2_0_common import *
from photdis.stage1 import require_compute
from baseline_v0_fit import truth_mu


def export(name,q,train_ids,reference='new spatial split'):
    q['model_id']=np.full(len(q['identity']),name,'U40')
    atomic_npz(DEST/f'training_{name}.npz',identity=train_ids)
    assert not np.intersect1d(q['identity'],train_ids).size
    q['in_model_training']=np.isin(q['identity'],train_ids)
    q['outside_train_precision']=q['optical_error']>=.03
    q['distance_kpc']=10**((q['mu']-10)/5)
    rows=[]
    for split,label in [(1,'cal'),(2,'development_test')]:
        for group,mask in strata(q).items():
            use=(q['split']==split)&mask
            for kind in ('chosen','Gaia','specdis','spdist'):
                rows.append(dict(model=name,split=label,group=group,reference=kind,
                                 **percent_metrics(q['mu'][use],q['ref_'+kind][use])))
    csv_rows(TABLES/f'performance_{name}.csv',rows)
    atomic_npz(DEST/f'chosen_{name}.npz',**q)
    sr=[];sc=[]
    for code,kind in enumerate(('Gaia','specdis','spdist')):
        selected=np.flatnonzero(np.isfinite(q['ref_'+kind]))
        sr.extend(selected);sc.extend([code]*len(selected))
    atomic_npz(DEST/f'sources_{name}.npz',chosen_view_row=np.asarray(sr),source=np.asarray(sc),
               identity=q['identity'][sr])
    return dict(model=name,chosen_view=f'data/v2_0/chosen_{name}.npz',train_n=len(train_ids),
                split_definition=reference,n=len(q['identity']),
                training_identity_file=f'data/v2_0/training_{name}.npz',
                training_identity_sha256=__import__('hashlib').sha256(train_ids.tobytes()).hexdigest())


def main():
    require_compute()
    snap=json.loads((TABLES/'snapshot.json').read_text())
    snap['models']=[m for m in snap['models'] if m['model'] in CFG['models']]
    with np.load(OUT/'references.npz') as z:
        exact_photometry={k:z[k] for k in ('identity','g0','r0','z0','r','w10','w20','i0')}
    for item in snap['models']:
        name=item['model']
        model_ids=load(OUT/'experiments'/f'{name}_identities.npz')
        train_ids=model_ids['distance_train_identity']
        legacy_path=OUT/'experiments'/f'{name}_legacy_training.npz'
        if legacy_path.exists():
            train_ids=np.r_[train_ids,load(legacy_path)['distance_train_identity']]
        atomic_npz(DEST/f'training_{name}.npz',identity=train_ids)
        item['training_identity_file']=f'data/v2_0/training_{name}.npz'
        item['training_identity_sha256']=__import__('hashlib').sha256(train_ids.tobytes()).hexdigest()
        saved=load(DEST/f'chosen_{name}.npz')
        exact_rows=unique_join(saved['identity'],exact_photometry['identity'])
        assert (exact_rows>=0).all()
        for key in ('g0','r0','z0','r','w10','w20','i0'):
            saved[key]=exact_photometry[key][exact_rows]
        saved['gr']=saved['g0']-saved['r0'];saved['rz']=saved['r0']-saved['z0']
        export(name,saved,train_ids)

    q=load(DEST/'chosen_expanded_A.npz')
    ids=load(OUT/'experiments/expanded_A_identities.npz')
    legacy=load(OUT/'experiments/expanded_A_legacy_training.npz')
    train=np.r_[ids['distance_train_identity'],legacy['distance_train_identity']]
    r0path=ROOT/'data/baseline_v1_5_r0'
    rp=load(audited(r0path/'evaluation.npz'))
    rs=json.loads((r0path/'result.json').read_text())
    assert sha(r0path/'evaluation.npz')==rs['evaluation_sha256']
    assert np.array_equal(rp['identity'],q['identity'])
    new=dict(q,mu=rp['mu'],mu_raw=rp['mu'],supported=rp['supported'],ambiguous=rp['ambiguous'])
    ga=q['parallax_eligible']
    for key in ('full_pit','full_log_density'):
        new[key]=np.full(len(ga),np.nan);new[key][ga]=rp[key]
    for kind in ('chosen','Gaia','specdis','spdist'):
        new['delta_mu_'+kind]=rp['mu'][:,2]-q['ref_'+kind]
        new['epsilon_pct_'+kind]=100*np.expm1(np.log(10)*new['delta_mu_'+kind]/5)
    item=export('expanded_A_r0',new,train)
    item.update(original_model_id='expanded_A + grz_r0',model_path='data/baseline_v1_5_r0/model.joblib',model_sha256=rs['model_sha256'],protocol='docs/baseline_v1_5_r0_protocol.md')
    snap['models'].append(item)
    audited(r0path/'model.joblib',rs['model_sha256'])
    wp=ROOT/'data/baseline_v1_6_wise'
    pred=load(audited(wp/'evaluation.npz'))
    result=json.loads((wp/'result.json').read_text())
    assert sha(wp/'evaluation.npz')==result['evaluation_sha256']
    assert np.array_equal(pred['identity'],q['identity'])
    for i,name in enumerate(('weak_optical','weak_w1','weak_w1w2')):
        new=dict(q,mu=pred['mu'][:,i],mu_raw=pred['mu'][:,i],cached_full_pdf=np.ones(len(q['identity']),bool))
        for key in ('supported','ambiguous'):
            new[key]=np.zeros(len(q['identity']),bool)
            new[key+'_available']=np.zeros(len(q['identity']),bool)
        for key,source in [('full_pit','pit'),('full_log_density','log_density')]:
            new[key]=np.full(len(ga),np.nan);new[key][ga]=pred[source][:,i]
        new['pdf_chunk_start']=np.arange(len(q['identity']))//192*192
        new['pdf_chunk_row']=np.arange(len(q['identity']))%192
        new['pdf_arm']=np.full(len(q['identity']),i,'i1')
        for kind in ('chosen','Gaia','specdis','spdist'):
            new['delta_mu_'+kind]=new['mu'][:,2]-q['ref_'+kind]
            new['epsilon_pct_'+kind]=100*np.expm1(np.log(10)*new['delta_mu_'+kind]/5)
        item=export(name,new,pred['train_identity'])
        item.update(original_model_id=('optical','w1','w1w2')[i],representation='joint empirical templates, no serialized forest',protocol='docs/baseline_v1_6_wise_protocol.md',source_evaluation_sha256=result['evaluation_sha256'])
        snap['models'].append(item)
    oldpath=ROOT/'data/baseline_v0/references.npz'
    old=load(audited(oldpath))
    old['cell']=old['field']
    for source,target in [('ra','RA'),('dec','DEC'),('ebv','EBV')]:
        old[target]=old[source]
    old['TYPE']=np.full(len(old['identity']),'PSF','U4')
    model_names=['A','B','C','A_on_B','A_on_C','B_on_C']
    old_models={}
    for name in ['recommended',*model_names]:
        stem='recommended_evaluation.npz' if name=='recommended' else name+'_wide_evaluation.npz'
        path=ROOT/'data/baseline_v1'/stem
        audited(path)
        with np.load(path) as z:
            p={k:z[k] for k in ('rows','mu','supported','ambiguous') if k in z.files}
            if 'mu_cal' in z.files:p['mu_cal']=z['mu_cal']
        tt=subset(old,p['rows'])
        tt.update(mu=p['mu'],supported=p['supported'],ambiguous=p['ambiguous'])
        tt['mu_raw']=p['mu']
        tt['mu_calibrated_available']=np.full(len(p['rows']),'mu_cal' in p)
        if 'mu_cal' in p:tt['mu_cal']=p['mu_cal']
        for kind in ('chosen','Gaia','specdis','spdist'):
            tt['ref_'+kind]=truth_mu(tt,kind)
            tt['delta_mu_'+kind]=tt['mu'][:,2]-tt['ref_'+kind]
            tt['epsilon_pct_'+kind]=100*np.expm1(np.log(10)*tt['delta_mu_'+kind]/5)
        tier='B' if name in ('B','A_on_B') else 'C' if name in ('C','A_on_C','B_on_C') else 'A'
        mask=(old['split']==0)&old['fit_eligible']
        if tier!='A':mask &= old[tier]
        train=old['identity'][mask]
        if name=='recommended':
            train=old['identity'][(old['split']==0)&old['fit_eligible']]
        item=export('baseline_v1_'+name,tt,train,'old whole-field split')
        if name!='recommended':
            mp=ROOT/'data/baseline_v1'/f'{name}.joblib'
            audited(mp)
            item.update(model_path=str(mp.relative_to(ROOT)),model_sha256=sha(mp))
            old_models[name]=dict(train=train,eval=tt['identity'])
        snap['models'].append(item)
    for a,b in [('A_on_B','B'),('A_on_C','C'),('B_on_C','C')]:
        assert np.array_equal(old_models[a]['train'],old_models[b]['train'])
        assert np.array_equal(old_models[a]['eval'],old_models[b]['eval'])
    app=snap['application_fields']
    prep=json.loads((ROOT/'manifests/baseline_v0_prepared.json').read_text())
    for row,item in zip(app,prep['fields']):
        with fitsio.FITS(str(ROOT/item['input'])) as h:
            row['field_diagnostics_rows']=h[1].get_nrows()
        row['input_parent_records']=item['parent_records']
    csv_rows(TABLES/'C_application_fields.csv',app)
    # Keep actual training r distributions and independent evidence counts.
    with np.load(OUT/'references.npz') as z:
        nt={k:z[k] for k in ('identity','r')}
    with np.load(ROOT/'data/baseline_v1_3/references.npz') as z:
        ot={k:z[k] for k in ('identity','r')}
    hist=[]
    edges=np.array([14,15,16,17,18,19,20,21,22,23,24,25.])
    for family,key in [('distance','distance_train_identity'),('class','class_train_identity')]:
        a=unique_join(ids[key],nt['identity']);b=unique_join(legacy[key],ot['identity'])
        assert (a>=0).all() and (b>=0).all()
        values=np.r_[nt['r'][a],ot['r'][b]]
        count=np.histogram(values,edges)[0]
        for lo,hi,n in zip(edges[:-1],edges[1:],count):
            hist.append(dict(family=family,lo=lo,hi=hi,n=int(n)))
    csv_rows(TABLES/'training_brightness_hist.csv',hist)
    pop=load(DEST/'population.npz')
    evidence=[]
    for split in (0,1,2):
        for lo,hi in [(-np.inf,18),(18,20),(20,22),(22,25)]:
            for name,mask in [('no_W1',~pop['B']),('W1_only',pop['B']&~pop['C']),('W1W2',pop['C'])]:
                use=(pop['split']==split)&(pop['r']>=lo)&(pop['r']<hi)&mask
                evidence.append(dict(split=split,rlo=lo,rhi=hi,WISE=name,n=int(use.sum()),
                    class_n=int((use&pop['class_eligible']).sum()),
                    chosen_n=int((use&(pop['label_source']>0)).sum()),
                    gaia_n=int((use&pop['parallax_eligible']).sum())))
    csv_rows(TABLES/'evidence_by_r_wise.csv',evidence)

    for filename in ('configs/baseline_v0.json','configs/baseline_v1.json','configs/baseline_v1_4.json','configs/v2_0_snapshot.json','data/baseline_v1/selection.json'):
        audited(ROOT/filename)
    snap['units']=dict(distance='kpc',parallax='mas',distance_modulus='mag',percentage_residual='percent',flux='nanomaggies')
    snap['inputs'].update(INPUTS)
    snap['report_code_hashes']={str(p.relative_to(ROOT)):sha(p) for p in [
        ROOT/'scripts/v2_0_collect.py',ROOT/'scripts/v2_0_extend.py',ROOT/'scripts/v2_0_figures.py',
        ROOT/'src/photdis/report_v2.py']}
    # Range table requested distance errors in physical units, alongside sigma_mu.
    # Restore exact SpecDis relative error from the frozen logarithmic sigma recipe.
    import csv
    with open(TABLES/'B_sample_ranges.csv') as f:
        records=list(csv.DictReader(f))
    for model in ('expanded_A',):
        t=load(DEST/f'chosen_{model}.npz')
        sigma=t['sigma_mu']*np.log(10)/5
        error=np.where(t['label_source']==1,t['ref_distance']*sigma,
                       t['ref_distance']*np.sqrt(np.expm1(sigma*sigma)))
        for split,label in [(1,'cal_chosen'),(2,'test_chosen')]:
            mask=(t['split']==split)&np.isfinite(t['ref_chosen'])
            records.append(dict(sample=label,variable='ref_distance_error_kpc',weighted=False,
                                range_definition='Gaia linear propagation or inverse SpecDis log-sigma recipe',
                                selection_threshold='source qualification',plot_range='not plotted',**ranges(error[mask])))
    csv_rows(TABLES/'B_sample_ranges.csv',records)
    save_json(TABLES/'snapshot.json',snap)
    save_json(DEST/'collection_complete.json',dict(snapshot_sha256=sha(TABLES/'snapshot.json')))
    print('COMPLETE extended report tables',len(snap['models']),flush=True)


if __name__=='__main__':
    main()
