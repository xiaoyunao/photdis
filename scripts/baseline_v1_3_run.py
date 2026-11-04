#!/usr/bin/env python3
"""Allocated reference inventory, fixed forest control and frozen candidate."""
from baseline_v1_2_run import *
from photdis.baseline_v1_3 import class_eligible,evolved_candidate,cells,CLASS_NAMES
from baseline_v1_2_score import counts,curves,distances
from photdis.baseline_v1_2 import conservative_threshold
from baseline_v1_1_risk import estimate,wilson
from baseline_v1_2_parallax import summarize
V12=OUT
OUT=ROOT/'data/baseline_v1_3'


def hist(t,mask):return np.bincount(cells(t)[mask],minlength=135).tolist()


def main():
    require_compute();OUT.mkdir(exist_ok=True,parents=True)
    if (OUT/'result.json').exists():raise FileExistsError('Completed v1.3 output is frozen')
    t=load(OLD/'references.npz');eligible=class_eligible(t)
    train=eligible&(t['optical_error']<.03)
    t['class_eligible']=eligible;t['class_train_eligible']=train;t['distance_train_eligible']=t['fit_eligible'].copy()
    np.savez_compressed(OUT/'references.npz',**t)
    checks={'unique_identity':len(np.unique(t['identity']))==len(t['identity']),
            'distance_mask_unchanged':np.array_equal(t['distance_train_eligible'],t['fit_eligible']),
            'class_no_conflict':not t['identity_conflict'][eligible].any(),
            'class_no_contaminants':not t['nonstellar_risk'][eligible].any()}
    inventory={}
    for sp in range(4):
        mask=t['split']==sp
        weak=t['astro_quality']&t['zpt_valid']&np.isfinite(t['p_corr'])&(t['gaia_parallax_error']>0)&(t['p_corr']/t['gaia_parallax_error']<5)
        inventory[str(sp)]={}
        for key,use in dict(class_labels=eligible,class_training=train,old_class_training=t['fit_eligible']&(t['class_label']>=0),
                            distance_labels=(t['label_source']>0)|t['specdis_valid']|t['spdist_valid'],weak_or_negative_parallax=weak).items():
            inventory[str(sp)][key]=dict(n=int((mask&use).sum()),cells=hist(t,mask&use),
                classes={str(k):int((mask&use&(t['class_label']==k)).sum()) for k in (-1,0,1,2)})
    parent=np.zeros(135,np.int64)
    needed=['r','optical_error','B','C','g0','r0']
    for i in range(13):
        with np.load(OLD/f'parent_{i:02}.npz') as z:p={k:z[k] for k in needed}
        owner=np.load(OLD/f'owner_{i:02}.npy');parent+=np.asarray(hist(p,owner));print('parent counted',i,flush=True)
    checks['parent_count']=int(parent.sum())==8899334
    physical={}
    for key in ['desi_teff','desi_logg','Mg']:
        z=t[key][eligible&(t['class_label']==2)];z=z[np.isfinite(z)]
        physical[key]=dict(n=len(z),quantiles=np.quantile(z,[0,.05,.5,.95,1]).tolist() if len(z) else [])
    result=dict(class_names=CLASS_NAMES,inventory=inventory,parent_cells=parent.tolist(),
                cell_axes=dict(r=['<18','18–20','20–22','22–25','>=25'],optical_error=['<.03','.03–.1','>=.1'],wise=['none','W1 only','W1W2'],gr=['<.5','.5–1','>=1']),class2_parameters=physical,models={})
    # The saved old forest and new data forest see exactly the same evaluation rows.
    for name,(dim,tier) in MODELS.items():
        tr=np.flatnonzero((t['split']==0)&train&available(t,tier))
        oldtr=load(V12/f'{name}_train.npz')['rows'];added=np.setdiff1d(tr,oldtr)
        rows=np.flatnonzero(np.isin(t['split'],[1,2])&eligible&available(t,tier))
        checks[name+':no_leakage']=not np.intersect1d(t['identity'][tr],t['identity'][rows]).size
        old=joblib.load(V12/f'{name}_forest.joblib');models={'old':old};fit=dict(n_old=len(oldtr),n_new=len(tr),added=len(added),removed=len(np.setdiff1d(oldtr,tr)),added_cells=hist(t,np.isin(np.arange(len(eligible)),added)))
        if len(added) and len(np.unique(t['class_label'][tr]))==3:
            new=DirectClassifier(dim,'forest',CFG).fit(arrays(t,tr)[0],t['class_label'][tr],balanced_weights(t['class_label'][tr]))
            joblib.dump(new,OUT/f'{name}_forest.joblib');models['expanded']=new
        else:fit['skip_reason']='No added training labels or missing class'
        np.savez_compressed(OUT/f'{name}_identities.npz',train_rows=tr,train_identity=t['identity'][tr],eval_rows=rows,eval_identity=t['identity'][rows])
        entry=dict(fit=fit,comparisons={});mags=arrays(t,rows)[0];labels=t['class_label'][rows];cal=t['split'][rows]==1;test=t['split'][rows]==2
        oldeval=load(V12/f'reference_{name}_forest.npz')['rows'];shared=np.isin(rows,oldeval)
        for key,clf in models.items():
            prob=clf.predict_proba(mags);np.savez_compressed(OUT/f'{name}_{key}_scores.npz',rows=rows,identity=t['identity'][rows],scores=prob)
            entry['comparisons'][key]={}
            for target in (1,2):
                q=dict(expanded_test=curves(labels,prob[:,target],target,test),original_test=curves(labels,prob[:,target],target,test&shared),workpoints={})
                for budget in (.01,.02,.05):
                    th=conservative_threshold(prob[cal&(labels==0),target],budget);sel=prob[:,target]>=th
                    q['workpoints'][str(budget)]=dict(threshold=th,cal=counts(labels,sel,target,cal),test=counts(labels,sel,target,test),original_test=counts(labels,sel,target,test&shared))
                    checks[f'{name}:{key}:{target}:{budget}:cal_budget']=(q['workpoints'][str(budget)]['cal']['dwarf_fpr'] or 0)<=budget+1e-12
                entry['comparisons'][key][str(target)]=q
        result['models'][name]=entry;print('fixed forest control',name,fit['added'],flush=True)
    # Fixed prediction-selected subgroup; truth labels never enter selection.
    p=load(V12/'reference_B_forest.npz');wide=load(PREV/'B_wide_evaluation.npz');rows=p['rows'];sel=evolved_candidate(p)
    checks['candidate_reference_identity']=np.array_equal(rows,wide['rows'])
    candidate=dict(status='candidate_only',parent_rule_relative_comparison_pass=True,relative_comparison_scope='Same pre-existing guards evaluated on the evolved-only cal subgroup; no threshold selection',absolute_risk_acceptance=None,definition='B/forest supported confidence>=0.9 margin>=0.4 main_index==1',reference={})
    for tag,sp in [('cal',1),('test',2)]:
        use=sel&(t['split'][rows]==sp);d=distances(t,rows,p,wide,use)
        perclass={}
        for k in (-1,0,1,2):
            take=use&(t['class_label'][rows]==k);perclass[str(k)]=distances(t,rows,p,wide,take)
            for ref,dd in perclass[str(k)].items():
                truth=truth_mu(t,ref)[rows];valid=take&np.isfinite(truth)
                for method,mu in [('main',p['main_mu']),('mixture',p['mu']),('wide',wide['mu'])]:
                    dd[method]['tail_1_ci95']=wilson(int((abs(mu[valid,2]-truth[valid])>1).sum()),int(valid.sum()))
        candidate['reference'][tag]=dict(counts=counts(t['class_label'][rows],sel,1,t['split'][rows]==sp),distances=d,by_class=perclass)
    cc=candidate['reference']['cal']
    candidate['relative_comparison_pass']=bool(cc['distances']['chosen']['main'].get('tail_1',1)<=.05 and all(
        cc['by_class'][str(k)]['chosen']['main']['n']>=30 and
        cc['by_class'][str(k)]['chosen']['main'].get('tail_1',1)<=cc['by_class'][str(k)]['chosen']['wide'].get('tail_1',0)+.02 for k in range(3)))
    np.savez_compressed(OUT/'candidate_reference.npz',identity=t['identity'][rows][sel],rows=rows[sel],split=t['split'][rows][sel])
    sample=load(PREV/'parent_sample.npz');pp=load(V12/'parent_B_forest.npz');psel=evolved_candidate(pp);expanded=np.zeros(len(sample['identity']),bool);expanded[pp['rows']]=psel
    candidate['parent']=estimate(sample,expanded)
    np.savez_compressed(OUT/'candidate_parent.npz',identity=sample['identity'][pp['rows']][psel],rows=pp['rows'][psel])
    ga=load(PREV/'parallax_sample.npz');gp=load(V12/'parallax_B_forest.npz');gs=evolved_candidate(gp);candidate['Gaia']={}
    for key in ['wide_mixture','forest_mixture','forest_main']:
        a=load(V12/f'parallax_B_{key}_checks.npz');checks[key+':Gaia_identity']=np.array_equal(a['rows'],gp['rows'])
        candidate['Gaia'][key]=summarize(ga,gp['rows'],a,gs)
    np.savez_compressed(OUT/'candidate_Gaia.npz',identity=ga['identity'][gp['rows']][gs],rows=gp['rows'][gs])
    result['candidate']=candidate
    for names in [('A_on_B','B'),('A_on_C','B_on_C','C')]:
        ids=[load(OUT/f'{name}_identities.npz') for name in names]
        for j in range(1,len(ids)):
            for k in ['train_identity','eval_identity']:checks[names[j]+':matched_'+k]=np.array_equal(ids[0][k],ids[j][k])
    assert all(checks.values()),checks
    result['validation']=dict(n_checks=len(checks),all_passed=True,checks=checks)
    result['hashes']={p.name:sha(p) for p in OUT.glob('*.npz')}
    result['model_hashes']={p.name:sha(p) for p in OUT.glob('*.joblib')}
    result['source_reference_sha256']=sha(OLD/'references.npz')
    write_json(OUT/'result.json',result);print('COMPLETE',len(checks),'checks',flush=True)

if __name__=='__main__':main()
