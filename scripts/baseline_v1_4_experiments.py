#!/usr/bin/env python3
"""Fixed frozen/new-data/matched-i experiments with immutable evaluation IDs."""
import concurrent.futures,json,multiprocessing as mp,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'.deps')]
import joblib
import numpy as np
from sklearn.metrics import average_precision_score
from photdis.stage1 import require_compute,write_json
from photdis.baseline_v0 import HistogramQRF,BANDS,ERR,metrics,distance_quantiles
from photdis.baseline_v1 import integrate
from photdis.baseline_v1_2 import DirectClassifier,balanced_weights,conservative_threshold,full_parallax_log_density
from photdis.optical_v1_4 import OpticalQRF,OpticalClassifier
from photdis.baseline_parallax import parallax_check
from baseline_v1_4_extract import sha,atomic_npz
from baseline_v1_4_assemble import load

OUT=ROOT/'data/baseline_v1_4'
SNR=np.array([-np.inf,0,5,10,20,np.inf])
_T=None;_MODEL=None;_MODE=None;_DEST=None
KEYS=['identity','split','cell','field','g0','r0','z0','w10','w20','i0','sigma_i','i_usable',*ERR,
      'r','B','C','class_label','class_eligible','class_train_eligible','photometric_eligible','nonstellar_risk','identity_conflict',
      'fit_eligible','Mg','label_source','ref_distance','specdis_distance','spdist_distance','specdis_valid','spdist_valid',
      'anchor_astrometry','p_corr','gaia_parallax_error','parallax_eligible','optical_error']


def clean_json(value):
    if isinstance(value,dict):return {str(k):clean_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)):return [clean_json(v) for v in value]
    if isinstance(value,(np.bool_,)):return bool(value)
    if isinstance(value,(np.integer,)):return int(value)
    if isinstance(value,(float,np.floating)):return float(value) if np.isfinite(value) else None
    return value


def save_json(path,obj):write_json(path,clean_json(obj))

def subset(t,rows):return {k:v[rows] for k,v in t.items()}

def arrays(t,rows,optical=False):
    keys=['g0','r0','i0','z0'] if optical else BANDS
    err=['sigma_g','sigma_r','sigma_i','sigma_z'] if optical else ERR
    return np.column_stack([t[k][rows] for k in keys]),np.column_stack([t[k][rows] for k in err]),t['identity'][rows]


def truth(t,kind):
    with np.errstate(invalid='ignore',divide='ignore'):
        if kind=='Gaia':return np.where(t['anchor_astrometry']&~t['nonstellar_risk'],10-5*np.log10(t['p_corr']),np.nan)
        if kind=='chosen':return np.where(t['label_source']>0,10+5*np.log10(t['ref_distance']),np.nan)
        return 10+5*np.log10(t[kind+'_distance'])


def groups(t):
    n=len(t['identity']);g={'all':np.ones(n,bool),'no_W1':~t['B'],'W1_only':t['B']&~t['C'],'W1W2':t['C']}
    for lo,hi in [(-np.inf,18),(18,20),(20,22),(22,25)]:
        m=(t['r']>=lo)&(t['r']<hi);g[f'r{lo}_{hi}']=m;g[f'r{lo}_{hi}:no_W1']=m&~t['B']
    for k in [-1,0,1,2]:g[f'class{k}']=t['class_label']==k
    for lo,hi in [(0,.03),(.03,.1),(.1,np.inf)]:g[f'error{lo}_{hi}']=(t['optical_error']>=lo)&(t['optical_error']<hi)
    for k in np.unique(t['cell']):g[f'cell{k}']=t['cell']==k
    return g


def class_scores(t,prob):
    labels=t['class_label'].copy();labels[~t['class_eligible']]=-1
    cal=t['split']==1;test=t['split']==2
    threshold=conservative_threshold(prob[cal&(labels==0),1],.02)
    result={'cal_dwarf_budget':.02,'threshold':threshold,'groups':{}}
    for name,mask in groups(t).items():
        use=test&mask;known=use&(labels>=0);y=labels[known]==1;s=prob[known,1]
        selected=use&(prob[:,1]>=threshold);positive=use&(labels==1);dwarf=use&(labels==0)
        hit=int((selected&(labels==1)).sum());nk=int((selected&(labels>=0)).sum())
        result['groups'][name]=dict(n=int(use.sum()),known=int(known.sum()),unknown=int((use&(labels<0)).sum()),
            evolved=int(positive.sum()),average_precision=float(average_precision_score(y,s)) if len(np.unique(y))==2 else None,
            selected=int(selected.sum()),known_selected=nk,recall=hit/int(positive.sum()) if positive.any() else None,
            known_contamination=1-hit/nk if nk else None,dwarf_fpr=float((selected&dwarf).sum()/dwarf.sum()) if dwarf.any() else None)
    result['cal_dwarf_fpr']=float(np.mean(prob[cal&(labels==0),1]>=threshold)) if np.any(cal&(labels==0)) else None
    assert result['cal_dwarf_fpr'] is None or result['cal_dwarf_fpr']<=.02+1e-12
    return result


def gaia_values(t,pdf,edges):
    n=len(t['identity']);args=(pdf,edges,t['g0'],t['p_corr'],t['gaia_parallax_error'])
    full=parallax_check(*args,np.full(n,-np.inf),np.full(n,np.inf))
    bins=np.column_stack([parallax_check(*args,np.full(n,lo),np.full(n,hi))['selection_probability'] for lo,hi in zip(SNR[:-1],SNR[1:])])
    snr=t['p_corr']/t['gaia_parallax_error'];k=np.searchsorted(SNR[1:-1],snr,side='right')
    conditional=parallax_check(*args,SNR[k],SNR[k+1]);logd=full_parallax_log_density(*args)
    return dict(full_pit=full['pit'],full_log_density=logd,full_mean=full['conditional_mean'],bin_probability=bins,
                conditional_pit=conditional['pit'],observed_bin_probability=bins[np.arange(n),k])


def distance_chunk(task):
    number,rows=task;path=_DEST/f'{number:06}.npz';marker=path.with_suffix('.json')
    if marker.exists():
        assert sha(path)==json.loads(marker.read_text())['sha256'];return path
    m,e,ids=arrays(_T,rows,_MODE=='optical')
    if _MODE=='optical':
        p=_MODEL.predict(m,e,ids,draws=128);mu=distance_quantiles(m[:,0],p['q_Meffective'])[0]
    else:p=integrate(_MODEL,m,e,ids,draws=128)['wide'];mu=p['mu']
    pdf=p['pdf'];assert np.isfinite(pdf).all() and np.allclose(pdf.sum(axis=1),1,atol=2e-5)
    result=dict(rows=rows,identity=ids,mu=mu,supported=p['supported'],ambiguous=p['ambiguous'])
    for kind in ['Gaia','chosen','specdis','spdist']:
        v=truth(subset(_T,rows),kind);absolute=m[:,0]-v
        cdf=np.cumsum(pdf,axis=1);cdf=np.column_stack([np.zeros(len(rows)),cdf])
        # CDF in mu reverses the effective-absolute-magnitude distribution.
        pit=np.array([1-np.interp(a,_MODEL.edges,c) if np.isfinite(a) else np.nan for a,c in zip(absolute,cdf)])
        result['distance_pit_'+kind]=pit
    good=_T['parallax_eligible'][rows]
    values=gaia_values(subset(_T,rows[good]),pdf[good],_MODEL.edges) if good.any() else {}
    for k in ['full_pit','full_log_density','full_mean','conditional_pit','observed_bin_probability','bin_probability']:
        result[k]=np.full((len(rows),5) if k=='bin_probability' else len(rows),np.nan)
        if good.any():result[k][good]=values[k]
    if number==0 and good.any():
        # Check independent higher-order quadrature on a bounded first batch.
        r=rows[good][:32];pp=pdf[good][:32]
        hi=parallax_check(pp,_MODEL.edges,_T['g0'][r],_T['p_corr'][r],_T['gaia_parallax_error'][r],np.full(len(r),-np.inf),np.full(len(r),np.inf),order=16)
        result['quadrature_check_difference']=np.full(len(rows),np.nan)
        result['quadrature_check_difference'][np.flatnonzero(good)[:len(r)]]=abs(hi['pit']-values['full_pit'][:len(r)])
    atomic_npz(path,**result);write_json(marker,dict(sha256=sha(path),n=len(rows)))
    return path


def score_distance(t,p):
    report={};test=t['split']==2
    for name,mask in groups(t).items():
        use=test&mask;ga=use&t['parallax_eligible'];v={}
        for kind in ['Gaia','specdis','spdist','chosen']:
            tr=truth(t,kind);v[kind]=metrics(p['mu'][use],tr[use])
            pp=p['distance_pit_'+kind][use];pp=pp[np.isfinite(pp)]
            v[kind]['pit_histogram']=np.histogram(pp,np.linspace(0,1,11))[0].tolist()
        gp=p['full_pit'][ga];logd=p['full_log_density'][ga]
        bp=p['bin_probability'][ga];observed=np.searchsorted(SNR[1:-1],t['p_corr'][ga]/t['gaia_parallax_error'][ga],side='right')
        v['full_Gaia']=dict(n=int(ga.sum()),negative=int((ga&(t['p_corr']<0)).sum()),
            mean_log_density=float(np.mean(logd)) if ga.any() else None,
            mean_pit=float(np.mean(gp)) if ga.any() else None,low05=float(np.mean(gp<.05)) if ga.any() else None,
            high95=float(np.mean(gp>.95)) if ga.any() else None,pit_histogram=np.histogram(gp,np.linspace(0,1,11))[0].tolist(),
            observed_snr_fraction=np.bincount(observed,minlength=5)/len(observed) if len(observed) else None,
            predicted_snr_fraction=np.mean(bp,axis=0) if len(bp) else None,
            snr_brier=float(np.mean(np.sum((bp-np.eye(5)[observed])**2,axis=1))) if len(bp) else None)
        report[name]=v
    return report


def evaluate_distance(name,model,t,rows,mode,reference_hash):
    global _T,_MODEL,_MODE,_DEST
    dest=OUT/'experiments'/name;dest.mkdir(parents=True,exist_ok=True)
    fingerprint=dict(reference_sha256=reference_hash,identity_sha256=__import__('hashlib').sha256(t['identity'][rows].tobytes()).hexdigest(),
                     draws=128,mode=mode,model_sha256=sha(OUT/'experiments'/(name+'.joblib')),
                     code_sha256=sha(__file__))
    freeze=dest/'freeze.json'
    if freeze.exists():assert json.loads(freeze.read_text())==fingerprint
    else:write_json(freeze,fingerprint)
    if (dest/'scores.json').exists():return json.loads((dest/'scores.json').read_text())
    _T,_MODEL,_MODE,_DEST=t,model,mode,dest
    batches=[(j//512,rows[j:j+512]) for j in range(0,len(rows),512)]
    assert len(batches)>0
    paths=[];start=time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=16,mp_context=mp.get_context('fork')) as pool:
        for j,path in enumerate(pool.map(distance_chunk,batches)):
            paths.append(path)
            if j%20==0:print(name,'distance chunks',j,'/',len(batches),'seconds',round(time.time()-start),flush=True)
    chunks=[load(path) for path in paths];keys=set.intersection(*[set(p) for p in chunks])
    p={k:np.concatenate([v[k] for v in chunks]) for k in keys}
    assert np.array_equal(p['rows'],rows) and np.array_equal(p['identity'],t['identity'][rows])
    ga=t['parallax_eligible'][rows]
    checks=dict(identity_aligned=True,mu_finite=bool(np.isfinite(p['mu']).all()),mu_ordered=bool(np.all(np.diff(p['mu'],axis=1)>=0)),
        gaia_log_density_finite=bool(np.isfinite(p['full_log_density'][ga]).all()),
        gaia_bin_mass=bool(np.allclose(p['bin_probability'][ga].sum(axis=1),1,atol=2e-5)),
        gaia_negative_retained=int((ga&(t['p_corr'][rows]<0)).sum()))
    assert checks['mu_finite'] and checks['mu_ordered'] and checks['gaia_log_density_finite'] and checks['gaia_bin_mass']
    atomic_npz(dest/'evaluation.npz',**p)
    report=dict(n=len(rows),checks=checks,groups=score_distance(subset(t,rows),p),fingerprint=fingerprint,
                evaluation_sha256=sha(dest/'evaluation.npz'),seconds=time.time()-start)
    save_json(dest/'scores.json',report);return report


def save_model(name,model,metadata):
    path=OUT/'experiments'/(name+'.joblib');marker=path.with_suffix('.json')
    if path.exists():
        saved=json.loads(marker.read_text())
        assert sha(path)==saved['sha256']
        assert all(saved[k]==metadata[k] for k in ['reference_sha256','stage','name'])
        return joblib.load(path)
    temp=path.with_suffix('.tmp.joblib');joblib.dump(model,temp);temp.replace(path)
    save_json(marker,dict(metadata,sha256=sha(path)));return model


def main(stage):
    require_compute();(OUT/'experiments').mkdir(exist_ok=True)
    audit=json.loads((OUT/'assembly.json').read_text());assert all(audit['checks'].values())
    assert sha(OUT/'references.npz')==audit['reference_sha256']
    with np.load(OUT/'references.npz') as z:t={k:z[k] for k in KEYS}
    reference_hash=audit['reference_sha256'];result={}
    clean=t['photometric_eligible']&~t['identity_conflict']&~t['nonstellar_risk']
    evaluation=np.isin(t['split'],[1,2])&clean
    distance_evidence=(t['label_source']>0)|t['specdis_valid']|t['spdist_valid']|t['parallax_eligible']
    if stage!='frozen':assert (OUT/'experiments/frozen_complete.json').exists()
    if stage=='matched':assert (OUT/'experiments/expanded_complete.json').exists()
    names=['grz','griz'] if stage=='matched' else ['A','B','C']
    for name in names:
        optical=stage=='matched';tier=np.ones(len(clean),bool) if name in ['A','grz','griz'] else t[name]
        if optical:tier=t['i_usable']
        rows=np.flatnonzero(evaluation&tier)
        drows=np.flatnonzero(evaluation&tier&distance_evidence)
        cltrain=np.flatnonzero((t['split']==0)&t['class_train_eligible']&tier)
        dtrain=np.flatnonzero((t['split']==0)&t['fit_eligible']&tier)
        assert not np.intersect1d(t['identity'][cltrain],t['identity'][rows]).size
        assert not np.intersect1d(t['identity'][dtrain],t['identity'][drows]).size
        prefix=stage+'_'+name
        metadata=dict(reference_sha256=reference_hash,stage=stage,name=name,class_train=len(cltrain),distance_train=len(dtrain))
        # Save all training/evaluation identities before fitting; both matched
        # variants use exactly the same masks and shared four-band Sobol noise.
        ids_path=OUT/'experiments'/(prefix+'_identities.npz')
        expected=dict(class_train_identity=t['identity'][cltrain],distance_train_identity=t['identity'][dtrain],class_eval_identity=t['identity'][rows],distance_eval_identity=t['identity'][drows])
        if stage=='frozen':
            with np.load(ROOT/'data/baseline_v1_3'/(name+'_identities.npz')) as z:expected['class_train_identity']=z['train_identity']
            with np.load(ROOT/'data/baseline_v0/references.npz') as z:
                fit=(z['split']==0)&z['fit_eligible']
                if name!='A':fit &= z[name]
                expected['distance_train_identity']=z['identity'][fit]
            metadata.update(class_train=len(expected['class_train_identity']),distance_train=len(expected['distance_train_identity']),training='frozen historical models; no fit')
            assert not np.intersect1d(expected['class_train_identity'],expected['class_eval_identity']).size
            assert not np.intersect1d(expected['distance_train_identity'],expected['distance_eval_identity']).size
        if ids_path.exists():
            saved=load(ids_path);assert all(np.array_equal(v,saved[k]) for k,v in expected.items())
        else:atomic_npz(ids_path,**expected)
        cp=OUT/'experiments'/(prefix+'_classifier.joblib');dp=OUT/'experiments'/(prefix+'_distance.joblib')
        if stage=='frozen':
            oldc=ROOT/'data/baseline_v1_3'/(name+'_forest.joblib');oldd=ROOT/'data/baseline_v1'/(name+'.joblib')
            clf=save_model(prefix+'_classifier',joblib.load(oldc),dict(metadata,frozen_source=str(oldc),source_sha256=sha(oldc)))
            model=save_model(prefix+'_distance',joblib.load(oldd).wide,dict(metadata,frozen_source=str(oldd),source_sha256=sha(oldd)))
        else:
            old=None
            if not optical:
                with np.load(ROOT/'data/baseline_v1_3/references.npz') as z:
                    old={k:z[k] for k in [*BANDS,*ERR,'identity','split','class_label','class_train_eligible','fit_eligible','Mg','B','C']}
                oldtier=np.ones(len(old['identity']),bool) if name=='A' else old[name]
                oc=np.flatnonzero((old['split']==0)&old['class_train_eligible']&oldtier)
                od=np.flatnonzero((old['split']==0)&old['fit_eligible']&oldtier)
                assert not np.intersect1d(old['identity'][np.r_[oc,od]],t['identity'][rows]).size
                assert not np.intersect1d(old['identity'][np.r_[oc,od]],t['identity'][np.r_[cltrain,dtrain]]).size
                atomic_npz(OUT/'experiments'/(prefix+'_legacy_training.npz'),class_train_identity=old['identity'][oc],distance_train_identity=old['identity'][od])
                metadata.update(legacy_class_train=len(oc),legacy_distance_train=len(od),training='old training plus new train; new cal/test fixed')
            if cp.exists():clf=save_model(prefix+'_classifier',None,metadata)
            else:
                mags=arrays(t,cltrain,optical)[0];labels=t['class_label'][cltrain]
                if old is not None:mags=np.concatenate([arrays(old,oc)[0],mags]);labels=np.r_[old['class_label'][oc],labels]
                if optical:clf=OpticalClassifier(name).fit(mags,labels)
                else:
                    cfg=json.loads((ROOT/'configs/baseline_v1_2.json').read_text())
                    clf=DirectClassifier({'A':2,'B':3,'C':4}[name],'forest',cfg).fit(mags,labels,balanced_weights(labels))
                clf=save_model(prefix+'_classifier',clf,metadata)
                print('trained classifier',prefix,len(labels),flush=True)
            if dp.exists():model=save_model(prefix+'_distance',None,metadata)
            else:
                mags=arrays(t,dtrain,optical)[0];y=t['Mg'][dtrain];labels=t['class_label'][dtrain]
                if old is not None:mags=np.concatenate([arrays(old,od)[0],mags]);y=np.r_[old['Mg'][od],y];labels=np.r_[old['class_label'][od],labels]
                model=OpticalQRF(name) if optical else HistogramQRF({'A':2,'B':3,'C':4}[name])
                model.forest.n_jobs=32;model.fit(mags,y,labels);model.forest.n_jobs=1
                model=save_model(prefix+'_distance',model,metadata)
                print('trained distance',prefix,len(y),flush=True)
        scorepath=OUT/'experiments'/(prefix+'_classification.json')
        if scorepath.exists():classification=json.loads(scorepath.read_text())
        else:
            m,e,ids=arrays(t,rows,optical);prob=clf.predict_proba(m)
            classification={'nominal':class_scores(subset(t,rows),prob)}
            payload=dict(rows=rows,identity=ids,nominal=prob)
            if optical:
                # Predict by bounded chunks with n_jobs8 in the fixed classifier.
                marginal=[]
                for j in range(0,len(rows),4096):
                    chunkpath=OUT/'experiments'/(prefix+f'_class_chunk_{j:07}.npz')
                    chunkmarker=chunkpath.with_suffix('.json')
                    if chunkmarker.exists():
                        assert sha(chunkpath)==json.loads(chunkmarker.read_text())['sha256']
                        chunk=load(chunkpath);assert np.array_equal(chunk['identity'],ids[j:j+4096])
                        marginal.append(chunk['probability'])
                    else:
                        value=clf.predict_marginalized(m[j:j+4096],e[j:j+4096],ids[j:j+4096],128)
                        atomic_npz(chunkpath,identity=ids[j:j+4096],probability=value)
                        write_json(chunkmarker,dict(sha256=sha(chunkpath)))
                        marginal.append(value)
                    if j%40960==0:print(prefix,'classification marginal',j,len(rows),flush=True)
                marginal=np.concatenate(marginal);payload['marginalized']=marginal
                classification['marginalized']=class_scores(subset(t,rows),marginal)
            atomic_npz(OUT/'experiments'/(prefix+'_class_scores.npz'),**payload)
            save_json(OUT/'experiments'/(prefix+'_class_integrity.json'),dict(sha256=sha(OUT/'experiments'/(prefix+'_class_scores.npz')),reference_sha256=reference_hash))
            save_json(scorepath,classification)
        distance=evaluate_distance(prefix+'_distance',model,t,drows,'optical' if optical else 'wide',reference_hash)
        result[name]=dict(metadata=metadata,classification=classification,distance_score_file=str(OUT/'experiments'/(prefix+'_distance')/'scores.json'),n_distance=distance['n'])
        print('completed',prefix,flush=True)
    if stage=='expanded':
        for name in names:
            a=load(OUT/'experiments'/('frozen_'+name+'_identities.npz'));b=load(OUT/'experiments'/('expanded_'+name+'_identities.npz'))
            assert all(np.array_equal(a[k],b[k]) for k in ['class_eval_identity','distance_eval_identity'])
    if stage=='matched':
        a=load(OUT/'experiments/matched_grz_identities.npz');b=load(OUT/'experiments/matched_griz_identities.npz')
        assert all(np.array_equal(v,b[k]) for k,v in a.items())
    save_json(OUT/'experiments'/(stage+'_complete.json'),dict(stage=stage,models=result,reference_sha256=reference_hash,
        scope='Fixed algorithms/hyperparameters; cal2% dwarf budget; no tuning on test; deployed v1 unchanged'))

if __name__=='__main__':main(sys.argv[1])
