#!/usr/bin/env python3
"""Fixed train/cal/test broad QRF, all references evaluated without type filtering."""
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'.deps')]
import numpy as np
import joblib
from photdis.stage1 import require_compute
from photdis.baseline_v0 import HistogramQRF,BANDS,ERR,distance_quantiles,metrics
from kgiant_distance_inventory import write_json,sha
CFG=json.load(open('configs/baseline_v0.json'));OUT=Path(CFG['output_dir'])
MODELS={'A':(2,'A'),'B':(3,'B'),'C':(4,'C'),'A_on_B':(2,'B'),'A_on_C':(2,'C'),'B_on_C':(3,'C')}

def load():
    with np.load(OUT/'references.npz') as f:return {k:f[k] for k in f.files}
def available(t,tier):return np.ones(len(t['g0']),bool) if tier=='A' else t[tier]
def predict_chunks(model,t,rows):
    parts=[]
    for j in range(0,len(rows),CFG['chunk_size']):
        ix=rows[j:j+CFG['chunk_size']];m=np.column_stack([t[k][ix] for k in BANDS]);e=np.column_stack([t[k][ix] for k in ERR])
        p=model.predict(m,e,t['identity'][ix],draws=CFG['photometry_draws']);p['mu'],p['distance']=distance_quantiles(t['g0'][ix],p['q_Meffective']);parts.append(p)
    return {k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
def truth_mu(t,kind):
    if kind=='Gaia':return np.where(t['anchor_astrometry']&~t['nonstellar_risk'],10-5*np.log10(np.maximum(t['p_corr'],1e-20)),np.nan)
    if kind=='chosen':return np.where(t['label_source']>0,10+5*np.log10(t['ref_distance']),np.nan)
    return 10+5*np.log10(t[kind+'_distance'])
def strata(t,ix):
    m={'all':np.ones(len(ix),bool),'train_precision':t['optical_error'][ix]<.03,'no_W1':~t['B'][ix]}
    for field in np.unique(t['field'][ix]):m['field_'+str(field)]=t['field'][ix]==field
    for lo,hi in [(12,16),(16,18),(18,20),(20,22),(22,25)]:m[f'r_{lo}_{hi}']=(t['r'][ix]>=lo)&(t['r'][ix]<hi)
    for lo,hi in [(0,.03),(.03,.06),(.06,.12),(.12,.25)]:m[f'err_{lo}_{hi}']=(t['optical_error'][ix]>=lo)&(t['optical_error'][ix]<hi)
    for k,name in [(-1,'unknown'),(0,'dwarf'),(1,'evolved'),(2,'transition')]:m['class_'+name]=t['class_label'][ix]==k
    m['nonstellar_risk']=t['nonstellar_risk'][ix]
    return m

def main():
    require_compute();t=load();result={'models':{},'same_star':{},'source':'development spatial holdout; not blind final test'}
    pred={};train=(t['split']==0)&t['fit_eligible'];seen=set()
    for name,(dim,tier) in MODELS.items():
        start=time.time();fit=train&available(t,tier);ids=t['identity'][fit];assert len(ids)>=100
        assert not np.intersect1d(ids,t['identity'][t['split']>0]).size
        if name=='A_on_B':assert np.array_equal(ids,train_ids['B'])
        if name in ('A_on_C','B_on_C'):assert np.array_equal(ids,train_ids['C'])
        if name=='A':train_ids={}
        train_ids[name]=ids
        model_path=OUT/(name+'.joblib')
        if model_path.exists():model=joblib.load(model_path)
        else:
            model=HistogramQRF(dim,seed=CFG['seed'],n_estimators=CFG['n_estimators'],min_samples_leaf=CFG['min_samples_leaf'],max_depth=CFG['max_depth'],bin_width=CFG['distribution_bin_mag']).fit(np.column_stack([t[k][fit] for k in BANDS]),t['Mg'][fit],t['class_label'][fit]);joblib.dump(model,model_path)
        rows=np.flatnonzero(np.isin(t['split'],[1,2])&available(t,tier));p=predict_chunks(model,t,rows)
        p['rows']=rows;p['centers']=model.centers;pred[name]=p
        cal=(t['split'][rows]==1)&(t['label_source'][rows]>0)&p['supported'];truth=truth_mu(t,'chosen')[rows]
        corrections=[]
        for alpha,lo,hi in [(.32,1,3),(.05,0,4)]:
            score=np.maximum(p['mu'][cal,lo]-truth[cal],truth[cal]-p['mu'][cal,hi]);n=len(score)
            corrections.append(max(0.,float(np.quantile(score,min(1,np.ceil((n+1)*(1-alpha))/n),method='higher'))) if n>=100 else 0.)
        p['mu_cal']=p['mu'].copy();p['mu_cal'][:,1]-=corrections[0];p['mu_cal'][:,3]+=corrections[0];p['mu_cal'][:,0]-=corrections[1];p['mu_cal'][:,4]+=corrections[1]
        p['mu_cal'][:,0]=np.minimum(p['mu_cal'][:,0],p['mu_cal'][:,1]);p['mu_cal'][:,4]=np.maximum(p['mu_cal'][:,4],p['mu_cal'][:,3])
        np.savez_compressed(OUT/(name+'_evaluation.npz'),**p)
        test=t['split'][rows]==2;layers=strata(t,rows);layers['supported']=p['supported'];layers['outside']=~p['supported'];layers['ambiguous']=p['ambiguous']
        scores={kind:{layer:metrics(p['mu'][test&mask],truth_mu(t,kind)[rows][test&mask]) for layer,mask in layers.items()} for kind in ['Gaia','specdis','spdist','chosen']}
        # Class scores conditional on known reference labels: empirical, not population probabilities.
        cl=p['class_weights'];den=cl[:,:3].sum(axis=1);known=test&(t['class_label'][rows]>=0)&(den>0)
        cls=np.argmax(cl[:,:3],axis=1);matrix=np.zeros((3,3),int);np.add.at(matrix,(t['class_label'][rows][known],cls[known]),1)
        entry=dict(n_train=int(fit.sum()),n_train_without_W1=int((fit&~t['B']).sum()),train_identity_sha256=__import__('hashlib').sha256(ids.tobytes()).hexdigest(),train_label_counts={str(k):int((fit&(t['label_source']==k)).sum()) for k in (1,2)},n_evaluation=len(rows),n_test=int(test.sum()),n_cal_supported=int(cal.sum()),cqr_expand68_95=corrections,metrics=scores,calibrated_test_chosen=metrics(p['mu_cal'][test],truth[test]),class_confusion=matrix.tolist(),class_known_fraction=float(np.median(den[test])),class_test_n=int(known.sum()),model_sha256=sha(model_path),elapsed_seconds=time.time()-start)
        result['models'][name]=entry;write_json(OUT/'fit_progress.json',result);print(name,'train',fit.sum(),'eval',len(rows),'seconds',time.time()-start,flush=True)
        del model
    # Honest paired tests use equal training IDs and exact same test positions.
    for tag,names in [('W1',['A_on_B','B']),('W1W2',['A_on_C','B_on_C','C'])]:
        ix=pred[names[0]]['rows'];assert all(np.array_equal(ix,pred[n]['rows']) for n in names)
        test=t['split'][ix]==2;result['same_star'][tag]={n:{k:metrics(pred[n]['mu'][test],truth_mu(t,k)[ix][test]) for k in ('Gaia','specdis','spdist','chosen')} for n in names}
    # Cal-only recommendation: compare main models on the same available, supported labels.
    route={};full_mu={};full_support={}
    for n in ('A','B','C'):
        p=pred[n];a=np.full(len(t['g0']),np.nan);a[p['rows']]=p['mu'][:,2];full_mu[n]=a
        a=np.zeros(len(t['g0']),bool);a[p['rows']]=p['supported'];full_support[n]=a
    truth=truth_mu(t,'chosen')
    for tier,previous in [('B','A'),('C','B')]:
        lower=previous if previous=='A' or route.get('B',{}).get('adopt',False) else 'A'
        common=(t['split']==1)&np.isfinite(truth)&np.isfinite(full_mu[tier])&np.isfinite(full_mu[lower])&full_support[tier]&full_support[lower]
        scores={n:float(np.median(abs(full_mu[n][common]-truth[common]))) for n in (lower,tier)}
        route[tier]=dict(n=int(common.sum()),compared_to=lower,median_absolute_mu=scores,adopt=bool(common.sum()>=100 and scores[tier]<scores[lower]))
    result['route']=route;write_json(OUT/'fit_result.json',result)
if __name__=='__main__':main()
