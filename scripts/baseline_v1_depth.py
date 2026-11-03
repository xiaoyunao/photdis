#!/usr/bin/env python3
from baseline_v1_common import *
import joblib
from baseline_v1_predict import load_selected,predict_all
from photdis.baseline_v0 import distance_quantiles
from photdis.stage1 import require_compute

def bootstrap(t,mu,use):
    ids=t['identity'][use];groups=np.unique(ids)
    if len(groups)<10:return None
    lookup=[np.flatnonzero(ids==g) for g in groups];truth=t['truth_mu'][use];q=mu[use];rng=np.random.default_rng(CFG['seed']);draw=[]
    for _ in range(CFG['bootstrap_sources']):
        ix=np.concatenate([lookup[k] for k in rng.integers(len(groups),size=len(groups))]);m=metrics(q[ix],truth[ix])
        draw.append([m[k] for k in ('bias_mag','nmad_mag','tail_1','coverage68','coverage95')])
    return dict(n_sources=len(groups),metrics=['bias','nmad','tail_1','coverage68','coverage95'],ci95=np.quantile(draw,[.025,.975],axis=0).T.tolist())

def main():
    require_compute();t=load(OUT/'depth_input.npz');lock,models=load_selected()
    oldmodels={k:joblib.load(OLD/(k+'.joblib')) for k in 'ABC'};oldfit=json.load(open(OLD/'fit_result.json'))
    n=len(t['identity']);v0=np.full((n,5),np.nan);v1=v0.copy();v1cal=v0.copy();quality=np.zeros(n,bool);branch=np.full(n,-1);route0=np.full(n,-1);route1=route0.copy()
    rows=np.flatnonzero(t['accepted']);supported=np.zeros(n,bool)
    for j in range(0,len(rows),CFG['chunk_size']):
        ix=rows[j:j+CFG['chunk_size']];a={k:v[ix] for k,v in t.items()}
        p,allp=predict_all(a,lock,models);v1[ix]=p['mu'];v1cal[ix]=p['mu_cal'];quality[ix]=p['research_quality'];route1[ix]=p['route'];supported[ix]=p['supported'];branch[ix]=np.argmax(p['weights'][:,:3],axis=1)
        m,e,ids=arrays(t,ix);mu=np.full((len(ix),5),np.nan);sp=np.zeros(len(ix),bool);route=np.zeros(len(ix),int)
        for code,name in enumerate('ABC'):
            use=np.ones(len(ix),bool) if name=='A' else t[name][ix]
            if not use.any():continue
            q=oldmodels[name].predict(m[use],e[use],ids[use],8)
            upgrade=np.ones(use.sum(),bool) if name=='A' else q['supported']&(oldfit['route'][name]['adopt']|~sp[use])
            dest=np.flatnonzero(use)[upgrade];mu[dest]=distance_quantiles(m[use,0],q['q_Meffective'])[0][upgrade];sp[dest]=q['supported'][upgrade];route[dest]=code
        v0[ix]=mu;route0[ix]=route
        print('Depth predictions',j+len(ix),len(rows),flush=True)
    saved=dict(v0_mu=v0,v1_mu=v1,v1_mu_cal=v1cal,research_quality=quality,route_v0=route0,route_v1=route1,branch=branch,supported=supported)
    np.savez(OUT/'depth_predictions.npz',**saved)
    groups={'all':np.ones(n,bool),'no_W1':~t['B']}
    for k in (-1,0,1,2):groups['class_'+str(k)]=t['class_label']==k
    for lo,hi in CFG['depth_r_bins']:groups[f'observed_r_{lo}_{hi}']=(t['r']>=lo)&(t['r']<hi)
    for d in range(4):groups['target_depth_'+str(d)]=t['target_depth']==d
    for k in (0,1,2):
        for lo,hi in CFG['depth_r_bins']:groups[f'class{k}_observed_r_{lo}_{hi}']=(t['class_label']==k)&(t['r']>=lo)&(t['r']<hi)
    result={}
    for key,mask in groups.items():
        use=mask&t['accepted'];known=use&(t['class_label']>=0)
        result[key]=dict(n_draws=int(mask.sum()),n_accepted=int(use.sum()),n_sources=int(len(np.unique(t['identity'][use]))),
            n_research_quality=int((use&quality).sum()),routes={str(k):int((use&(route1==k)).sum()) for k in range(3)},
            branch_agreement=float(np.mean(branch[known]==t['class_label'][known])) if known.any() else None,
            models={name:dict(metrics=metrics(mu[use],t['truth_mu'][use]),source_cluster_ci=bootstrap(t,mu,use)) for name,mu in [('v0',v0),('v1',v1),('v1_cal',v1cal)]})
    write_json(OUT/'depth_result.json',dict(groups=result,failures=dict(no_grz_snr=int((~t['grz_base']).sum()),no_grz_clean=int((~t['grz_clean']).sum()),
        missing_template_wise=int((~t['template_wise_finite']).sum())),
        template_noise_exceeds_target=np.nanmean(t['template_noise_ratio']>1,axis=0).tolist(),
        selection_sha256=sha(OUT/'selection.json'),inputs_sha256=sha(OUT/'depth_input.npz'),
        interpretation='Fixed bright-template flux degradation; not a demonstration that real faint populations or image completeness match.'))
if __name__=='__main__':main()
