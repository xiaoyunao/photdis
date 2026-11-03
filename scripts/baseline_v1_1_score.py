#!/usr/bin/env python3
"""Reference scores, matched fallback decomposition, and cal-only adoption."""
from baseline_v1_1_common import *
from scipy.special import ndtr
import joblib
from photdis.stage1 import require_compute

def source_sigma(t,source):
    if source=='chosen':return t['sigma_mu']
    if source=='Gaia':return 5/np.log(10)*t['gaia_parallax_error']/np.maximum(t['p_corr'],1e-20)
    return t[source+'_sigma_mu']

def logscore(t,rows,p,source):
    truth=truth_mu(t,source)[rows];s=source_sigma(t,source)[rows]
    ok=np.isfinite(truth)&np.isfinite(s)&(s>0);answer=np.full(len(rows),np.nan)
    edges=p['edges'];y=t['g0'][rows][ok]-truth[ok]
    z=(edges[None,:]-y[:,None])/s[ok,None];a,b=z[:,:-1],z[:,1:]
    kernel=np.where(a>0,ndtr(-a)-ndtr(-b),ndtr(b)-ndtr(a))/np.diff(edges)
    density=np.sum(p['pdf'][ok]*kernel,axis=1)
    answer[ok]=np.log(np.maximum(density,1e-300))
    return answer

def extended_scores(t,p):
    rows=p['rows'];layers=strata(t,rows);result={}
    for source in ('chosen','Gaia','specdis','spdist'):
        truth=truth_mu(t,source)[rows];ls=p.get('log_'+source)
        if ls is None:ls=logscore(t,rows,p,source)
        by_split={}
        for sp,tag in [(1,'cal'),(2,'test')]:
            out={}
            for layer,mask in layers.items():
                use=mask&(t['split'][rows]==sp)&np.isfinite(truth)
                m=metrics(p['mu'][use],truth[use]);valid=use&np.isfinite(ls)
                m['mean_log_density']=float(np.mean(ls[valid])) if valid.any() else None
                m['n_log_density']=int(valid.sum());out[layer]=m
            by_split[tag]=out
        result[source]=by_split
    return result

def decomposition(t,p,variant):
    rows=p['rows'];result={};labels=t['class_label'][rows]
    for source in ('chosen','Gaia','specdis','spdist'):
        truth=truth_mu(t,source)[rows]
        for sp,tag in [(1,'cal'),(2,'test')]:
            for cl in (-1,0,1,2):
                use=(t['split'][rows]==sp)&(labels==cl)&np.isfinite(truth)
                if not use.any():continue
                w=p['weights'][use];bs=p['branch_support'][use]
                result[f'{source}:{tag}:class{cl}']=dict(n=int(use.sum()),
                    unknown_fraction_quantiles=np.quantile(p['unknown_fraction'][use],[.1,.5,.9]).tolist(),
                    branch_support_failure_fraction=np.mean(~bs,axis=0).tolist(),
                    true_branch_support_failure_fraction=float(np.mean(~bs[:,cl])) if cl>=0 else None,
                    mean_support_fallback=float(np.mean(p['support_fallback'][use])),
                    mean_residual_weight=1. if variant=='wide' else float(np.mean(w[:,3])),
                    predicted_component_counts=([0,0,0,int(use.sum())] if variant=='wide' else np.bincount(np.argmax(w,axis=1),minlength=4).tolist()),
                    metrics=metrics(p['mu'][use],truth[use]))
    return result

def adoption(old,new,optimization):
    checks={'optimizer_converged':bool(optimization['success'])};classes=['class_dwarf','class_evolved','class_transition']
    checks['class_sample']=all(old[k]['n']>=CFG['cal_min_class'] and new[k]['n']==old[k]['n'] for k in classes)
    if checks['class_sample']:
        gain=np.mean([old[k]['tail_1']-new[k]['tail_1'] for k in classes]);checks['macro_improvement']=gain>=CFG['cal_macro_tail_improvement']
        for k in ['all','class_dwarf']:checks[k+'_tail']=new[k]['tail_1']<=old[k]['tail_1']+CFG['cal_tail_guard']
        for k in classes:
            checks[k+'_coverage95']=new[k]['coverage95']>=old[k]['coverage95']-CFG['cal_coverage95_guard']
            checks[k+'_width95']=new[k]['width95']<=old[k]['width95']*CFG['cal_width95_ratio_guard']
        checks['log_density']=new['all']['mean_log_density']>=old['all']['mean_log_density']-CFG['cal_logscore_guard']
    else:gain=None
    return dict(adopt=all(checks.values()),checks=checks,macro_tail_gain=gain)

def route_scores(t,selection):
    # Keep the v1 band policy and nominal support decisions fixed for all variants.
    lock=json.load(open(EXPERT/'selection.json'));base=load(OUT/'A_wide_evaluation.npz')
    rows=base['rows'];route=np.zeros(len(rows),int);supported=base['supported'].copy();positions={}
    for code,name in enumerate('BC',1):
        p=load(OUT/f'{name}_wide_evaluation.npz');pos=np.searchsorted(rows,p['rows']);positions[name]=pos
        assert np.array_equal(rows[pos],p['rows'])
        use=p['supported']&(lock['route'][name]['adopt']|~supported[pos])
        route[pos[use]]=code;supported[pos[use]]=p['supported'][use]
    result={}
    for variant in ('wide','evidence','selected'):
        v=selection['A']['variant'] if variant=='selected' else variant
        p=load(OUT/f'A_{v}_evaluation.npz')
        out={k:a.copy() for k,a in p.items() if k not in ('pdf','centers','edges')}
        mids=np.full((len(rows),3),np.nan);mids[:,0]=p['mu'][:,2]
        for source in ('chosen','Gaia','specdis','spdist'):out['log_'+source]=logscore(t,rows,p,source)
        for code,name in enumerate('BC',1):
            v=selection[name]['variant'] if variant=='selected' else variant
            p=load(OUT/f'{name}_{v}_evaluation.npz');pos=positions[name];use=route[pos]==code
            mids[pos,code]=p['mu'][:,2]
            for k in out:
                if not k.startswith('log_'):out[k][pos[use]]=p[k][use]
            for source in ('chosen','Gaia','specdis','spdist'):out['log_'+source][pos[use]]=logscore(t,p['rows'],p,source)[use]
        out['disagreement']=np.nanmax(mids,axis=1)-np.nanmin(mids,axis=1)
        out['route']=route;np.savez_compressed(OUT/f'routed_{variant}_evaluation.npz',**out)
        result[variant]=extended_scores(t,out)
    write_json(OUT/'routed_scores.json',dict(scores=result,route_counts=np.bincount(route,minlength=3).tolist(),policy='v1 frozen band route and support; no new routing optimization'))


def main():
    require_compute();t=load(OLD/'references.npz');result={};selection={}
    for name in MODELS:
        scored={};parts={}
        for variant in VARIANTS:
            p=load(OUT/f'{name}_{variant}_evaluation.npz')
            scored[variant]=extended_scores(t,p);parts[variant]=decomposition(t,p,variant)
            if variant=='wide':old=p
            if variant=='evidence':
                new=p;raw=joblib.load(OUT/f'{name}_gate.joblib').predict_proba(arrays(t,p['rows'])[0])
                entropy=-np.sum(raw*np.log(np.maximum(raw,1e-300)),axis=1)
                np.savez_compressed(OUT/f'{name}_weight_diagnostics.npz',rows=p['rows'],identity=p['identity'],raw_weights=raw,
                    gate_entropy=entropy,final_weights=p['weights'],support_fallback=p['support_fallback'],
                    label_missing_fraction=p['unknown_fraction'],known_label_condition_defined=p['unknown_fraction']<1-1e-7,
                    known_label_conditional_max=p['known_condmax'],known_label_conditional_entropy=p['known_entropy'])
        train=json.load(open(OUT/f'{name}_train.json'))
        choice=adoption(scored['wide']['chosen']['cal'],scored['evidence']['chosen']['cal'],train['optimization'])
        rows=old['rows'];truth=truth_mu(t,'chosen')[rows]
        cal=(t['split'][rows]==1)&np.isfinite(truth)
        ae={k:float(np.median(abs(p['mu'][cal,2]-truth[cal]))) for k,p in [('wide',old),('evidence',new)]}
        choice['checks']['median_absolute_error']=ae['evidence']<=ae['wide']+CFG['cal_median_error_guard']
        choice['adopt']=all(choice['checks'].values());choice['median_absolute_error']=ae
        choice['variant']='evidence' if choice['adopt'] else 'wide';selection[name]=choice
        result[name]=dict(scores=scored,decomposition=parts,training=train)
        print('Scored',name,choice,flush=True)
    write_json(OUT/'selection.json',dict(models=selection,source='cal only; no test-based tuning',
        config_sha256=sha(ROOT/'configs/baseline_v1_1.json'),frozen_v1_selection_sha256=sha(EXPERT/'selection.json')))
    write_json(OUT/'reference_scores.json',result)
    route_scores(t,selection)

if __name__=='__main__':main()
