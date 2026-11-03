#!/usr/bin/env python3
"""Frozen calibration-only decisions; test summaries never enter selection."""
from baseline_v1_common import *
from photdis.stage1 import require_compute

def criterion(t,rows,mu):
    truth=truth_mu(t,'chosen')[rows];ok=np.isfinite(truth);r=abs(mu[:,2]-truth)
    byclass={str(k):dict(n=int((ok&(t['class_label'][rows]==k)).sum()),
        tail=float(np.mean(r[ok&(t['class_label'][rows]==k)]>1))) for k in range(3)
        if (ok&(t['class_label'][rows]==k)).any()}
    informative=[v['tail'] for v in byclass.values() if v['n']>=CFG['cal_min_class']]
    return dict(n=int(ok.sum()),median_abs=float(np.median(r[ok])),tail=float(np.mean(r[ok]>1)),
        classes=byclass,macro_tail=float(np.mean(informative)) if informative else 1.)

def admissible(candidate,base):
    dw=candidate['classes'].get('0',{});bd=base['classes'].get('0',{})
    return bool(candidate['n']>=100 and
        candidate['tail']<=base['tail']+CFG['cal_overall_tail_guard'] and
        candidate['median_abs']<=base['median_abs']+CFG['cal_median_error_guard_mag'] and
        (dw.get('n',0)<30 or dw['tail']<=bd['tail']+CFG['cal_dwarf_tail_guard']))

def recommend(t,models,route):
    rows=models['A']['rows'];n=len(rows);result={k:v.copy() for k,v in models['A'].items() if k not in ('pdf','centers','rows')}
    result['rows']=rows;result['route']=np.zeros(n,np.int8)
    all_mu=np.full((n,3),np.nan);all_mu[:,0]=result['mu'][:,2]
    for code,name in enumerate(('B','C'),1):
        p=models[name];pos=np.searchsorted(rows,p['rows']);assert np.array_equal(rows[pos],p['rows'])
        all_mu[pos,code]=p['mu'][:,2]
        use=p['supported']&(route[name]['adopt']|~result['supported'][pos]);dest=pos[use]
        for k in ('mu','mu_cal','distance','modes','weights','supported','ambiguous','draws'):
            result[k][dest]=p[k][use]
        result['route'][dest]=code
    result['disagreement']=np.nanmax(all_mu,axis=1)-np.nanmin(all_mu,axis=1)
    return result

def quality_mask(t,rows,p,cuts):
    if cuts is None:return np.zeros(len(rows),bool)
    return (p['supported']&(p['mu'][:,4]-p['mu'][:,0]<=cuts['width95'])&
        (np.max(p['weights'][:,:3],axis=1)>=cuts['weight'])&
        (p['disagreement']<=cuts['disagreement'])&
        (t['optical_error'][rows]<cuts['error']))

def main():
    require_compute();t=load(OLD/'references.npz');chosen={};selection={}
    for name in MODELS:
        comparisons={};ps={}
        for v in ('wide','natural','balanced'):
            p=load(OUT/f'{name}_{v}_evaluation.npz');cal=t['split'][p['rows']]==1
            comparisons[v]=criterion(t,p['rows'][cal],p['mu'][cal]);ps[v]=p
        valid=[v for v in ('natural','balanced') if admissible(comparisons[v],comparisons['wide']) and comparisons[v]['macro_tail']<comparisons['wide']['macro_tail']]
        best=min(valid,key=lambda v:comparisons[v]['macro_tail']) if valid else 'wide'
        p=ps[best];cal=(t['split'][p['rows']]==1)&(t['label_source'][p['rows']]>0)&p['supported']
        truth=truth_mu(t,'chosen')[p['rows']];correction=[];p['mu_cal']=p['mu'].copy()
        for alpha,lo,hi in ((.32,1,3),(.05,0,4)):
            s=np.maximum(p['mu'][cal,lo]-truth[cal],truth[cal]-p['mu'][cal,hi]);n=len(s)
            c=max(0.,float(np.quantile(s,min(1,np.ceil((n+1)*(1-alpha))/n),method='higher'))) if n else 0.
            correction.append(c);p['mu_cal'][:,lo]-=c;p['mu_cal'][:,hi]+=c
        p['mu_cal'][:,0]=np.minimum(p['mu_cal'][:,0],p['mu_cal'][:,1]);p['mu_cal'][:,4]=np.maximum(p['mu_cal'][:,4],p['mu_cal'][:,3])
        np.savez_compressed(OUT/f'{name}_selected_evaluation.npz',**p)
        if name in 'ABC':chosen[name]={k:v for k,v in p.items() if k!='pdf'}
        selection[name]=dict(variant=best,cal_criteria=comparisons,cqr=correction)
        print('Selected',name,best,comparisons,flush=True)
        del ps,p
    route={}
    for name,lower in [('B','A'),('C','B')]:
        if lower=='B' and not route['B']['adopt']:lower='A'
        a=chosen[lower];b=chosen[name];common,ia,ib=np.intersect1d(a['rows'],b['rows'],return_indices=True)
        use=(t['split'][common]==1)&a['supported'][ia]&b['supported'][ib]
        base=criterion(t,common[use],a['mu'][ia][use]);candidate=criterion(t,common[use],b['mu'][ib][use])
        adopt=admissible(candidate,base) and candidate['macro_tail']<=base['macro_tail'] and candidate['median_abs']<base['median_abs']
        route[name]=dict(adopt=bool(adopt),lower=lower,base=base,candidate=candidate)
    p=recommend(t,chosen,route);rows=p['rows'];cal=t['split'][rows]==1
    candidates=[];best=None;best_n=-1
    pc={k:v[cal] for k,v in p.items() if k!='rows'};rc=rows[cal];truth=truth_mu(t,'chosen')[rc]
    for width in (1,2,4,8):
        for weight in (.5,.7,.9):
            for disagreement in (.25,.5,1):
                for error in (.03,.06,.12,.25):
                    cuts=dict(width95=width,weight=weight,disagreement=disagreement,error=error)
                    use=quality_mask(t,rc,pc,cuts)&np.isfinite(truth)
                    counts=[int((use&(t['class_label'][rc]==k)).sum()) for k in range(3)]
                    valid=use.sum()>=200 and min(counts)>=30
                    tails=[float(np.mean(abs(pc['mu'][use&(t['class_label'][rc]==k),2]-truth[use&(t['class_label'][rc]==k)])>1)) if counts[k] else None for k in range(3)]
                    valid=bool(valid and max(x for x in tails if x is not None)<=.1)
                    candidates.append(dict(cuts=cuts,n=int(use.sum()),class_counts=counts,class_tail=tails,eligible=valid))
                    if valid and use.sum()>best_n:best=cuts;best_n=int(use.sum())
    p['research_quality']=quality_mask(t,rows,p,best)
    np.savez_compressed(OUT/'recommended_evaluation.npz',**p)
    write_json(OUT/'selection.json',dict(models=selection,route=route,quality_rule=best,
        quality_candidates=candidates,numerical=json.load(open(OUT/'numerical.json'))['selection'],
        source='cal only; no test metrics used',config_sha256=sha(ROOT/'configs/baseline_v1.json')))
    test=t['split'][rows]==2
    write_json(OUT/'recommended_scores.json',dict(test=score(t,rows[test],p['mu'][test]),
        calibrated_test=score(t,rows[test],p['mu_cal'][test]),
        quality_test=score(t,rows[test&p['research_quality']],p['mu'][test&p['research_quality']]),
        quality_retained_test=int((test&p['research_quality']).sum())))
    print('Frozen route',route,'research quality',best,flush=True)
if __name__=='__main__':main()
