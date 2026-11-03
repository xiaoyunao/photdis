#!/usr/bin/env python3
"""Cal-fixed ranking workpoints and distance-decision risk, with all contaminants."""
from baseline_v1_2_run import *
from photdis.baseline_v1_2 import conservative_threshold,single_mask,fraction_interval
from baseline_v1_1_risk import estimate
from sklearn.metrics import roc_curve,precision_recall_curve,roc_auc_score,average_precision_score

_BASE_TRUTH=truth_mu
_TRUTH_CACHE={}
def truth_mu(t,source):
    key=(id(t),source)
    if key not in _TRUTH_CACHE:_TRUTH_CACHE[key]=_BASE_TRUTH(t,source)
    return _TRUTH_CACHE[key]


def counts(labels,selected,target,domain):
    use=selected&domain;known=labels>=0;positive=domain&(labels==target);dwarf=domain&(labels==0)
    hit=int((use&(labels==target)).sum());false=int((use&(labels==0)).sum())
    npos=int(positive.sum());nd=int(dwarf.sum());nk=int((use&known).sum())
    return dict(n_domain=int(domain.sum()),n_selected=int(use.sum()),
        class_counts={str(k):int((use&(labels==k)).sum()) for k in (-1,0,1,2)},
        recall=hit/npos if npos else None,recall_ci95=fraction_interval(hit,npos),
        dwarf_fpr=false/nd if nd else None,dwarf_fpr_ci95=fraction_interval(false,nd),
        known_precision=hit/nk if nk else None,n_target=npos,n_dwarf=nd)


def distances(t,rows,p,wide,mask):
    out={}
    for ref in ('chosen','Gaia','specdis','spdist'):
        truth=truth_mu(t,ref)[rows];use=mask&np.isfinite(truth)
        out[ref]={kind:metrics(mu[use],truth[use]) for kind,mu in [('mixture',p['mu']),('main',p['main_mu']),('wide',wide['mu'])]}
        out[ref]['class_counts']={str(k):int((use&(t['class_label'][rows]==k)).sum()) for k in (-1,0,1,2)}
    return out


def subgroup_stats(t,rows,p,wide,sel,labels,target):
    test=t['split'][rows]==2;groups=strata(t,rows)
    color=t['g0'][rows]-t['r0'][rows]
    for lo,hi in [(-np.inf,.5),(.5,1.),(1.,1.5),(1.5,np.inf)]:groups[f'gr{lo}_{hi}']=(color>=lo)&(color<hi)
    result={}
    truth=truth_mu(t,'chosen')[rows]
    for name,mask in groups.items():
        domain=test&mask;take=sel&domain
        result[name]=counts(labels,sel,target,domain)
        result[name]['chosen_distance']={kind:metrics(mu[take],truth[take]) for kind,mu in [('mixture',p['mu']),('main',p['main_mu']),('wide',wide['mu'])]}
    return result


def curves(labels,scores,target,mask):
    known=mask&(labels>=0);y=labels[known]==target;s=scores[known]
    if len(np.unique(y))<2:return dict(n=int(known.sum()))
    # ROC/PR one-vs-rest includes transition/dwarf negatives as appropriate.
    fpr,tpr,_=roc_curve(y,s);precision,recall,_=precision_recall_curve(y,s)
    ix=np.unique(np.linspace(0,len(fpr)-1,min(len(fpr),201)).astype(int));jx=np.unique(np.linspace(0,len(precision)-1,min(len(precision),201)).astype(int))
    return dict(n=len(y),auc=float(roc_auc_score(y,s)),average_precision=float(average_precision_score(y,s)),
        fpr=fpr[ix],tpr=tpr[ix],precision=precision[jx],recall=recall[jx])


def score_model(name):
    require_compute();t=load(OLD/'references.npz');sample=load(PREV/'parent_sample.npz')
    wide=load(PREV/f'{name}_wide_evaluation.npz');results={}
    for variant in VARS:
        p=load(OUT/f'reference_{name}_{variant}.npz');rows=p['rows'];assert np.array_equal(rows,wide['rows'])
        labels=t['class_label'][rows];cal=t['split'][rows]==1;test=t['split'][rows]==2
        parent=load(OUT/f'parent_{name}_{variant}.npz') if name in 'ABC' else None
        entry=dict(working_points={},curves={},single_grid={},all_distances=distances(t,rows,p,wide,test))
        for state in ('raw','final'):
            weights=p['raw_weights' if state=='raw' else 'weights']
            parentw=parent['raw_weights' if state=='raw' else 'weights'] if parent is not None else None
            for target in (1,2):
                score=weights[:,target];key=f'{state}_class{target}'
                entry['curves'][key]={tag:curves(labels,score,target,mask) for tag,mask in [('cal',cal),('test',test)]}
                for budget in CFG['false_positive_budgets']:
                    threshold=conservative_threshold(score[cal&(labels==0)],budget);sel=score>=threshold
                    record=dict(threshold=threshold,budget=budget,target=target,score_state=state,
                        cal=counts(labels,sel,target,cal),test=counts(labels,sel,target,test),
                        distances=distances(t,rows,p,wide,sel&test),strata=subgroup_stats(t,rows,p,wide,sel,labels,target))
                    if parent is not None:
                        expanded=np.zeros(len(sample['identity']),bool);expanded[parent['rows']]=parentw[:,target]>=threshold
                        record['parent']=estimate(sample,expanded)
                    entry['working_points'][f'{key}_budget{budget}']=record
        # Same photometrically-defined groups for every true label, including unknown.
        choices=[]
        for confidence in CFG['confidence_grid']:
            for margin in CFG['margin_grid']:
                key=f'confidence{confidence}_margin{margin}'
                mask=single_mask(p['weights'],p['branch_support'],p['supported'],confidence,margin)
                scores={tag:distances(t,rows,p,wide,mask&sp) for tag,sp in [('cal',cal),('test',test)]}
                ok=scores['cal']['chosen']['main'].get('tail_1',1)<=CFG['single_max_tail']
                class_scores={}
                truth=truth_mu(t,'chosen')[rows]
                for k in range(3):
                    use=mask&cal&(labels==k);cm=metrics(p['main_mu'][use],truth[use]);wm=metrics(wide['mu'][use],truth[use])
                    passed=cm['n']>=CFG['single_min_class'] and cm.get('tail_1',1)<=wm.get('tail_1',0)+CFG['single_class_tail_guard']
                    class_scores[str(k)]=dict(main=cm,wide=wm,passed=passed);ok=ok and passed
                record=dict(cal_acceptable=bool(ok),cal_class_scores=class_scores,**scores,
                    n_cal_selected=int((mask&cal).sum()),n_test_selected=int((mask&test).sum()))
                if parent is not None:
                    pm=single_mask(parent['weights'],parent['branch_support'],parent['supported'],confidence,margin)
                    expanded=np.zeros(len(sample['identity']),bool);expanded[parent['rows']]=pm
                    record['parent']=estimate(sample,expanded)
                entry['single_grid'][key]=record
                if ok:choices.append((record['n_cal_selected'],key))
        entry['cal_single_rule']=max(choices)[1] if choices else None
        # Distance improvements for all true classes, without post-hoc correctness selection.
        entry['test_by_class']={str(k):distances(t,rows,p,wide,test&(labels==k)) for k in (-1,0,1,2)}
        results[variant]=entry;print('Scored',name,variant,flush=True)
    write_json(OUT/f'{name}_scores.json',results)

if __name__=='__main__':score_model(sys.argv[1])
