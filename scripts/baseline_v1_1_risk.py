#!/usr/bin/env python3
"""Prediction-defined risk groups and design-weighted parent coverage."""
from baseline_v1_1_common import *
from photdis.stage1 import require_compute


def states(p):
    w=p['weights'];winner=w.argmax(axis=1);conf=w.max(axis=1);width=p['mu'][:,3]-p['mu'][:,1]
    cb=np.searchsorted([.5,.8],conf,side='right');wb=np.searchsorted([.5,1.,2.],width,side='left')
    supported=p['supported'];out={'all':np.ones(len(w),bool)}
    for k in range(4):out[f'component{k}']=winner==k
    for k in range(3):out[f'confidence{k}']=cb==k
    for k in range(4):out[f'width{k}']=wb==k
    out['supported']=supported;out['unsupported']=~supported
    joint=winner*24+cb*8+wb*2+supported
    for k in range(96):out[f'joint{k:02}']=joint==k
    for c in (.5,.8):
        for width_max in (.5,1.,2.):
            out[f'cut_conf{c}_width{width_max}']=(conf>=c)&(width<=width_max)&supported
    return out


def wilson(k,n):
    if n==0:return None
    z=1.959963984540054;p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [float(max(0,center-half)),float(min(1,center+half))]


def estimate(sample,selected,domain=None):
    """Stratified SRS ratio estimate; finite-population sampling error only."""
    if domain is None:domain=np.ones(len(selected),bool)
    w=sample['sample_weight'];total=float(w[domain].sum())
    if total==0:return dict(fraction=None,se=None,n_sample=0)
    fraction=float(w[selected&domain].sum()/total);variance=0.
    y=(selected&domain).astype(float)-fraction*domain
    for field in np.unique(sample['field']):
        use=sample['field']==field;n=int(use.sum());size=float(w[use].sum())
        if n>1:variance+=size**2*max(0,1-n/size)*np.var(y[use],ddof=1)/n
    se=float(np.sqrt(variance)/total)
    return dict(fraction=fraction,se=se,ci95=[max(0,fraction-1.96*se),min(1,fraction+1.96*se)],
                n_sample=int((selected&domain).sum()),domain_sample=int(domain.sum()),estimated_count=float(w[selected&domain].sum()))


def risk(t,p,wide,mask,sp,source):
    rows=p['rows'];truth=truth_mu(t,source)[rows];pool=(t['split'][rows]==sp)&np.isfinite(truth)
    use=mask&pool;known=t['class_label'][rows];res=p['mu'][:,2]-truth
    n=int(use.sum());m=metrics(p['mu'][use],truth[use]);m.update(retention=n/int(pool.sum()) if pool.any() else None,
        class_counts={str(k):int((use&(known==k)).sum()) for k in (-1,0,1,2)},
        tail_1_ci95=wilson(int((use&(abs(res)>1)).sum()),n),wide_same_objects=metrics(wide['mu'][use],truth[use]))
    return m


def main():
    require_compute();t=load(OLD/'references.npz');sample=load(OUT/'parent_sample.npz');out={}
    for name in 'ABC':
        p=load(OUT/f'{name}_evidence_evaluation.npz');wide=load(OUT/f'{name}_wide_evaluation.npz')
        ps=load(OUT/f'parent_{name}_evidence.npz');groups=states(p);parentgroups=states(ps);entries={}
        for group,mask in groups.items():
            # Joint groups retain every contaminating true class. Cal determines
            # whether group evidence is sufficient; test never changes this flag.
            entry={}
            for source in ('chosen','Gaia','specdis','spdist'):
                cal=risk(t,p,wide,mask,1,source);test=risk(t,p,wide,mask,2,source)
                entry[source]=dict(cal=cal,test=test,cal_sufficient=cal['n']>=CFG['cal_min_class'])
            expanded=np.zeros(len(sample['identity']),bool);expanded[ps['rows']]=parentgroups[group]
            domain=np.zeros(len(expanded),bool);domain[ps['rows']]=True
            entry['parent']=estimate(sample,expanded)
            entry['parent_given_band']=estimate(sample,expanded,domain);entries[group]=entry
        rows=p['rows'];matrix=np.zeros((3,4),int);known=(t['class_label'][rows]>=0)&(t['split'][rows]==2)
        np.add.at(matrix,(t['class_label'][rows][known],p['weights'][known].argmax(axis=1)),1)
        out[name]=dict(groups=entries,test_confusion=matrix.tolist(),
            state_definition='evidence component/score, W68 and wide nominal support; scores are not population probabilities')
    write_json(OUT/'risk.json',out)
    print('Risk states and weighted parent estimates complete',flush=True)
if __name__=='__main__':main()
