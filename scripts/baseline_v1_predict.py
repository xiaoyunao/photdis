"""Deploy only frozen photometric decisions; also used by the stress experiment."""
from baseline_v1_common import *
import joblib
from photdis.baseline_v1 import integrate

def load_selected():
    lock=json.load(open(OUT/'selection.json'));models={}
    for name in 'ABC':
        base=OLD if lock['models'][name]['variant']=='wide' else OUT
        models[name]=joblib.load(base/(name+'.joblib'))
    return lock,models

def predict_all(t,lock,models):
    n=len(t['identity']);pred={};m,e,ids=arrays(t,np.arange(n))
    for name in 'ABC':
        use=np.ones(n,bool) if name=='A' else t[name];ix=np.flatnonzero(use)
        p={}
        for layer,mask in [('low',t['optical_error'][ix]<.03),('high',t['optical_error'][ix]>=.03)]:
            if not mask.any():continue
            rows=ix[mask];variant=lock['models'][name]['variant'];ndraw=lock['numerical'][layer]['draws']
            values=integrate(models[name],m[rows],e[rows],ids[rows],ndraw,variants=(variant,),keep_pdf=False)[variant]
            for k,a in values.items():
                if k not in p:p[k]=np.zeros((n,)+a.shape[1:],a.dtype) if a.dtype.kind=='b' else np.full((n,)+a.shape[1:],np.nan)
                p[k][rows]=a
        if not len(ix):
            p=dict(mu=np.full((n,5),np.nan),distance=np.full((n,5),np.nan),modes=np.full((n,4),np.nan),
                   weights=np.full((n,4),np.nan),supported=np.zeros(n,bool),ambiguous=np.zeros(n,bool))
        c=lock['models'][name]['cqr'];p['mu_cal']=p['mu'].copy()
        p['mu_cal'][:,1]-=c[0];p['mu_cal'][:,3]+=c[0];p['mu_cal'][:,0]-=c[1];p['mu_cal'][:,4]+=c[1]
        p['mu_cal'][:,0]=np.minimum(p['mu_cal'][:,0],p['mu_cal'][:,1]);p['mu_cal'][:,4]=np.maximum(p['mu_cal'][:,4],p['mu_cal'][:,3])
        pred[name]=p
    chosen={k:v.copy() for k,v in pred['A'].items()};chosen['route']=np.zeros(n,'i1')
    for code,name in enumerate('BC',1):
        use=t[name]&pred[name]['supported']&(lock['route'][name]['adopt']|~chosen['supported'])
        for k in pred[name]:chosen[k][use]=pred[name][k][use]
        chosen['route'][use]=code
    mids=np.column_stack([pred[k]['mu'][:,2] for k in 'ABC']);chosen['disagreement']=np.nanmax(mids,axis=1)-np.nanmin(mids,axis=1)
    from baseline_v1_select import quality_mask
    chosen['research_quality']=quality_mask(t,np.arange(n),chosen,lock['quality_rule'])
    chosen['draws']=np.where(t['optical_error']<.03,lock['numerical']['low']['draws'],lock['numerical']['high']['draws']).astype('i2')
    chosen['numerical_converged']=np.where(t['optical_error']<.03,lock['numerical']['low']['converged'],lock['numerical']['high']['converged'])
    return chosen,pred
