"""Predictive checks that retain one shared population draw across a catalog."""
import numpy as np
from scipy.special import logsumexp,softmax


class SharedPredictive:
    def __init__(self,weights):
        w=np.asarray(weights,float)
        if w.ndim!=1 or not np.isfinite(w).all() or (w<0).any() or not np.isclose(w.sum(),1,atol=1e-12):
            raise ValueError('Normalized shared draw weights required')
        with np.errstate(divide='ignore'):self.logweight=np.log(w)
        self.total=np.zeros(len(w));self.design_total=np.zeros(len(w));self.n=0

    def add(self,log_selected_density,design_weight):
        value=np.asarray(log_selected_density,float);design=np.asarray(design_weight,float)
        if value.ndim!=2 or value.shape[0]!=len(self.total) or design.shape!=(value.shape[1],):raise ValueError('Shared draw x object densities and object weights required')
        if not np.isfinite(value).all() or not np.isfinite(design).all() or (design<=0).any():raise ValueError('Finite densities and positive design weights required')
        self.total+=value.sum(axis=1);self.design_total+=value@design;self.n+=len(design)
        return logsumexp(self.logweight[:,None]+value,axis=0)

    def result(self):
        result=dict(n=self.n,log_density_per_draw=self.total.copy(),design_log_density_per_draw=self.design_total.copy())
        for name,total in [('joint',self.total),('generalized_design_joint',self.design_total)]:
            lw=self.logweight+total;logz=float(logsumexp(lw));w=softmax(lw)
            result[name+'_log_density']=logz;result[name+'_importance_ess']=float(1/(w@w))
            halves=[]
            for j in (0,1):
                prior=logsumexp(self.logweight[j::2])
                if not np.isfinite(prior):halves=[];break
                halves.append(float(logsumexp(lw[j::2])-prior))
            result[name+'_half_log_density_delta']=abs(halves[0]-halves[1]) if halves else np.inf
        return result


def mix_predictive(results,state_weights):
    weights=np.asarray(state_weights,float)
    if len(results)!=len(weights) or not np.isfinite(weights).all() or (weights<0).any() or not np.isclose(weights.sum(),1,atol=1e-12):
        raise ValueError('All states and normalized prior state weights required')
    n=int(results[0]['n'])
    if any(int(r['n'])!=n for r in results):raise ValueError('Same observed catalog required')
    prior=np.concatenate([w*np.asarray(r['draw_weights']) for w,r in zip(weights,results)])
    parity=np.concatenate([np.arange(len(r['draw_weights']))%2 for r in results])
    with np.errstate(divide='ignore'):logprior=np.log(prior)
    summary=dict(n=n)
    for label,key in [('joint','log_density_per_draw'),('generalized_design_joint','design_log_density_per_draw')]:
        total=np.concatenate([r[key] for r in results]);posterior=softmax(logprior+total)
        halves=[]
        for j in (0,1):
            mask=parity==j;den=float(logsumexp(logprior[mask]))
            if not np.isfinite(den):halves=[];break
            halves.append(float(logsumexp(logprior[mask]+total[mask])-den))
        summary[label+'_log_density']=float(logsumexp(logprior+total))
        summary[label+'_importance_ess']=float(1/(posterior@posterior))
        summary[label+'_half_log_density_delta']=abs(halves[0]-halves[1]) if halves else np.inf
        summary[label+'_supported']=bool(summary[label+'_importance_ess']>=100 and summary[label+'_half_log_density_delta']<.2)
    return summary
