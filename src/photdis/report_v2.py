"""Frozen-report metrics and weighted range summaries in explicit physical units."""
import numpy as np


def weighted_quantile(value, probability, weight=None):
    value = np.asarray(value, float)
    probability = np.asarray(probability)
    if weight is None:
        return np.quantile(value, probability)
    weight = np.asarray(weight, float)
    order = np.argsort(value)
    x,w = value[order],weight[order]
    return np.interp(probability,(np.cumsum(w)-.5*w)/w.sum(),x,left=x[0],right=x[-1])


def percent_metrics(mu, reference):
    good = np.isfinite(reference)&np.isfinite(mu).all(axis=1)
    mu,reference = np.asarray(mu)[good],np.asarray(reference)[good]
    if not len(reference):
        return dict(n=0)
    delta = mu[:,2]-reference
    error = 100*np.expm1(np.log(10)*delta/5)
    median = np.median(error)
    return dict(n=len(error),median_pct=median,nmad_pct=1.4826*np.median(abs(error-median)),
                median_abs_pct=np.median(abs(error)),p16_pct=np.quantile(error,.16),
                p84_pct=np.quantile(error,.84),tail20=np.mean(abs(error)>20),
                tail50=np.mean(abs(error)>50),tail1mag=np.mean(abs(delta)>1),
                bias_mag=np.median(delta),coverage68=np.mean((reference>=mu[:,1])&(reference<=mu[:,3])),
                coverage95=np.mean((reference>=mu[:,0])&(reference<=mu[:,4])),
                width68_mag=np.median(mu[:,3]-mu[:,1]),width95_mag=np.median(mu[:,4]-mu[:,0]))


def ranges(value,weight=None):
    value = np.asarray(value,float)
    finite = np.isfinite(value)
    if weight is not None:
        weight = np.asarray(weight)
        finite &= np.isfinite(weight)&(weight>0)
    record = dict(n=len(value),finite=int(finite.sum()),missing=int((~finite).sum()))
    if finite.any():
        q = weighted_quantile(value[finite],np.array([0,.01,.05,.5,.95,.99,1]),None if weight is None else weight[finite])
        record.update(dict(zip(['min','p1','p5','p50','p95','p99','max'],q)))
    return record
