"""Paired, design-weighted diagnostic contrasts with sky-block uncertainty."""
import numpy as np


def paired_mean(values, weights, cells, *, repeats=1024, seed=311302):
    x, w, cells = np.asarray(values,float), np.asarray(weights,float), np.asarray(cells)
    if x.ndim != 1 or w.shape != x.shape or cells.shape != x.shape:
        raise ValueError('Aligned one-dimensional paired values, weights and cells required')
    if not np.isfinite(x).all() or not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError('Finite paired differences and positive design weights required')
    if len(x) == 0: return dict(n=0,cells=0,mean=None,lo95=None,hi95=None,effective_n=0.)
    groups, inverse = np.unique(cells,return_inverse=True)
    numerator = np.bincount(inverse,weights=x*w); denominator = np.bincount(inverse,weights=w)
    result = dict(n=len(x),cells=len(groups),mean=float(w@x/w.sum()),
        effective_n=float(w.sum()**2/(w@w)),lo95=None,hi95=None)
    if len(groups) > 1:
        draw = np.random.default_rng(seed).integers(0,len(groups),(repeats,len(groups)))
        estimates = numerator[draw].sum(axis=1)/denominator[draw].sum(axis=1)
        result.update(lo95=float(np.quantile(estimates,.025)),hi95=float(np.quantile(estimates,.975)))
    return result


def match_controls(records, control, baseline='response12'):
    """Join by exact identity, require the same sampling design and sky ownership."""
    indexed = {}
    for r in records:
        key = (r['identity'],r['control'])
        if key in indexed: raise ValueError('Duplicate diagnostic identity/control')
        indexed[key] = r
    pairs = []
    for r in records:
        if r['control'] != control: continue
        key = (r['identity'],baseline)
        if key not in indexed: raise ValueError('Missing same-identity baseline')
        base = indexed[key]
        for name in ('TYPE','cell','stratum','inclusion_probability','class_label'):
            if r[name] != base[name]: raise ValueError('Paired sample mismatch: '+name)
        if r['TYPE'] != 'PSF': raise ValueError('Native PSF required')
        pairs.append((r,base))
    return pairs
