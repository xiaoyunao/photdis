"""Low-parameter forward RGB color model with shared-band errors."""
from __future__ import annotations

import numpy as np

RG, RR, RZ = 3.214, 2.165, 1.211


def basis(mg, feh):
    m = np.asarray(mg, float) - 1.0
    z = np.broadcast_to(np.asarray(feh, float), m.shape) + 1.7
    return np.stack((np.ones_like(m), m, m*m, z), axis=-1)


def residual_measurement_cov(sg, sr, sz, slope, ncolor):
    """Covariance of c_obs - c_model(Mg_obs) for shared measured bands."""
    sg2, sr2 = np.asarray(sg)**2, np.asarray(sr)**2
    s1 = np.asarray(slope)[..., 0]
    v1 = sg2 + sr2 + s1*s1*sg2 - 2*s1*sg2
    if ncolor == 1:
        return v1[..., None, None]
    sz2 = np.asarray(sz)**2
    s2 = np.asarray(slope)[..., 1]
    v2 = sr2 + sz2 + s2*s2*sg2
    c12 = -sr2 - s2*sg2 + s1*s2*sg2
    out = np.empty(v1.shape + (2, 2), float)
    out[..., 0, 0] = v1
    out[..., 1, 1] = v2
    out[..., 0, 1] = out[..., 1, 0] = c12
    return out


def model_mean(beta, mg, feh):
    return basis(mg, feh) @ np.asarray(beta).T


def model_slope(beta, mg):
    b = np.asarray(beta)
    m = np.asarray(mg, float) - 1.0
    return b[:, 1] + 2*m[..., None]*b[:, 2]


def stable_cov(a, floor):
    eig, vec = np.linalg.eigh((a+a.T)/2)
    return (vec*np.maximum(eig, floor*floor)) @ vec.T


def fit_gls(mg, feh, color, sg, sr, sz, cluster, *, ridge=4.0, floor=.01):
    """Iterated GLS with equal total cluster weight and a fixed Fe ridge."""
    mg, feh, color = np.asarray(mg), np.asarray(feh), np.asarray(color)
    n, nc = color.shape
    x = basis(mg, feh)
    unique, inverse, counts = np.unique(cluster, return_inverse=True, return_counts=True)
    weight = 1.0 / (len(unique)*counts[inverse])
    beta = np.stack([np.linalg.lstsq(x, color[:, j], rcond=None)[0]
                     for j in range(nc)])
    intrinsic = np.eye(nc)*(.03**2)
    for _ in range(5):
        slope = model_slope(beta, mg)
        measurement = residual_measurement_cov(sg, sr, sz, slope, nc)
        inv = np.linalg.inv(measurement + intrinsic)
        normal = np.einsum("n,njk,na,nb->jakb", weight, inv, x, x).reshape(4*nc, 4*nc)
        rhs = np.einsum("n,njk,nk,na->ja", weight, inv, color, x).reshape(4*nc)
        for j in range(nc):
            normal[j*4+3,j*4+3] += ridge
        beta = np.linalg.solve(normal, rhs).reshape(nc, 4)
        resid = color-model_mean(beta, mg, feh)
        empirical = (resid.T*weight) @ resid / weight.sum()
        typical_measurement = np.einsum("n,nij->ij",weight,measurement)/weight.sum()
        intrinsic = stable_cov(empirical-typical_measurement, floor)
    return beta, intrinsic


def log_gaussian(residual, cov):
    residual = np.asarray(residual)
    cov = np.asarray(cov)
    sign, logdet = np.linalg.slogdet(cov)
    if np.any(sign <= 0):
        raise ValueError("nonpositive predictive covariance")
    inv_resid = np.linalg.solve(cov, residual[..., None])[..., 0]
    return -.5*(np.sum(residual*inv_resid,axis=-1)+logdet
                + residual.shape[-1]*np.log(2*np.pi))


def support_bins(data, edges, minimum):
    """Return stage-specific cells shared by M53 and another development system."""
    out = {}
    for stage, companion in (("upper_rgb", "ngc7006"), ("lower_rgb", "m3")):
        a = data[("m53", stage)]
        b = data[(companion, stage)]
        valid = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            na = np.sum((a >= lo) & (a < hi))
            nb = np.sum((b >= lo) & (b < hi))
            if na >= minimum and nb >= minimum:
                valid.append([float(lo), float(hi), int(na), int(nb)])
        out[stage] = valid
    return out


def grid_from_bins(cells, step):
    parts = [np.arange(lo+step/2, hi, step) for lo,hi,*_ in cells]
    return np.concatenate(parts) if parts else np.empty(0)


def posterior_on_grid(models, grid, feh_components, feh_weights, observed,
                      sg, sr, sz):
    """Uniform-Mg prior on the registered grid, averaging shared-nuisance fits."""
    if len(grid) == 0:
        return np.empty(0)
    nc = len(observed)
    terms = []
    for model in models:
        beta = np.asarray(model["beta"])
        intrinsic = np.asarray(model["intrinsic"])
        for z, weight in zip(feh_components, feh_weights):
            slope = model_slope(beta, grid)
            cov = residual_measurement_cov(sg, sr, sz, slope, nc)+intrinsic
            mean = model_mean(beta, grid, z)
            ll = log_gaussian(np.asarray(observed)-mean,cov)
            terms.append(ll + np.log(weight/len(models)))
    values = np.stack(terms)
    peak = np.max(values)
    likelihood = np.exp(values-peak).sum(axis=0)
    return likelihood / likelihood.sum()


def quantiles(grid, probability, percentiles=(.025,.16,.5,.84,.975)):
    cdf = np.cumsum(probability)
    return np.asarray(grid)[np.clip(np.searchsorted(cdf, percentiles),0,len(grid)-1)]
