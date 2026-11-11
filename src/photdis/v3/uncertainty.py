"""Shared reference-response uncertainty, with explicit approximation provenance.

Case/sky Bayesian-bootstrap weights are common to every SED component. Source
calibration draws reweight the actual continuous reference-mu quadrature before
latent sufficient statistics are accumulated. This is one EM influence update
around the fitted mixture, not independent per-component Gaussian error bars or
a claim of an exact fully refitted mixture posterior.
"""
import copy
import multiprocessing as mp
import numpy as np
from scipy.special import logsumexp, ndtr
from scipy.stats import norm
from .sed import projected_conditionals

_CONTEXT = None


def reference_quadrature_components(mu, channel, observed_mu, sigma, calibration_map, repeats=2):
    """Recover and verify the frozen core/broad Hermite node membership."""
    from numpy.polynomial.hermite import hermgauss
    nodes = mu.shape[1]//repeats
    if nodes*repeats != mu.shape[1]: raise ValueError('Reference quadrature shape mismatch')
    x1, _ = hermgauss(nodes//2); x2, _ = hermgauss(nodes-nodes//2)
    out = np.full(mu.shape, -1, 'i1')
    for c in (1, 2):
        rows = np.flatnonzero(channel == c)
        if not len(rows): continue
        center = observed_mu[rows]+calibration_map[c-1, 0]
        values = np.c_[center[:, None]+np.sqrt(2)*np.hypot(sigma[rows], calibration_map[c-1, 1])[:, None]*x1,
                       center[:, None]+np.sqrt(2)*np.hypot(sigma[rows], 2.)[:, None]*x2]
        order = np.argsort(values, axis=1)
        expected = np.tile(np.take_along_axis(values, order, axis=1), (1, repeats))
        np.testing.assert_allclose(mu[rows], expected, rtol=1e-12, atol=1e-12)
        components = np.broadcast_to(np.r_[np.zeros(len(x1), 'i1'), np.ones(len(x2), 'i1')], values.shape)
        out[rows] = np.tile(np.take_along_axis(components, order, axis=1), (1, repeats))
    return out


def reference_logratio(mu, channel, observed_mu, sigma, calibration_map, calibration_draws,
                       reference_weight, broad_node):
    """All calibrated offset/scatter/broad-fraction parameters enter together."""
    n, nodes = mu.shape; draws = len(calibration_draws)
    result = np.zeros((draws, n, nodes))
    def density(x, center, reported, parameters, broad):
        offset, scatter, outlier = parameters
        core = np.hypot(reported, scatter)
        return np.where(broad, np.log(outlier)+norm.logpdf(x, center+offset, np.hypot(reported, 2.)),
                        np.log1p(-outlier)+norm.logpdf(x, center+offset, core))
    for c in (1, 2):
        rows = np.flatnonzero(channel == c)
        if not len(rows): continue
        if not np.isin(broad_node[rows], [0, 1]).all(): raise ValueError('Missing reference channel node provenance')
        base = density(mu[rows], observed_mu[rows, None], sigma[rows, None], calibration_map[c-1], broad_node[rows])
        for s in range(draws):
            result[s, rows] = density(mu[rows], observed_mu[rows, None], sigma[rows, None], calibration_draws[s, c-1], broad_node[rows])-base
    normalization = logsumexp(result+np.log(reference_weight)[None, :, :], axis=2)
    result -= normalization[:, :, None]
    return result, normalization


def shared_case_weights(identity, cells, draws=32, seed=301801):
    identity = np.asarray(identity); cells = np.asarray(cells)
    if len(np.unique(identity)) != len(identity): raise ValueError('Bootstrap exact identities, never quadrature nodes')
    unique, index = np.unique(cells, return_inverse=True)
    rng = np.random.default_rng(seed)
    sky = rng.dirichlet(np.ones(len(unique)), size=draws)*len(unique)
    case = rng.exponential(size=(draws, len(identity)))
    weights = case*sky[:, index]
    weights *= len(identity)/weights.sum(axis=1)[:, None]
    return weights, unique, sky


def _component(task):
    k, statistics = task
    cloud, model, logratio, case_weights, normalizer = _CONTEXT
    y, noise, mask = cloud['y'], cloud['noise'], cloud['mask']
    labels = cloud['labels']; node_weight = cloud['logweight']
    draws, n, _ = logratio.shape; dim = y.shape[-1]
    values = np.empty((draws, n)) if not statistics else None
    counts = np.zeros(draws); first = np.zeros((draws, dim)); second = np.zeros((draws, dim, dim)); classes = np.zeros((draws, 3))
    bitmask = (mask.astype('i8')*(1 << np.arange(dim))).sum(axis=1)
    for pattern in np.unique(bitmask):
        rows = np.flatnonzero(bitmask == pattern); observed = mask[rows[0]]
        for start in range(0, len(rows), 128):
            rr = rows[start:start+128]
            b, B, likelihood = projected_conditionals(y[rr], noise[rr], observed, model.means[k], model.covariances[k])
            label_log = np.zeros(len(rr)); known = labels[rr] >= 0
            label_log[known] = np.log(model.classes[k, labels[rr][known]])
            joint = likelihood+node_weight[rr]+np.log(model.weights[k])+label_log[:, None]
            weighted = joint[None, :, :]+logratio[:, rr]
            if not statistics:
                values[:, rr] = logsumexp(weighted, axis=2)
                continue
            q = np.exp(weighted-normalizer[:, rr, None])*case_weights[:, rr, None]
            flat = q.reshape(draws, -1)
            counts += q.sum(axis=(1, 2))
            first += flat@b.reshape(-1, dim)
            moment = B+b[..., :, None]*b[..., None, :]
            second += (flat@moment.reshape(-1, dim*dim)).reshape(draws, dim, dim)
            for c in range(3): classes[:, c] += q[:, labels[rr] == c].sum(axis=(1, 2))
    return (counts, first, second, classes) if statistics else values


def one_step_ensemble(model, cloud, logratio, case_weights, *, workers=1):
    global _CONTEXT
    _CONTEXT = (cloud, model, logratio, case_weights, None)
    def run(statistics):
        tasks = [(k, statistics) for k in range(len(model.weights))]
        if workers == 1: return [_component(x) for x in tasks]
        with mp.get_context('fork').Pool(workers) as pool: return pool.map(_component, tasks, chunksize=1)
    likelihood = np.asarray(run(False))
    normalizer = logsumexp(likelihood, axis=0)
    del likelihood
    _CONTEXT = (*_CONTEXT[:-1], normalizer)
    statistics = run(True); _CONTEXT = None
    ensemble = []
    for s in range(len(logratio)):
        draw = copy.deepcopy(model); counts = []
        for k, (count, first, second, classes) in enumerate(statistics):
            if count[s] <= 0: raise RuntimeError('Bootstrap component lost all support')
            counts.append(count[s]); draw.means[k] = first[s]/count[s]
            covariance = second[s]/count[s]-np.outer(draw.means[k], draw.means[k])
            scale = np.maximum(abs(draw.means[k]), .001); scale[-1] = 1.
            normalized = (covariance+covariance.T)/2/scale[:, None]/scale[None, :]
            eigen, vector = np.linalg.eigh(normalized)
            draw.covariances[k] = (vector*np.maximum(eigen, .005**2))@vector.T*scale[:, None]*scale[None, :]
            draw.classes[k] = (classes[s]+.5)/(classes[s].sum()+1.5)
        draw.weights = np.asarray(counts)/sum(counts)
        draw.negative_intrinsic_flux_probability = ndtr(-draw.means[:, :6]/np.sqrt(np.diagonal(draw.covariances[:, :6, :6], axis1=1, axis2=2)))
        draw.uncertainty_method = 'one-step joint source-calibration and sky/identity Bayesian-bootstrap influence update'
        draw.global_draw_id = s
        ensemble.append(draw)
    return ensemble, normalizer
