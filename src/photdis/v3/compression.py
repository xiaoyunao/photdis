"""Auditable joint component/binary/continuous-distance posterior compression.

The stored prior is joint: reference SED weight, binary weight and proper mu
prior. Class probabilities live in the versioned SED model. CDF compression is
conditional on each retained branch, so small distant modes are not merged into
one Gaussian or a point distance. Extreme reweighting outside verified support
must be rejected using the omitted-mass bound.
"""
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm
from .inference import distance_basis

DISTANCE_MU_BOUNDARIES = 10+5*np.log10(np.array([2., 5., 10., 20., 50.]))


def cdf_knots(x, cdf, tolerance=2e-4, mandatory=()):
    """Piecewise linear monotone CDF with a verified discrete sup-norm bound."""
    x = np.asarray(x); y = np.asarray(cdf)
    keep = {0, len(x)-1}
    keep.update(int(np.argmin(abs(x-v))) for v in mandatory if x[0] < v < x[-1])
    initial = sorted(keep); stack = list(zip(initial[:-1], initial[1:]))
    while stack:
        lo, hi = stack.pop()
        if hi <= lo+1: continue
        predicted = y[lo]+(y[hi]-y[lo])*(x[lo+1:hi]-x[lo])/(x[hi]-x[lo])
        errors = abs(y[lo+1:hi]-predicted); index = int(np.argmax(errors))+lo+1
        if errors[index-lo-1] > tolerance:
            keep.add(index); stack.extend([(lo, index), (index, hi)])
    indices = np.asarray(sorted(keep), int)
    maximum = float(abs(np.interp(x, x[indices], y[indices])-y).max())
    if maximum > tolerance*(1+1e-8): raise RuntimeError('CDF compression bound failed')
    return indices, maximum


def compress(posterior, n_components, *, tolerance=2e-4, tail_budget=1e-7):
    if len(posterior.joint_density) != 2*n_components:
        raise ValueError('Expected both binary states of every SED component')
    if not 0 <= tail_budget < 1: raise ValueError('Invalid omitted-mass budget')
    oldmu = posterior.mu
    boundaries = DISTANCE_MU_BOUNDARIES[(DISTANCE_MU_BOUNDARIES > oldmu[0]) & (DISTANCE_MU_BOUNDARIES < oldmu[-1])]
    mu = np.unique(np.r_[oldmu, boundaries])
    density = np.array([np.interp(mu, oldmu, row) for row in posterior.joint_density])
    cumulative = np.c_[np.zeros(len(density)),
        np.cumsum((density[:, 1:]+density[:, :-1])*np.diff(mu)[None, :]/2, axis=1)]
    mass = cumulative[:, -1]
    order = np.argsort(-mass, kind='stable')
    count = np.searchsorted(np.cumsum(mass[order]), 1-tail_budget)+1
    retained = np.sort(order[:count]); omitted = float(mass.sum()-mass[retained].sum())
    xpieces = []; cpieces = []; offsets = [0]; bound = 0.
    for k in retained:
        cdf = cumulative[k]/mass[k]
        # Reporting boundaries and major local peaks remain explicit knots.
        peak = mu[int(np.argmax(density[k]))]
        indices, error = cdf_knots(mu, cdf, tolerance, (*DISTANCE_MU_BOUNDARIES, peak))
        xpieces.append(mu[indices]); cpieces.append(cdf[indices]); offsets.append(offsets[-1]+len(indices))
        bound += mass[k]*error
    return dict(component=(retained % n_components).astype('i2'),
        binary=(retained//n_components).astype('i1'), mass=mass[retained],
        offsets=np.asarray(offsets, 'i4'), mu=np.concatenate(xpieces).astype('f4'),
        cdf=np.concatenate(cpieces).astype('f4'), omitted_mass=omitted,
        weighted_cdf_bound=bound, source_log_evidence=posterior.log_evidence,
        reference_mu_mean=15., reference_mu_sigma=5.)


def quadrature(compressed, order=3, luminosity_shift=None, extra_knots=()):
    """Gauss-Legendre integration within each stored conditional-CDF interval."""
    roots, weights = np.polynomial.legendre.leggauss(order)
    output = {k: [] for k in ('mu', 'mass', 'component', 'binary')}
    for j, (k, b, total) in enumerate(zip(compressed['component'], compressed['binary'], compressed['mass'])):
        lo, hi = compressed['offsets'][j:j+2]
        x = compressed['mu'][lo:hi].astype(float); cdf = compressed['cdf'][lo:hi].astype(float)
        shift = 0. if luminosity_shift is None else np.asarray(luminosity_shift)[k]
        boundaries = np.r_[DISTANCE_MU_BOUNDARIES, extra_knots]-shift
        boundaries = boundaries[(boundaries > x[0]) & (boundaries < x[-1])]
        newx = np.unique(np.r_[x, boundaries])
        cdf = np.interp(newx, x, cdf); x = newx
        widths = np.diff(x); probability = np.diff(cdf)
        valid = probability > 0
        if (widths <= 0).any() or (probability < -1e-7).any(): raise ValueError('Nonmonotone stored CDF')
        nodes = ((x[:-1]+x[1:])/2)[:, None]+widths[:, None]*roots[None, :]/2
        mass = total*probability[:, None]*weights[None, :]/2
        output['mu'].extend(nodes[valid].ravel()); output['mass'].extend(mass[valid].ravel())
        output['component'].extend(np.full(valid.sum()*order, k, 'i2'))
        output['binary'].extend(np.full(valid.sum()*order, b, 'i1'))
    return {k: np.asarray(v, dtype='i2' if k in ('component', 'binary') else 'f8') for k, v in output.items()}


def reweight(compressed, model, *, theta=None, component_weights=None, classes=None,
             luminosity_shift=None, binary_fraction=None, reference_prior=False,
             order=3, prior_ratio_limit=100.):
    """Reweight full joint latent support, including common luminosity shifts.

    A per-kernel common magnitude shift exactly translates its native likelihood
    in mu when mean and intrinsic covariance scale together. It is not a license
    to reinterpret arbitrary color/shape perturbations as distance shifts.
    """
    if compressed['omitted_mass']*prior_ratio_limit > .001:
        raise ValueError('Omitted support cannot certify the requested prior range')
    q = quadrature(compressed, order, luminosity_shift); k = q['component']; b = q['binary']; x = q['mu']
    weights = model.weights if component_weights is None else np.asarray(component_weights)
    cls = model.classes if classes is None else np.asarray(classes)
    if (weights <= 0).any() or not np.isclose(weights.sum(), 1): raise ValueError('Invalid SED weights')
    if (cls <= 0).any() or not np.allclose(cls.sum(axis=1), 1): raise ValueError('Invalid physical class frequencies')
    y = x if luminosity_shift is None else x+np.asarray(luminosity_shift)[k]
    binary = model.binary_reference_fraction if binary_fraction is None else binary_fraction
    if not 0 < binary < 1: raise ValueError('Binary fraction must be interior')
    base_binary = np.array([1-model.binary_reference_fraction, model.binary_reference_fraction])
    new_binary = np.array([1-binary, binary])
    base_log = np.log(model.weights[k])+np.log(base_binary[b])+norm.logpdf(x, 15., 5.)
    if reference_prior:
        new_log = (np.log(weights[k])[:, None]+np.log(cls[k])+
                   np.log(new_binary[b])[:, None]+norm.logpdf(y, 15., 5.)[:, None])
    else:
        theta = np.asarray(theta).reshape(3, 5)
        if (theta < 0).any() or not np.isclose(theta.sum(), 1): raise ValueError('Invalid joint population prior')
        component_given_class = weights[:, None]*cls
        component_given_class /= component_given_class.sum(axis=0)[None, :]
        radial = (theta@distance_basis(y)).T
        new_log = (np.log(component_given_class[k])+np.log(new_binary[b])[:, None]
                   +np.log(np.maximum(radial, 1e-300)))
    logratio = new_log-base_log[:, None]
    maximum_ratio = float(np.exp(np.minimum(logratio.max(), 700)))
    if maximum_ratio > prior_ratio_limit:
        raise ValueError('Reweighting exceeds verified joint prior ratio support')
    logmass = np.log(q['mass'])[:, None]+logratio
    logevidence_ratio = logsumexp(logmass)
    p = np.exp(logmass-logevidence_ratio)
    group = np.searchsorted(DISTANCE_MU_BOUNDARIES, y, side='right')
    joint = np.array([[p[group == j, c].sum() for j in range(6)] for c in range(3)])
    sort = np.argsort(y); cum = np.cumsum(p.sum(axis=1)[sort]); cum /= cum[-1]
    quantiles = np.interp([.025, .16, .5, .84, .975], cum, y[sort])
    return dict(class_distance=joint, mu_quantiles=quantiles,
        qnear=float(joint[:, :3].sum()), binary_probability=float(p[b == 1].sum()),
        log_evidence_ratio=float(logevidence_ratio), maximum_prior_ratio=maximum_ratio)


def pack(records):
    """Ragged numeric arrays; no pickle/object arrays in per-star products."""
    branches = [0]; knots = [0]
    result = {key: [] for key in ('component', 'binary', 'mass', 'mu', 'cdf')}
    for record in records:
        branches.append(branches[-1]+len(record['component']))
        knots.extend(record['offsets'][1:]+knots[-1])
        for key in result: result[key].append(record[key])
    result = {key: np.concatenate(value) for key, value in result.items()}
    result['star_branch_offset'] = np.asarray(branches, 'i8')
    result['branch_knot_offset'] = np.asarray(knots, 'i8')
    for key in ('omitted_mass', 'weighted_cdf_bound', 'source_log_evidence'):
        result[key] = np.array([r[key] for r in records])
    return result


def unpack(payload, star):
    lo, hi = payload['star_branch_offset'][star:star+2]
    start, stop = payload['branch_knot_offset'][[lo, hi]]
    result = {key: payload[key][lo:hi] for key in ('component', 'binary', 'mass')}
    result.update(mu=payload['mu'][start:stop], cdf=payload['cdf'][start:stop],
        offsets=payload['branch_knot_offset'][lo:hi+1]-start,
        reference_mu_mean=15., reference_mu_sigma=5.)
    for key in ('omitted_mass', 'weighted_cdf_bound', 'source_log_evidence'):
        result[key] = payload[key][star]
    return result
