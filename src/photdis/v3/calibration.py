"""Source-conditioned signed-parallax reference calibration.

This is empirical consistency calibration of Gaia-dependent VACs, not a new
independent absolute-distance experiment. One source is calibrated at a time.
"""
import numpy as np
from scipy.special import logsumexp, expit
from scipy.optimize import minimize
from numpy.polynomial.hermite import hermgauss


def signed_parallax_loglike(mu, sigma_mu, parallax, parallax_error, offset=0., extra=0., order=48):
    """Integrate normal reference mu against signed Gaussian parallax.

    Positive offset means the corrected reference distance modulus is larger.
    Integrate in the narrower coordinate for high-precision positive Gaia.
    Never invert a weak/negative parallax into a reference point distance.
    """
    mu, s, p, e = np.broadcast_arrays(np.asarray(mu, float), np.hypot(sigma_mu, extra),
                                     np.asarray(parallax, float), np.asarray(parallax_error, float))
    if not (np.isfinite(mu).all() and np.isfinite(s).all() and (s > 0).all()
            and np.isfinite(p).all() and np.isfinite(e).all() and (e > 0).all()):
        raise ValueError('Invalid reference or signed-parallax measurement')
    x, w = hermgauss(order); lw = np.log(w/np.sqrt(np.pi))
    center = mu+offset
    m = center[..., None]+np.sqrt(2)*s[..., None]*x
    model = 10.**((10-m)/5)
    z = (p[..., None]-model)/e[..., None]
    value = logsumexp(-.5*z*z-np.log(e[..., None]*np.sqrt(2*np.pi))+lw, axis=-1)
    ls = np.log(10)/5*s
    narrow = (p > 5*e) & (p*ls > e)
    if narrow.any():
        observed = p[narrow, None]+np.sqrt(2)*e[narrow, None]*x
        safe = np.maximum(observed, 1e-300)
        logcenter = (10-center[narrow])*np.log(10)/5
        lp = -.5*((np.log(safe)-logcenter[:, None])/ls[narrow, None])**2
        lp -= np.log(safe*ls[narrow, None]*np.sqrt(2*np.pi))
        lp[observed <= 0] = -np.inf
        value[narrow] = logsumexp(lp+lw, axis=-1)
    return value


def source_parameters(theta):
    """Offset mag, excess scatter mag, broad-reference mixture fraction."""
    return float(theta[0]), float(np.exp(theta[1])), float(.3*expit(theta[2]))


def source_logposterior(theta, mu, sigma, parallax, error, weights=None, order=48):
    offset, scatter, outlier = source_parameters(theta)
    core = signed_parallax_loglike(mu, sigma, parallax, error, offset, scatter, order)
    broad = signed_parallax_loglike(mu, sigma, parallax, error, offset, 2., order)
    ll = np.logaddexp(np.log1p(-outlier)+core, np.log(outlier)+broad)
    w = np.ones(len(ll)) if weights is None else np.asarray(weights)
    # Explicit weakly regularized priors; log scatter parameterized directly.
    prior = -.5*(offset/.3)**2-.5*((theta[1]-np.log(.15))/1.)**2
    prior -= .5*((theta[2]+2.)/1.5)**2
    return float(w@ll+prior)


def fit_source(mu, sigma, parallax, error, weights=None, initial=None, order=48):
    initial = [0., np.log(.15), -2.] if initial is None else initial
    result = minimize(lambda x: -source_logposterior(x, mu, sigma, parallax, error, weights, order),
                      initial, method='L-BFGS-B', bounds=[(-1., 1.), (-5., .5), (-8., 4.)],
                      options=dict(maxiter=150, ftol=1e-10, gtol=1e-5))
    if not result.success: raise RuntimeError('Source calibration did not converge: '+result.message)
    return result


def reference_nodes(channel, distance, sigma_mu, parallax, parallax_error, *,
                    offset=0., extra=0., outlier=0., nodes=24, scale_length_kpc=1.35):
    """Full signed-Gaia or VAC normal/broad-mixture distance integration nodes.

    VAC uses a flat-mu reference measure; Gaia-only uses explicit EDSD reference
    prior d^2 exp(-d/L), including Jacobian d dd/dmu. Nodes integrate that
    documented reference posterior, not an unrecorded inverse-parallax point.
    Equal-probability deterministic quadrature avoids importance-weight collapse.
    """
    from scipy.special import ndtr
    from .likelihood import trapezoid_weights
    c = np.asarray(channel); n = len(c)
    mu_nodes = np.full((n, nodes), np.nan)
    quantiles = (np.arange(nodes)+.5)/nodes
    grid = np.linspace(-5, 30, 3501)
    w = trapezoid_weights(grid)
    d = 10.**((grid-10)/5)
    for start in range(0, n, 256):
        stop = min(n, start+256); cc = c[start:stop]
        for k in (1, 2):
            ii = np.flatnonzero(cc == k)+start
            if not len(ii): continue
            center = 10+5*np.log10(np.asarray(distance)[ii])+offset
            ss = np.hypot(np.asarray(sigma_mu)[ii], extra)
            cdf = (1-outlier)*ndtr((grid[None, :]-center[:, None])/ss[:, None])
            cdf += outlier*ndtr((grid[None, :]-center[:, None])/np.hypot(sigma_mu[ii], 2.)[:, None])
            for j, row in enumerate(ii):
                if cdf[j, 0] > 1e-6 or cdf[j, -1] < 1-1e-6:
                    raise ValueError('Reference distance nodes require wider bounds')
                mu_nodes[row] = np.interp(quantiles, cdf[j], grid)
        ii = np.flatnonzero(cc == 3)+start
        for row in ii:
            p = float(parallax[row]); e = float(parallax_error[row])
            if not np.isfinite(p) or not np.isfinite(e) or e <= 0:
                raise ValueError('Invalid Gaia likelihood')
            local_grid = grid
            if p > 5*e:
                # Numerical proposal only for precise positive parallax. Evaluate
                # the original signed likelihood and prior on the union grid.
                center = 10-5*np.log10(p)
                width = 5/np.log(10)*e/p
                local_grid = np.unique(np.r_[grid, np.linspace(center-12*width, center+12*width, 513)])
            distance_grid = 10.**((local_grid-10)/5)
            logp = -.5*((p-1/distance_grid)/e)**2+3*np.log(distance_grid)-distance_grid/scale_length_kpc
            logp -= logsumexp(logp+np.log(trapezoid_weights(local_grid)))
            density = np.exp(logp)
            cdf = np.r_[0, np.cumsum(np.diff(local_grid)*(density[1:]+density[:-1])/2)]
            mu_nodes[row] = np.interp(quantiles, cdf/cdf[-1], local_grid)
    if np.isnan(mu_nodes).any(): raise ValueError('No primary distance evidence; retain outside SED-distance fit')
    return mu_nodes, np.full((n, nodes), 1/nodes)
