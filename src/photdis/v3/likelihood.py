"""Native-band likelihood and continuous distance quadrature.

Templates are expressed as intrinsic flux at 1 kpc. Distance, dust and survey
calibration are forward operators. No observed weak flux is log-transformed.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp, log_ndtr
from scipy.signal import find_peaks
from scipy.stats import multivariate_normal

LOG10 = np.log(10.)


def distance_kpc(mu):
    return 10.**((np.asarray(mu)-10.)/5.)


def trapezoid_weights(grid):
    grid = np.asarray(grid, float)
    if grid.ndim != 1 or len(grid) < 2 or not np.isfinite(grid).all() or (np.diff(grid) <= 0).any():
        raise ValueError('Strictly increasing finite quadrature grid required')
    dx = np.diff(grid)
    return np.r_[dx[0]/2, (dx[:-1]+dx[1:])/2, dx[-1]/2]


def intrinsic_covariance(samples, measurement_covariance, weights=None, floor=1e-8):
    """Method-of-moments initialization, explicitly subtracting reference noise.

    This initializer is not a claim that noisy individual templates are exact.
    Negative eigenvalues / removed variance are reported for fitting diagnostics.
    Distance uncertainty must appear as an all-band common term in the supplied
    measurement covariance, not six independent magnitude errors.
    """
    x = np.asarray(samples, float); noise = np.asarray(measurement_covariance, float)
    w = np.ones(len(x))/len(x) if weights is None else np.asarray(weights, float)
    w = w/w.sum()
    mean = w@x; residual = x-mean
    denom = 1-w@w
    if denom <= 0: raise ValueError('At least two effective reference sources required')
    empirical = (residual.T*w)@residual/denom
    # For independently noisy observations, this weighted unbiased covariance
    # contains sum w_i(1-w_i) E_i / (1-sum w_i^2), not simply sum w_i E_i.
    correction = np.einsum('i,ijk->jk', w*(1-w)/denom, noise)
    raw = (empirical-correction); raw = (raw+raw.T)/2
    val, vec = np.linalg.eigh(raw)
    cov = (vec*np.maximum(val, floor))@vec.T
    return mean, cov, dict(negative_eigenvalues=int((val < 0).sum()),
                           clipped_variance=float(np.maximum(floor-val, 0).sum()),
                           effective_sources=float(1/(w@w)))


def reference_flux_covariance(flux_at_1kpc, flux_error_at_1kpc, sigma_mu,
                              extinction_vector=None, sigma_ebv=0.):
    f = np.asarray(flux_at_1kpc, float)
    e = np.asarray(flux_error_at_1kpc, float)
    cov = np.eye(f.shape[-1])*e[..., :, None]**2
    common = .4*LOG10*f*np.asarray(sigma_mu)[..., None]
    cov = cov + common[..., :, None]*common[..., None, :]
    if extinction_vector is not None:
        dust = .4*LOG10*f*np.asarray(extinction_vector)*np.asarray(sigma_ebv)[..., None]
        cov += dust[..., :, None]*dust[..., None, :]
    return cov


def _censored_logpdf(value, mean, cov, censored):
    """Exact Gaussian conditioning; upper-limit event is integrated, not a point."""
    detected = ~censored
    total = 0.
    if detected.any():
        cdd = cov[np.ix_(detected, detected)]
        delta = value[detected]-mean[detected]
        sign, det = np.linalg.slogdet(cdd)
        if sign <= 0: raise ValueError('Non-positive observation covariance')
        total = -.5*(det + delta@np.linalg.solve(cdd, delta) + detected.sum()*np.log(2*np.pi))
    if censored.any():
        conditional_mean = mean[censored].copy()
        conditional_cov = cov[np.ix_(censored, censored)].copy()
        if detected.any():
            cross = cov[np.ix_(censored, detected)]
            conditional_mean += cross@np.linalg.solve(cdd, delta)
            conditional_cov -= cross@np.linalg.solve(cdd, cross.T)
        if censored.sum() == 1:
            total += log_ndtr((value[censored][0]-conditional_mean[0])/np.sqrt(conditional_cov[0, 0]))
        else:
            # Fixed integration seed makes likelihood reproducible. Numerical CDF
            # accuracy must still be checked for actual multi-limit observations.
            args = dict(mean=conditional_mean, cov=conditional_cov, maxpts=1000000,
                        abseps=1e-9, releps=1e-9)
            import inspect
            if 'rng' in inspect.signature(multivariate_normal.cdf).parameters:
                args['rng'] = np.random.default_rng(301112)
            p = multivariate_normal.cdf(value[censored], **args)
            if p <= 0: return -np.inf
            total += np.log(p)
    return float(total)


def flux_loglikelihood(flux, error, mask, transmission, template_flux,
                       template_covariance, mu, *, calibration_mag=None,
                       upper_limit=None, system='DECam_DR11', template_system='DECam_DR11'):
    """Return component x distance log likelihood for one observed star.

    Full intrinsic native-flux covariance transforms with distance and dust.
    Inputs marked missing are never evaluated; invalid observed values fail.
    Unmodeled systems cannot silently borrow another survey's templates.
    """
    if system != template_system: raise ValueError('Photometric system mismatch')
    mask = np.asarray(mask, bool)
    if not mask.any(): raise ValueError('No measured bands')
    f = np.asarray(flux, float)[mask]; e = np.asarray(error, float)[mask]
    tr = np.asarray(transmission, float)[mask]
    if not (np.isfinite(f).all() and np.isfinite(e).all() and (e > 0).all()
            and np.isfinite(tr).all() and (tr > 0).all()):
        raise ValueError('Invalid measured flux, error or transmission')
    sed = np.atleast_2d(template_flux)[:, mask]
    cov = np.asarray(template_covariance)[:, mask][:, :, mask]
    if not np.isfinite(sed).all() or not np.isfinite(cov).all():
        raise ValueError('Unsupported template bands must not be mean-filled')
    if calibration_mag is not None: tr = tr*10.**(-.4*np.asarray(calibration_mag)[mask])
    amplitude = 10.**(-.4*(np.asarray(mu)-10.))
    factor = amplitude[:, None]*tr[None, :]
    mean = sed[:, None, :]*factor[None, :, :]
    v = cov[:, None, :, :]*factor[None, :, :, None]*factor[None, :, None, :]
    v = v + np.diag(e**2)[None, None, :, :]
    censored = np.zeros(len(f), bool) if upper_limit is None else np.asarray(upper_limit, bool)[mask]
    if censored.any():
        return np.array([[_censored_logpdf(f, m, c, censored) for m, c in zip(mm, vv)]
                         for mm, vv in zip(mean, v)])
    delta = f[None, None, :]-mean
    chol = np.linalg.cholesky(v)
    whitened = np.linalg.solve(chol, delta[..., None])[..., 0]
    logdet = 2*np.log(np.diagonal(chol, axis1=-2, axis2=-1)).sum(axis=-1)
    return -.5*(np.sum(whitened**2, axis=-1)+logdet+len(f)*np.log(2*np.pi))


@dataclass
class ContinuousPosterior:
    mu: np.ndarray
    joint_density: np.ndarray
    log_evidence: float
    convergence: dict

    def component_mass(self):
        return self.joint_density@trapezoid_weights(self.mu)

    def cdf(self, point):
        """Integrate a linearly interpolated density to an arbitrary boundary."""
        x = float(point)
        if x <= self.mu[0]: return np.zeros(len(self.joint_density))
        if x >= self.mu[-1]: return self.component_mass()
        use = self.mu < x
        grid = np.r_[self.mu[use], x]
        density = np.column_stack([self.joint_density[:, use],
            [np.interp(x, self.mu, row) for row in self.joint_density]])
        return density@trapezoid_weights(grid)

    def distance_bins(self, bounds=(0, 2, 5, 10, 20, 50, np.inf)):
        values = []
        for d in bounds:
            values.append(np.zeros(len(self.joint_density)) if d <= 0 else
                          self.component_mass() if not np.isfinite(d) else self.cdf(10+5*np.log10(d)))
        return np.diff(np.asarray(values), axis=0).T

    def quantiles(self, levels=(.025, .16, .5, .84, .975)):
        density = self.joint_density.sum(axis=0)
        cdf = np.r_[0, np.cumsum(np.diff(self.mu)*(density[1:]+density[:-1])/2)]
        return np.interp(levels, cdf/cdf[-1], self.mu)

    def modes(self):
        p = self.joint_density.sum(axis=0)
        peaks, _ = find_peaks(p)
        if not len(peaks): peaks = np.array([p.argmax()])
        minima = np.array([a+np.argmin(p[a:b+1]) for a, b in zip(peaks[:-1], peaks[1:])], dtype=int)
        bounds = np.r_[self.mu[0], self.mu[minima], self.mu[-1]]
        mass = np.array([(self.cdf(b)-self.cdf(a)).sum() for a, b in zip(bounds[:-1], bounds[1:])])
        return self.mu[peaks], mass


def normalize_joint(mu, log_density):
    w = trapezoid_weights(mu)
    logz = logsumexp(np.asarray(log_density)+np.log(w)[None, :])
    if not np.isfinite(logz): raise ValueError('No finite posterior support')
    return ContinuousPosterior(np.asarray(mu), np.exp(log_density-logz), float(logz), {})


def adaptive_posterior(log_density, *, lower=-5., upper=25., step=.25,
                       probability_tolerance=.001, quantile_tolerance=.01,
                       edge_tolerance=1e-5, max_refinements=8, max_extensions=6):
    """Global refinement/extension reference implementation, not pruned production.

    Initial step must resolve the narrowest likelihood scale (derived from flux
    errors in the caller); later global halvings verify probabilities, modes and
    quantiles. This implementation is the full-grid compression oracle.
    """
    previous = None; extensions = 0; pdelta = qdelta = np.inf; stable_modes = False
    for refinement in range(max_refinements+1):
        while True:
            grid = np.linspace(lower, upper, int(np.ceil((upper-lower)/step))+1)
            p = normalize_joint(grid, log_density(grid))
            edge = np.r_[p.cdf(lower+step).sum(), (p.component_mass()-p.cdf(upper-step)).sum()]
            if edge.max() <= edge_tolerance or extensions >= max_extensions: break
            if edge[0] > edge_tolerance: lower -= 5
            if edge[1] > edge_tolerance: upper += 5
            extensions += 1
        # Include component/mode probability, not just total near probability.
        vector = np.r_[p.distance_bins().ravel(), p.component_mass()]
        quantiles = p.quantiles()
        if previous is not None:
            oldvector, oldq, oldmodes = previous
            pdelta = float(abs(vector-oldvector).max()); qdelta = float(abs(quantiles-oldq).max())
            modes, mass = p.modes()
            stable_modes = sum(mass > .01) == oldmodes
            if pdelta <= probability_tolerance and qdelta <= quantile_tolerance and stable_modes:
                break
        previous = (vector, quantiles, int(sum(p.modes()[1] > .01)))
        step /= 2
    p.convergence = dict(probability_delta=pdelta, quantile_delta_mag=qdelta,
        boundary_mass=edge.tolist(), extensions=extensions, refinements=refinement,
        nodes=len(p.mu), converged=bool(pdelta <= probability_tolerance and
        qdelta <= quantile_tolerance and edge.max() <= edge_tolerance and stable_modes))
    return p
