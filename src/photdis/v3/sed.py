"""Projected heteroscedastic empirical-SED mixture with reference-mu quadrature.

The Gaussian latent-flux approximation is explicit and carries positivity/support
flags. Reference distance is integrated jointly, never added to intrinsic scatter.
Missing FeH/bands have no likelihood term. Missing class is marginalized over the
three physical reference labels, never made a fourth physical population.
"""
import multiprocessing as mp
import numpy as np
from scipy.special import logsumexp
from .likelihood import intrinsic_covariance

_CONTEXT = None


def projected_conditionals(y, noise, observed, mean, covariance):
    """Gaussian latent posterior under an arbitrary observed-coordinate projection.

    y/noise have arbitrary leading batch dimensions; returned full latent moments
    retain correlations with missing coordinates. Projection indices, not a fake
    huge error / zero measurement, implement missingness.
    """
    ix = np.flatnonzero(observed)
    if not len(ix): raise ValueError('No projected observations')
    yy = np.asarray(y)[..., ix]; ss = np.asarray(noise)[..., ix, :][..., :, ix]
    v = np.asarray(covariance); m = np.asarray(mean)
    c = ss+v[np.ix_(ix, ix)]
    # Per-component units prevent bright-flux / dex condition-number artifacts.
    scale = np.sqrt(np.diag(v)[ix])
    scale = np.maximum(scale, 1e-15)
    normalized = c/scale[:, None]/scale[None, :]
    delta = (yy-m[ix])/scale
    chol = np.linalg.cholesky(normalized)
    whitened = np.linalg.solve(chol, delta[..., None])[..., 0]
    inverse = np.linalg.inv(normalized)/scale[:, None]/scale[None, :]
    gain = np.einsum('ab,...bc->...ac', v[:, ix], inverse)
    posterior_mean = m+np.einsum('...ab,...b->...a', gain, yy-m[ix])
    posterior_cov = v-np.einsum('...ab,bc->...ac', gain, v[ix, :])
    logdet = 2*np.log(np.diagonal(chol, axis1=-2, axis2=-1)).sum(axis=-1)+2*np.log(scale).sum()
    logp = -.5*((whitened**2).sum(axis=-1)+logdet+len(ix)*np.log(2*np.pi))
    return posterior_mean, posterior_cov, logp


def _component(task):
    k, statistics = task
    y, noise, mask, node_logweight, labels, weights, means, covariances, classes, normalizer = _CONTEXT
    n, nodes, dim = y.shape
    marginal = np.empty(n)
    count = 0.; first = np.zeros(dim); second = np.zeros((dim, dim)); class_count = np.zeros(3)
    bitmask = (mask.astype('i8')*(1 << np.arange(dim))).sum(axis=1)
    for pattern in np.unique(bitmask):
        rows = np.flatnonzero(bitmask == pattern)
        observed = mask[rows[0]]
        for start in range(0, len(rows), 256):
            rr = rows[start:start+256]
            b, B, l = projected_conditionals(y[rr], noise[rr], observed, means[k], covariances[k])
            known = labels[rr] >= 0
            class_term = np.zeros(len(rr))
            class_term[known] = np.log(classes[k, labels[rr][known]])
            logjoint = l+node_logweight[rr]+np.log(weights[k])+class_term[:, None]
            marginal[rr] = logsumexp(logjoint, axis=1)
            if statistics:
                q = np.exp(logjoint-normalizer[rr, None])
                count += q.sum()
                first += np.einsum('ij,ijd->d', q, b)
                second += np.einsum('ij,ijab->ab', q, B)+np.einsum('ij,ija,ijb->ab', q, b, b)
                for c in range(3): class_count[c] += q[labels[rr] == c].sum()
    return (count, first, second, class_count) if statistics else marginal


class EmpiricalSED:
    """One fixed mixture family; component count is a compression setting.

    Initialization uses optical colors/absolute magnitude only to distribute
    centers, not as an application classifier. Optional missing values are used
    nowhere as observed zero/mean flux. Positive-support adequacy is audited.
    """
    def __init__(self, components=128, max_iter=64, tol=1e-4, seed=301111):
        self.components = components; self.max_iter = max_iter
        self.tol = tol; self.seed = seed

    def initialize(self, y, noise, mask, features, labels, initial_node=None):
        from sklearn.cluster import KMeans
        n, _, d = y.shape
        scale = np.nanstd(features, axis=0); scale[scale < 1e-6] = 1
        # Center spread in color/luminosity coordinates preserves faint branches;
        # arbitrary filled optional flux cannot determine cluster assignments.
        clusters = KMeans(self.components, random_state=self.seed, n_init=1,
                          max_iter=100).fit_predict(features/scale)
        node = y.shape[1]//2 if initial_node is None else initial_node
        middle = y[:, node]; nn = noise[:, node]
        self.weights = np.bincount(clusters, minlength=self.components).astype(float)
        self.weights /= self.weights.sum()
        self.means = np.zeros((self.components, d)); self.covariances = np.zeros((self.components, d, d))
        self.classes = np.zeros((self.components, 3))
        self.initialization_audit = []
        global_mean = np.array([np.median(middle[mask[:, j], j]) for j in range(d)])
        for k in range(self.components):
            rows = clusters == k
            # Initial missing-coordinate values are prior initialization only;
            # the E step projects them out and learns them from observed bands.
            filled = np.where(mask[rows], middle[rows], global_mean)
            sample = filled
            if len(sample) >= 2:
                mean, cov, audit = intrinsic_covariance(sample, nn[rows], floor=1e-10)
            else:
                mean = sample[0]; cov = np.diag(np.maximum(abs(mean)*.15, .01)**2)
                audit = dict(negative_eigenvalues=0, clipped_variance=0., effective_sources=1.)
            self.means[k] = mean
            # A local relative floor avoids collapsing a narrow optical branch;
            # floors are explicit numerical regularization, not teacher noise.
            floor = np.maximum(abs(mean)*.005, 1e-5)**2; floor[-1] = .02**2
            cov += np.diag(floor)
            self.covariances[k] = cov
            counts = np.bincount(labels[rows & (labels >= 0)], minlength=3)+.5
            self.classes[k] = counts/counts.sum()
            self.initialization_audit.append(dict(component=k, n=int(rows.sum()), **audit))

    def fit(self, y, noise, mask, node_logweight, labels, features, workers=1, callback=None, initial_node=None):
        global _CONTEXT
        y, noise, mask = np.asarray(y), np.asarray(noise), np.asarray(mask, bool)
        labels = np.asarray(labels, int)
        if not np.isin(labels, [-1, 0, 1, 2]).all(): raise ValueError('Invalid reference class')
        if not np.isfinite(y[mask[:, None, :].repeat(y.shape[1], axis=1)]).all():
            raise ValueError('Nonfinite observed training coordinate')
        self.initialize(y, noise, mask, features, labels, initial_node=initial_node)
        self.history = []; self.converged = False
        def run(statistics):
            tasks = [(k, statistics) for k in range(self.components)]
            if workers == 1: return [_component(x) for x in tasks]
            # Read-only clouds/parameters inherited, not serialized per component.
            with mp.get_context('fork').Pool(workers) as pool:
                return pool.map(_component, tasks, chunksize=1)
        for iteration in range(self.max_iter):
            _CONTEXT = (y, noise, mask, node_logweight, labels, self.weights,
                        self.means, self.covariances, self.classes, None)
            logcomponent = np.array(run(False)).T
            norm = logsumexp(logcomponent, axis=1)
            likelihood = float(norm.mean()); self.history.append(likelihood)
            if callback: callback(iteration, likelihood)
            if iteration > 3 and abs(self.history[-1]-self.history[-2]) < self.tol*(1+abs(likelihood)):
                self.converged = True; break
            _CONTEXT = (*_CONTEXT[:-1], norm)
            statistics = run(True)
            counts = np.array([x[0] for x in statistics])
            if (counts < 1e-8).any(): raise RuntimeError('Empty SED component; inspect support before changing design')
            for k, (count, first, second, classes) in enumerate(statistics):
                self.means[k] = first/count
                cov = second/count-np.outer(self.means[k], self.means[k])
                scale = np.maximum(abs(self.means[k]), .001); scale[-1] = 1.
                normalized = (cov+cov.T)/2/scale[:, None]/scale[None, :]
                eigen, vector = np.linalg.eigh(normalized)
                normalized = (vector*np.maximum(eigen, .005**2))@vector.T
                self.covariances[k] = normalized*scale[:, None]*scale[None, :]
                self.classes[k] = (classes+.5)/(classes.sum()+1.5)
            self.weights = counts/counts.sum()
        _CONTEXT = None
        # A Gaussian flux kernel can be an inadequate approximation in broad,
        # low-support branches. Record this instead of pretending physical support.
        from scipy.special import ndtr
        self.negative_intrinsic_flux_probability = ndtr(-self.means[:, :6]/np.sqrt(np.diagonal(self.covariances[:, :6, :6], axis1=1, axis2=2)))
        return self
