"""Frozen-tree reference contributions and explicitly scoped near-event scores."""
import numpy as np
from scipy.sparse import csr_matrix
from .baseline_v1 import qmc_noise, quantiles_interpolated


def psf_admission(types):
    types = np.char.upper(np.char.strip(np.asarray(types).astype('U')))
    missing = np.isin(types, ['', 'NONE', 'NAN', 'UNKNOWN'])
    good = types == 'PSF'
    return good, dict(total=len(types), admitted=int(good.sum()),
                      missing_type=int(missing.sum()),
                      non_psf=int((~missing & ~good).sum()))


def near_mass(pdf, edges, g0, limit=15.):
    """P(mu<limit) for a piecewise uniform effective-M histogram, no Gaussian."""
    fraction = np.clip((edges[1:][None, :] - (np.asarray(g0) - limit)[:, None])
                       / np.diff(edges)[None, :], 0, 1)
    return np.sum(pdf * fraction, axis=-1)


def interval_score(mu, truth, alpha=.32):
    lo, hi = mu[:, 1], mu[:, 3]
    return hi-lo + 2/alpha*np.maximum(lo-truth, 0) + 2/alpha*np.maximum(truth-hi, 0)


def migration(reference_mu, selected, reference_sigma=None):
    ref = np.asarray(reference_mu)
    selected = np.asarray(selected, bool)
    valid = np.isfinite(ref)
    near = ref < 15
    nn = int((valid & near & selected).sum())
    nf = int((valid & near & ~selected).sum())
    fn = int((valid & ~near & selected).sum())
    ff = int((valid & ~near & ~selected).sum())
    out = dict(n=int(valid.sum()), near_selected=nn, near_rejected=nf,
               far_selected=fn, far_rejected=ff,
               completeness=nn/(nn+nf) if nn+nf else None,
               far_contamination=fn/(nn+fn) if nn+fn else None)
    if reference_sigma is not None:
        sigma = np.asarray(reference_sigma)
        known = np.isfinite(sigma) & (sigma > 0)
        crossing = valid & known & (ref-sigma <= 15) & (ref+sigma >= 15)
        out.update(boundary_crossing=int(crossing.sum()),
                   boundary_selected=int((crossing & selected).sum()),
                   unknown_reference_error=int((valid & ~known).sum()))
    assert nn+nf+fn+ff == out['n']
    return out


def color_covariance(errors):
    """Independent adjacent colors: negative covariance from shared middle band."""
    errors = np.asarray(errors, float)
    nband = errors.shape[1]
    difference = np.eye(nband)[:-1] - np.eye(nband)[1:]
    return np.einsum('ai,ni,bi->nab', difference, errors**2, difference)


class JointReferenceForest:
    """Rebuild frozen class×Mg and distance-source×Mg, without changing a tree.

    Two separate joint decompositions are stored (four class channels, plus the
    supplied source vocabulary). They are NOT an independent-marginal product.
    Sparse leaf tables retain all members including missing class labels.
    """
    def __init__(self, model, mags, luminosity, labels, sources):
        self.model = model
        self.source_values = np.unique(sources)
        source_index = np.searchsorted(self.source_values, sources)
        classes = np.where(np.asarray(labels) < 0, 3, labels).astype(int)
        if not np.isin(classes, np.arange(4)).all():
            raise ValueError('Unexpected training class')
        self.nchannel = 4 + len(self.source_values)
        self.tables = []
        x = model.feature_matrix(mags)
        bins = np.searchsorted(model.edges, luminosity, side='right')-1
        nb = len(model.centers)
        if not ((bins >= 0) & (bins < nb)).all():
            raise ValueError('Training luminosity outside frozen grid')
        checks = []
        for tree, mapping, original, types in zip(model.forest.estimators_, model.maps,
                                                  model.tables, model.types):
            leaf = mapping[tree.apply(x)]
            count = np.bincount(leaf, minlength=len(original))
            if (count == 0).any():
                raise ValueError('Missing frozen leaf members; wrong training identities')
            row = np.r_[leaf, leaf]
            col = np.r_[classes*nb+bins, (4+source_index)*nb+bins]
            # Coalesce integer counts BEFORE division. Summing float32 1/N for
            # repeated members of a large leaf drifts from the frozen histogram.
            table = csr_matrix((np.ones(len(row), dtype=np.int64), (row, col)),
                               shape=(len(original), self.nchannel*nb))
            table = table.astype(np.float64)
            table.data /= np.repeat(count, np.diff(table.indptr))
            table = table.astype(np.float32)
            # Audit one tree at a time, keeping temporary dense allocation bounded.
            dense = table.toarray().reshape(len(original), self.nchannel, nb)
            dp = float(np.max(abs(dense[:, :4].sum(axis=1)-original)))
            dc = float(np.max(abs(dense[:, :4].sum(axis=2)-types)))
            ds = float(np.max(abs(dense[:, 4:].sum(axis=1)-original)))
            if max(dp, dc, ds) > 2e-6:
                raise ValueError(f'Frozen joint marginal errors: PDF={dp}, class={dc}, source={ds}')
            checks.append([dp, dc, ds])
            self.tables.append(table)
        self.reconstruction_max_errors = np.max(checks, axis=0)

    def distribution(self, x):
        model = self.model
        out = np.zeros((len(x), self.nchannel, len(model.centers)), dtype=np.float64)
        for tree, mapping, table in zip(model.forest.estimators_, model.maps, self.tables):
            out += table[mapping[tree.apply(x)]].toarray().reshape(out.shape)
        return out / len(self.tables)

    def predict(self, mags, errors, ids, draws=128):
        model = self.model
        err = np.nan_to_num(errors, nan=0, posinf=0)
        total = np.zeros((len(mags), self.nchannel, len(model.centers)))
        nominal = self.distribution(model.feature_matrix(mags))
        for draw in range(draws):
            delta = qmc_noise(ids, draw)*err
            joint = self.distribution(model.feature_matrix(mags+delta))
            total += shift_joint(joint, delta[:, 0], model.edges)
        total /= draws
        # Each decomposition has unit total mass; preserve relative channel weights.
        for sl in (slice(0, 4), slice(4, None)):
            total[:, sl] /= total[:, sl].sum(axis=(1, 2))[:, None, None]
        pdf = total[:, :4].sum(axis=1)
        mu = mags[:, 0, None]-quantiles_interpolated(pdf, model.edges)[:, ::-1]
        return dict(joint=total, nominal=nominal, pdf=pdf, mu=mu)


def shift_joint(joint, delta_g, edges):
    """Linear CDF transport with channel-specific boundary mass (never set to1).

    The endpoint mass of a class channel is its contribution, not a normalized
    class-conditional PDF. Thus both class and source decompositions conserve mass.
    """
    n, nc, nb = joint.shape
    cumulative = np.concatenate([np.zeros((n, nc, 1)), np.cumsum(joint, axis=2)], axis=2)
    coordinate = (edges[None, :]+delta_g[:, None]-edges[0])/(edges[1]-edges[0])
    lower = np.floor(coordinate).astype(int)
    fraction = coordinate-lower
    lo = np.broadcast_to(np.clip(lower, 0, nb)[:, None, :], (n, nc, nb+1))
    hi = np.broadcast_to(np.clip(lower+1, 0, nb)[:, None, :], (n, nc, nb+1))
    c = (np.take_along_axis(cumulative, lo, axis=2)*(1-fraction[:, None, :])
         + np.take_along_axis(cumulative, hi, axis=2)*fraction[:, None, :])
    c[:, :, 0] = 0
    c[:, :, -1] = cumulative[:, :, -1]
    return np.maximum(np.diff(c, axis=2), 0)
