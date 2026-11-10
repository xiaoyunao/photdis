"""Fixed probability calibration and covariance-aware local information audit."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from scipy.spatial import cKDTree
from .probability_v2_1 import color_covariance


def calibrated(probabilities, parameters):
    p = np.clip(probabilities, 1e-12, 1)
    # Reference class0 fixes the redundant additive offset.
    return softmax(np.log(p)/np.exp(parameters[0]) + np.r_[0., parameters[1:]], axis=1)


def fit_calibration(probabilities, labels):
    labels = np.asarray(labels, int)
    if not np.array_equal(np.unique(labels), np.arange(3)):
        raise ValueError('Calibration requires all three reliable classes')
    def objective(theta):
        p = calibrated(probabilities, theta)
        return -np.log(np.maximum(p[np.arange(len(labels)), labels], 1e-15)).mean() + 1e-5*np.sum(theta**2)
    fit = minimize(objective, np.zeros(3), method='L-BFGS-B', bounds=[(-2, 2), (-5, 5), (-5, 5)])
    if not fit.success:
        raise RuntimeError('Fixed calibration did not converge: '+fit.message)
    return fit.x


def class_score(probability, labels, baseline):
    n = len(labels)
    if not n:
        return dict(n=0)
    obs = probability[np.arange(n), labels]
    base = baseline[np.arange(n), labels]
    return dict(n=n, log_score=float(np.mean(np.log(np.maximum(obs, 1e-12)))),
                log_gain_over_prior=float(np.mean(np.log(np.maximum(obs, 1e-12))-np.log(np.maximum(base, 1e-12)))),
                brier=float(np.mean(np.sum((probability-np.eye(3)[labels])**2, axis=1))),
                floor_events=int((obs < 1e-12).sum()))


def leaf_class_control(model, train_mags, train_labels, eval_mags, prior):
    """Frozen partitions, controlled identical label pool; disclose tree provenance.

    Unlike original class contributions, this conditional diagnostic excludes
    unverified/nonPSF and unlabeled members. Empty labeled leaves use common prior.
    """
    tx, ex = model.feature_matrix(train_mags), model.feature_matrix(eval_mags)
    result = np.zeros((len(ex), 3))
    empty = np.zeros(len(ex), dtype=int)
    for tree in model.forest.estimators_:
        leaf = tree.apply(tx)
        counts = np.bincount(leaf*3+train_labels, minlength=tree.tree_.node_count*3).reshape(-1, 3)
        mass = counts.sum(axis=1)
        p = np.broadcast_to(prior, counts.shape).copy()
        good = mass > 0
        p[good] = counts[good]/mass[good, None]
        el = tree.apply(ex)
        result += p[el]
        empty += ~good[el]
    return result/len(model.forest.estimators_), empty/len(model.forest.estimators_)


def local_neighbors(train_mags, train_errors, labels, luminosity, eval_mags, eval_errors, eval_labels, candidates=64):
    """Candidate-limited Mahalanobis neighborhood; no diagonal-covariance shortcut.

    Search is bounded and explicitly not an exhaustive information bound.
    The covariance is C_train+C_eval with a1e-8 diagonal numerical regularizer.
    """
    tx = np.diff(train_mags, axis=1)*-1
    ex = np.diff(eval_mags, axis=1)*-1
    tc, ec = color_covariance(train_errors), color_covariance(eval_errors)
    scale = np.maximum(np.std(tx, axis=0), .01)
    k = min(candidates, len(tx))
    _, ix = cKDTree(tx/scale).query(ex/scale, k=k)
    if k == 1:
        ix = ix[:, None]
    diff = tx[ix]-ex[:, None, :]
    cov = tc[ix]+ec[:, None, :, :]+np.eye(tx.shape[1])[None, None]*1e-8
    chi = np.einsum('nki,nki->nk', diff, np.linalg.solve(cov, diff[..., None])[..., 0])
    near = chi <= 9*tx.shape[1]
    weight = np.exp(-.5*np.minimum(chi, 1400))*near
    total = weight.sum(axis=1)
    local = np.column_stack([(weight*(labels[ix] == c)).sum(axis=1) for c in range(3)])
    local = np.divide(local, total[:, None], out=np.full_like(local, np.nan), where=total[:, None] > 0)
    different = labels[ix] != eval_labels[:, None]
    other_min = np.min(np.where(different, chi, np.inf), axis=1)
    entropy = -np.nansum(local*np.log(np.maximum(local, 1e-30)), axis=1)
    entropy[total == 0] = np.nan
    span = np.full(len(ex), np.nan)
    for i in range(len(ex)):
        if near[i].sum() >= 2:
            span[i] = np.ptp(luminosity[ix[i, near[i]]])
    return dict(neighbors=near.sum(axis=1), min_other_chi2=other_min,
                entropy=entropy, luminosity_span=span,
                same_class_mass=local[np.arange(len(ex)), eval_labels], local_class=local)
