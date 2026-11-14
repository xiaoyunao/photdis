"""Posterior transfer between matched population entry integration levels."""
import numpy as np
from scipy.special import logsumexp

from photdis.v3.target import joint_coefficients


def posterior_transfer(data, model, theta, beta, fields, *, batch=32):
    """Integrate entry drift and isolate its effect on the population posterior.

    The numerator is fixed in this diagnostic. Conditional class-by-radial-basis
    probabilities are identical across entry levels at a given draw. Changes
    in their posterior averages therefore arise solely from draw reweighting.
    """
    entry = np.asarray(data['entry'], float)
    response = np.asarray(data['response'], float)
    field = np.asarray(data['field'])
    case_weight = np.array(data['design_weight'], dtype=float, copy=True)
    if (len(theta) != len(beta) or len(beta) < 2 or theta.shape[1] != len(fields)
            or not np.array_equal(np.unique(field), fields)
            or entry.shape != (len(field), 3, *response.shape[1:])
            or not np.isfinite(case_weight).all() or (case_weight <= 0).any()):
        raise ValueError('Aligned shared draws and all three matched entry levels required')
    case_weight /= case_weight.mean()
    log_ratios = np.zeros((3, len(beta)))
    largest_entry_delta = np.zeros(3)
    for j, f in enumerate(fields):
        use = field == f
        flat_entry = np.ascontiguousarray(entry[use].transpose(1, 0, 2, 3)).reshape(3, use.sum(), -1)
        for start in range(0, len(beta), batch):
            stop = min(start+batch, len(beta))
            coef = joint_coefficients(model, theta[start:stop, j], beta[start:stop])
            normalizer = flat_entry @ coef.sum(axis=-1).reshape(stop-start, -1).T
            if not np.isfinite(normalizer).all() or (normalizer <= 0).any():
                raise ValueError('A fitted draw lacks entry support')
            delta = abs(normalizer-normalizer[1:2]).max(axis=(1, 2))
            largest_entry_delta = np.maximum(largest_entry_delta, delta)
            log_entry = np.log(normalizer)
            log_ratios[:, start:stop] -= (log_entry-log_entry[1:2]).transpose(0, 2, 1) @ case_weight[use]
    log_normalization = logsumexp(log_ratios, axis=1)
    weights = np.exp(log_ratios-log_normalization[:, None])
    baseline = weights[1]
    maximum_probability_delta = np.zeros(3)
    counts = np.zeros((3, 3, response.shape[-1]))
    conditional_counts = np.zeros((len(beta), 3, response.shape[-1]))
    for j, f in enumerate(fields):
        rows = np.flatnonzero(field == f)
        for start in range(0, len(rows), 256):
            selected = rows[start:start+256]
            means = np.zeros((3, len(selected), 3, response.shape[-1]))
            matrix = np.ascontiguousarray(response[selected].transpose(2, 0, 1))
            for a in range(0, len(beta), batch):
                b = min(a+batch, len(beta))
                coefficient = joint_coefficients(model, theta[a:b, j], beta[a:b])
                radial = coefficient.transpose(2, 1, 0, 3).reshape(matrix.shape[0], matrix.shape[2], -1)
                value = (matrix @ radial).reshape(matrix.shape[0], len(selected), b-a, 3).transpose(2, 1, 3, 0)
                total = value.sum(axis=(2, 3), keepdims=True)
                if not np.isfinite(total).all() or (total <= 0).any():
                    raise ValueError('A fitted draw lacks numerator support')
                value /= total
                conditional_counts[a:b] += value.sum(axis=1)
                means += np.einsum('ls,sncd->lncd', weights[:, a:b], value)
            maximum_probability_delta = np.maximum(
                maximum_probability_delta, abs(means-means[1:2]).max(axis=(1, 2, 3)))
            counts += means.sum(axis=1)
    np.testing.assert_allclose(counts.sum(axis=(1, 2)), len(field), rtol=0, atol=1e-7)
    np.testing.assert_allclose(counts, np.einsum('ls,scd->lcd', weights, conditional_counts), rtol=0, atol=1e-7)
    checks = []
    for level in (0, 2):
        count_delta = abs(counts[level]-counts[1]).max()
        probability_delta = maximum_probability_delta[level]
        checks.append(dict(
            level=level, draws=len(beta), maximum_fixed_draw_entry_delta=float(largest_entry_delta[level]),
            absolute_log_normalizer_delta=float(log_normalization[level]-np.log(len(beta))),
            relative_log_weight_span=float(np.ptp(log_ratios[level])),
            weight_ess=float(1/np.sum(weights[level]**2)),
            total_variation=float(.5*abs(weights[level]-baseline).sum()),
            maximum_joint_basis_probability_delta=float(probability_delta),
            maximum_expected_basis_count_delta=float(count_delta)))
    return checks, dict(log_ratios=log_ratios, weights=weights, basis_counts=counts,
                        conditional_basis_counts=conditional_counts)
