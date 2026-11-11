"""Evidence-weighted total count covariance across shared response models."""
import numpy as np
from scipy.special import softmax


def combine_counts(records, log_evidence):
    """Preserve latent, target-parameter and between-response covariance terms."""
    weights = softmax(np.asarray(log_evidence, float))
    if len(records) != len(weights) or not np.isfinite(weights).all(): raise ValueError('Finite evidence for every response required')
    n = np.array([int(r['n']) for r in records])
    if not (n == n[0]).all(): raise ValueError('Shared response catalog identities/counts must agree')
    means = np.array([r['mean'] for r in records]); mean = weights@means
    conditional = np.einsum('r,rij->ij', weights, np.array([r['conditional_covariance'] for r in records]))
    within = np.einsum('r,rij->ij', weights, np.array([r['shared_covariance'] for r in records]))
    delta = means-mean; between = (delta.T*weights)@delta
    covariance = conditional+within+between
    count_draws = []; draw_weights = []; draw_ids = []
    for response, (r, weight) in enumerate(zip(records, weights)):
        values = np.asarray(r['count_draws'])
        if values.dtype != np.int64 or not (values.sum(axis=1) == n[0]).all(): raise ValueError('Exact int64 observed counts required')
        count_draws.append(values); draw_weights.append(weight*r['draw_weights'])
        draw_ids.append(response*10**9+np.arange(len(values), dtype='i8'))
    return dict(n=int(n[0]), mean=mean, covariance=covariance,
        conditional_covariance=conditional, within_response_covariance=within, between_response_covariance=between,
        response_weights=weights, response_effective_sample_size=float(1/(weights@weights)),
        count_draws=np.concatenate(count_draws), draw_weights=np.concatenate(draw_weights), draw_ids=np.concatenate(draw_ids))


def combine_star_means(means, half_a, half_b, log_evidence):
    weights = softmax(np.asarray(log_evidence, float))
    mean = np.einsum('r,rik->ik', weights, means)
    a = np.einsum('r,rik->ik', weights, half_a); b = np.einsum('r,rik->ik', weights, half_b)
    return mean, float(abs(a-b).max())


def merge_chunk_counts(records):
    """Merge disjoint object chunks before taking shared-parameter covariance."""
    first=records[0]; ids=first['draw_ids']; weights=first['draw_weights']
    for r in records:
        if not np.array_equal(ids,r['draw_ids']) or not np.array_equal(weights,r['draw_weights']):
            raise ValueError('Every chunk must use identical global parameter draws')
    n=sum(int(r['n']) for r in records)
    means=sum(r['conditional_means'] for r in records)
    conditional=sum(r['conditional_covariance'] for r in records)
    mean=weights@means; delta=means-mean
    shared=(delta.T*weights)@delta
    realized=sum(r['count_draws'] for r in records)
    if realized.dtype != np.int64 or not (realized.sum(axis=1)==n).all():
        raise ValueError('Invalid chunk count totals')
    return dict(n=n,mean=mean,covariance=conditional+shared,conditional_covariance=conditional,
        shared_covariance=shared,conditional_means=means,count_draws=realized,
        draw_ids=ids,draw_weights=weights,
        marginal_half_delta=max(float(r['marginal_half_delta']) for r in records))
