"""Observed-object responsibilities and evidence-weighted response inference."""
import numpy as np
from scipy.special import logsumexp
from photdis.population_v2_5 import selected_probabilities


def responsibilities(eta, response, cells):
    mass = np.asarray(eta)[:, None, :] * np.asarray(response)[:, cells].T[None, :, :]
    norm = mass.sum(axis=2, keepdims=True)
    if np.any(norm <= 0):
        raise ValueError('Observed cell has zero probability')
    return mass / norm


def latent_counts(eta_draws, response, cells, seed=261111):
    """One eta per joint draw; all stars share it. No marginal independence."""
    rng = np.random.default_rng(seed)
    total = np.zeros((len(cells), eta_draws.shape[1]))
    half = total.copy()
    draws = np.zeros((len(eta_draws), eta_draws.shape[1]), dtype=np.int16)
    for start in range(0, len(eta_draws), 256):
        stop = min(start+256, len(eta_draws))
        p = responsibilities(eta_draws[start:stop], response, cells)
        total += p.sum(axis=0)
        if start < len(eta_draws)//2:
            half += p[:min(stop, len(eta_draws)//2)-start].sum(axis=0)
        label = (rng.random(p.shape[:2])[..., None] > np.cumsum(p, axis=2)).sum(axis=2)
        label = np.minimum(label, p.shape[2]-1)
        for j in range(p.shape[2]):
            draws[start:stop, j] = np.sum(label == j, axis=1)
    mean = total/len(eta_draws)
    delta = 2*half/len(eta_draws) - 2*(total-half)/len(eta_draws)
    return mean, draws, float(np.max(abs(delta)))


def response_logposterior(counts, eta, responses):
    """Uniform finite response prior and uniform theta quadrature measure.

    Returns normalized within-response theta weights and response evidence weights.
    The selected-catalog size is conditioned on, consistently with v2.5.
    """
    weights=[]; evidence=[]
    for response in responses:
        prob=selected_probabilities(eta,response)
        ll=np.atleast_2d(counts)@np.log(np.maximum(prob,1e-300)).T
        z=logsumexp(ll,axis=1)
        weights.append(np.exp(ll-z[:,None]))
        evidence.append(z-np.log(len(eta)))
    evidence=np.array(evidence).T
    rw=np.exp(evidence-logsumexp(evidence,axis=1)[:,None])
    return np.array(weights),rw,evidence
