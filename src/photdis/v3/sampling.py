"""Metropolis-corrected, fixed-metric HMC with warmup-only step adaptation."""
import numpy as np


def hmc(objective, initial, metric, *, draws=512, warmup=512, seed=302301,
        step_size=.1, leapfrog_steps=8):
    rng = np.random.default_rng(seed)
    initial = np.asarray(initial, float); factor = np.linalg.cholesky(metric)
    # z = initial + factor @ y; sample unit momentum in whitened coordinates.
    def evaluate(y):
        value, gradient = objective(initial+factor@y)
        return value, factor.T@gradient
    y = rng.normal(0, .05, len(initial)); value, grad = evaluate(y)
    samples = []; acceptance = []; errors = []; eps = step_size
    running_log_eps = np.log(eps); average_log_eps = np.log(eps)
    mu = np.log(10*eps); hbar = 0.
    for iteration in range(warmup+draws):
        p0 = rng.normal(size=len(y)); p = p0.copy(); trial = y.copy(); trial_grad = grad.copy()
        p -= .5*eps*trial_grad
        steps = int(rng.integers(max(2, leapfrog_steps//2), leapfrog_steps*3//2+1))
        trial_value = np.inf
        for j in range(steps):
            trial += eps*p
            trial_value, trial_grad = evaluate(trial)
            if not np.isfinite(trial_value) or not np.isfinite(trial_grad).all(): break
            p -= eps*trial_grad*(.5 if j == steps-1 else 1.)
        energy_error = trial_value+.5*(p@p)-value-.5*(p0@p0)
        probability = float(np.exp(min(0., -energy_error))) if np.isfinite(energy_error) else 0.
        if rng.random() < probability: y, value, grad = trial, trial_value, trial_grad
        if iteration < warmup:
            t = iteration+1; eta = 1/(t+10)
            hbar = (1-eta)*hbar+eta*(.8-probability)
            running_log_eps = mu-np.sqrt(t)/.05*hbar
            running_log_eps = np.clip(running_log_eps, -12, 1)
            eta_bar = t**(-.75)
            average_log_eps = eta_bar*running_log_eps+(1-eta_bar)*average_log_eps
            eps = float(np.exp(average_log_eps if t == warmup else running_log_eps))
        else:
            samples.append(initial+factor@y); acceptance.append(probability); errors.append(energy_error)
    return dict(samples=np.asarray(samples), acceptance=np.asarray(acceptance),
                energy_error=np.asarray(errors), step_size=eps,
                divergent=int(np.sum(np.abs(errors) > 100)), seed=seed)


def chain_diagnostics(samples):
    """Conservative split-chain Rhat and initial-positive autocorrelation ESS."""
    x = np.asarray(samples, float)
    chains, draws, dimension = x.shape
    if chains < 2 or draws < 8: raise ValueError('At least two chains with eight draws required')
    half = draws//2; x = np.concatenate([x[:, :half], x[:, -half:]], axis=0)
    m, n, _ = x.shape
    within = x.var(axis=1, ddof=1).mean(axis=0)
    between = n*x.mean(axis=1).var(axis=0, ddof=1)
    variance = (n-1)/n*within+between/n
    rhat = np.sqrt(variance/np.maximum(within, 1e-300))
    centered = x-x.mean(axis=1, keepdims=True)
    spectrum = np.fft.rfft(centered, n=2*n, axis=1)
    autocov = np.fft.irfft(spectrum*np.conj(spectrum), n=2*n, axis=1)[:, :n].mean(axis=0)/n
    rho = 1-(within[None]-autocov)/np.maximum(variance[None], 1e-300)
    rho[0] = 1
    pair = rho[::2][:n//2]+rho[1::2]
    valid = np.cumprod(pair > 0, axis=0).astype(bool)
    pair = np.minimum.accumulate(pair, axis=0)
    tau = np.maximum(-1+2*np.sum(np.where(valid, pair, 0), axis=0), 1.)
    ess = np.minimum(m*n/tau, m*n)
    return dict(rhat=rhat, ess=ess, max_rhat=float(rhat.max()), minimum_ess=float(ess.min()))
