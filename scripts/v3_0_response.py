#!/usr/bin/env python3
"""Allocated numerical/performance benchmarks for the common v3 forward core."""
import argparse
import json
import os
import resource
import time
import joblib
import numpy as np
from v2_1_near import ROOT, load, save_json, sha, atomic_npz
from photdis.stage1 import require_compute
from photdis.v3.data import stable_rank
from photdis.v3.fast import spectral_flux_loglikelihood
from photdis.v3.likelihood import flux_loglikelihood
from photdis.v3.inference import infer_reference
from photdis.v3.selection import effective_errors, component_entry, integrate_entry

D = ROOT/'data/v3_0'
OBSERVATION_KEYS = ('flux', 'flux_error', 'transmission', 'band_mask',
                    'photometric_system', 'EBV')


def observation(table, row):
    return {key: table[key][row] for key in OBSERVATION_KEYS}


def main():
    require_compute()
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['benchmark', 'fit'], default='benchmark')
    args = parser.parse_args()
    model_path = D/f'sed_{args.model}.joblib'; model = joblib.load(model_path)
    output = D/f'response_benchmark_{args.model}'; output.mkdir(exist_ok=True)
    t = load(D/'references.npz'); parent = load(D/'application_inputs/field_00.npz')
    rows = np.flatnonzero(t['role'] == 4)
    rows = rows[np.argsort(stable_rank(t['identity'][rows], 301601))[:16]]
    app_rows = np.argsort(stable_rank(parent['identity'], 301601))[:16]
    sample = {key: np.concatenate([t[key][rows], parent[key][app_rows]])
              for key in OBSERVATION_KEYS+('identity', 'TYPE')}
    assert (sample['TYPE'] == 'PSF').all()
    atomic_npz(output/'photometry.npz', **sample)
    del t, parent
    grid = np.linspace(-5., 30., 701)
    timing = []; logdelta = []; posterior = []
    for i in range(len(sample['identity'])):
        o = observation(sample, i)
        a = (o['flux'], o['flux_error'], o['band_mask'], o['transmission'],
             model.means[:, :6], model.covariances[:, :6, :6], grid)
        start = time.monotonic(); fast = spectral_flux_loglikelihood(*a)
        fast_seconds = time.monotonic()-start
        start = time.monotonic(); direct = flux_loglikelihood(*a)
        direct_seconds = time.monotonic()-start
        # Discard only numerically irrelevant log-likelihoods, never posterior modes.
        significant = direct > direct.max()-50
        delta = float(abs(fast[significant]-direct[significant]).max())
        assert delta < 1e-5, ('spectral oracle mismatch', i, delta)
        logdelta.append(delta)
        start = time.monotonic(); result = infer_reference(model, o, initial_step=.1)
        seconds = time.monotonic()-start
        timing.append(dict(row=i, source='reference' if i < len(rows) else 'application',
            direct_seconds=direct_seconds, spectral_seconds=fast_seconds,
            adaptive_seconds=seconds, **result['posterior'].convergence))
        posterior.append(dict(mu=result['mu_quantiles'].tolist(), qnear=result['qnear'],
            class_distance=result['class_distance'].tolist(), modes=result['modes'].tolist(),
            mass=result['mode_mass'].tolist()))
        print('RESPONSE BENCH', i, seconds, result['converged'], flush=True)
    errors = effective_errors(sample)
    selection_checks = []
    for i in (0, 8, 16, 24):
        previous = None
        for draws in (128, 512, 2048):
            start = time.monotonic()
            response = component_entry(model, errors[i], grid, draws=draws)
            alpha = integrate_entry(model, response, grid)
            delta = None if previous is None else float(abs(alpha-previous).max())
            selection_checks.append(dict(row=i, draws=draws, maximum_delta=delta,
                seconds=time.monotonic()-start, alpha=alpha.tolist()))
            previous = alpha
    save_json(output/'result.json', dict(model_sha256=sha(model_path), model_stage=args.model,
        inference_not_scientifically_accepted=True, job=os.getenv('SLURM_JOB_ID'),
        ids=sample['identity'].tolist(), maximum_significant_log_delta=max(logdelta),
        timings=timing, posterior=posterior, selection_checks=selection_checks,
        peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    print(json.dumps(dict(stars=len(timing), median_seconds=np.median([r['adaptive_seconds'] for r in timing]))), flush=True)


if __name__ == '__main__': main()
