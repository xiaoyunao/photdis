#!/usr/bin/env python3
"""Continuous same-catalog point/PDF/target recovery with honest count intervals."""
import argparse
import json
import os
import time
import joblib
import numpy as np
from v2_1_near import ROOT, load, sha, save_json, atomic_npz, csv_rows
from photdis.stage1 import require_compute
from photdis.v3.model import inference_model
from photdis.v3.compression import unpack, DISTANCE_MU_BOUNDARIES
from photdis.v3.target import component_response, joint_coefficients, joint_draw_probabilities
from photdis.v3.hierarchy import FieldPopulation
from photdis.v3.fast_hierarchy import FastFieldPopulation
from photdis.v3.sampling import hmc, chain_diagnostics
from photdis.v3.selection import effective_errors
from photdis.v3.selection_table import SelectionTable
from photdis.v3.population import SharedCounts
from photdis.v3.marginal import marginal_coefficients, marginal_star
from photdis.v3.metrics import distance_summary, observed_parallax
from photdis.v3.inference import kernel_loglike, distance_basis
from scipy.special import logsumexp
from photdis.v3.likelihood import trapezoid_weights
from v3_0_response import observation
D = ROOT/'data/v3_0'


def main():
    require_compute(); parser = argparse.ArgumentParser()
    parser.add_argument('--catalog', type=int, required=True)
    parser.add_argument('--engine', choices=['original', 'cached'], default='cached')
    parser.add_argument('--view', choices=['grz', 'multiband'], default='multiband')
    parser.add_argument('--draws', type=int, default=512); parser.add_argument('--warmup', type=int, default=512)
    parser.add_argument('--chains', type=int, default=4); parser.add_argument('--profiles', type=int, default=256)
    args = parser.parse_args(); start = time.monotonic()
    pred = D/'predictions'/f'simulations_{args.view}_fit_0'
    qrf = D/'predictions/simulations_qrf_0'
    for path in (pred, qrf): assert (path/'complete.json').exists()
    source_path = D/'cohorts/simulations.npz'; truth_path = D/'simulations/truth.npz'
    selection_path = D/'selection'/f'fit_-1_{args.profiles}/table.npz'; model_path = D/'sed_fit.joblib'
    selection_record = json.loads(selection_path.with_name('complete.json').read_text())
    assert selection_record['passed'] and selection_record['sha256'] == sha(selection_path)
    paths = [source_path, truth_path, model_path, selection_path, pred/'complete.json', qrf/'complete.json', ROOT/'scripts/v3_0_recovery.py']
    paths += [ROOT/'src/photdis/v3'/x for x in ('hierarchy.py', 'fast_hierarchy.py', 'target.py', 'sampling.py', 'marginal.py', 'population.py', 'metrics.py')]
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in paths}; inputs.update(vars(args))
    out = D/'recovery'/args.view/f'catalog_{args.catalog:03}_s{args.draws}_{args.engine}'
    out.mkdir(parents=True, exist_ok=True)
    if (out/'inputs.json').exists(): assert json.loads((out/'inputs.json').read_text()) == inputs
    else: save_json(out/'inputs.json', inputs)
    if (out/'complete.json').exists():
        old = json.loads((out/'complete.json').read_text())
        assert old['inputs'] == inputs
        for name, digest in old['outputs'].items(): assert sha(out/name) == digest
        return
    source = load(source_path); truth = load(truth_path)
    use = np.flatnonzero(truth['catalog'] == args.catalog); assert len(use) == 128
    phot = {k: v[use] for k, v in source.items()}; truth = {k: v[use] for k, v in truth.items()}
    assert (phot['TYPE'] == 'PSF').all(); np.testing.assert_array_equal(phot['identity'], truth['identity'])
    model = inference_model(joblib.load(model_path)); table = SelectionTable(**load(selection_path))
    compact = []; summary = []; grids = []; qrecords = []
    for first in range(int(use[0]), int(use[-1])+1, 32):
        path = pred/f'chunk_{first:08}.npz'; marker = json.loads(path.with_suffix('.json').read_text())
        joint = path.with_name(path.stem+'_joint.npz')
        assert sha(path) == marker['summary_sha256'] and sha(joint) == marker['joint_sha256']
        s = load(path); p = load(joint); summary.append(s)
        for j in range(len(s['identity'])):
            compact.append(unpack(p, j)); lo, hi = p['full_grid_offset'][j:j+2]
            grids.append((p['full_mu'][lo:hi], p['full_logpdf'][lo:hi]))
    s = {k: np.concatenate([r[k] for r in summary]) for k in summary[0]}
    np.testing.assert_array_equal(s['identity'], phot['identity'])
    for first in sorted(set((use//256)*256)):
        path = qrf/f'chunk_{first:08}.npz'; record = json.loads(path.with_suffix('.json').read_text())
        assert sha(path) == record['sha256']; q = load(path)
        idx = np.flatnonzero(np.isin(q['identity'], phot['identity']))
        qrecords.append({k: v[idx] for k, v in q.items() if k != 'Mg_edges'} | {'Mg_edges': q['Mg_edges']})
    q = qrecords[0]; np.testing.assert_array_equal(q['identity'], phot['identity'])
    response_path = out/'response.npz'
    if response_path.exists(): r = load(response_path); response, alpha = r['response'], r['selection']
    else:
        response = np.array([component_response(r, model) for r in compact])
        alpha, audit = table.evaluate(effective_errors(phot))
        if not audit['inside_profile_hull'].all(): raise ValueError('Uncertified simulation selection profile')
        atomic_npz(response_path, response=response, selection=alpha)
    hierarchy_type = FastFieldPopulation if args.engine == 'cached' else FieldPopulation
    hierarchy = hierarchy_type(response.sum(axis=3), alpha, model, phot['field'])
    fit_path = out/'map.joblib'
    if fit_path.exists(): fit = joblib.load(fit_path)
    else: fit = hierarchy.fit(); joblib.dump(fit, fit_path)
    print('RECOVERY MAP', args.catalog, args.view, fit['objective'], flush=True)
    records = []
    for i in range(args.chains):
        path = out/f'chain_{i}.joblib'
        if path.exists(): record = joblib.load(path)
        else:
            record = hmc(hierarchy.objective, fit['map'], fit['metric'], draws=args.draws,
                         warmup=args.warmup, seed=304000+args.catalog*10+i)
            joblib.dump(record, path)
        records.append(record); print('RECOVERY CHAIN', args.catalog, i, flush=True)
    samples = np.array([r['samples'] for r in records]); diagnostics = chain_diagnostics(samples)
    z = samples.transpose(1, 0, 2).reshape(-1, samples.shape[-1])
    decoded = [hierarchy.decode(x)[:2] for x in z]
    theta = np.array([x[0][0] for x in decoded]); beta = np.array([x[1] for x in decoded])
    coefficients = joint_coefficients(model, theta, beta)
    probabilities = joint_draw_probabilities(response, coefficients)
    counts = SharedCounts(np.arange(len(z)), 18, seed=305000+args.catalog)
    target_joint = counts.add(probabilities.reshape(len(z), len(use), 18), np.arange(len(z))).reshape(-1, 3, 6)
    count_result = counts.result(); atomic_npz(out/'counts.npz', **count_result)
    average, evidence = marginal_coefficients(response, coefficients)
    target = [marginal_star(r, model, c) for r, c in zip(compact, average)]
    np.testing.assert_allclose(np.array([r['class_distance'] for r in target]), target_joint, atol=2e-10)
    target_mu = np.array([r['mu'] for r in target]); target_logscore = []; target_gaia = []; reference_gaia = []
    direct_error = []; bands = np.array([1, 1, 0, 1, 0, 0], bool) if args.view == 'grz' else None
    for i, (grid, reference_logpdf) in enumerate(grids):
        # Recompute native likelihood on the certified nominal integration grid.
        # This verifies compressed reweighting without using mock truth as input.
        likelihood, _ = kernel_loglike(model, observation(phot, i), grid, bands=bands)
        radial_prior = average[i].sum(axis=2)@distance_basis(grid)
        b = np.array([1-model.binary_reference_fraction, model.binary_reference_fraction])
        logpdf = logsumexp(likelihood+np.log(b)[:, None, None]+np.log(np.maximum(radial_prior, 1e-300))[None], axis=(0, 1))
        lognorm = logsumexp(logpdf+np.log(trapezoid_weights(grid))); logpdf -= lognorm
        direct_error.append(abs(np.exp(lognorm-compact[i]['source_log_evidence'])-1))
        target_logscore.append(np.interp(truth['mu'][i], grid, logpdf, left=-np.inf, right=-np.inf))
        target_gaia.append(observed_parallax(grid, logpdf, truth['parallax'][i], truth['parallax_error'][i]))
        reference_gaia.append(observed_parallax(grid, reference_logpdf, truth['parallax'][i], truth['parallax_error'][i]))
    truth_bin = np.searchsorted(DISTANCE_MU_BOUNDARIES, truth['mu'], side='right')
    truth_counts = np.bincount(truth['class_label']*6+truth_bin, minlength=18).reshape(3, 6)
    distance_truth = truth_counts.sum(axis=0)
    qedges = q['g0'][:, None]-q['Mg_edges'][None, ::-1]; qpdf = q['pdf'][:, ::-1]
    qcdfs = np.array([np.sum(qpdf*np.clip((x-qedges[:, :-1])/np.diff(qedges), 0, 1), axis=1) for x in DISTANCE_MU_BOUNDARIES]).T
    qdist = np.diff(np.c_[np.zeros(len(qpdf)), qcdfs, np.ones(len(qpdf))], axis=1)
    # Count intervals include latent membership and shared population variation.
    draws = count_result['count_draws'].reshape(-1, 3, 6)
    rows = []
    models = [('qrf_point', np.eye(6)[np.searchsorted(DISTANCE_MU_BOUNDARIES, q['mu'][:, 2], side='right')]),
              ('qrf_pdf', qdist), ('v3_reference', s['class_distance'].sum(axis=1)), ('v3_target', target_joint.sum(axis=1))]
    for method, probability in models:
        for b in range(6):
            selected_draws = draws.sum(axis=1)[:, b] if method == 'v3_target' else None
            lo, hi = np.quantile(selected_draws, [.025, .975]) if selected_draws is not None else (np.nan, np.nan)
            rows.append(dict(method=method, axis='distance', bin=b, truth=int(distance_truth[b]), mean=float(probability[:, b].sum()), lower95=lo, upper95=hi,
                covered95=bool(lo <= distance_truth[b] <= hi) if selected_draws is not None else None))
    for c in range(3):
        for b in range(6):
            lo, hi = np.quantile(draws[:, c, b], [.025, .975])
            rows.append(dict(method='v3_target', axis='class_distance', bin=c*6+b, truth=int(truth_counts[c, b]),
                mean=float(target_joint[:, c, b].sum()), lower95=lo, upper95=hi, covered95=bool(lo <= truth_counts[c, b] <= hi)))
    csv_rows(out/'recovery.csv', rows)
    metrics = []
    for name, mu in [('qrf', q['mu']), ('reference', s['mu']), ('target', target_mu)]:
        for label, mask in [('all', np.ones(len(use), bool))]+[(f'class_{c}', truth['class_label'] == c) for c in range(3)]+[('2_5kpc', truth_bin == 1)]:
            metrics.append(dict(method=name, group=label, **distance_summary(mu[mask], truth['mu'][mask])))
    csv_rows(out/'distance.csv', metrics)
    atomic_npz(out/'per_star.npz', identity=phot['identity'], TYPE=phot['TYPE'], mu=target_mu,
        class_distance=target_joint, truth_mu=truth['mu'], truth_class=truth['class_label'],
        logscore=target_logscore, gaia_pit=[r['pit'] for r in target_gaia], gaia_logscore=[r['log_score'] for r in target_gaia],
        reference_gaia_pit=[r['pit'] for r in reference_gaia], reference_gaia_logscore=[r['log_score'] for r in reference_gaia],
        direct_normalization_error=direct_error, omitted_bound=[r['omitted_posterior_bound'] for r in target])
    sampler_passed = diagnostics['max_rhat'] < 1.05 and diagnostics['minimum_ess'] >= 100 and not sum(x['divergent'] for x in records)
    passed = sampler_passed and count_result['marginal_half_delta'] < .005 and max(direct_error) < .005 and max(r['omitted_posterior_bound'] for r in target) < .001
    save_json(out/'complete.json', dict(inputs=inputs, catalog=args.catalog, scenario=str(truth['scenario'][0]),
        n=len(use), shared_response_scope='Conditional on fitted SED; shared response comparison is a separate required artifact',
        max_rhat=diagnostics['max_rhat'], minimum_ess=diagnostics['minimum_ess'], sampler_passed=bool(sampler_passed), passed=bool(passed),
        marginal_half_delta=count_result['marginal_half_delta'], max_direct_normalization_error=float(max(direct_error)),
        max_omitted_bound=max(r['omitted_posterior_bound'] for r in target),
        outputs={p.name: sha(p) for p in out.iterdir() if p.suffix in ('.npz', '.joblib', '.csv')},
        elapsed_seconds=time.monotonic()-start, job=os.getenv('SLURM_JOB_ID')))
    print('RECOVERY COMPLETE', args.catalog, passed, diagnostics['max_rhat'], flush=True)


if __name__ == '__main__': main()
