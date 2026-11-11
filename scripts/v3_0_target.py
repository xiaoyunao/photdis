#!/usr/bin/env python3
"""Build joint responses and fit one selection-normalized sky hierarchy."""
import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
import time
import joblib
import numpy as np
from v2_1_near import ROOT, load, sha, save_json, atomic_npz
from photdis.stage1 import require_compute
from photdis.v3.model import inference_model
from photdis.v3.compression import unpack
from photdis.v3.target import component_response, joint_coefficients, joint_draw_probabilities
from photdis.v3.hierarchy import FieldPopulation
from photdis.v3.fast_hierarchy import FastFieldPopulation
from photdis.v3.sampling import hmc, chain_diagnostics
from photdis.v3.selection import effective_errors
from photdis.v3.selection_table import SelectionTable
from photdis.v3.population import SharedCounts

D = ROOT/'data/v3_0'
_HIERARCHY = _FIT = _ARGS = None


def chain(index):
    return hmc(_HIERARCHY.objective, _FIT['map'], _FIT['metric'],
               draws=_ARGS.draws, warmup=_ARGS.warmup, seed=302401+index)


def main():
    global _HIERARCHY, _FIT, _ARGS
    require_compute(); parser = argparse.ArgumentParser()
    parser.add_argument('cohort', choices=['application', 'development'])
    parser.add_argument('--engine', choices=['original', 'cached'], default='original')
    parser.add_argument('--reuse-response', action='store_true')
    parser.add_argument('--model', choices=['benchmark', 'fit'], default='fit')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--profiles', type=int, default=64)
    parser.add_argument('--draws', type=int, default=512)
    parser.add_argument('--warmup', type=int, default=512)
    parser.add_argument('--chains', type=int, default=4)
    _ARGS = args = parser.parse_args(); start = time.monotonic()
    prediction = D/'predictions'/f'{args.cohort}_multiband_{args.model}_{args.limit}'
    assert (prediction/'complete.json').exists(), 'Incomplete nominal predictions'
    selection_dir = D/'selection'/f'{args.model}_-1_{args.profiles}'
    selection_record = json.loads((selection_dir/'complete.json').read_text())
    assert selection_record['passed'], 'Selection table failed direct validation'
    table_path = selection_dir/'table.npz'; assert sha(table_path) == selection_record['sha256']
    table = SelectionTable(**load(table_path))
    model_path = D/f'sed_{args.model}.joblib'; model = inference_model(joblib.load(model_path))
    source = load(D/'cohorts'/(args.cohort+'.npz'))
    base_output = D/'targets'/f'{args.cohort}_{args.model}_{args.limit}'
    output = base_output if args.engine == 'original' else base_output.with_name(base_output.name+f'_cached_s{args.draws}')
    output.mkdir(parents=True, exist_ok=True)
    paths = [prediction/'complete.json', table_path, model_path, D/'cohorts'/(args.cohort+'.npz'),
        ROOT/'scripts/v3_0_target.py']
    paths += [ROOT/'src/photdis/v3'/x for x in ('target.py', 'hierarchy.py', 'fast_hierarchy.py', 'sampling.py', 'compression.py', 'population.py')]
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    inputs.update(draws=args.draws, warmup=args.warmup, chains=args.chains, engine=args.engine, reuse_response=args.reuse_response)
    if args.reuse_response:
        assert args.engine == 'cached'
        original_inputs = json.loads((base_output/'inputs.json').read_text())
        for p in [prediction/'complete.json', table_path, model_path, D/'cohorts'/(args.cohort+'.npz')]:
            assert original_inputs[str(p.relative_to(ROOT))] == sha(p)
        for p in [base_output/'component_response.npz', base_output/'map.joblib']:
            inputs[str(p.relative_to(ROOT))] = sha(p)
    freeze = output/'inputs.json'
    if freeze.exists(): assert json.loads(freeze.read_text()) == inputs, 'Explicit target-run migration required'
    else: save_json(freeze, inputs)
    marker = output/'complete.json'
    if marker.exists():
        record = json.loads(marker.read_text()); assert record['inputs'] == inputs
        for name, digest in record['outputs'].items(): assert sha(output/name) == digest
        if not record['passed']: raise RuntimeError('Cached target sampler failed; explicit refinement required')
        return
    response_path = (base_output if args.reuse_response else output)/'component_response.npz'
    if not response_path.exists():
        responses = []; rows = []; identities = []; evidence = []
        order = np.argsort(source['identity'])
        for path in sorted(prediction.glob('chunk_*_joint.npz')):
            summary_path = path.with_name(path.name.replace('_joint', ''))
            metadata = json.loads(summary_path.with_suffix('.json').read_text())
            assert sha(path) == metadata['joint_sha256'] and sha(summary_path) == metadata['summary_sha256']
            payload = load(path); summary = load(summary_path)
            index = np.searchsorted(source['identity'][order], summary['identity'])
            selected = order[index]
            np.testing.assert_array_equal(source['identity'][selected], summary['identity'])
            for j in range(len(selected)):
                record = unpack(payload, j)
                responses.append(component_response(record, model).astype('f4'))
                evidence.append(record['source_log_evidence'])
            rows.extend(selected); identities.extend(summary['identity'])
        rows = np.asarray(rows, 'i8')
        alpha, audit = table.evaluate(effective_errors({k: v[rows] for k, v in source.items()}))
        if not audit['inside_profile_hull'].all(): raise RuntimeError('Target error profile outside certified selection')
        atomic_npz(response_path, response=np.array(responses), selection=alpha.astype('f4'),
            identity=np.asarray(identities), rows=rows, reference_log_evidence=evidence)
    r = load(response_path); rows = r['rows']; response = r['response']; selection = r['selection']
    assert (source['TYPE'][rows] == 'PSF').all()
    hierarchy_type = FastFieldPopulation if args.engine == 'cached' else FieldPopulation
    _HIERARCHY = hierarchy_type(response.sum(axis=3), selection, model, source['field'][rows], source['design_weight'][rows])
    fit_path = output/'map.joblib'
    if fit_path.exists(): _FIT = joblib.load(fit_path)
    elif args.reuse_response:
        _FIT = joblib.load(base_output/'map.joblib')
        value, grad = _HIERARCHY.objective(_FIT['map'])
        assert abs(value-_FIT['objective']) < 1e-6 and abs(grad).max() < .001
        joblib.dump(_FIT, fit_path)
    else:
        _FIT = _HIERARCHY.fit(); joblib.dump(_FIT, fit_path)
    print('TARGET MAP', _FIT['objective'], _FIT['gradient_max'], _HIERARCHY.dimension, flush=True)
    chain_path = output/'chain_records.joblib'
    if chain_path.exists(): records = joblib.load(chain_path)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.chains, mp_context=mp.get_context('fork')) as pool:
            records = list(pool.map(chain, range(args.chains)))
        joblib.dump(records, chain_path)
    samples = np.array([x['samples'] for x in records]); diagnostic = chain_diagnostics(samples)
    atomic_npz(output/'chains.npz', samples=samples, rhat=diagnostic['rhat'], ess=diagnostic['ess'],
        acceptance=np.array([x['acceptance'] for x in records]),
        energy_error=np.array([x['energy_error'] for x in records]))
    # All stars/chunks use the same field/global posterior draw indices. Marginal
    # precision is checked after combining chains, never claimed from Hessian widths.
    chosen = samples.transpose(1, 0, 2).reshape(-1, samples.shape[-1])
    draw_ids = np.arange(len(chosen), dtype='i8')
    counts = SharedCounts(draw_ids, 18, seed=302499)
    probability_mean = np.zeros((len(rows), 3, 6))
    fields = source['field'][rows]
    decoded = [_HIERARCHY.decode(z)[:2] for z in chosen]
    all_theta = np.array([x[0] for x in decoded]); all_beta = np.array([x[1] for x in decoded])
    for fi, field in enumerate(_HIERARCHY.fields):
        coefficients = joint_coefficients(model, all_theta[:, fi], all_beta)
        field_rows = np.flatnonzero(fields == field)
        for first in range(0, len(field_rows), 32):
            use = field_rows[first:first+32]
            probabilities = joint_draw_probabilities(response[use], coefficients)
            mean = counts.add(probabilities.reshape(len(chosen), len(use), 18), draw_ids)
            probability_mean[use] = mean.reshape(-1, 3, 6)
    result = counts.result()
    atomic_npz(output/'counts.npz', **result)
    atomic_npz(output/'joint_probability.npz', identity=r['identity'], TYPE=source['TYPE'][rows],
        probability=probability_mean, field=fields, design_weight=source['design_weight'][rows])
    # These count totals concern the sampled objects. Parent-expansion design
    # covariance is a separate downstream artifact, never hidden in this covariance.
    passed = diagnostic['max_rhat'] < 1.05 and diagnostic['minimum_ess'] >= 100 and sum(x['divergent'] for x in records) == 0 and result['marginal_half_delta'] < .005
    save_json(output/'complete.json', dict(inputs=inputs, n=len(rows), model=args.model,
        fields=_HIERARCHY.fields.tolist(), dimensions=_HIERARCHY.dimension,
        max_rhat=diagnostic['max_rhat'], minimum_ess=diagnostic['minimum_ess'],
        divergent=sum(x['divergent'] for x in records),
        marginal_half_delta=result['marginal_half_delta'], passed=passed,
        case_weight_sum=_HIERARCHY.case_weight_sum, effective_case_count=_HIERARCHY.effective_case_count,
        conditional_scope='Fixed SED/reference calibration; shared response ensemble still required',
        evidence_constant=float((source['design_weight'][rows]/source['design_weight'][rows].mean())@r['reference_log_evidence']),
        unit_weight_evidence_constant=float(np.sum(r['reference_log_evidence'])),
        weight_scope='Design-rescaled generalized likelihood, sum of weights equals observed sample size; not expanded parent independent evidence',
        outputs={p.name: sha(p) for p in output.iterdir() if p.suffix in ('.npz', '.joblib')},
        elapsed_seconds=time.monotonic()-start, job=os.getenv('SLURM_JOB_ID')))
    print('TARGET COMPLETE', passed, diagnostic['max_rhat'], diagnostic['minimum_ess'], flush=True)
    if not passed: raise RuntimeError('Target chains require numerical refinement; do not claim converged counts')


if __name__ == '__main__': main()
