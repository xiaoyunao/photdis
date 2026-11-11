#!/usr/bin/env python3
"""Freeze continuous empirical transfer catalogs, then use the common inference."""
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
from photdis.v3.simulation import SCENARIOS, empirical_library, draw_transfer, model_library
from photdis.v3.model import inference_model
import v3_0_predict as predict
import v3_0_qrf_predict as qrf
D = ROOT/'data/v3_0'


def prepare(repeats=8, catalog_size=128):
    output = D/'simulations'; output.mkdir(exist_ok=True)
    paths = [D/'empirical_transfer_bank.npz', D/'cohorts/application.npz', D/'sed_fit.joblib', D/'sed_cloud.npz',
        ROOT/'src/photdis/v3/simulation.py', ROOT/'scripts/v3_0_simulate.py']
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    inputs.update(repeats=repeats, catalog_size=catalog_size)
    freeze = output/'inputs.json'
    if freeze.exists(): assert json.loads(freeze.read_text()) == inputs
    else: save_json(freeze, inputs)
    marker = output/'prepared.json'
    if marker.exists():
        old = json.loads(marker.read_text()); assert old['inputs'] == inputs
        for name, digest in old['outputs'].items(): assert sha(D/name) == digest
        return
    bank = load(paths[0]); context = load(paths[1]); model = joblib.load(paths[2])
    assert (bank['TYPE'] == 'PSF').all() and (context['TYPE'] == 'PSF').all()
    assert not np.intersect1d(bank['identity'], model.training_identity).size
    library = empirical_library(bank)
    atomic_npz(output/'shared_empirical_library.npz', **library)
    phot = []; truth = []; records = []; fields = np.unique(context['field'])
    cloud = load(D/'sed_cloud.npz')
    nodes = cloud['reference_mu'].shape[1]//2
    weight = cloud['reference_weight'][:, :nodes]; weight = weight/weight.sum(axis=1)[:, None]
    mu_mean = (weight*cloud['reference_mu'][:, :nodes]).sum(axis=1)
    mu_sigma = np.sqrt((weight*(cloud['reference_mu'][:, :nodes]-mu_mean[:, None])**2).sum(axis=1))
    factor = 10**(.4*(mu_mean-cloud['reference_mu'][:, 0]))
    same_bank = dict(identity=cloud['identity'], class_label=cloud['labels'], band_mask=cloud['mask'][:, :6],
        flux_at_1kpc=cloud['y'][:, 0, :6]*factor[:, None],
        flux_error_at_1kpc=np.sqrt(np.diagonal(cloud['noise'][:, 0, :6, :6], axis1=1, axis2=2))*factor[:, None],
        reference_sigma=mu_sigma, corrected_feh=cloud['y'][:, 0, 6], corrected_feh_valid=cloud['mask'][:, 6])
    same_library = empirical_library(same_bank, seed=302598)
    del cloud, same_bank
    scenarios = list(SCENARIOS)+['known_model', 'same_library']
    index = 0
    for scenario in scenarios:
        for repeat in range(repeats):
            field = fields[repeat % len(fields)]; use = context['field'] == field
            local = {k: v[use] for k, v in context.items()}
            active_library = model_library(inference_model(model), seed=302700+repeat) if scenario == 'known_model' else same_library if scenario == 'same_library' else library
            generator_scenario = 'reference_like' if scenario in ('known_model', 'same_library') else scenario
            p, t, info = draw_transfer(active_library, local, generator_scenario, n=catalog_size,
                seed=302600+index, first_identity=index*catalog_size, known_distance=scenario in ('known_model', 'same_library'))
            t.update(identity=p['identity'], catalog=np.full(catalog_size, index, 'i4'),
                     scenario=np.full(catalog_size, scenario), repeat=np.full(catalog_size, repeat, 'i2'))
            # Synthetic signed Gaia is a diagnostic only and never an inference input.
            rng = np.random.default_rng(303600+index)
            t['parallax_error'] = np.exp(rng.uniform(np.log(.02), np.log(.3), catalog_size))
            t['parallax'] = 10**((10-t['mu'])/5)+rng.normal(size=catalog_size)*t['parallax_error']
            phot.append(p); truth.append(t)
            records.append(dict(catalog=index, scenario=scenario, repeat=repeat, field=int(field),
                unique_template_ids=len(np.unique(t['template_identity'])),
                template_training_overlap=int(np.isin(t['template_identity'], model.training_identity).sum()), **info))
            index += 1
    p = {k: np.concatenate([x[k] for x in phot]) for k in phot[0]}
    t = {k: np.concatenate([x[k] for x in truth]) for k in truth[0]}
    atomic_npz(D/'cohorts/simulations.npz', **p)
    atomic_npz(output/'truth.npz', **t)
    save_json(output/'catalogs.json', records)
    names = ['cohorts/simulations.npz', 'simulations/truth.npz', 'simulations/catalogs.json', 'simulations/shared_empirical_library.npz']
    save_json(marker, dict(inputs=inputs, outputs={name: sha(D/name) for name in names},
        rows=len(p['identity']), catalogs=index, independent_source_bank=len(bank['identity']),
        used_source_ids=len(np.unique(t['template_identity'])),
        scenarios=scenarios, job=os.getenv('SLURM_JOB_ID'),
        truth_scope='Fixed independent source library, teacher-scale and sampled flux uncertainty; synthetic PSF conditional noise/entry. Repeats do not create new real reference sources.',
        prior_regression_controls='Known model draws, same fitted source library (deliberately optimistic), independent empirical transfer; historical v2.6 artifacts preserved.'))
    print('SIMULATION PREPARED', len(p['identity']), 'catalogs', index, flush=True)


def run_predictions(view, workers):
    source = D/'cohorts/simulations.npz'; model_path = D/'sed_fit.joblib'; control_path = D/'controls/qrf_grz.joblib'
    t = load(source); model = joblib.load(model_path); control = joblib.load(control_path)
    assert model.converged and (t['TYPE'] == 'PSF').all()
    paths = [source, model_path, control_path, ROOT/'scripts/v3_0_simulate.py',
        ROOT/'scripts/v3_0_predict.py', ROOT/'scripts/v3_0_qrf_predict.py']
    paths += [ROOT/'src/photdis/v3'/name for name in ('model.py', 'inference.py', 'fast.py', 'likelihood.py', 'compression.py', 'controls.py')]
    frozen = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    frozen.update(view=view, simulation=True)
    folder = 'simulations_qrf_0' if view == 'qrf' else f'simulations_{view}_fit_0'
    output = D/'predictions'/folder; output.mkdir(parents=True, exist_ok=True)
    freeze = output/'inputs.json'
    if freeze.exists(): assert json.loads(freeze.read_text()) == frozen
    else: save_json(freeze, frozen)
    if view == 'qrf':
        qrf._T = t; qrf._MODEL = control; qrf._OUTPUT = output; qrf._INPUTS = frozen
        worker = qrf.chunk; chunk_size = 256
    else:
        predict._TABLE = t; predict._MODEL = inference_model(model); predict._CONTROL = control
        predict._DEST = output; predict._FROZEN = frozen
        predict._BANDS = np.array([1, 1, 0, 1, 0, 0], bool) if view == 'grz' else np.ones(6, bool)
        worker = predict.chunk; chunk_size = 32
    tasks = [np.arange(i, min(i+chunk_size, len(t['identity']))) for i in range(0, len(t['identity']), chunk_size)]
    start = time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork')) as pool:
        records = list(pool.map(worker, tasks))
    save_json(output/'complete.json', dict(inputs=frozen, chunks={p: sha(p) for p in records},
        rows=len(t['identity']), elapsed_seconds=time.monotonic()-start, job=os.getenv('SLURM_JOB_ID')))
    print('SIMULATION PREDICTIONS', view, len(t['identity']), flush=True)


def main():
    require_compute(); parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'predict'])
    parser.add_argument('--view', choices=['grz', 'multiband', 'qrf'], default='multiband')
    parser.add_argument('--workers', type=int, default=64)
    parser.add_argument('--repeats', type=int, default=8)
    args = parser.parse_args()
    if args.stage == 'prepare': prepare(args.repeats)
    else: run_predictions(args.view, args.workers)


if __name__ == '__main__': main()
