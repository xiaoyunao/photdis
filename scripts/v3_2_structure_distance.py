#!/usr/bin/env python3
"""Independent narrow-prior entry and native response on added NGC288 nodes."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path

import joblib
import numpy as np
from scipy.special import logsumexp, roots_hermitenorm, softmax

from photdis.stage1 import require_compute
from photdis.v3.model import inference_model
from photdis.v3.selection import OPTICAL, gaussian_component_entry
from photdis.v3.systematics import calibration_vectors
from photdis.v3_1.candidate_selection import candidate_thresholds
from photdis.v3_1.gaussian_response import class_gaussian_logresponse
from photdis.v3_1.selection_dust import entry_model_with_local_dust
from photdis.v3_1.structure import structure_quadrature, withdrawn_gaussian_logresponse
from v2_1_near import ROOT, atomic_npz, csv_rows, load, save_json, sha
from v3_2_radial_checkpoint import compute_rows
from v3_2_tilt_inputs import verified

ROWS = (273, 436, 224)
_STATE = None


def narrow_entry(model, error, means, *, order, draws, seed):
    """Integrate the registered 0.02-mag narrow radial kernel at each node."""
    nodes, weights = roots_hermitenorm(order)
    points = (means[:, None] + .02 * nodes).ravel()
    raw = gaussian_component_entry(model, error, points, draws=draws, seed=seed)
    prior = model.weights[:, None] * model.classes
    prior /= prior.sum(axis=0)
    fraction = model.binary_reference_fraction
    curves = prior.T @ ((1-fraction)*raw[0] + fraction*raw[1])
    return curves.reshape(3, len(means), order) @ (weights/weights.sum())


def native_responses(table, means):
    sigmas = np.full(len(means), .02)
    values = []
    for grid in ('coarse', 'fine'):
        native = class_gaussian_logresponse(
            table[f'{grid}_class_loglike'], table[f'{grid}_grid'], means,
            sigmas, np.zeros((1, 3, 2)))[0]
        gaussian = withdrawn_gaussian_logresponse(
            table[f'{grid}_reference_logevidence'], table[f'{grid}_reference_mean'],
            table[f'{grid}_reference_sigma'], means, sigmas)
        values.append((native, gaussian))
    return np.asarray(values)


def one(row):
    photo, model, cal, frozen, means, old_nodes, draw = _STATE
    identity = int(photo['identity'][row])
    root = ROOT / 'data/v3_1/structure'
    refined = root / f'native_refined/draw_{draw:02}'
    key = f'object_{identity}/complete.json'
    record = json.loads(verified(refined/key, frozen['objects'][key]).read_text())
    data = load(verified(ROOT/record['selected_payload'], record['selected_sha256']))
    if record['row'] != row or record['identity'] != identity or record['TYPE'] != 'PSF':
        raise ValueError('Mismatched frozen native-likelihood context')
    np.testing.assert_array_equal(data['identity'], [identity])
    np.testing.assert_array_equal(data['TYPE'], ['PSF'])
    np.testing.assert_array_equal(data['common_mu'], old_nodes)
    frequency = data['class_frequency']
    log_frequency = np.log(frequency)[:, None]
    targets = np.r_[means, old_nodes]
    response = native_responses(data, targets)
    checks = []
    for k, branch in enumerate(('full', 'gaussian')):
        coarse, fine = response[:, k]
        relevant = np.maximum(coarse, fine) > max(coarse.max(), fine.max())-30
        log_delta = float(abs(coarse-fine)[relevant].max())
        delta = float(abs(softmax(coarse+log_frequency, axis=0)
                          - softmax(fine+log_frequency, axis=0)).max())
        checks.append(dict(row=row, identity=identity, TYPE='PSF', check=branch+'_native',
                           maximum_absolute_delta=log_delta, maximum_secondary_delta=delta,
                           passed=log_delta < .02 and delta < .003))
    error, _ = candidate_thresholds(photo['flux_error'][row, OPTICAL],
                                    photo['transmission'][row, OPTICAL], cal[row, OPTICAL])
    dust = entry_model_with_local_dust(model, photo['transmission'][row], photo['EBV'][row])
    # Include old nodes as an independent closure test against the published
    # selected narrow entry, after recomputing under the same selection.
    entry = np.asarray([narrow_entry(dust, error, targets, order=order, draws=draws, seed=seed)
                        for order, draws, seed in [(9, 256, 328211), (9, 256, 328212), (5, 128, 328211)]])
    if not np.isfinite(entry).all() or entry.min() < -1e-12 or entry.max() > 1+1e-12:
        raise ValueError('Invalid narrow-entry probability')
    base = entry[0]
    for j, name in [(1, 'entry_second_seed'), (2, 'entry_previous_level')]:
        delta = abs(base-entry[j])
        absolute, relative = float(delta.max()), float((delta/np.maximum(base, .001)).max())
        checks.append(dict(row=row, identity=identity, TYPE='PSF', check=name,
                           maximum_absolute_delta=absolute, maximum_secondary_delta=relative,
                           passed=absolute < .0005 and relative < .01))
    mixed_response = logsumexp(response+log_frequency[None, None], axis=-2)
    mixed_entry = frequency @ entry
    return mixed_response, mixed_entry, checks


def checkpoint_row(row):
    response, entry, checks = one(row)
    return response, entry, np.array(json.dumps(checks))


def main():
    global _STATE
    require_compute()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draw', type=int, choices=range(32), default=12)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    draw = args.draw
    root = ROOT / 'data/v3_1/structure'
    manifests = ROOT / 'manifests/v3_1/structure'
    marker = verified(root/'matrices/complete.json', sha(manifests/'matrices/complete.json'))
    published = json.loads(marker.read_text())
    if not published['passed'] or not published['all_PSF']:
        raise ValueError('Frozen NGC288 structure matrices required')
    matrix_path = verified(marker.parent/'matrices.npz', published['outputs']['matrices.npz'])
    with np.load(matrix_path) as data:
        old_nodes, old_entry = data['common_mu'], data['entry'][0, draw, :, 5:]
        old_response = data['logresponse'][:, :, draw, :, 5:]
        identities, types = data['identity'], data['TYPE']
    refined = root/f'native_refined/draw_{draw:02}/complete.json'
    frozen = json.loads(verified(refined, published['inputs'][str(refined.relative_to(ROOT))]).read_text())
    native = json.loads(verified(root/'complete.json', sha(manifests/'complete.json')).read_text())
    photo_path = verified(root/'photometry.npz', native['outputs']['photometry.npz'])
    photo = load(photo_path)
    np.testing.assert_array_equal(identities, photo['identity'])
    np.testing.assert_array_equal(types, photo['TYPE'])
    if not (types == 'PSF').all() or len(types) != 512 or len(np.unique(identities)) != 512:
        raise ValueError('The complete frozen 512-object native PSF cohort is required')
    old = ROOT/'data/v3_0'
    model_path = old/f'uncertainty_refit/draw_{draw:02}/model.joblib'
    shared_path = old/'uncertainty/shared_draws.npz'
    for path in (model_path, shared_path, photo_path):
        verified(path, frozen['inputs'][str(path.relative_to(ROOT))])
    model = inference_model(joblib.load(model_path))
    cal, _ = calibration_vectors(photo, load(shared_path), draw)
    means = np.unique(np.concatenate([structure_quadrature(n, 64)[0] for n in (128, 256)]))
    rows = np.asarray(ROWS if args.pilot else range(len(types)))
    _STATE = photo, model, cal, frozen, means, old_nodes, draw
    name = 'pilot' if args.pilot else f'draw_{draw:02}'
    out = ROOT/'data/v3_2/structure_distance'/name
    if out.exists():
        raise FileExistsError(out)
    sources = [Path(__file__).resolve(), marker, refined, photo_path, model_path, shared_path,
               ROOT/'docs/v3_2_structure_protocol.md', ROOT/'scripts/v3_2_radial_checkpoint.py']
    sources += [ROOT/'src/photdis'/p for p in ('v3/selection.py', 'v3_1/structure.py',
                 'v3_1/selection_dust.py', 'v3_1/gaussian_response.py')]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    checkpoint_state = {}
    if args.pilot:
        with ProcessPoolExecutor(max_workers=min(16, len(rows)), mp_context=mp.get_context('fork')) as pool:
            results = list(pool.map(one, rows))
    else:
        checkpoint = out.parent/'row_checkpoints'/name
        parts = compute_rows(checkpoint, dict(draw=draw, inputs=hashes, new_nodes=len(means)),
                             identities, checkpoint_row, workers=16, keys=('response', 'entry', 'checks'))
        results = [(response, entry, json.loads(checks.item())) for response, entry, checks in parts]
        checkpoint_state['row_checkpoint_sha256'] = sha(checkpoint/'inputs.json')
    response, entry, checks = zip(*results)
    response, entry = np.asarray(response), np.asarray(entry)
    checks = [check for context in checks for check in context]
    archived_response = np.moveaxis(old_response[:, :, rows], 2, 0)
    np.testing.assert_allclose(response[..., len(means):], archived_response, rtol=0, atol=1e-8)
    ref = old_entry[rows]
    delta = abs(entry[:, 0, len(means):]-ref)
    for k, row in enumerate(rows):
        absolute = float(delta[k].max())
        relative = float((delta[k]/np.maximum(ref[k], .001)).max())
        checks.append(dict(row=row, identity=int(identities[row]), TYPE='PSF', check='frozen_entry_closure',
                           maximum_absolute_delta=absolute, maximum_secondary_delta=relative,
                           passed=absolute < .0005 and relative < .01))
    out.mkdir(parents=True, exist_ok=False)
    atomic_npz(out/'nodes.npz', response=response[..., :len(means)], entry=entry[:, :, :len(means)],
               closure_entry=entry[:, :, len(means):], mu=means, row=rows,
               identity=identities[rows], TYPE=types[rows])
    csv_rows(out/'checks.csv', checks)
    save_json(out/'complete.json', dict(
        inputs=hashes, **checkpoint_state,
        outputs={p.name: sha(p) for p in out.iterdir()}, job=os.getenv('SLURM_JOB_ID'),
        draw=draw, pilot=args.pilot, all_PSF=True, n=len(rows), new_nodes=len(means),
        archived_native_maximum_delta=float(abs(response[..., len(means):]-archived_response).max()),
        failures=sum(not c['passed'] for c in checks), passed=all(c['passed'] for c in checks)))


if __name__ == '__main__':
    main()
