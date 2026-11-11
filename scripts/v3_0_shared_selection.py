#!/usr/bin/env python3
"""Selection response under the identical global model/system draw as prediction."""
import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
import time
import joblib
import numpy as np
from v2_1_near import ROOT, sha, load, save_json, atomic_npz
from photdis.stage1 import require_compute
from photdis.v3.model import inference_model
from photdis.v3.data import stable_rank
from photdis.v3.selection import effective_errors
from photdis.v3.selection_table import profile_centers, SelectionTable
from photdis.v3.systematics import calibration_vectors
import v3_0_selection as nominal
D = ROOT/'data/v3_0'


def main():
    require_compute(); start = time.monotonic(); p = argparse.ArgumentParser()
    p.add_argument('--draw', type=int, default=int(os.getenv('SLURM_ARRAY_TASK_ID', 0)))
    p.add_argument('--ensemble', choices=['influence', 'refit'], default='refit')
    p.add_argument('--profiles', type=int, default=256); p.add_argument('--workers', type=int, default=32)
    args = p.parse_args(); shared_path = D/'uncertainty/shared_draws.npz'; shared = load(shared_path)
    model_path = D/'uncertainty/ensemble.joblib' if args.ensemble == 'influence' else D/'uncertainty_refit'/f'draw_{args.draw:02}/model.joblib'
    fitted = joblib.load(model_path)
    if args.ensemble == 'influence': fitted = fitted[args.draw]
    else:
        meta = json.loads(model_path.with_name('complete.json').read_text())
        assert meta['converged'] and sha(model_path) == meta['sha256']
    nominal._MODEL = inference_model(fitted)
    out = D/'shared_selection'/f'{args.ensemble}_{args.draw:02}_p{args.profiles}'; out.mkdir(parents=True, exist_ok=True)
    paths = [model_path, shared_path, ROOT/'scripts/v3_0_shared_selection.py', ROOT/'scripts/v3_0_selection.py']
    paths += [ROOT/'src/photdis/v3'/x for x in ('selection.py', 'selection_table.py', 'systematics.py', 'model.py')]
    paths += [D/'cohorts'/(name+'.npz') for name in ('hyper', 'development', 'application', 'simulations')]
    inputs = {str(x.relative_to(ROOT)): sha(x) for x in paths}; inputs.update(vars(args))
    if (out/'inputs.json').exists(): assert json.loads((out/'inputs.json').read_text()) == inputs
    else: save_json(out/'inputs.json', inputs)
    if (out/'complete.json').exists():
        old = json.loads((out/'complete.json').read_text()); assert old['inputs'] == inputs
        assert old['passed'] and sha(out/'table.npz') == old['sha256']; return
    errors = []; ids = []
    for name in ('hyper', 'development', 'application', 'simulations'):
        t = load(D/'cohorts'/(name+'.npz')); assert (t['TYPE'] == 'PSF').all()
        calibration, _ = calibration_vectors(t, shared, args.draw)
        # Exactly the same target-relative transmission factor used by native
        # inference: f_model -> f_model*10**(-.4*delta_mag).
        errors.append(effective_errors(t)*10**(.4*calibration[:, [0, 1, 3]])); ids.append(t['identity'])
    errors = np.concatenate(errors); ids = np.concatenate(ids); centers = profile_centers(errors, args.profiles)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('fork')) as pool:
        records = list(pool.map(nominal.build_profile, centers))
        table = SelectionTable(centers, records[0][0], np.array([r[1] for r in records], 'f4'))
        checked = np.argsort(stable_rank(ids, 305301))[:64]
        checked = np.unique(np.r_[checked, errors.argmin(axis=0), errors.argmax(axis=0)]).astype(int)
        direct = np.array(list(pool.map(nominal.direct_error, errors[checked])))
    approximate, audit = table.evaluate(errors[checked]); maximum = float(abs(approximate-direct).max())
    inside = True
    for first in range(0, len(errors), 1024):
        _, a = table.evaluate(errors[first:first+1024]); inside &= bool(a['inside_profile_hull'].all())
    atomic_npz(out/'table.npz', profiles=table.profiles, shift=table.shift, alpha=table.alpha)
    atomic_npz(out/'validation.npz', identity=ids[checked], direct=direct, approximate=approximate,
        errors=errors[checked], inside=audit['inside_profile_hull'])
    passed = maximum <= .005 and inside
    save_json(out/'complete.json', dict(inputs=inputs, sha256=sha(out/'table.npz'),
        validation_sha256=sha(out/'validation.npz'), passed=passed, maximum_component_entry_difference=maximum,
        profiles=len(centers), checked=len(checked), construction_objects=len(errors), all_inside=inside,
        global_draw_id=args.draw, elapsed_seconds=time.monotonic()-start, job=os.getenv('SLURM_JOB_ID')))
    print('SHARED SELECTION', args.ensemble, args.draw, passed, maximum, flush=True)
    if not passed: raise RuntimeError('Refine shared selection before target inference')


if __name__ == '__main__': main()
