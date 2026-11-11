#!/usr/bin/env python3
"""One chunked photometry-only inference path for development and catalog products."""
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
from photdis.baseline_v1 import support
from photdis.v3.controls import optical_magnitudes
from photdis.v3.data import stable_rank
from photdis.v3.inference import infer_reference, reference_log_density
from scipy.special import logsumexp
from photdis.v3.model import inference_model
from photdis.v3.compression import compress, pack, reweight
from v3_0_response import observation

D = ROOT/'data/v3_0'
_TABLE = _MODEL = _CONTROL = _DEST = _FROZEN = _BANDS = None


def chunk(rows):
    marker = _DEST/f'chunk_{rows[0]:08}.json'; output = marker.with_suffix('.npz')
    compact_path = marker.with_name(marker.stem+'_joint.npz')
    if marker.exists():
        old = json.loads(marker.read_text()); assert old['inputs'] == _FROZEN
        assert sha(output) == old['summary_sha256'] and sha(compact_path) == old['joint_sha256']
        return str(marker)
    start = time.monotonic(); result = []; compact = []; full_mu = []; full_logpdf = []; full_offsets = [0]
    data = {key: _TABLE[key][rows] for key in ('flux', 'flux_error', 'transmission', 'band_mask')}
    mag, _ = optical_magnitudes(data); supported = support(_CONTROL, mag)
    for j, row in enumerate(rows):
        o = observation(_TABLE, row)
        if str(o['photometric_system']) != _MODEL.system:
            raise ValueError('Separate unmodeled-system status product required; no system aliasing')
        p = infer_reference(_MODEL, o, bands=_BANDS, initial_step=.1)
        record = compress(p['posterior'], len(_MODEL.weights))
        # Real per-star compression check, not only an implementation unit test.
        checked = reweight(record, _MODEL, reference_prior=True)
        error = float(abs(checked['class_distance']-p['class_distance']).max())
        if error > .001: raise RuntimeError('Joint compression failed at identity '+str(_TABLE['identity'][row]))
        compact.append(record)
        grid = p['posterior'].mu
        full_mu.append(grid)
        full_logpdf.append(logsumexp(reference_log_density(_MODEL, o, grid, bands=_BANDS), axis=0)-p['posterior'].log_evidence)
        full_offsets.append(full_offsets[-1]+len(grid))
        mode_order = np.argsort(-p['mode_mass'])[:8]
        modes = np.full(8, np.nan); mode_mass = np.zeros(8)
        modes[:len(mode_order)] = p['modes'][mode_order]; mode_mass[:len(mode_order)] = p['mode_mass'][mode_order]
        result.append(dict(mu=p['mu_quantiles'], qnear=p['qnear'], class_distance=p['class_distance'],
            modes=modes, mode_mass=mode_mass, minor_mode_mass=float(1-mode_mass.sum()),
            binary_probability=p['binary_probability'],
            discrepancy_probability=float(p['component_probability']@_MODEL.discrepancy_channel),
            negative_intrinsic_support=p['negative_intrinsic_support'], supported=bool(supported[j]),
            converged=p['converged'], grid_nodes=p['posterior'].convergence['nodes'],
            numerical_probability_delta=p['posterior'].convergence['probability_delta'],
            compression_error=error, omitted_mass=record['omitted_mass'],
            status=0 if p['converged'] else 2))
    payload = {key: np.asarray([r[key] for r in result]) for key in result[0]}
    payload.update(identity=_TABLE['identity'][rows], TYPE=_TABLE['TYPE'][rows], rows=rows,
        ls_id=_TABLE['ls_id'][rows], photometric_system=_TABLE['photometric_system'][rows],
        band_mask=_TABLE['band_mask'][rows], band_state=_TABLE['band_state'][rows],
        field=_TABLE['field'][rows], cell=_TABLE['cell'][rows])
    assert (payload['TYPE'] == 'PSF').all()
    np.testing.assert_allclose(payload['class_distance'].sum(axis=(1, 2)), 1., atol=1e-10)
    joint = pack(compact)
    joint.update(full_mu=np.concatenate(full_mu), full_logpdf=np.concatenate(full_logpdf),
                 full_grid_offset=np.asarray(full_offsets, 'i8'))
    atomic_npz(output, **payload); atomic_npz(compact_path, **joint)
    save_json(marker, dict(inputs=_FROZEN, n=len(rows), summary_sha256=sha(output),
        joint_sha256=sha(compact_path), elapsed_seconds=time.monotonic()-start,
        joint_bytes=compact_path.stat().st_size, summary_bytes=output.stat().st_size,
        converged=int(payload['converged'].sum()), job=os.getenv('SLURM_JOB_ID')))
    return str(marker)


def main():
    global _TABLE, _MODEL, _CONTROL, _DEST, _FROZEN, _BANDS
    require_compute(); parser = argparse.ArgumentParser()
    parser.add_argument('cohort', choices=['development', 'hyper', 'application'])
    parser.add_argument('--view', choices=['grz', 'griz', 'multiband'], default='multiband')
    parser.add_argument('--model', choices=['fit', 'benchmark'], default='fit')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    input_path = D/'cohorts'/(args.cohort+'.npz')
    model_path = D/f'sed_{args.model}.joblib'; control_path = D/'controls/qrf_grz.joblib'
    _TABLE = load(input_path)
    if args.limit:
        selected = np.argsort(stable_rank(_TABLE['identity'], 302001))[:args.limit]
        _TABLE = {k: v[selected] for k, v in _TABLE.items()}
    assert (_TABLE['TYPE'] == 'PSF').all()
    fitted = joblib.load(model_path); _MODEL = inference_model(fitted); _CONTROL = joblib.load(control_path)
    if args.cohort != 'application': assert not np.intersect1d(_TABLE['identity'], fitted.training_identity).size
    _BANDS = np.array([1, 1, 0, 1, 0, 0], bool) if args.view == 'grz' else np.array([1, 1, 1, 1, 0, 0], bool) if args.view == 'griz' else np.ones(6, bool)
    paths = [input_path, model_path, control_path, ROOT/'scripts/v3_0_predict.py',
             ROOT/'scripts/v3_0_response.py', ROOT/'src/photdis/baseline_v0.py', ROOT/'src/photdis/baseline_v1.py']
    paths += [ROOT/'src/photdis/v3'/name for name in ('model.py', 'inference.py', 'fast.py', 'likelihood.py', 'compression.py', 'controls.py', 'data.py')]
    _FROZEN = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    _FROZEN.update(view=args.view, limit=args.limit, inference_version=_MODEL.inference_version)
    _DEST = D/'predictions'/f'{args.cohort}_{args.view}_{args.model}_{args.limit}'
    _DEST.mkdir(parents=True, exist_ok=True)
    freeze = _DEST/'inputs.json'
    if freeze.exists(): assert json.loads(freeze.read_text()) == _FROZEN, 'Explicit inference migration required'
    else: save_json(freeze, _FROZEN)
    tasks = [np.arange(i, min(i+32, len(_TABLE['identity']))) for i in range(0, len(_TABLE['identity']), 32)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('fork')) as pool:
        paths = []
        for path in pool.map(chunk, tasks):
            paths.append(path)
            if len(paths) % 16 == 0: print('V3 PREDICT', args.cohort, args.view, len(paths), len(tasks), flush=True)
    save_json(_DEST/'complete.json', dict(inputs=_FROZEN, rows=len(_TABLE['identity']),
        chunks={str(p): sha(p) for p in paths}, scientific_acceptance=False,
        prospective_outcomes_opened=False, job=os.getenv('SLURM_JOB_ID')))
    print('V3 PREDICT COMPLETE', args.cohort, args.view, len(_TABLE['identity']), flush=True)


if __name__ == '__main__': main()
