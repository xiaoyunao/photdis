#!/usr/bin/env python3
"""Separate profile interpolation error from Sobol integration error."""
import concurrent.futures
import multiprocessing as mp
import time
import joblib
import numpy as np
from v2_1_near import ROOT, load, save_json
from photdis.stage1 import require_compute
from photdis.v3.model import inference_model
from photdis.v3.selection import component_entry
from photdis.v3.inference import distance_basis
from photdis.v3.likelihood import trapezoid_weights
_MODEL = None


def direct(error):
    grid = np.arange(-15., 45.0001, .05)
    p = component_entry(_MODEL, error, grid, draws=128)
    curve = (1-_MODEL.binary_reference_fraction)*p[0]+_MODEL.binary_reference_fraction*p[1]
    return (curve*trapezoid_weights(grid))@distance_basis(grid).T


def main():
    global _MODEL
    require_compute(); start = time.monotonic()
    d = ROOT/'data/v3_0'; _MODEL = inference_model(joblib.load(d/'sed_benchmark.joblib'))
    t = load(d/'selection/benchmark_-1_64/validation.npz')
    with concurrent.futures.ProcessPoolExecutor(max_workers=16, mp_context=mp.get_context('fork')) as pool:
        same = np.array(list(pool.map(direct, t['errors'])))
    result = {}
    for label, difference in [('interpolation', abs(t['approximate']-same)), ('sobol_128_512', abs(t['direct']-same))]:
        index = np.unravel_index(difference.argmax(), difference.shape)
        result[label] = dict(maximum=float(difference.max()), q99=float(np.quantile(difference, .99)),
            max_index=index, identity=int(t['identity'][index[0]]))
    result['seconds'] = time.monotonic()-start
    save_json(d/'selection/benchmark_-1_64/error_decomposition.json', result)
    print(result, flush=True)


if __name__ == '__main__': main()
