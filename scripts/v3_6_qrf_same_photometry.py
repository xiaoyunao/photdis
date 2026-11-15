#!/usr/bin/env python3
"""Evaluate both frozen QRFs on identical current native photometry and noise draws."""
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import fitsio
import joblib
import numpy as np

from photdis.stage1 import require_compute
from photdis.v3.controls import optical_magnitudes
from photdis.v3_3 import frozen_qrf
from v2_1_near import ROOT, atomic_npz, csv_rows, load, save_json, sha
from v3_0_native import RAW, native_payload
from v3_3_catalog import FREEZE, read_frozen
from v3_3_sed import verified
from v3_5_dr11_inventory import CONFIG
from v3_5_dr11_validate import join_parts
from v3_6_qrf_cache_comparison import OLD, OUT as CACHE, compare


OUT = ROOT/'data/v3_6/qrf_same_photometry'
_STATE = None


def native_magnitudes(path,rows,expected):
    order = np.argsort(rows,kind='stable')
    with fitsio.FITS(str(path)) as f:
        available = set(f[1].get_colnames())
        raw = f[1].read(columns=[k for k in RAW if k in available],rows=rows[order])
    native = native_payload(raw[np.argsort(order)],'DECam_DR11')
    np.testing.assert_array_equal(native['ls_id'],expected)
    if not (native['TYPE']=='PSF').all():
        raise ValueError(f'Same-input diagnostic requires verified native PSF: {path}')
    return dict(zip(('magnitudes','errors'),optical_magnitudes(native)))


def predict(index):
    photo,models = _STATE
    t = {k:photo[k][index] for k in ('magnitudes','errors','identity')}
    result = {}
    for name,model in models.items():
        q = frozen_qrf(model,**t)
        result[f'{name}_mu'] = q['mu']
        result[f'{name}_supported'] = q['supported']
        result[f'{name}_distance'] = q['distance']
    return result


def main():
    global _STATE
    require_compute()
    cache = CACHE/'complete.json'
    rec = json.loads(cache.read_text())
    path = verified(CACHE/'paired.npz',rec['outputs']['paired.npz'])
    p = load(path)
    snapshot = OLD/'manifests/snapshot.json'
    old = next(m for m in json.loads(verified(snapshot,rec['inputs'][str(snapshot.relative_to(ROOT))]).read_text())
               ['models'] if m['model']=='expanded_A')
    current = read_frozen()['models']['qrf']
    paths = {'expanded':verified(ROOT/old['model_path'],old['model_sha256']),
             'current':verified(ROOT/current['path'],current['sha256'])}
    models = {k:joblib.load(x) for k,x in paths.items()}
    if {m.dim for m in models.values()}!={2}:
        raise ValueError('Both frozen controls must use the two grz colors')
    raw = Path(json.loads(CONFIG.read_text())['sweep_root'])
    names = p['sweep']
    jobs = [(raw/name,np.flatnonzero(names==name)) for name in np.unique(names)]
    ncpu = int(os.getenv('SLURM_CPUS_PER_TASK','1'))
    def read(item):
        file,j = item
        return native_magnitudes(file,p['parent_row'][j],p['ls_id'][j])
    with ThreadPoolExecutor(max_workers=min(12,ncpu)) as pool:
        inputs = join_parts(list(pool.map(read,jobs)))
    original_order = np.argsort(np.concatenate([j for _,j in jobs]))
    photo = {k:v[original_order] for k,v in inputs.items()}
    photo['identity'] = p['identity']
    print(f'Same raw grz photometry recovered for {len(names)} native PSFs',flush=True)
    _STATE = photo,models
    batches = [slice(a,min(a+384,len(names))) for a in range(0,len(names),384)]
    with ProcessPoolExecutor(max_workers=ncpu,mp_context=mp.get_context('fork')) as pool:
        predicted = join_parts(list(pool.map(predict,batches)))
    if not predicted['current_supported'].all():
        raise ValueError('Current QRF support differs from original DR11 admission')
    p.update(archived_expanded_mu=p['expanded_mu'],archived_expanded_supported=p['expanded_supported'],
             published_qrf_quantiles_kpc=p['qrf_quantiles_kpc'],
             same_input_magnitudes=photo['magnitudes'],same_input_errors=photo['errors'],**predicted)
    p['qrf_quantiles_kpc'] = predicted['current_distance']
    OUT.mkdir(parents=True,exist_ok=True)
    marker = OUT/'complete.json'
    if marker.exists():
        raise FileExistsError(marker)
    atomic_npz(OUT/'paired.npz',**p)
    csv_rows(OUT/'performance.csv',compare(p))
    published = 10+5*np.log10(p['published_qrf_quantiles_kpc'][:,2])
    shift = np.abs(predicted['current_mu'][:,2]-published)
    save_json(marker,dict(job=os.getenv('SLURM_JOB_ID'),rows=len(names),
        same_input_current_vs_published_median_abs_mu_delta=float(np.median(shift)),
        same_input_current_vs_published_max_abs_mu_delta=float(shift.max()),
        inputs={str(x.relative_to(ROOT)):sha(x) for x in (Path(__file__),cache,FREEZE,CONFIG,*paths.values())},
        interpretation='Both frozen QRFs use the same native grz and identical Sobol perturbations. Only the historical same-star cache intersection is re-evaluated, without any refits.',
        outputs={x:sha(OUT/x) for x in ('paired.npz','performance.csv')}))


if __name__=='__main__':
    main()
