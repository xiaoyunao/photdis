#!/usr/bin/env python3
"""Predict from sealed photometry, then score the frozen Gaia comparison."""
import argparse
import concurrent.futures
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path

import joblib
import numpy as np
from scipy.special import logsumexp

from photdis.gaia import SOURCE_ID_HEALPIX_FACTOR, derive_gaia_labels, join_gaia
from photdis.stage1 import require_compute
from photdis.v3.controls import optical_magnitudes
from photdis.v3.metrics import histogram_parallax, observed_parallax
from photdis.v3.model import inference_model
from photdis.v3_1.validation_report import paired_field_statistic
from photdis.v3_3 import class_log_likelihood, frozen_qrf, reference_log_curves
from v2_1_near import ROOT, atomic_npz, csv_rows, load, save_json, sha
from v3_0_response import observation
from v3_3_catalog import FREEZE, read_frozen
from v3_3_prospective_inputs import OUT
from v3_3_sed import verified

ROOT_OUT = OUT.parent
FORECAST = ROOT_OUT/'forecast'
SCORES = ROOT_OUT/'score'
GAIA = Path('/global/cfs/cdirs/cosmo/data/gaia/dr3/healpix')
MU = np.linspace(-5., 30., 1401)
_STATE = None


def one(i):
    photo, models = _STATE
    return [logsumexp(reference_log_curves(class_log_likelihood(model, observation(photo, i), MU),
                                          MU, prior), axis=0) for model, prior in models.values()]


def inputs(frozen):
    marker = OUT/'complete.json'
    record = json.loads(verified(marker, frozen['prospective_inputs_sha256']).read_text())
    photo = load(verified(OUT/'photometry.npz', record['sha256']))
    if not record['all_PSF'] or record['outcomes_read'] or not (photo['TYPE'] == 'PSF').all():
        raise ValueError('Require sealed native PSF roster')
    if len(np.unique(photo['identity'])) != record['n'] or len(photo['identity']) != record['n']:
        raise ValueError('Duplicated or missing validation identities')
    return photo


def fit():
    frozen = read_frozen()
    photo = inputs(frozen)
    global _STATE
    models = {key: joblib.load(verified(ROOT/item['path'], item['sha256']))
              for key, item in frozen['models'].items()}
    for key, model in models.items():
        if np.intersect1d(model.training_identity, photo['identity']).size:
            raise ValueError(f'Training identities in the new Gaia cohort for {key}')
    qrf = models.pop('qrf')
    _STATE = photo, {key: (inference_model(model), model.weights@model.classes) for key, model in models.items()}
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', '16')),
                                                mp_context=mp.get_context('fork')) as pool:
        curves = np.asarray(list(pool.map(one, range(len(photo['identity'])), chunksize=8)))
    mags, errors = optical_magnitudes(photo)
    q = frozen_qrf(qrf, mags, errors, photo['identity'])
    mass, edges = q['mass'], q['mu_edges']
    FORECAST.mkdir(parents=True, exist_ok=False)
    for i, view in enumerate(models):
        atomic_npz(FORECAST/f'{view}.npz', identity=photo['identity'], TYPE=photo['TYPE'], mu=MU, logp=curves[:, i])
    atomic_npz(FORECAST/'qrf.npz', identity=photo['identity'], TYPE=photo['TYPE'],
               mass=mass, edges=edges, supported=q['supported'])
    save_json(FORECAST/'complete.json', dict(n=len(photo['identity']), model_freeze=sha(FREEZE),
        script=sha(Path(__file__)), outcomes_read=False, all_PSF=True,
        outputs={p.name: sha(p) for p in sorted(FORECAST.iterdir())}, job=os.getenv('SLURM_JOB_ID')))


def score():
    frozen = read_frozen()
    photo = inputs(frozen)
    marker = FORECAST/'complete.json'
    record = json.loads(marker.read_text())
    if record['model_freeze'] != sha(FREEZE) or record['script'] != sha(Path(__file__)):
        raise ValueError('Frozen scoring method or forecasts changed')
    models = {}
    for name, expected in record['outputs'].items():
        models[Path(name).stem] = load(verified(FORECAST/name, expected))
        t = models[Path(name).stem]
        np.testing.assert_array_equal(t['identity'], photo['identity'])
        np.testing.assert_array_equal(t['TYPE'], photo['TYPE'])
    tiles = [GAIA/f'healpix-{pixel:05d}.fits' for pixel in np.unique(photo['identity']//SOURCE_ID_HEALPIX_FACTOR)]
    def archive():
        return {str(p): [p.stat().st_size, p.stat().st_mtime_ns] if p.exists() else None for p in tiles}
    opening = dict(forecast=sha(marker), model_freeze=sha(FREEZE), evaluator=sha(Path(__file__)), tiles=archive())
    SCORES.mkdir(parents=True, exist_ok=True)
    if (SCORES/'complete.json').exists():
        raise FileExistsError('Scored outcomes already recorded')
    started = SCORES/'started.json'
    if started.exists():
        if json.loads(started.read_text())['opening'] != opening:
            raise ValueError('Changed inputs on retry after outcome access')
    else:
        with started.open('x') as stream:
            json.dump(dict(opening=opening, utc=datetime.now(timezone.utc).isoformat()), stream, indent=2)
    gaia = join_gaia(photo['identity'], GAIA)
    g = derive_gaia_labels(gaia)
    if opening['tiles'] != archive():
        raise ValueError('Gaia archive changed after first read')
    eligible = g['astro_quality'] & g['zpt_valid'] & g['finite_parallax']
    p, err = g['p_corr'], gaia['parallax_error']
    w1 = photo['band_state'][:, 4]
    mag = photo['observed_r']
    groups = {'all': eligible, 'negative': eligible & (p < 0),
              'low_precision': eligible & (p < 5*err),
              'high_precision': eligible & (p >= 5*err)}
    groups.update({f'r_{lo}_{hi}': eligible & (mag >= lo) & (mag < hi) for lo, hi in [(16, 18), (18, 20), (20, 22)]})
    groups.update({f'w1_state_{i}': eligible & (w1 == i) for i in np.unique(w1)})
    outcomes, contrasts = [], []
    for view, t in models.items():
        out = np.full((len(eligible), 2), np.nan)
        for i in np.flatnonzero(eligible):
            if view == 'qrf':
                item = histogram_parallax(t['edges'][i], t['mass'][i], p[i], err[i], order=32)
            else:
                item = observed_parallax(t['mu'], t['logp'][i], p[i], err[i])
            out[i] = item['log_score'], item['pit']
        t['scores'] = out
        atomic_npz(SCORES/f'{view}.npz', identity=photo['identity'], TYPE=photo['TYPE'], log_score=out[:, 0],
                   pit=out[:, 1], eligible=eligible, corrected_parallax=p, parallax_error=err, cell=photo['cell'])
        for group, use in groups.items():
            a = out[use]
            n = len(a)
            outcomes.append(dict(view=view, group=group, n=n, finite=int(np.isfinite(a[:, 0]).sum()),
                                 mean_log_score=float(a[:, 0].mean()) if n else np.nan,
                                 pit_outside_90=float(((a[:, 1] < .05) | (a[:, 1] > .95)).mean()) if n else np.nan))
    for group, use in groups.items():
        for base in ('nominal', 'qrf'):
            good = use & models['qrf']['supported'] if base == 'qrf' else use
            contrasts.append(dict(group=group, baseline=base, **paired_field_statistic(
                models['candidate']['scores'][good, 0], models[base]['scores'][good, 0], photo['cell'][good])))
    csv_rows(SCORES/'scores.csv', outcomes)
    csv_rows(SCORES/'paired_scores.csv', contrasts)
    save_json(SCORES/'complete.json', dict(n=len(eligible), all_PSF=True, signed_gaia=int(eligible.sum()),
        negative=int((eligible & (p < 0)).sum()), outcomes_opened=True, freeze=opening,
        outputs={p.name: sha(p) for p in sorted(SCORES.iterdir())}, job=os.getenv('SLURM_JOB_ID')))


if __name__ == '__main__':
    require_compute()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['forecast', 'score'])
    args = parser.parse_args()
    (fit if args.stage == 'forecast' else score)()
