#!/usr/bin/env python3
"""Measure the registered primary giant-only endpoint as soon as its fit is ready."""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import norm

from photdis.audit import rvs_metallicity
from photdis.stage1 import require_compute
from photdis.v3.likelihood import normalize_joint
from photdis.v3.metrics import distance_summary, observed_parallax
from photdis.v3.model import inference_model
from photdis.v3_1.validation_report import paired_field_statistic
from photdis.v3_3 import class_log_likelihood
from v2_1_near import ROOT, atomic_npz, csv_rows, load, save_json, sha
import v3_0_evaluate as historical
from v3_0_response import observation
from v3_3_development import MU
from v3_3_sed import BASE, MANIFESTS, OUT as FIT, verified

OUT = ROOT/'data/v3_3/development/giant_conditional'
_STATE = None


def evaluate(i):
    photo, models = _STATE
    o = observation(photo, i)
    curves = [class_log_likelihood(model, o, MU)[1] + norm.logpdf(MU, 15., 5.) for model in models]
    q = [normalize_joint(MU, c[None]).quantiles() for c in curves]
    scores = [observed_parallax(MU, c, photo['p_corr'][i], photo['gaia_parallax_error'][i])['log_score']
              if photo['parallax_eligible'][i] else np.nan for c in curves]
    return q, scores


def main():
    require_compute()
    manifest = json.loads((MANIFESTS/'cohorts.json').read_text())['outputs']
    cohort_paths = [verified(BASE/name, manifest[name]) for name in
                    ('cohorts/development.npz', 'evaluation/development.npz')]
    photo, evidence = map(load, cohort_paths)
    np.testing.assert_array_equal(photo['identity'], evidence['identity'])
    strict = (photo['TYPE'] == 'PSF') & (evidence['TYPE'] == 'PSF')
    if not (evidence['role'] == 4).all():
        raise ValueError('Expected exposed development identities')
    selected = strict & evidence['reference_stellar_eligible'] & (evidence['class_label'] == 1)
    t = {k: a[selected] for k, a in (photo | evidence).items()}
    record = json.loads((FIT/'giant/complete.json').read_text())
    if not record['all_PSF'] or not record['converged'] or record['part']['label'] != 1 or not selected.any():
        raise ValueError('Expected completed giant partition and nonempty paired sample')
    candidate = verified(FIT/'giant/model.joblib', record['sha256'])
    nominal = verified(BASE/'sed_fit.joblib', json.loads((MANIFESTS/'sed_fit.json').read_text())['sha256'])
    cal = verified(BASE/'calibration.npz', json.loads((MANIFESTS/'calibration.json').read_text())['sha256'])
    historical._CAL = load(cal)
    ref = historical.evidence_views(t)['primary'][0]
    models = [joblib.load(path) for path in (nominal, candidate)]
    for model in models:
        if np.intersect1d(model.training_identity, t['identity']).size:
            raise ValueError('Giant diagnostic training overlap')
    old, new = models
    # With no class labels in the giant-only EM each kernel has the same class
    # vector. This yields exactly the normalized giant kernels used at assembly.
    np.testing.assert_allclose(new.classes, np.broadcast_to(new.classes[0], new.classes.shape))
    for key in ('system', 'binary_reference_fraction', 'binary_brightness_factor'):
        setattr(new, key, getattr(old, key))
    global _STATE
    _STATE = t, list(map(inference_model, models))
    with ProcessPoolExecutor(max_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', '8')),
                             mp_context=mp.get_context('fork')) as pool:
        results = list(pool.map(evaluate, range(len(t['identity'])), chunksize=8))
    q, scores = [np.asarray(a) for a in zip(*results)]
    feh, qualified = rvs_metallicity(t['rvs_feh'], t['rvs_teff'], t['rvs_logg'])
    feh_good = qualified & t['rvs_parameter_quality']
    distance = 10**((ref-10.)/5.)
    groups = {'all_giants': np.ones(len(ref), bool), 'metal_poor_giants': feh_good & (feh < -1.5)}
    groups.update({f'reference_{i}': t['primary_channel'] == i for i in (1, 2, 3)})
    groups.update({f'distance_{lo}_{hi}': (distance >= lo) & (distance < hi)
                   for lo, hi in ((0, 2), (2, 5), (5, 10), (10, 20), (20, 50), (50, np.inf))})
    OUT.mkdir(parents=True, exist_ok=False)
    csv_rows(OUT/'distance.csv', [dict(group=group, view=view, **distance_summary(q[use, j], ref[use]))
        for group, use in groups.items() for j, view in enumerate(('nominal', 'candidate'))])
    csv_rows(OUT/'gaia_paired.csv', [dict(group=group, **paired_field_statistic(scores[use, 1], scores[use, 0], t['cell'][use]))
        for group, use in groups.items()])
    atomic_npz(OUT/'pairs.npz', identity=t['identity'], TYPE=t['TYPE'], reference_mu=ref,
               quantiles=q, gaia_log_score=scores, cell=t['cell'])
    inputs = [*cohort_paths, nominal, candidate, cal, Path(__file__).resolve(), ROOT/'src/photdis/v3_3.py']
    save_json(OUT/'complete.json', dict(n=len(q), all_PSF=True, development_only=True,
        missing_TYPE=int(((photo['TYPE'] == '') | (evidence['TYPE'] == '')).sum()),
        non_PSF=int((~strict & (photo['TYPE'] != '') & (evidence['TYPE'] != '')).sum()),
        inputs={str(p.relative_to(ROOT)): sha(p) for p in inputs},
        outputs={p.name: sha(p) for p in sorted(OUT.iterdir())}, job=os.getenv('SLURM_JOB_ID')))


if __name__ == '__main__':
    main()
