#!/usr/bin/env python3
"""Refine only the signed Gaia curves with material grid sensitivity."""
import json
import os

import joblib
import numpy as np
from scipy.special import logsumexp

from photdis.stage1 import require_compute
from photdis.v3.metrics import observed_parallax
from photdis.v3.model import inference_model
from photdis.v3_1.validation_report import paired_field_statistic
from photdis.v3_3 import class_log_likelihood, reference_log_curves
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_0_response import observation
from v3_3_catalog import FREEZE, read_frozen
from v3_3_prospective import FORECAST, SCORES, inputs
from v3_3_score_sensitivity import payload
from v3_3_sed import verified

OUT = ROOT/'data/v3_3/gaia_grid_refinement'
TOLERANCE = .02


def main():
    require_compute()
    freeze = read_frozen()
    photo = inputs(freeze)
    rows, summary = [], {}
    saved = {}
    for name in ('candidate', 'nominal'):
        score = payload(SCORES, f'{name}.npz')
        forecast = payload(FORECAST, f'{name}.npz')
        np.testing.assert_array_equal(score['identity'], photo['identity'])
        np.testing.assert_array_equal(forecast['identity'], photo['identity'])
        if not (score['TYPE'] == 'PSF').all():
            raise ValueError('Refinement requires the original prospective PSF roster')
        mu, curves = forecast['mu'], forecast['logp']
        selected = np.flatnonzero(score['eligible'])
        revised = score['log_score'].copy()
        recorded = freeze['models'][name]
        model = inference_model(joblib.load(verified(ROOT/recorded['path'], recorded['sha256'])))
        prior = model.weights @ model.classes
        for i in selected:
            p, e = score['corrected_parallax'][i], score['parallax_error'][i]
            coarse = observed_parallax(mu[::2], curves[i, ::2], p, e)['log_score']
            fine = score['log_score'][i]
            if abs(coarse-fine) <= TOLERANCE:
                continue
            step = np.linspace(mu[0], mu[-1], 2*len(mu)-1)
            ll = class_log_likelihood(model, observation(photo, i), step)
            curve = logsumexp(reference_log_curves(ll, step, prior), axis=0)
            refined = observed_parallax(step, curve, p, e)['log_score']
            revised[i] = refined
            rows.append(dict(view=name, identity=int(score['identity'][i]), TYPE='PSF', coarse=coarse,
                             original=fine, refined=refined, delta=refined-fine))
        saved[name] = (score, revised)
        summary[name] = int(sum(row['view'] == name for row in rows))
    a, b = (saved[key] for key in ('candidate', 'nominal'))
    use = a[0]['eligible']
    np.testing.assert_array_equal(use, b[0]['eligible'])
    cells = a[0]['cell'][use]
    summary.update(primary=paired_field_statistic(a[0]['log_score'][use], b[0]['log_score'][use], cells),
                   refined_sidecar=paired_field_statistic(a[1][use], b[1][use], cells))
    OUT.mkdir(parents=True, exist_ok=False)
    csv_rows(OUT/'flagged_scores.csv', rows)
    atomic_npz(OUT/'refined_scores.npz', identity=a[0]['identity'], TYPE=a[0]['TYPE'],
               candidate=a[1], nominal=b[1], eligible=use)
    save_json(OUT/'complete.json', dict(job=os.getenv('SLURM_JOB_ID'), threshold_nat=TOLERANCE,
        summary=summary, inputs={str(p.relative_to(ROOT)): sha(p) for p in
            (FREEZE, FORECAST/'complete.json', SCORES/'complete.json')},
        outputs={p.name: sha(p) for p in sorted(OUT.iterdir())}))


if __name__ == '__main__':
    main()
