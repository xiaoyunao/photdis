#!/usr/bin/env python3
"""Score the frozen grz QRF on the opened v3.1 field cohort."""
import json
import os
from pathlib import Path
import sys

import joblib
import numpy as np

from photdis.baseline_v1 import integrate
from photdis.stage1 import require_compute
from photdis.v3.controls import optical_magnitudes
from photdis.v3.metrics import histogram_parallax, histogram_reference_score
from photdis.v3_1.validation_report import paired_field_statistic
from v2_1_near import ROOT, csv_rows, save_json, sha
from v3_1_validation_report import read_cohort
from v3_2_opened_diagnostics import PATH, ROLES


def main():
    require_compute()
    output = Path(sys.argv[1]).resolve()
    parent = ROOT / 'data/v3_2'
    if not output.is_relative_to(parent) or output == parent or output.exists():
        raise ValueError('Choose a new directory below data/v3_2')
    controls = ROOT / 'data/v3_0/controls'
    marker = ROOT / 'manifests/v3_0/controls/complete.json'
    expected = json.loads(marker.read_text())['outputs']['qrf_grz.joblib']
    model_file = controls / 'qrf_grz.joblib'
    if sha(model_file) != expected:
        raise ValueError('Frozen QRF hash mismatch')
    model = joblib.load(model_file)
    paths = [marker, model_file, Path(__file__).resolve()]
    paths += [ROOT / p for p in ('src/photdis/baseline_v1.py', 'src/photdis/baseline_v0.py',
                                'src/photdis/v3/metrics.py', 'src/photdis/v3/controls.py',
                                'scripts/v3_1_validation_report.py')]
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    photo, evidence, records, _, metrics = read_cohort(ROOT / 'data/v3_1', 'field', inputs)
    ids = photo['identity']
    optical = [0, 1, 3]
    clean = photo['band_mask'][:, optical].all(axis=1)
    for key in ('flux', 'flux_error', 'transmission'):
        values = photo[key][:, optical]
        clean &= (np.isfinite(values) & (values > 0)).all(axis=1)
    overlap = np.isin(ids, model.training_identity)
    eligible = photo['candidate_admission'] & np.isin(photo['role'], ROLES)
    indices = np.flatnonzero(eligible & clean & ~overlap)
    if not len(indices):
        raise ValueError('No eligible QRF objects')
    mag, error = optical_magnitudes({k: photo[k][indices] for k in
                                     ('flux', 'flux_error', 'transmission', 'band_mask')})
    shape = (len(indices), 2)
    result = integrate(model, np.column_stack([mag, np.zeros(shape)]),
                       np.column_stack([error, np.zeros(shape)]), ids[indices], draws=128)['wide']
    gaia = np.full(len(ids), np.nan)
    astronn = gaia.copy()
    qnear = gaia.copy()
    in_support = np.zeros(len(ids), bool)
    gaia_stable = in_support.copy()
    objects = {i: dict(identity=int(ids[i]), TYPE='PSF', role=str(photo['role'][i]),
                       cell=int(photo['cell'][i]), reference_class=int(evidence['class_label'][i]),
                       status=('training_identity_overlap' if overlap[i] else 'invalid_native_grz'))
               for i in np.flatnonzero(eligible)}
    for j, i in enumerate(indices):
        edges = mag[j, 0] - model.edges[::-1]
        mass = result['pdf'][j, ::-1].astype(float)
        mass /= mass.sum()
        qnear[i] = mass @ np.clip((15 - edges[:-1]) / np.diff(edges), 0, 1)
        in_support[i] = result['supported'][j]
        obj = objects[i]
        obj.update(status='in_support' if in_support[i] else 'outside_qrf_support',
                   qnear=float(qnear[i]), median_mu=float(result['mu'][j, 2]))
        if evidence['parallax_eligible'][i]:
            args = (edges, mass, evidence['p_corr'][i], evidence['gaia_parallax_error'][i])
            old = histogram_parallax(*args, order=32)
            new = histogram_parallax(*args, order=64)
            gaia[i] = new['log_score']
            delta = abs(new['log_score'] - old['log_score'])
            pit_delta = abs(new['pit'] - old['pit'])
            gaia_stable[i] = delta < .02 and pit_delta < .003
            obj.update(gaia_log_score=gaia[i], gaia_pit=new['pit'], gaia_log_score_delta=delta,
                       gaia_pit_delta=pit_delta, gaia_quadrature_passed=bool(gaia_stable[i]))
        if evidence['distance_available'][i]:
            distance = evidence['distance_kpc'][i]
            mu = 10 + 5 * np.log10(distance)
            sigma = (5 / np.log(10)) * evidence['distance_error_kpc'][i] / distance
            astronn[i] = histogram_reference_score(edges, mass, mu, sigma)
            obj.update(astronn_log_score=astronn[i], astronn_mu_residual=obj['median_mu'] - mu)

    paired = []
    for role in ROLES:
        role_mask = eligible & (photo['role'] == role)
        for label, mask in [('all', role_mask), ('dwarf', role_mask & (evidence['class_label'] == 0)),
                            ('giant', role_mask & (evidence['class_label'] == 1))]:
            for region, use in [('all_QRF_predictions', mask), ('in_QRF_support', mask & in_support)]:
                for view in (PATH[0], PATH[2], PATH[3], PATH[-1]):
                    values, accepted = metrics[view]
                    for metric, baseline in [('gaia/log_score', gaia), ('astronn/log_score', astronn),
                                             ('qnear', qnear)]:
                        qualified = use & ~np.isnan(baseline) & ~np.isnan(values[metric])
                        paired.append(dict(role=role, stratum=label, support=region, view=view, metric=metric,
                                           group_n=int(mask.sum()),
                                           view_failed=int((qualified & ~accepted).sum()),
                                           qrf_gaia_failed=int((qualified & ~gaia_stable).sum())
                                           if metric == 'gaia/log_score' else None,
                                           **paired_field_statistic(values[metric][qualified], baseline[qualified],
                                                                    photo['cell'][qualified])))
    output.mkdir(parents=True, exist_ok=False)
    csv_rows(output / 'objects.csv', objects.values())
    csv_rows(output / 'paired.csv', paired)
    save_json(output / 'complete.json', dict(
        inputs=inputs, outputs={p.name: sha(p) for p in sorted(output.glob('*.csv'))},
        job=os.environ.get('SLURM_JOB_ID'), n=int(eligible.sum()), predicted=len(indices),
        in_support=int(in_support.sum()), training_overlap=int((eligible & overlap).sum()),
        invalid_grz=int((eligible & ~clean & ~overlap).sum()), all_PSF=True,
        scope='Retrospective frozen distance QRF. Gaia scores use native histogram bins. '
              'Leaf label contributions are not scored as physical probabilities.'))


if __name__ == '__main__':
    main()
