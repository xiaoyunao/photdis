#!/usr/bin/env python3
"""Evaluate the registered posterior adjustment on opened field forecasts."""
import csv
import json
import os
from pathlib import Path

import joblib
import numpy as np
from scipy.special import logsumexp

from photdis.stage1 import require_compute
from photdis.v3.likelihood import ContinuousPosterior, trapezoid_weights
from photdis.v3_1.radial_products import bin_integrals
from photdis.v3_1.validation_report import paired_field_statistic
from photdis.v3_1.validation_scores import compare_score_levels, score_pdf
from photdis.v3_2.giant_branch import adjust_giant_branch
from v2_1_near import ROOT, csv_rows, load, save_json, sha
from v3_1_validation_report import read_cohort
from v3_2_tilt_inputs import verified


def summarize(grid, curves, evidence):
    joint = bin_integrals(np.exp(curves), grid)
    logpdf = logsumexp(curves, axis=0)
    q = ContinuousPosterior(grid, np.exp(logpdf)[None], 0., {}).quantiles()
    value = score_pdf(grid, logpdf, q, joint, evidence)
    stats = dict(gaia_score=value.get('gaia', {}).get('log_score'),
                 gaia_pit=value.get('gaia', {}).get('pit'),
                 class_score=value.get('class', {}).get('log_score'),
                 p_giant=value['class_probability'][1], qnear=value['qnear'],
                 predicted_class=int(np.argmax(value['class_probability'])),
                 median_mu=q[2], width68=value['width68'], width95=value['width95'])
    teacher = value['references'].get('astronn', {})
    stats.update(teacher_score=teacher.get('log_score'), teacher_residual=teacher.get('mu_residual'),
                 teacher68=teacher.get('reference_agreement68'), teacher95=teacher.get('reference_agreement95'))
    if value.get('class', {}).get('zero_probability'):
        stats['class_score'] = -np.inf
    return stats, joint, value


def main():
    require_compute()
    root = ROOT/'data/v3_2'
    source = root/'tilt_predictions/initial'
    marker = source/'complete.json'
    compact = ROOT/'manifests/v3_2/tilt_predictions/initial/complete.json'
    record = json.loads(verified(marker, sha(compact)).read_text())
    model_path = 'data/v3_0/uncertainty_refit/draw_12/model.joblib'
    model = joblib.load(verified(ROOT/model_path, record['inputs'][model_path]))
    frequency = model.weights @ model.classes
    inputs = {}
    photo, evidence, _, _, metrics = read_cohort(ROOT/'data/v3_1', 'field', inputs)
    identities = photo['identity']
    if record['all_PSF'] is not True or record['n'] != len(identities):
        raise ValueError('Opened factorial forecast is incomplete')
    for path in (marker, Path(__file__).resolve(), ROOT/'docs/v3_2_branch_protocol.md',
                 ROOT/'src/photdis/v3_2/giant_branch.py'):
        inputs[str(path.relative_to(ROOT))] = sha(path)
    table = verified(source/'objects.csv', record['outputs']['objects.csv'])
    with table.open(newline='') as stream:
        base_rows = {int(r['identity']): r for r in csv.DictReader(stream) if r['view']=='candidate_free'}
    if base_rows.keys() != set(identities):
        raise ValueError('Opened identities differ from the frozen factorial')
    qrf_root = ROOT/'manifests/v3_2/opened_qrf'
    qrf_marker = qrf_root/'complete.json'
    qrf_path = verified(qrf_root/'objects.csv',
                        json.loads(qrf_marker.read_text())['outputs']['objects.csv'])
    with qrf_path.open(newline='') as stream:
        qrf = {int(r['identity']): r for r in csv.DictReader(stream)}
    for path in (qrf_marker, qrf_path, table, ROOT/model_path):
        inputs[str(path.relative_to(ROOT))] = sha(path)
    rows, masses = [], []
    for i, identity in enumerate(identities):
        name = f'object_{identity}.npz'
        obj = load(verified(source/name, record['objects'][name]))
        np.testing.assert_array_equal(obj['identity'], [identity])
        np.testing.assert_array_equal(obj['TYPE'], ['PSF'])
        grid, base = obj['grid'], obj['candidate_free_class_logpdf']
        adjusted = adjust_giant_branch(grid, base, obj['reference_class_log_curve'], frequency)
        log_width = np.log(trapezoid_weights(grid))
        def normalized(curves):
            return curves - logsumexp(curves + log_width, axis=1)[:, None]
        np.testing.assert_allclose(normalized(adjusted), normalized(base), rtol=0, atol=1e-9)
        ev = {k: v[i] for k, v in evidence.items()}
        left, old_joint, _ = summarize(grid, base, ev)
        right, new_joint, fine = summarize(grid, adjusted, ev)
        coarse_grid = grid[::2]
        coarse_curve = adjust_giant_branch(coarse_grid, base[:, ::2],
                                           obj['reference_class_log_curve'][:, ::2], frequency)
        _, coarse_joint, coarse = summarize(coarse_grid, coarse_curve, ev)
        checks = compare_score_levels(coarse, fine)
        joint_delta = float(abs(coarse_joint - new_joint).max())
        mu_delta = float(abs(np.asarray(coarse['mu']) - fine['mu']).max())
        b = base_rows[identity]
        np.testing.assert_allclose([left['qnear'],left['p_giant']], [float(b['qnear']),float(b['p_giant'])], rtol=0, atol=1e-10)
        rows.append(dict(identity=int(identity), TYPE='PSF', role=str(photo['role'][i]),
                         cell=int(photo['cell'][i]), candidate_admission=bool(photo['candidate_admission'][i]),
                         label=int(ev['class_label']), coarse_joint_delta=joint_delta,
                         coarse_mu_delta=mu_delta, coarse_scores_passed=checks['passed'],
                         numerical_pass=checks['passed'] and joint_delta < .003 and mu_delta < .03,
                         **{key+'_'+k:v for key, stats in (('base',left),('adjusted',right))
                            for k,v in stats.items()}))
        masses.append(np.stack([old_joint,new_joint]))
    masses = np.asarray(masses)
    verify = np.ones(len(rows), bool)
    score_names = dict(gaia_score='gaia/log_score', gaia_pit='gaia/pit', class_score='class/log_score',
                       qnear='qnear', width68='width68', width95='width95',
                       teacher_score='astronn/log_score', teacher_residual='astronn/mu_residual',
                       teacher68='astronn/reference_agreement68', teacher95='astronn/reference_agreement95')
    anchors = {name: {key: metrics[name][0][source] for key, source in score_names.items()
                      if source in metrics[name][0]}
               for name in ('reference/nominal', 'old_target/old_target', 'candidate/bootstrap_evidence')}
    qrf_names = dict(gaia_score='gaia_log_score', gaia_pit='gaia_pit', qnear='qnear',
                     median_mu='median_mu', teacher_score='astronn_log_score',
                     teacher_residual='astronn_mu_residual')
    anchors['qrf'] = {key: np.array([float(qrf.get(i, {}).get(source) or 'nan')
                                    for i in identities]) for key, source in qrf_names.items()}
    anchors['candidate_free'] = {key: np.array([r['base_'+key] for r in rows], dtype=float)
                                 for key in ('p_giant', 'median_mu', *score_names)}
    adjusted = {key: np.array([r['adjusted_'+key] for r in rows], dtype=float)
                for key in anchors['candidate_free']}
    pairs, counts, confusion = [], [], []
    for role in np.unique(photo['role']):
        for scope, gate in [('all_opened',verify),('candidate_admitted',photo['candidate_admission'])]:
            cohort = gate & (photo['role']==role)
            for label in (-1,0,1):
                use = cohort & (verify if label==-1 else evidence['class_label']==label)
                for anchor, values in anchors.items():
                    for metric, before in values.items():
                        pairs.append(dict(role=role,scope=scope,label=label,metric=metric,baseline=anchor,
                                          **paired_field_statistic(adjusted[metric][use],before[use],photo['cell'][use])))
                if label != -1:
                    for view in ('base','adjusted'):
                        prediction = np.array([r[view+'_predicted_class'] for r in rows])
                        for predicted in range(3):
                            confusion.append(dict(role=role,scope=scope,view=view,label=label,
                                                  predicted=predicted,n=int(use.sum()),
                                                  count=int((use & (prediction==predicted)).sum())))
            means = masses[cohort].sum(axis=0)
            np.testing.assert_allclose(means.sum(axis=(1,2)), cohort.sum(), rtol=0, atol=1e-9)
            for c in range(3):
                for b in range(6):
                    counts.append(dict(role=role,scope=scope,label=c,distance_bin=b,
                                       base=means[0,c,b],adjusted=means[1,c,b]))
    destination = root/'branch_development'
    destination.mkdir(exist_ok=False)
    for name, records in [('objects',rows),('paired',pairs),('joint_counts',counts),('confusion',confusion)]:
        csv_rows(destination/f'{name}.csv',records)
    save_json(destination/'complete.json',dict(
        inputs=inputs, outputs={p.name:sha(p) for p in sorted(destination.iterdir())},
        job=os.getenv('SLURM_JOB_ID'), n=len(rows), all_PSF=True,
        new_outcomes_read=False, class_conditional_invariance=True,
        numerical_flags=sum(not r['numerical_pass'] for r in rows),
        scope='Opened development on frozen fine grids with a thinning check. '
              'Prospective forecasts independently recompute both grids.'))


if __name__ == '__main__':
    main()
