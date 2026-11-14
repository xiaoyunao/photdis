#!/usr/bin/env python3
"""Compare the frozen DR19 forecasts on matched, independently qualified rows."""
import json
import os
from pathlib import Path

import numpy as np

from photdis.stage1 import require_compute
from photdis.v3.metrics import histogram_parallax, histogram_reference_score
from photdis.v3_1.freeze import verify_freeze
from photdis.v3_1.validation_report import field_statistic
from photdis.v3_1.validation_scores import compare_score_levels, reference_center_agreement, score_pdf
from v2_1_near import ROOT, csv_rows, load, save_json, sha
from v3_2_reference_freeze import VIEWS, aligned, read_admission
from v3_2_tilt_inputs import verified


def score_one(view, data, i, e):
    status = bool(data['numerical_pass'][i])
    if 'available' in data and not data['available'][i]:
        return dict(numerical_pass=False, prediction_available=False)
    q = data['mu_quantiles'][i]
    if view == 'qrf':
        edges = data['mu_edges'][i]
        mass = data['mass'][i]
        if not np.isfinite(edges).all() or not (np.diff(edges) > 0).all():
            raise ValueError('Invalid QRF edges')
        if not np.isfinite(mass).all() or (mass < 0).any() or not np.isclose(mass.sum(), 1.):
            raise ValueError('Invalid QRF mass')
        near = mass @ np.clip((15.-edges[:-1])/np.diff(edges), 0, 1)
        value = dict(qnear=float(near), width68=float(q[3]-q[1]), width95=float(q[4]-q[0]))
        reference_center_agreement(q, q[2])
        if e['parallax_eligible']:
            args = edges, mass, e['p_corr'], e['gaia_parallax_error']
            coarse = histogram_parallax(*args, order=32)
            fine = histogram_parallax(*args, order=64)
            value.update(gaia_log_score=fine['log_score'], gaia_pit=fine['pit'],
                         gaia_score_stable=abs(fine['log_score']-coarse['log_score']) < .02
                         and abs(fine['pit']-coarse['pit']) < .003)
        if e['distance_available']:
            mu = 10 + 5*np.log10(e['distance_kpc'])
            error = 5/np.log(10)*e['distance_error_kpc']/e['distance_kpc']
            score = histogram_reference_score(edges, mass, mu, error)
            value.update(reference_center_agreement(q, mu), teacher_log_score=score,
                         teacher_score_stable=bool(np.isfinite(score)))
    else:
        results = []
        for prefix in ('coarse_', ''):
            results.append(score_pdf(data[prefix+'mu_grid'], data[prefix+'logpdf'][i],
                                     data[prefix+'mu_quantiles'][i], data[prefix+'class_distance'][i], e))
        old, current = results
        check = compare_score_levels(old, current)
        stable = {r['channel']: r['passed'] for r in check['checks']}
        value = {name: current[name] for name in ('qnear', 'width68', 'width95')}
        if 'gaia' in current:
            value.update(gaia_log_score=current['gaia']['log_score'], gaia_pit=current['gaia']['pit'],
                         gaia_score_stable=stable['gaia'])
        if 'astronn' in current['references']:
            d = current['references']['astronn']
            for name in ('mu_residual', 'reference_agreement68', 'reference_agreement95', 'severe'):
                value[name] = d[name]
            value.update(teacher_log_score=d['log_score'], teacher_score_stable=stable['astronn'])
        value['p_giant'] = current['class_probability'][1]
        if 'class' in current:
            c = current['class']
            value.update(class_log_score=-np.inf if c['zero_probability'] else c['log_score'],
                         class_brier=c['brier'], predicted_class=c['predicted'],
                         class_correct=c['predicted'] == e['class_label'],
                         class_score_stable=stable['class'])
    value.update(numerical_pass=status, prediction_available=True, median_mu=float(q[2]),
                 high_qnear=bool(value['qnear'] > .95))
    if view == 'qrf':
        value['in_qrf_support'] = bool(data['in_qrf_support'][i]) if 'in_qrf_support' in data else None
    if e['distance_precise']:
        value['teacher_near'] = bool(e['distance_kpc'] < 10)
        value['teacher_near_brier'] = (value['qnear']-value['teacher_near'])**2
        value['teacher_near_probability_error'] = value['qnear']-value['teacher_near']
        if value['high_qnear']:
            value['teacher_far_given_high_q'] = not value['teacher_near']
        if value['teacher_near']:
            value['high_q_given_teacher_near'] = value['high_qnear']
    return value


def summarize(rows, evidence, candidate_admitted):
    d = evidence['distance_kpc']
    label = evidence['class_label']
    precise = evidence['distance_precise']
    cells = evidence['cell']
    groups = {'all': np.ones(len(d), bool), 'qualified_teacher': precise,
              'giant': label == 1, 'dwarf': label == 0,
              'metal_poor_giant': (label == 1) & evidence['feh_valid'] & (evidence['feh'] < -1.5)}
    for lo, hi in ((0, 2), (2, 5), (5, 10), (10, 20), (20, 50)):
        use = precise & (d >= lo) & (d < hi)
        for group in ('all', 'giant', 'metal_poor_giant'):
            groups[f'{group}_{lo}_{hi}_teacher_kpc'] = use & groups[group]
    scopes = {'native_photometric': groups['all'], 'matched_candidate_admission': candidate_admitted}
    scoped_groups = {(scope, group): base & mask for scope, base in scopes.items()
                     for group, mask in groups.items()}
    metrics = {'gaia_log_score': 'gaia_score_stable', 'teacher_log_score': 'teacher_score_stable',
               'class_log_score': 'class_score_stable', 'class_brier': 'class_score_stable',
               'class_correct': 'class_score_stable', 'p_giant': None,
               'mu_residual': 'teacher_score_stable', 'severe': 'teacher_score_stable',
               'reference_agreement68': 'teacher_score_stable', 'reference_agreement95': 'teacher_score_stable',
               'width68': None, 'width95': None, 'teacher_near_brier': 'teacher_score_stable',
               'teacher_near_probability_error': 'teacher_score_stable',
               'teacher_far_given_high_q': 'teacher_score_stable',
               'high_q_given_teacher_near': 'teacher_score_stable'}
    summaries, pairs = [], []
    for metric, flag in metrics.items():
        series = {view: np.asarray([r.get(metric, np.nan) for r in rows[view]], float) for view in VIEWS}
        fixed = {view: np.asarray([r['numerical_pass'] and (flag is None or r.get(flag, False))
                                  for r in rows[view]], bool) for view in VIEWS}
        for (scope, group), mask in scoped_groups.items():
            for accepted in (False, True):
                cand = series['candidate']
                for view in VIEWS:
                    x = series[view]
                    use = mask & ~np.isnan(x) & (fixed[view] if accepted else True)
                    summaries.append(dict(view=view, admission_scope=scope, group=group, metric=metric,
                                          numerical_only=accepted,
                                          **field_statistic(x[use], cells[use])))
                    # The high-q conditional denominators differ by view.
                    if view == 'candidate' or metric == 'teacher_far_given_high_q':
                        continue
                    pair = use & ~np.isnan(cand) & (fixed['candidate'] if accepted else True)
                    with np.errstate(invalid='ignore'):
                        delta = cand[pair]-x[pair]
                    undefined = np.isnan(delta)
                    delta[undefined] = np.inf
                    pairs.append(dict(anchor=view, admission_scope=scope, group=group, metric=metric,
                                      numerical_only=accepted,
                                      undefined_nonfinite=int(undefined.sum()),
                                      **field_statistic(delta, cells[pair])))
    return summaries, pairs


def main():
    require_compute()
    out = ROOT/'data/v3_2/final_validation_dr19'
    marker = out/'freeze.json'
    freeze_sha = verify_freeze(ROOT, marker)
    freeze = json.loads(marker.read_text())
    if freeze['evaluator'] != str(Path(__file__).resolve().relative_to(ROOT)):
        raise ValueError('Current scorer differs from registered evaluator')
    evidence_marker = out/'evidence_complete.json'
    complete = json.loads(evidence_marker.read_text())
    if complete['freeze_sha256'] != freeze_sha or not complete['outcomes_opened']:
        raise ValueError('Evidence and forecast freeze must match')
    evidence = load(verified(out/'evidence.npz', complete['outputs']['evidence.npz']))
    photo = load(ROOT/'data/v3_2/reference_native/photometry.npz')
    aligned(photo, evidence)
    forecast = json.loads((ROOT/freeze['forecast']).read_text())
    admission = forecast['candidate_admission']
    candidate_admitted = read_admission(verified(ROOT/admission['payload'], admission['sha256']),
                                        photo, admission['admitted'])
    output = out/'scores'
    if output.exists():
        raise FileExistsError(output)
    by_view = {}
    for name in VIEWS:
        view = forecast['views'][name]
        data = load(verified(ROOT/view['payload'], view['sha256']))
        aligned(photo, data)
        by_view[name] = [
            dict(identity=int(identity), ls_id=int(evidence['ls_id'][i]), TYPE='PSF',
                 candidate_admission=bool(candidate_admitted[i]),
                 cell=int(evidence['cell'][i]), view=name, teacher_precise=bool(evidence['distance_precise'][i]),
                 reference_class=int(evidence['class_label'][i]), **score_one(
                     name, data, i, {k: v[i] for k, v in evidence.items()}))
            for i, identity in enumerate(evidence['identity'])]
    absolute, paired = summarize(by_view, evidence, candidate_admitted)
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in [('objects', sum(by_view.values(), [])), ('absolute', absolute), ('paired', paired)]:
        csv_rows(output/f'{name}.csv', rows)
    save_json(output/'complete.json', dict(freeze_sha256=freeze_sha, evidence_sha256=sha(evidence_marker),
              all_PSF=True, views=list(VIEWS), n=len(evidence['identity']), job=os.getenv('SLURM_JOB_ID'),
              candidate_admitted=int(candidate_admitted.sum()),
              outputs={p.name: sha(p) for p in output.iterdir()},
              scope='Paired field-reference diagnostics. AstroNNDist agreement is conditional on its reference selection.'))


if __name__ == '__main__':
    main()
