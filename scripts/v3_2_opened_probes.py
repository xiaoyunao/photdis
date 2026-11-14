#!/usr/bin/env python3
"""Evaluate frozen photometric class probes on the opened field identities."""
import json
import os
from pathlib import Path

import joblib
import numpy as np

from photdis.stage1 import require_compute
from photdis.v3.controls import apply_temperature, probe_features
from photdis.v3_1.validation_report import field_statistic
from v2_1_near import ROOT, csv_rows, load, save_json, sha
from v3_1_validation_report import read_cohort
from v3_2_tilt_inputs import verified

VIEWS = ('no_color', 'grz', 'griz', 'multiband')
ANCHORS = ('reference/nominal', 'reference/response12', 'old_target/old_target', 'candidate/zero')


def scores(probabilities, label):
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (len(label), 3) or (p < 0).any() or not np.allclose(p.sum(axis=1), 1):
        raise ValueError('Invalid class probability')
    with np.errstate(divide='ignore'):
        log_score = np.log(p[np.arange(len(label)), label])
    return dict(log_score=log_score, brier=np.square(p - np.eye(3)[label]).sum(axis=1),
                accuracy=p.argmax(axis=1) == label, p_giant=p[:, 1])


def main():
    require_compute()
    root = ROOT / 'data/v3_0'
    control = ROOT / 'manifests/v3_0/controls/complete.json'
    record = json.loads(control.read_text())
    fitted_path = verified(root / 'controls/probes.joblib', record['outputs']['probes.joblib'])
    history_path = verified(root / 'references.npz', record['inputs']['data/v3_0/references.npz'])
    training_path = verified(root / 'controls/training.npz', record['outputs']['training.npz'])
    history, training = load(history_path), load(training_path)
    # Temperature calibration uses role 3 with available class labels.
    calibration = history['identity'][(history['role'] == 3) & (history['class_label'] >= 0)]
    calibration_roles = (history['role'] == 3) & (history['class_label'] >= 0)
    if (len(calibration) != record['temperature_calibration_sources'] or
            not (history['TYPE'][calibration_roles] == 'PSF').all() or
            not (training['TYPE'] == 'PSF').all()):
        raise ValueError('Frozen probe role identity mismatch')
    fitted = joblib.load(fitted_path)
    if set(fitted) != set(VIEWS):
        raise ValueError('Frozen probe feature sets changed')
    inputs = {}
    photo, evidence, records, _, _ = read_cohort(ROOT / 'data/v3_1', 'field', inputs)
    identity, label = photo['identity'], evidence['class_label']
    optical = [0, 1, 3]
    grz = photo['band_mask'][:, optical].all(axis=1)
    for key in ('flux', 'flux_error', 'transmission'):
        value = photo[key][:, optical]
        grz &= (np.isfinite(value) & (value > 0)).all(axis=1)
    eligible = photo['candidate_admission']
    seen = np.isin(identity, np.concatenate([training['identity'], calibration]))
    select = eligible & grz & ~seen
    labels = select & (label >= 0)
    indices = np.flatnonzero(select)
    known = label[select] >= 0
    admitted = {k: v[select] for k, v in photo.items()}
    models = {}
    for view in VIEWS:
        this = fitted[view]
        if not np.array_equal(this['model'].classes_, np.arange(3)):
            raise ValueError(f'Mismatched frozen classes for {view}')
        p = this['model'].predict_proba(probe_features(admitted, view))
        models[view] = apply_temperature(p, this['temperature'])
    anchors = {name: np.array([records[i]['views'][name]['class_probability'] for i in indices])
               for name in ANCHORS}
    absolute, contrasts, objects = [], [], []
    for i in np.flatnonzero(eligible):
        objects.append(dict(identity=int(identity[i]), TYPE=photo['TYPE'][i], role=str(photo['role'][i]),
                            cell=int(photo['cell'][i]), class_label=int(label[i]),
                            status='historical_fit_or_calibration_id' if seen[i] else
                            'ineligible_grz' if not grz[i] else 'predicted'))
    rows_by_id = {r['identity']: r for r in objects}
    for view, p in models.items():
        for i, value in zip(indices, p):
            for c, v in enumerate(value):
                rows_by_id[int(identity[i])][f'{view}_p{c}'] = float(v)
        new = scores(p[known], label[labels])
        for role in ('new_sky_field', 'new_source_seen_sky_field'):
            for stratum, c in [('all', None), ('dwarf', 0), ('giant', 1)]:
                mask = photo['role'][labels] == role
                if c is not None:
                    mask &= label[labels] == c
                cells = photo['cell'][labels][mask]
                for metric, values in new.items():
                    absolute.append(dict(role=role, stratum=stratum, probe=view, metric=metric,
                                         **field_statistic(values[mask], cells)))
                for name, prob in anchors.items():
                    old = scores(prob[known], label[labels])
                    for metric, values in new.items():
                        with np.errstate(invalid='ignore'):
                            delta = values[mask].astype(float) - old[metric][mask]
                        undefined = np.isnan(delta)
                        delta[undefined] = np.inf
                        contrasts.append(dict(role=role, stratum=stratum, probe=view, anchor=name,
                                              metric=metric, undefined_nonfinite=int(undefined.sum()),
                                              **field_statistic(delta, cells)))
    out = ROOT / 'data/v3_2/opened_probes'
    out.mkdir(parents=True, exist_ok=False)
    for name, rows in [('objects', objects), ('absolute', absolute), ('paired', contrasts)]:
        csv_rows(out / f'{name}.csv', rows)
    paths = [fitted_path, history_path, training_path, control, Path(__file__).resolve(),
             ROOT / 'src/photdis/v3/controls.py']
    inputs.update({str(p.relative_to(ROOT)): sha(p) for p in paths})
    save_json(out / 'complete.json', dict(inputs=inputs,
        outputs={p.name: sha(p) for p in out.iterdir()}, all_PSF=True,
        n=int(eligible.sum()), predicted=int(select.sum()), labeled=int(labels.sum()),
        training_or_calibration_overlaps=int((eligible & seen).sum()),
        missing_grz=int((eligible & ~grz).sum()), job=os.getenv('SLURM_JOB_ID'),
        scope='Frozen classification control on opened references. Identity exclusion does not '
              'establish sky independence or prevalence calibration. No probe enters likelihood.'))


if __name__ == '__main__':
    main()
