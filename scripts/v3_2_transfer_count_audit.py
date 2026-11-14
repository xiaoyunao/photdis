#!/usr/bin/env python3
"""Trace frozen conditional count failures to weights and entry level/seed."""
import csv
import json
import os
from pathlib import Path

import numpy as np
from scipy.special import logsumexp, softmax

from photdis.stage1 import require_compute
from v2_1_near import ROOT, csv_rows, load, save_json, sha
from v3_2_tilt_inputs import verified

VARIANTS = ('base', 'entry_seed', 'entry_level', 'native_fine')


def compare(cond, base, lw):
    """Whole-catalog shift at one conditional state and stable draw IDs."""
    w = softmax(lw) if lw is not None else np.full(len(base), 1 / len(base))
    if not np.isfinite(w).all() or not np.isclose(w.sum(), 1):
        raise ValueError('Invalid transferred population weight')
    baseline = base.mean(axis=0)
    weight_effect = w @ base - baseline
    response_effect = w @ (cond - base)
    total = weight_effect + response_effect
    ix = int(np.argmax(np.abs(total)))
    return dict(weight=w, maximum_bin=ix, count_delta=float(total[ix]),
                weight_effect=float(weight_effect[ix]), response_effect=float(response_effect[ix]),
                maximum_count_delta=float(abs(total).max()), weight_ess=float(1 / (w @ w)),
                absolute_log_normalizer_delta=(float(logsumexp(lw) - np.log(len(w)))
                                               if lw is not None else 0.),
                relative_log_weight_span=(float(np.ptp(lw)) if lw is not None else 0.))


def main():
    require_compute()
    root = ROOT / 'data/v3_1/heldout_transfer/candidate'
    frozen = ROOT / 'manifests/v3_1/heldout_transfer/candidate'
    rows, provenance = [], {}
    for correction in range(34):
        name = f'correction_{correction:02}'
        active, manifest = root / name, frozen / name
        path = verified(active / 'complete.json', sha(manifest / 'complete.json'))
        marker = json.loads(path.read_text())
        if marker['case'] != 'candidate' or marker['correction'] != correction or not marker['all_PSF']:
            raise ValueError(f'Unexpected conditional state: {name}')
        checks = verified(active / 'checks.csv', marker['outputs']['checks.csv'])
        with checks.open(newline='') as stream:
            previous = {(r['envelope'], r['variant']): r for r in csv.DictReader(stream)}
        suffix = '/joint_counts/numerical_weights.npz'
        weights_path, digest = next((p, h) for p, h in marker['inputs'].items() if p.endswith(suffix))
        weights = load(verified(ROOT / weights_path, digest))
        provenance[str(path.relative_to(ROOT))] = sha(path)
        provenance[weights_path] = digest
        for envelope in ('broad', 'candidate'):
            counts = {}
            for variant in VARIANTS:
                filename = f'{envelope}_{variant}_counts.npz'
                counts[variant] = load(verified(active / filename, marker['outputs'][filename]))
                np.testing.assert_array_equal(counts[variant]['draw_ids'], weights['draw_ids'])
            base = counts['base']['conditional_means']
            n = int(counts['base']['n'])
            for variant in VARIANTS:
                value = counts[variant]
                cond = value['conditional_means']
                if cond.shape != base.shape or int(value['n']) != n or not np.allclose(cond.sum(axis=1), n):
                    raise ValueError(f'Invalid conditional counts: {name} {envelope} {variant}')
                numeric = compare(cond, base, weights[variant + '_logratio'] if variant != 'base' else None)
                weight = numeric.pop('weight')
                np.testing.assert_allclose(weight, value['draw_weights'], atol=1e-12, rtol=0)
                np.testing.assert_allclose(weight @ cond, value['mean'], atol=1e-9, rtol=0)
                original = previous[envelope, variant]
                np.testing.assert_allclose(numeric['maximum_count_delta'],
                                           float(original['maximum_count_mean_delta']), atol=1e-9, rtol=0)
                rows.append(dict(state=correction, envelope=envelope, variant=variant, n=n,
                                 draws=len(weight), v31_passed=original['passed'] == 'True',
                                 v31_pointwise_score_delta=float(original['maximum_pointwise_logscore_delta']),
                                 v31_probability_delta=float(original['maximum_probability_delta']), **numeric))
    out = ROOT / 'data/v3_2/transfer_count_audit'
    out.mkdir(parents=True, exist_ok=False)
    csv_rows(out / 'conditional_counts.csv', rows)
    provenance[str(Path(__file__).resolve().relative_to(ROOT))] = sha(__file__)
    save_json(out / 'complete.json', dict(inputs=provenance, n_states=34, n_rows=len(rows),
        failures=sum(not r['v31_passed'] for r in rows), all_PSF=True,
        outputs={'conditional_counts.csv': sha(out / 'conditional_counts.csv')},
        job=os.getenv('SLURM_JOB_ID'),
        scope='Frozen count sensitivity decomposition. No fits, validation outcomes, or acceptance changed.'))


if __name__ == '__main__':
    main()
