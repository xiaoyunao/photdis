#!/usr/bin/env python3
"""Audit Gaussian intrinsic-flux support separately in measured/missing bands.

Component-weighted prior negative tails are approximation-risk diagnostics, not
the posterior probability that a physical star has negative emitted flux.
"""
import argparse
import json
import os
import joblib
import numpy as np
from v2_1_near import ROOT, load, sha, save_json, atomic_npz, csv_rows
from photdis.stage1 import require_compute
from photdis.v3.model import inference_model
D = ROOT/'data/v3_0'


def main():
    require_compute(); p = argparse.ArgumentParser(); p.add_argument('cohort', choices=['development', 'application']); a = p.parse_args()
    source = D/'predictions'/f'{a.cohort}_multiband_fit_0'; model_path = D/'sed_fit.joblib'
    model = inference_model(joblib.load(model_path)); rows = []
    for path in sorted(source.glob('chunk_*_joint.npz')):
        summary_path = path.with_name(path.name.replace('_joint', '')); meta = json.loads(summary_path.with_suffix('.json').read_text())
        assert sha(path) == meta['joint_sha256'] and sha(summary_path) == meta['summary_sha256']
        s = load(summary_path)
        with np.load(path) as z: component, mass, offset = z['component'], z['mass'], z['star_branch_offset']
        for j in range(len(s['identity'])):
            lo, hi = offset[j:j+2]; k = component[lo:hi]; w = mass[lo:hi]; w = w/w.sum()
            band = s['band_mask'][j]
            risk = model.negative_intrinsic_flux_probability[k]
            observed = risk[:, band].max(axis=1)
            missing = risk[:, ~band].max(axis=1) if (~band).any() else np.zeros(len(k))
            rows.append(dict(identity=int(s['identity'][j]), TYPE='PSF', observed_negative_tail_proxy=float(w@observed),
                missing_negative_tail_proxy=float(w@missing), any_band_negative_tail_proxy=float(w@risk.max(axis=1)),
                discrepancy_probability=float(s['discrepancy_probability'][j]), supported=bool(s['supported'][j]),
                numerical_converged=bool(s['converged'][j]), qnear=float(s['qnear'][j])))
    out = D/'support_audit'/a.cohort; out.mkdir(parents=True, exist_ok=True)
    data = {k: np.asarray([r[k] for r in rows]) for k in rows[0]}; atomic_npz(out/'per_star.npz', **data)
    summaries = []
    for name, use in [('all', np.ones(len(rows), bool)), ('supported', data['supported']), ('near95', data['qnear'] > .95)]:
        r = data['observed_negative_tail_proxy'][use]
        summaries.append(dict(group=name, n=int(use.sum()), median_observed_proxy=float(np.median(r)),
            q95_observed_proxy=float(np.quantile(r, .95)), above_point01=int((r > .01).sum()), above_point1=int((r > .1).sum()),
            nonconverged=int((~data['numerical_converged'][use]).sum()),
            mean_discrepancy_probability=float(data['discrepancy_probability'][use].mean())))
    csv_rows(out/'summary.csv', summaries)
    save_json(out/'complete.json', dict(n=len(rows), model_sha256=sha(model_path), prediction_sha256=sha(source/'complete.json'),
        output_sha256=sha(out/'per_star.npz'), csv_sha256=sha(out/'summary.csv'), summary=summaries,
        meaning='Posterior component-weighted prior Gaussian negative-tail proxy, not conditional intrinsic-flux probability or a nonstellar classifier.', job=os.getenv('SLURM_JOB_ID')))
    print(json.dumps(summaries), flush=True)


if __name__ == '__main__': main()
