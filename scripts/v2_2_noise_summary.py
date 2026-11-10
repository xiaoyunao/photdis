#!/usr/bin/env python3
"""Paired source-cluster bootstrap of cached finite noise repetitions."""
import numpy as np
from v2_1_near import load, csv_rows
from v2_2_noise import DEST
from photdis.stage1 import require_compute


def main():
    require_compute()
    d = load(DEST/'noise_design.npz')
    clean = load(DEST/'noise_chunks/replica_-1.npz')
    noisy = [load(DEST/'noise_chunks'/f'replica_{i}.npz') for i in range(4)]
    y = d['class_label'].astype(int); ref = d['truth_mu']; rng = np.random.default_rng(271828)
    records = []
    def loss(p):
        delta = p['mu'][:, 2]-ref
        known = np.isin(y, [0, 1, 2]); ii = np.arange(len(y)); safe = np.maximum(y, 0)
        point = np.log(np.maximum(p['point_class'][ii, safe], 1e-12))
        averaged = np.log(np.maximum(p['averaged_class'][ii, safe], 1e-12))
        point[~known] = np.nan; averaged[~known] = np.nan
        return dict(tail=(abs(delta) > 1).astype(float), coverage=((p['mu'][:, 1] <= ref) & (p['mu'][:, 3] >= ref)).astype(float),
            point_log=point, averaged_log=averaged)
    lc = loss(clean); ln = [loss(p) for p in noisy]
    for lo in (18, 20, 22):
        for group, mask in [('all', np.ones(len(y), bool)), ('no_W1', ~d['W1']), ('evolved', y == 1), ('dwarf', y == 0)]:
            for metric in lc:
                diff = np.mean([r[metric] for r in ln], axis=0)-lc[metric]
                use = mask & (d['r_low'] == lo) & np.isfinite(diff)
                values = diff[use]
                # Exactly one matched target per source per magnitude stratum;
                # averaging its four replicas before bootstrap preserves clustering.
                assert len(np.unique(d['identity'][use])) == len(values)
                if not len(values):
                    continue
                b = values[rng.integers(0, len(values), size=(2000, len(values)))].mean(axis=1)
                records.append(dict(r_low=lo, group=group, metric=metric, templates=len(values),
                    paired_noise_minus_clean=float(values.mean()), low95=float(np.quantile(b, .025)), high95=float(np.quantile(b, .975))))
    csv_rows(DEST/'tables/noise_cluster_bootstrap.csv', records)
    print('NOISE CLUSTER SUMMARY', len(records), flush=True)


if __name__ == '__main__':
    main()
