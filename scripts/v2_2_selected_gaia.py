#!/usr/bin/env python3
"""Selected full-observation Gaia checks retaining weak and negative parallaxes."""
import numpy as np
from v2_1_near import ROOT, DEST as OLD, load, strata, csv_rows, save_json, sha
from photdis.stage1 import require_compute

DEST = ROOT/'data/v2_2'


def main():
    require_compute()
    t = load(OLD/'development.npz'); p = load(DEST/'probability_canonical.npz')
    assert (t['TYPE'] == 'PSF').all() and np.array_equal(t['identity'], p['identity'])
    rows = []; bins = []
    for split, role in [(1, 'cal'), (2, 'development_test')]:
        for threshold in (.5, .8, .95):
            for group, mask in strata(t, cross=False).items():
                use = (t['split'] == split) & mask & t['parallax_eligible'] & (p['qnear'] > threshold)
                pit, score = t['full_pit'][use], t['full_log_density'][use]
                assert np.isfinite(pit).all() and ((pit >= 0) & (pit <= 1)).all()
                hist = np.histogram(pit, bins=np.linspace(0, 1, 11))[0]
                assert hist.sum() == use.sum()
                key = dict(split=role, threshold=threshold, group=group)
                rows.append(dict(**key, n=int(use.sum()), negative=int((t['p_corr'][use] < 0).sum()),
                    log_mean=float(score.mean()) if len(score) and np.isfinite(score).all() else None,
                    log_median=float(np.median(score)) if len(score) else None,
                    nonfinite_log=int((~np.isfinite(score)).sum()),
                    pit_low=float((pit < .05).mean()) if len(pit) else None,
                    pit_high=float((pit > .95).mean()) if len(pit) else None))
                bins.extend(dict(**key, low=i/10, high=(i+1)/10, n=int(n)) for i, n in enumerate(hist))
    csv_rows(DEST/'tables/selected_gaia_observation.csv', rows)
    csv_rows(DEST/'tables/selected_gaia_pit.csv', bins)
    save_json(DEST/'selected_gaia.json', dict(input_sha256=sha(OLD/'development.npz'),
        probability_sha256=sha(DEST/'probability_canonical.npz'), strata=len(rows),
        interpretation='Frozen expanded_A full parallax likelihood/PIT; no parallax inversion cut, weak and negative eligible observations retained. Not independent of shared Gaia training scale.'))
    print('SELECTED GAIA', len(rows), flush=True)


if __name__ == '__main__':
    main()
