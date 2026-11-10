#!/usr/bin/env python3
"""Inspect retained extreme Gaia failures against local luminosity support."""
import numpy as np
from scipy.spatial import cKDTree
from v2_1_near import ROOT, DEST as OLD, load, subset, csv_rows, save_json
from photdis.stage1 import require_compute

DEST = ROOT/'data/v2_2'


def colors(t):
    return np.column_stack((t['g0']-t['r0'], t['r0']-t['z0'], t['z0']-t['w10']))


def main():
    require_compute()
    t = load(OLD/'development.npz'); t = subset(t, t['C'] & t['i_usable'])
    original = load(OLD/'frozen_training_audit.npz')
    full = subset(original, (original['TYPE'] == 'PSF') & np.isfinite(colors(original)).all(axis=1))
    known = load(OLD/'information_train.npz')
    assert (known['TYPE'] == 'PSF').all()
    trees = {name: (cat, cKDTree(colors(cat))) for name, cat in [('full_verified', full), ('candidate_68004', known)]}
    records = []
    for model in ('control', 'candidate'):
        pred = load(OLD/'partition_candidate'/(model+'_evaluation.npz'))
        assert np.array_equal(pred['identity'], t['identity'])
        valid = np.flatnonzero(t['parallax_eligible'])
        worst = valid[np.argsort(pred['gaia_log'][valid])[:32]]
        for i in worst:
            p = t['p_corr'][i]
            implied = t['g0'][i]-10+5*np.log10(p) if p > 0 else np.nan
            row = dict(model=model, identity=int(t['identity'][i]), TYPE=t['TYPE'][i],
                class_label=int(t['class_label'][i]), gaia_log=float(pred['gaia_log'][i]),
                corrected_parallax=p, parallax_error=t['gaia_parallax_error'][i],
                implied_Mg=implied, ref_chosen=t['ref_chosen'][i],
                ref_specdis=t['ref_specdis'][i], ref_spdist=t['ref_spdist'][i],
                predicted_mu50=pred['mu'][i, 2],
                nonstellar_risk=bool(t['nonstellar_risk'][i]), anchor_astrometry=bool(t['anchor_astrometry'][i]))
            for name, (catalog, tree) in trees.items():
                neighbors = np.asarray(tree.query_ball_point(colors(t)[i], .3), int)
                mg = catalog['Mg'][neighbors]; mg = mg[np.isfinite(mg)]
                row[name+'_local_n'] = len(mg)
                row[name+'_Mg_min'] = float(mg.min()) if len(mg) else None
                row[name+'_Mg_max'] = float(mg.max()) if len(mg) else None
                row[name+'_within1mag_n'] = int((abs(mg-implied) < 1).sum()) if np.isfinite(implied) else None
            records.append(row)
    csv_rows(DEST/'tables/extreme_gaia_support.csv', records)
    save_json(DEST/'extreme_support.json', dict(rows=len(records), radius_mag=.3,
        space='Observed gr,rz,z-W1 Euclidean colors; luminosity only evaluated after neighbor selection',
        interpretation='Diagnostic support availability, not a causal retraining experiment. Gaia implied luminosity is conditional on corrected positive parallax, not certified truth. Match/reference anomalies still require external evidence; no failure removed.',
        original_quadrature='v2_1/final_validation.json already checks worst32 with32vs64 nodes; results retained unchanged'))
    print('EXTREME SUPPORT', len(records), flush=True)


if __name__ == '__main__':
    main()
