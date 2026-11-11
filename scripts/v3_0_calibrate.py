#!/usr/bin/env python3
"""Shared-sky bootstrap source calibration, with original binary strata retained."""
import json
import os
import time
import numpy as np
from v2_1_near import ROOT, load, sha, atomic_npz, save_json, csv_rows
from photdis.stage1 import require_compute
from photdis.v3.data import stable_rank
from photdis.v3.calibration import fit_source, source_parameters, signed_parallax_loglike
D = ROOT/'data/v3_0'; T = D/'tables'


def main():
    require_compute(); start = time.monotonic()
    cfg = json.loads((ROOT/'configs/v3_0.json').read_text())
    files = [ROOT/'configs/v3_0.json', ROOT/'scripts/v3_0_calibrate.py',
             ROOT/'src/photdis/v3/calibration.py', D/'references.npz']
    freeze = {str(p.relative_to(ROOT)): sha(p) for p in files}
    marker = D/'calibration_freeze.json'
    if marker.exists(): assert json.loads(marker.read_text()) == freeze
    else: save_json(marker, freeze)
    if (D/'calibration.json').exists():
        assert sha(D/'calibration.npz') == json.loads((D/'calibration.json').read_text())['sha256']; return
    t = load(D/'references.npz'); selected = []; records = []; inputs = []; initial = []
    eligible = (t['role'] == 2) & t['parallax_eligible'] & t['reference_stellar_eligible']
    assert (t['TYPE'][eligible] == 'PSF').all() and not t['historical_fit'][eligible].any()
    for source in ('specdis', 'spdist'):
        rows = np.flatnonzero(eligible & t[source+'_valid'])
        rank = stable_rank(t['identity'][rows], salt=301201)
        rows = rows[np.argsort(rank, kind='stable')[:cfg['calibration_max_sources']]]
        mu = 10+5*np.log10(t[source+'_distance'][rows])
        data = (mu, t[source+'_sigma_mu'][rows], t['p_corr'][rows], t['gaia_parallax_error'][rows])
        result = fit_source(*data)
        selected.append(rows); inputs.append(data); initial.append(result.x)
        offset, scatter, outlier = source_parameters(result.x)
        # Actual signed likelihood quadrature refinement at the fitted parameters.
        coarse = signed_parallax_loglike(*data, offset, scatter, 48)
        fine = signed_parallax_loglike(*data, offset, scatter, 128)
        record = dict(source=source, n=len(rows), offset_mag=offset, extra_scatter_mag=scatter,
                      outlier_fraction=outlier, negative_parallax=int((data[2] < 0).sum()),
                      binary_flag_not1=int((t['specdis_raw_binary_flag'][rows] != 1).sum()),
                      sky_blocks=len(np.unique(t['cell'][rows])),
                      max_quadrature_loglike_change=float(abs(fine-coarse).max()),
                      mean_quadrature_loglike_change=float(abs(fine-coarse).mean()))
        print('SOURCE CALIBRATION', record, flush=True); records.append(record)
    cells = np.unique(np.concatenate([t['cell'][r] for r in selected]))
    rng = np.random.default_rng(301201)
    draws = []; parameters = []; cell_weights = []
    for draw in range(cfg['calibration_bootstrap_draws']):
        # The same sky weights apply to both sources and every repeated identity.
        weight = rng.dirichlet(np.ones(len(cells)))*len(cells)
        theta = []; physical = []
        for rows, data, center in zip(selected, inputs, initial):
            w = weight[np.searchsorted(cells, t['cell'][rows])]
            result = fit_source(*data, weights=w, initial=center)
            theta.append(result.x); physical.append(source_parameters(result.x))
        draws.append(theta); parameters.append(physical); cell_weights.append(weight)
        if draw % 8 == 0: print('SHARED SKY CALIBRATION DRAW', draw, flush=True)
    # Store each reference separately; correlated sources are never multiplied.
    for source, rows in zip(('specdis', 'spdist'), selected):
        atomic_npz(D/('calibration_'+source+'_inputs.npz'), identity=t['identity'][rows],
            TYPE=t['TYPE'][rows], rows=rows, cell=t['cell'][rows],
            binary_flag=t['specdis_raw_binary_flag'][rows], mu=10+5*np.log10(t[source+'_distance'][rows]),
            sigma_mu=t[source+'_sigma_mu'][rows], parallax=t['p_corr'][rows],
            parallax_error=t['gaia_parallax_error'][rows])
    physical = np.asarray(parameters)
    atomic_npz(D/'calibration.npz', theta_map=np.asarray(initial), theta_draws=np.asarray(draws),
        parameters=physical, map_parameters=np.array([source_parameters(x) for x in initial]),
        draw_ids=np.arange(len(draws), dtype='i8'), sky_cells=cells, sky_weights=np.asarray(cell_weights))
    summary = []
    for j, source in enumerate(('specdis', 'spdist')):
        for k, name in enumerate(('offset_mag', 'extra_scatter_mag', 'broad_fraction')):
            values = physical[:, j, k]
            summary.append(dict(source=source, parameter=name, mean=float(values.mean()),
                low=float(np.quantile(values, .025)), high=float(np.quantile(values, .975)),
                uncertainty='shared sky Bayesian bootstrap of source consistency fits'))
    csv_rows(T/'source_calibration.csv', records); csv_rows(T/'calibration_uncertainty.csv', summary)
    save_json(D/'calibration.json', dict(sha256=sha(D/'calibration.npz'), elapsed_seconds=time.monotonic()-start,
        jobs=os.getenv('SLURM_JOB_ID'), n_sources=[len(r) for r in selected], bootstrap_draws=len(draws),
        source_dependence='Gaia-dependent teachers; empirical consistency calibration, not independent scale proof',
        prospective_outcomes_opened=False, corrected_mu_convention='mu_corrected=mu_VAC+offset',
        quality='No BINARY_FLAG cut; full signed Gaia with astrometric quality, no parallax SNR selection'))


if __name__ == '__main__': main()
