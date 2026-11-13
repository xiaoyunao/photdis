#!/usr/bin/env python3
"""New common source-calibration/bootstrap/systematic draws, originals frozen."""
import json
import os
import numpy as np
from v2_1_near import ROOT, load, sha, save_json, atomic_npz
from photdis.stage1 import require_compute
from photdis.v3.calibration import fit_source, source_parameters
from photdis.v3.uncertainty import shared_case_weights


def main():
    require_compute(); old = ROOT/'data/v3_0'; out = ROOT/'data/v3_1/response_library'; out.mkdir(exist_ok=True)
    paths = [old/'calibration.npz',old/'calibration_specdis_inputs.npz',old/'calibration_spdist_inputs.npz',
        old/'sed_cloud.npz',old/'references.npz',ROOT/'docs/v3_1_response_protocol.md',
        ROOT/'scripts/v3_1_response_prepare.py',ROOT/'src/photdis/v3/calibration.py',ROOT/'src/photdis/v3/uncertainty.py']
    inputs = {str(p.relative_to(ROOT)):sha(p) for p in paths}; freeze = out/'inputs.json'; marker = out/'complete.json'
    if freeze.exists(): assert json.loads(freeze.read_text()) == inputs
    else: save_json(freeze,inputs)
    if marker.exists():
        done = json.loads(marker.read_text()); assert done['inputs'] == inputs
        for name,digest in done['outputs'].items(): assert sha(out/name) == digest
        return
    original = load(paths[0]); source = [load(p) for p in paths[1:3]]
    for t in source: assert (t['TYPE'] == 'PSF').all() and len(np.unique(t['identity'])) == len(t['identity'])
    cells = np.unique(np.concatenate([t['cell'] for t in source])); rng = np.random.default_rng(319103)
    sky = rng.dirichlet(np.ones(len(cells)),size=32)*len(cells); parameters = []; theta = []; fit_records = []
    for j in range(32):
        point = []; physical = []; records = []
        for c,t in enumerate(source):
            weight = sky[j,np.searchsorted(cells,t['cell'])]
            fit = fit_source(t['mu'],t['sigma_mu'],t['parallax'],t['parallax_error'],weights=weight,initial=original['theta_map'][c])
            assert np.isfinite(fit.fun) and (fit.success or abs(fit.jac).max() < 1e-4)
            point.append(fit.x); physical.append(source_parameters(fit.x))
            records.append(dict(success=bool(fit.success),objective=float(fit.fun),gradient_max=float(abs(fit.jac).max())))
        theta.append(point); parameters.append(physical); fit_records.append(records)
        print('NEW RESPONSE CALIBRATION',j,flush=True)
    cloud = load(paths[3])
    with np.load(paths[4]) as z:
        assert (z['TYPE'][cloud['rows']] == 'PSF').all() and (z['role'][cloud['rows']] == 1).all()
        source_cells = z['cell'][cloud['rows']]
    case,cells_sed,sky_sed = shared_case_weights(cloud['identity'],source_cells,draws=32,seed=319101)
    fields = np.r_[np.arange(14),[100,101]]; rng = np.random.default_rng(319102)
    atomic_npz(out/'source_calibration.npz',map_parameters=original['map_parameters'],theta_map=original['theta_map'],
        parameters=np.array(parameters),theta_draws=np.array(theta),draw_ids=np.arange(1000,1032),sky_cells=cells,sky_weights=sky)
    atomic_npz(out/'sed_case_weights.npz',identity=cloud['identity'],TYPE=cloud['TYPE'],case_weights=case,
        sky_cells=cells_sed,sky_weights=sky_sed,draw_ids=np.arange(1000,1032))
    atomic_npz(out/'shared_draws.npz',draw_id=np.arange(1000,1032),source_calibration=np.array(parameters),
        calibration_draw_id=np.arange(1000,1032),training_identity=cloud['identity'],TYPE=cloud['TYPE'],
        training_sky_cells=cells_sed,training_sky_weights=sky_sed,field_ids=fields,
        global_band_zero_point_mag=rng.normal(size=(32,6))*np.array([.005,.005,.005,.005,.01,.01]),
        field_dust_offset=rng.normal(size=(32,len(fields)))*.003)
    save_json(out/'calibration_fit.json',fit_records)
    save_json(marker,dict(inputs=inputs,outputs={name:sha(out/name) for name in ['source_calibration.npz','sed_case_weights.npz','shared_draws.npz','calibration_fit.json']},
        job=os.getenv('SLURM_JOB_ID'),draws=32,draw_ids=list(range(1000,1032)),
        seeds=dict(source_calibration=319103,sed_case_sky=319101,band_dust=319102),
        scope='Independent Monte Carlo source/sky/calibration/systematic regeneration, conditional on same physical reference data. SED refits and target evidence pending.'))


if __name__ == '__main__': main()
