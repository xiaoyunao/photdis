#!/usr/bin/env python3
"""Build error-aware SED training quadrature and benchmark/fit on allocated CPUs."""
import argparse
import json
import os
import resource
import time
import joblib
import numpy as np
from v2_1_near import ROOT, load, sha, save_json, atomic_npz
from photdis.stage1 import require_compute
from photdis.v3.calibration import reference_nodes
from photdis.v3.data import stable_rank
from photdis.v3.sed import EmpiricalSED
from photdis.audit import rvs_metallicity
D = ROOT/'data/v3_0'


def cloud():
    cfg = json.loads((ROOT/'configs/v3_0_sed.json').read_text())
    marker = D/'sed_cloud.json'
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in [D/'references.npz', D/'calibration.npz',
        ROOT/'configs/v3_0_sed.json', ROOT/'scripts/v3_0_sed.py', ROOT/'src/photdis/v3/calibration.py']}
    if marker.exists():
        record = json.loads(marker.read_text()); assert record['inputs'] == inputs
        assert sha(D/'sed_cloud.npz') == record['sha256']; return
    t = load(D/'references.npz'); cal = load(D/'calibration.npz')
    eligible = (t['role'] == 1) & t['reference_stellar_eligible'] & (t['primary_channel'] > 0)
    rows = np.flatnonzero(eligible)
    rows = rows[np.argsort(stable_rank(t['identity'][rows], cfg['seed']), kind='stable')[:cfg['max_training_sources']]]
    assert (t['TYPE'][rows] == 'PSF').all()
    n = len(rows); nodes = cfg['reference_mu_nodes']; binary = cfg['binary_reference_fraction']
    mus = np.zeros((n, nodes)); weights = mus.copy()
    channel = t['primary_channel'][rows]
    for c in (1, 2, 3):
        ix = np.flatnonzero(channel == c); rr = rows[ix]
        if not len(ix): continue
        source = 'specdis' if c == 1 else 'spdist'
        offset, extra, fraction = cal['map_parameters'][c-1] if c < 3 else (0., 0., 0.)
        mu, w = reference_nodes(channel[ix], t[source+'_distance'][rr], t[source+'_sigma_mu'][rr],
            t['p_corr'][rr], t['gaia_parallax_error'][rr], offset=offset, extra=extra,
            outlier=fraction, nodes=nodes)
        mus[ix], weights[ix] = mu, w
    # Brighter unresolved systems are a constrained latent branch during fitting
    # AND inference. They are not added only after the SED has already fitted them.
    brightness = np.r_[np.ones(nodes), np.full(nodes, cfg['binary_brightness_factor'])]
    mus = np.tile(mus, (1, 2)); weights = np.c_[weights*(1-binary), weights*binary]
    binary_state = np.r_[np.zeros(nodes, 'i1'), np.ones(nodes, 'i1')]
    amplitude = 10.**(-.4*(mus-10.))*brightness[None, :]
    tr = t['transmission'][rows]; band_mask = t['band_mask'][rows]
    factor = amplitude[:, :, None]*np.where(band_mask, tr, 1.)[:, None, :]
    rawflux = np.where(band_mask, t['flux'][rows], 0.)
    rawerror = np.where(band_mask, t['flux_error'][rows], 0.)
    y = np.zeros((n, nodes*2, 7)); y[:, :, :6] = rawflux[:, None, :]/factor
    noise = np.zeros((n, nodes*2, 7, 7))
    for b in range(6): noise[:, :, b, b] = (rawerror[:, None, b]/factor[:, :, b])**2
    # Small-angle linearization of per-object dust uncertainty in calibration.
    # Global extinction / band-system draws are separately propagated at inference.
    ebv = t['EBV'][rows]; sigebv = np.hypot(cfg['dust_sigma_ebv_floor'], cfg['dust_fractional_sigma']*ebv)
    extinct = -2.5*np.log10(np.where(band_mask, tr, 1.))
    coefficient = np.divide(extinct, ebv[:, None], out=np.zeros_like(extinct), where=ebv[:, None] > 0)
    dust = .4*np.log(10)*y[:, :, :6]*coefficient[:, None, :]*sigebv[:, None, None]
    noise[:, :, :6, :6] += dust[:, :, :, None]*dust[:, :, None, :]
    feh, correction_valid = rvs_metallicity(t['rvs_feh'][rows], t['rvs_teff'][rows], t['rvs_logg'][rows])
    ferr = t['rvs_feh_err'][rows]
    has_feh = correction_valid & t['rvs_parameter_quality'][rows] & np.isfinite(feh) & np.isfinite(ferr) & (ferr > 0) & (feh > -5) & (feh < 1)
    y[:, :, 6] = np.where(has_feh, feh, 0.)[:, None]
    noise[:, :, 6, 6] = np.where(has_feh, ferr**2, 0.)[:, None]
    mask = np.c_[band_mask, has_feh]
    # Change-of-variables Jacobian from scaled absolute flux back to observed
    # native flux. Without it broad reference-distance quadrature is biased.
    jacobian = -(np.log(factor)*band_mask[:, None, :]).sum(axis=2)
    logweight = np.log(weights)+jacobian
    f = t['flux'][rows][:, [0, 1, 3]]/tr[:, [0, 1, 3]]
    mag = 22.5-2.5*np.log10(f)
    mu_middle = np.median(mus[:, :nodes], axis=1)
    features = np.c_[mag[:, 0]-mag[:, 1], mag[:, 1]-mag[:, 2], mag[:, 1]-mu_middle]
    atomic_npz(D/'sed_cloud.npz', y=y, noise=noise, mask=mask, logweight=logweight,
        identity=t['identity'][rows], TYPE=t['TYPE'][rows], rows=rows, channel=channel,
        labels=t['class_label'][rows], features=features, reference_mu=mus,
        reference_weight=weights, binary_state=binary_state)
    save_json(marker, dict(inputs=inputs, sha256=sha(D/'sed_cloud.npz'), n=n,
        eligible_before_sampling=int(eligible.sum()), probability_sample='uniform exact-identity hash without replacement',
        class_missing=int((t['class_label'][rows] < 0).sum()), feh_missing=int((~has_feh).sum()),
        binary_brightness_factor=cfg['binary_brightness_factor'], binary_reference_fraction=binary,
        calibration='source consistency MAP for initial SED; shared ensemble propagation remains pending',
        gaia_reference_prior='EDSD L=1.35kpc, full signed likelihood',
        vac_reference_prior='flat mu with source offset/scatter/broad reference mixture',
        prior_dominated_Gaia_sources=int(((channel == 3) & (t['p_corr'][rows]/t['gaia_parallax_error'][rows] < 5)).sum())))
    print('SED CLOUD', n, 'sources;', y.shape[1], 'joint mu/binary nodes', flush=True)


def main():
    require_compute(); parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['cloud', 'benchmark', 'fit']); args = parser.parse_args()
    cfg = json.loads((ROOT/'configs/v3_0_sed.json').read_text())
    cloud()
    if args.stage == 'cloud': return
    start = time.monotonic(); t = load(D/'sed_cloud.npz')
    n = cfg['benchmark_sources'] if args.stage == 'benchmark' else len(t['identity'])
    iterations = cfg['benchmark_iterations'] if args.stage == 'benchmark' else cfg['max_iter']
    path = D/('sed_'+args.stage+'.joblib'); marker = path.with_suffix('.json')
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in [D/'sed_cloud.npz', ROOT/'src/photdis/v3/sed.py', ROOT/'configs/v3_0_sed.json']}
    if marker.exists():
        record = json.loads(marker.read_text()); assert record['inputs'] == inputs
        assert sha(path) == record['sha256']; return
    model = EmpiricalSED(cfg['components'], max_iter=iterations, tol=cfg['relative_likelihood_tolerance'], seed=cfg['seed'])
    def progress(iteration, value): print('SED ITERATION', iteration, value, 'elapsed', time.monotonic()-start, flush=True)
    model.fit(t['y'][:n], t['noise'][:n], t['mask'][:n], t['logweight'][:n], t['labels'][:n], t['features'][:n], workers=cfg['workers'], callback=progress)
    model.training_identity = t['identity'][:n]; model.system = 'DECam_DR11'
    model.binary_reference_fraction = cfg['binary_reference_fraction']; model.binary_brightness_factor = cfg['binary_brightness_factor']
    joblib.dump(model, path.with_suffix('.tmp.joblib')); path.with_suffix('.tmp.joblib').replace(path)
    elapsed = time.monotonic()-start
    save_json(marker, dict(inputs=inputs, sha256=sha(path), stage=args.stage, sources=n, components=cfg['components'],
        iterations=len(model.history), converged=model.converged, elapsed_seconds=elapsed,
        peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        output_bytes=path.stat().st_size, source_nodes=n*t['y'].shape[1],
        positivity_flagged_components=int((model.negative_intrinsic_flux_probability.max(axis=1) > .01).sum()),
        job=os.getenv('SLURM_JOB_ID'), not_for_application=args.stage != 'fit'))
    print(json.dumps(json.loads(marker.read_text()), indent=2), flush=True)


if __name__ == '__main__': main()
