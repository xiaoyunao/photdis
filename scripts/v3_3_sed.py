#!/usr/bin/env python3
"""Refit class-partitioned projected SED mixtures and assemble one candidate."""
import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np

from photdis.audit import rvs_metallicity
from photdis.stage1 import require_compute
from photdis.v3.calibration import reference_nodes
from photdis.v3.data import stable_rank
from photdis.v3.sed import EmpiricalSED
from v2_1_near import ROOT, atomic_npz, load, save_json, sha


BASE = ROOT/'data/v3_0'
OUT = ROOT/'data/v3_3/sed/initial'
CONFIG = ROOT/'configs/v3_3_sed.json'
NOMINAL_CONFIG = ROOT/'configs/v3_0_sed.json'
MANIFESTS = ROOT/'manifests/v3_0'


def verified(path, expected):
    if sha(path) != expected:
        raise ValueError(f'Unexpected input bytes: {path}')
    return path


def select_rows(identity, role, types, stellar, channel, labels, partitions, seed):
    types = np.char.strip(types.astype(str))
    training = role == 1
    psf = types == 'PSF'
    eligible = training & psf & stellar & np.isin(channel, (1, 2, 3))
    exclusions = dict(missing_type=int((training & (types == '')).sum()),
                      non_psf=int((training & ~psf & (types != '')).sum()),
                      missing_class=int((eligible & (labels < 0)).sum()))
    if len(np.unique(identity)) != len(identity):
        raise ValueError('Training and reference roles must be identity-disjoint')
    selection = []
    for part in partitions:
        rows = np.flatnonzero(eligible & (labels == part['label']))
        n = part['sources']
        if not part['components'] <= n <= len(rows):
            raise ValueError(f'Insufficient reference support for {part["name"]}')
        order = np.argsort(stable_rank(identity[rows], seed), kind='stable')
        selection.append((np.sort(rows[order[:n]]), n / len(rows), len(rows)))
    return selection, exclusions


def training_cloud(t, cal, nodes, recipe):
    channel = t['primary_channel']
    mu = np.empty((len(channel), nodes))
    weight = np.empty_like(mu)
    centers = np.empty(len(channel))
    for index, source in ((1, 'specdis'), (2, 'spdist'), (3, 'spdist')):
        use = channel == index
        params = cal[index - 1] if index < 3 else (0., 0., 0.)
        mu[use], weight[use] = reference_nodes(channel[use], t[source+'_distance'][use],
            t[source+'_sigma_mu'][use], t['p_corr'][use], t['gaia_parallax_error'][use],
            offset=params[0], extra=params[1], outlier=params[2], nodes=nodes)
        centers[use] = (10 + 5*np.log10(t[source+'_distance'][use]) + params[0]
                        if index < 3 else np.median(mu[use], axis=1))
    np.testing.assert_allclose(weight.sum(axis=1), 1, rtol=0, atol=1e-12)
    # Put the nearest reference-center node at the fitter's initializer index.
    # Swap the corresponding weights to preserve the reference integral.
    stars = np.arange(len(mu))
    middle = nodes // 2
    nearest = np.argmin(np.abs(mu-centers[:, None]), axis=1)
    for array in (mu, weight):
        array[:, middle], array[stars, nearest] = array[stars, nearest], array[:, middle].copy()
    binary = recipe['binary_reference_fraction']
    brightness = np.repeat([1., recipe['binary_brightness_factor']], nodes)
    amplitude = 10**(-.4*(np.tile(mu, (1, 2))-10)) * brightness
    observed = t['band_mask']
    trans = np.where(observed, t['transmission'], 1.)
    factor = amplitude[:, :, None] * trans[:, None, :]
    flux = np.where(observed, t['flux'], 0.)
    error = np.where(observed, t['flux_error'], 0.)
    y = np.zeros((*amplitude.shape, 7))
    y[:, :, :6] = flux[:, None, :] / factor
    noise = np.zeros((*amplitude.shape, 7, 7))
    bands = np.arange(6)
    noise[:, :, bands, bands] = (error[:, None, :] / factor)**2
    ebv = t['EBV']
    sigebv = np.hypot(recipe['dust_sigma_ebv_floor'], recipe['dust_fractional_sigma']*ebv)
    dust_coeff = np.divide(-2.5*np.log10(trans), ebv[:, None], out=np.zeros_like(trans),
                          where=ebv[:, None] > 0)
    dust = .4*np.log(10)*y[:, :, :6]*dust_coeff[:, None, :]*sigebv[:, None, None]
    noise[:, :, :6, :6] += dust[:, :, :, None]*dust[:, :, None, :]
    feh, valid = rvs_metallicity(t['rvs_feh'], t['rvs_teff'], t['rvs_logg'])
    error_feh = t['rvs_feh_err']
    measured = (valid & t['rvs_parameter_quality'] & np.isfinite(feh) & np.isfinite(error_feh)
                & (error_feh > 0) & (feh > -5) & (feh < 1))
    y[:, :, 6] = np.where(measured, feh, 0.)[:, None]
    noise[:, :, 6, 6] = np.where(measured, error_feh**2, 0.)[:, None]
    jacobian = -np.sum(np.log(factor)*observed[:, None, :], axis=2)
    logweight = np.log(np.concatenate([(1-binary)*weight, binary*weight], axis=1)) + jacobian
    optical = [0, 1, 3]
    magnitude = 22.5 - 2.5*np.log10(flux[:, optical] / trans[:, optical])
    features = np.c_[magnitude[:, 0]-magnitude[:, 1], magnitude[:, 1]-magnitude[:, 2],
                     magnitude[:, 1]-centers]
    if not all(np.isfinite(a).all() for a in (y, noise, logweight, features)):
        raise ValueError('Nonfinite training cloud')
    return dict(y=y, noise=noise, mask=np.c_[observed, measured], logweight=logweight, features=features)


def dump(value, path):
    temporary = path.with_suffix('.tmp')
    joblib.dump(value, temporary)
    temporary.replace(path)


def prepare(cfg, recipe):
    source = verified(BASE/'references.npz', json.loads((MANIFESTS/'preparation.json').read_text())['output_sha256'])
    cal_path = verified(BASE/'calibration.npz', json.loads((MANIFESTS/'calibration.json').read_text())['sha256'])
    with np.load(cal_path) as cal:
        params = cal['map_parameters']
    keys = ('primary_channel', 'specdis_distance', 'specdis_sigma_mu', 'spdist_distance', 'spdist_sigma_mu',
            'p_corr', 'gaia_parallax_error', 'band_mask', 'transmission', 'flux', 'flux_error', 'EBV',
            'rvs_feh', 'rvs_teff', 'rvs_logg', 'rvs_feh_err', 'rvs_parameter_quality')
    with np.load(source) as ref:
        ids, types, channel, labels = (ref[k] for k in ('identity', 'TYPE', 'primary_channel', 'class_label'))
        selection, exclusions = select_rows(ids, ref['role'], types, ref['reference_stellar_eligible'],
            channel, labels, cfg['partitions'], cfg['seed'])
        selected = np.concatenate([rows for rows, _, _ in selection])
        t = {key: ref[key][selected] for key in keys}
    knots = cfg['reference_mu_nodes']
    common = {str(p.relative_to(ROOT)): sha(p) for p in (source, cal_path, CONFIG, NOMINAL_CONFIG,
              Path(__file__), ROOT/'src/photdis/v3/calibration.py', ROOT/'src/photdis/v3/data.py')}
    offset = 0
    for part, (rows, probability, eligible) in zip(cfg['partitions'], selection):
        stop = offset + len(rows)
        folder = OUT/part['name']
        folder.mkdir(parents=True, exist_ok=True)
        path = folder/'cloud.npz'
        marker = folder/'cloud.json'
        if marker.exists():
            receipt = json.loads(marker.read_text())
            if receipt['inputs'] != common or receipt['partition'] != part:
                raise ValueError(f'Existing cloud has a different recipe: {folder}')
            verified(path, receipt['sha256'])
        else:
            if not (np.char.strip(types[rows].astype(str)) == 'PSF').all():
                raise ValueError('Missing PSF admission')
            cloud = training_cloud({k: a[offset:stop] for k, a in t.items()}, params, knots, recipe)
            atomic_npz(path, **cloud, identity=ids[rows], TYPE=np.full(len(rows), 'PSF'), rows=rows,
                       channel=channel[rows], class_label=labels[rows], inclusion_probability=np.full(len(rows), probability))
            save_json(marker, dict(inputs=common, sha256=sha(path), partition=part, eligible=eligible,
                inclusion_probability=probability, exclusions=exclusions, all_PSF=True, job=os.getenv('SLURM_JOB_ID')))
        print('TRAINING CLOUD', part['name'], len(rows), flush=True)
        offset = stop


def fit(cfg, part):
    folder = OUT/part['name']
    source = folder/'cloud.json'
    receipt = json.loads(source.read_text())
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in (source, CONFIG, Path(__file__), ROOT/'src/photdis/v3/sed.py')}
    cloud = load(verified(folder/'cloud.npz', receipt['sha256']))
    if receipt['partition'] != part or not (cloud['TYPE'] == 'PSF').all() or not (cloud['class_label'] == part['label']).all():
        raise ValueError('Invalid class training cloud')
    model_path = folder/'model.joblib'
    marker = folder/'complete.json'
    if marker.exists():
        done = json.loads(marker.read_text())
        if done['inputs'] != inputs:
            raise ValueError('Existing fit has different inputs')
        verified(model_path, done['sha256'])
        return
    checkpoint = folder/'checkpoint.joblib'
    resume = checkpoint.exists()
    if resume:
        state = joblib.load(checkpoint)
        if state['inputs'] != inputs:
            raise ValueError('Checkpoint has different inputs')
        model = state['model']
    else:
        model = EmpiricalSED(part['components'], max_iter=cfg['max_iter'],
                             tol=cfg['relative_likelihood_tolerance'], seed=cfg['seed']+part['label'])

    labels = np.full(len(cloud['identity']), -1)
    y, noise, mask, logweight = (cloud[k] for k in ('y', 'noise', 'mask', 'logweight'))

    def progress(iteration, value):
        if not model.converged:
            dump(dict(inputs=inputs, model=model), checkpoint)
        print('SED', part['name'], iteration + 1, value, flush=True)

    # Each fit already conditions on its known class. The label term is constant.
    model.fit(y, noise, mask, logweight, labels, cloud['features'], workers=cfg['workers'],
              callback=progress, initial_node=cfg['reference_mu_nodes']//2, resume=resume)
    responsibilities = model.training_responsibilities(y, noise, mask, logweight, labels, workers=cfg['workers'])
    mass = responsibilities.sum(axis=0)
    model.effective_reference_sources = mass**2 / np.sum(responsibilities**2, axis=0)
    model.reference_source_fraction = np.stack([responsibilities[cloud['channel'] == c].sum(axis=0)/mass
                                                for c in (1, 2, 3)], axis=1)
    model.training_identity = cloud['identity']
    dump(model, model_path)
    save_json(marker, dict(inputs=inputs, sha256=sha(model_path), part=part, all_PSF=True,
        converged=model.converged, history=model.history, job=os.getenv('SLURM_JOB_ID')))


def assemble(cfg, recipe):
    nominal_marker = MANIFESTS/'sed_fit.json'
    nominal = joblib.load(verified(BASE/'sed_fit.joblib', json.loads(nominal_marker.read_text())['sha256']))
    prior = nominal.weights @ nominal.classes
    models, receipts = [], []
    for part in cfg['partitions']:
        folder = OUT/part['name']
        path = folder/'complete.json'
        record = json.loads(path.read_text())
        if record['part'] != part or record['all_PSF'] is not True:
            raise ValueError('Unexpected class partition receipt')
        models.append(joblib.load(verified(folder/'model.joblib', record['sha256'])))
        receipts.append(path)
    combined = EmpiricalSED(sum(m.components for m in models))
    labels = np.repeat([p['label'] for p in cfg['partitions']], [m.components for m in models])
    combined.classes = np.eye(3)[labels]
    combined.weights = np.concatenate([prior[p['label']]*m.weights for m, p in zip(models, cfg['partitions'])])
    np.testing.assert_allclose(combined.weights @ combined.classes, prior, rtol=0, atol=1e-12)
    for key in ('means', 'covariances', 'negative_intrinsic_flux_probability', 'training_identity',
                'effective_reference_sources', 'reference_source_fraction'):
        setattr(combined, key, np.concatenate([getattr(m, key) for m in models]))
    if len(np.unique(combined.training_identity)) != len(combined.training_identity):
        raise ValueError('Duplicate identities across class fit partitions')
    combined.converged = all(m.converged for m in models)
    combined.system = nominal.system
    combined.binary_reference_fraction = recipe['binary_reference_fraction']
    combined.binary_brightness_factor = recipe['binary_brightness_factor']
    path = OUT/'model.joblib'
    marker = OUT/'complete.json'
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in
              (*receipts, nominal_marker, CONFIG, NOMINAL_CONFIG, Path(__file__))}
    if marker.exists():
        record = json.loads(marker.read_text())
        if record['inputs'] != inputs:
            raise ValueError('Existing assembly has different inputs')
        verified(path, record['sha256'])
        return
    dump(combined, path)
    save_json(marker, dict(inputs=inputs, sha256=sha(path),
        training_sources=len(combined.training_identity), components=combined.components,
        class_prior=prior.tolist(), converged=combined.converged, all_training_PSF=True, job=os.getenv('SLURM_JOB_ID')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'fit', 'assemble'))
    parser.add_argument('--partition', type=int, choices=range(3))
    args = parser.parse_args()
    require_compute()
    cfg = json.loads(CONFIG.read_text())
    recipe = json.loads(NOMINAL_CONFIG.read_text())
    if args.phase == 'prepare':
        prepare(cfg, recipe)
    elif args.phase == 'fit':
        if args.partition is None:
            parser.error('fit requires --partition')
        fit(cfg, cfg['partitions'][args.partition])
    else:
        assemble(cfg, recipe)


if __name__ == '__main__':
    main()
