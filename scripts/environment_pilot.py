#!/usr/bin/env python3
"""Census and frozen distance forecasts for registered native DECam fields."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path

from astropy.coordinates import SkyCoord
import joblib
import numpy as np

from photdis.baseline_v1 import support
from photdis.gaia import NPIX, SOURCE_ID_HEALPIX_FACTOR, derive_gaia_labels, join_gaia
from photdis.reference_v1_4 import unique_join
from photdis.stage1 import galactic, require_compute, tile_bounds
from photdis.v3.controls import apply_temperature, probe_features
from photdis.v3.data import stable_rank
from photdis.v3_3 import frozen_qrf
from v2_1_near import ROOT, atomic_npz, csv_rows, load, save_json, sha
from v3_0_native import read_native
from v3_1_external_history import HISTORY, identity_position
from v3_3_sed import verified
from v3_4_near_forecast import prepare, qrf_bins
from v3_4_near_validate import GAIA
from v3_5_dr11_production import CUTS
from v3_6_gaia_census import distances
from v3_9_catalog import LEDGER, scan_chunk, settings
from v3_9_references import join, matched, raw_catalogs, subset
from v3_10_selection import DEST as CLASSIFIERS

CONFIG = ROOT/'configs/environment_20261120.json'
C = json.loads(CONFIG.read_text())
OUT = ROOT/'data/environment_20261120'
MODEL = None
PREDICT_INPUT = None
NSIDE = 32
MAG = (14., 17., 19., 20., 21., 22., 23.)


def finish(folder, inputs):
    save_json(folder/'complete.json', dict(config_sha256=sha(CONFIG), inputs=inputs,
        job=os.environ.get('SLURM_JOB_ID'), outputs={
            p.name: sha(p) for p in sorted(folder.iterdir()) if p.is_file() and p.name != 'complete.json'
        }))


def population(gaia, distance, ebv):
    """Conservative Gaia HRD proxies at both zero and full-column extinction."""
    g, bp_rp = gaia['phot_g_mean_mag'], gaia['bp_rp']
    with np.errstate(divide='ignore', invalid='ignore'):
        absolute = g - (10 + 5*np.log10(distance))
    label = np.full(len(g), 'uncertain', dtype='U12')
    full_color = bp_rp - 1.35*ebv
    full_abs = absolute - 2.75*ebv
    finite = np.isfinite(absolute + full_color)
    dwarf = (full_color >= .65) & (bp_rp < 2.5)
    dwarf &= (absolute > 3 + 1.8*bp_rp) & (full_abs > 3 + 1.8*full_color)
    dwarf &= (absolute < 5.7 + 2.7*bp_rp) & (full_abs < 5.7 + 2.7*full_color)
    giant = (full_color >= .65) & (absolute < 2.5) & (full_abs < 2.5)
    label[finite & dwarf] = 'dwarf_proxy'
    label[finite & giant] = 'giant_proxy'
    return label


def field_rows(field):
    name, longitude, latitude = field
    center = SkyCoord(longitude, latitude, unit='deg', frame='galactic').icrs
    ra, dec = center.ra.deg, center.dec.deg
    radius = C['field_radius_deg']
    ra_width = np.rad2deg(np.arcsin(np.sin(np.deg2rad(radius))/np.cos(np.deg2rad(dec))))
    cfg = dict(settings()['south'], maximum_grz_error_mag=C['max_grz_error_mag'],
               domains={'pilot':dict(r_min=C['observed_r'][0], r_max=C['observed_r'][1])})
    rows, ledger, paths = [], [], {}
    for path in sorted(Path(cfg['sweep_root']).glob('sweep-*.fits')):
        bounds = tile_bounds(path)
        r0, d0, r1, d1 = bounds
        middle = (r0 + (r1-r0)%360/2) % 360
        if abs((middle-ra+180)%360-180) >= (r1-r0)%360/2 + ra_width:
            continue
        if not (d0 < dec+radius and d1 > dec-radius):
            continue
        raw = read_native(path)
        delta = np.deg2rad((raw['RA'] - ra + 180) % 360 - 180)
        decl = np.deg2rad(raw['DEC'])
        dec0 = np.deg2rad(dec)
        cos_angle = np.sin(decl)*np.sin(dec0) + np.cos(decl)*np.cos(dec0)*np.cos(delta)
        chosen = raw[cos_angle >= np.cos(np.deg2rad(radius))]
        if not len(chosen):
            continue
        stat = path.stat()
        paths[str(path)] = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        for photo, codes in scan_chunk(chosen, bounds, cfg, 0):
            if photo:
                valid = (photo['row_state'] == 7) & ~photo['north_primary_region']
                take = subset(photo, valid)
                take['field'] = np.repeat(name, len(take['ls_id']))
                rows.append(take)
        ledger.append(dict(field=name, sweep=path.name, rows=len(chosen),
            south_precision=int(valid.sum()) if np.any(codes>=5) else 0,
            **dict(zip(LEDGER, np.bincount(codes, minlength=len(LEDGER))))))
    return rows, ledger, paths


def historical(t):
    direct = np.zeros(len(t['ls_id']), bool)
    ledger = []
    for filename, category in HISTORY:
        path = ROOT/filename
        if not path.exists():
            ledger.append(dict(path=filename, role=category, found=False))
            continue
        ids, _ = identity_position(path)
        take = np.isin(t['identity'], ids)
        direct |= take
        ledger.append(dict(path=filename, role=category, found=True, rows=len(ids),
                           overlaps=int(take.sum())))
    return direct, ledger


def census():
    folder = OUT/'census'
    folder.mkdir(parents=True, exist_ok=True)
    parts, ledger, input_paths = [], [], {}
    with ProcessPoolExecutor(max_workers=6) as pool:
        for rows, counts, paths in pool.map(field_rows, C['fields']):
            parts.extend(rows)
            ledger.extend(counts)
            input_paths.update(paths)
            print('Scanned', counts[0]['field'] if counts else 'empty', flush=True)
    t = join([p for p in parts if len(p['ls_id'])])
    assert len(np.unique(t['ls_id'])) == len(t['ls_id']) and (t['TYPE']=='PSF').all()
    unique = (t['source_id']>0)&(unique_join(t['source_id'], t['source_id'])>=0)
    t['unique_G3'] = unique
    t['classification'] = np.full(len(unique), -1, 'i1')
    t['historical_overlap'], history = historical(t)
    catalogs, inputs = raw_catalogs()
    inputs.update(input_paths)
    for source in ('specdis', 'spdist', 'spectra'):
        cat = catalogs[source]
        idx = matched(t, cat, legacy=(source=='spectra'))
        exists = (idx>=0)&unique
        t[source+'_matched'] = exists
        if source=='spectra':
            use = exists.copy()
            use[exists] = cat['class_eligible'][idx[exists]]
            t['classification'][use] = cat['class_label'][idx[use]]
            t['spectra_split'] = np.full(len(idx), 'missing', dtype='U14')
            t['spectra_split'][exists] = cat['split'][idx[exists]].astype('U14')
            continue
        good = exists.copy()
        good[exists] = cat['historical_quality'][idx[exists]] & cat['usable_uncertainty'][idx[exists]]
        for key in ('distance_kpc', 'sigma_mu'):
            t[source+'_'+key] = np.full(len(idx), np.nan)
            t[source+'_'+key][good] = cat[key][idx[good]]
    ids = t['source_id']
    gaia_valid = unique & (ids<NPIX*SOURCE_ID_HEALPIX_FACTOR)
    g = join_gaia(np.where(gaia_valid, ids, 0), GAIA)
    flags = derive_gaia_labels(g)
    geometry = json.loads(CUTS.read_text())
    t['gaia_state'], d, lo, hi = distances(t, gaia_valid, flags, g, geometry)
    t.update(gaia_distance_kpc=d, gaia_sigma_mu=5/np.log(10) *
             np.divide(g['parallax_error'], flags['p_corr'], out=np.full(len(d),np.nan), where=np.isfinite(d)),
             gaia_parallax=flags['p_corr'], gaia_parallax_error=g['parallax_error'],
             gaia_proxy=population(g, d, t['EBV']), gaia_lo=lo, gaia_hi=hi)
    t['longitude'], t['latitude'] = galactic(t['RA'], t['DEC'])
    for pixel in np.unique(ids[gaia_valid]//SOURCE_ID_HEALPIX_FACTOR):
        path = GAIA/f'healpix-{pixel:05d}.fits'
        if path.is_file():
            stat = path.stat()
            inputs[str(path)] = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        else:
            inputs[str(path)] = 'absent'
    inputs[str(CUTS)] = sha(CUTS)
    csv_rows(folder/'raw_admission.csv', ledger)
    csv_rows(folder/'historical_identity_audit.csv', history)
    atomic_npz(folder/'qualified.npz', **t)
    finish(folder, inputs)
    for name, _, _ in C['fields']:
        mask=t['field']==name
        print(name, 'precise', int(mask.sum()), 'Gaia', int((mask & np.isfinite(d)).sum()),
              'spec', int((mask & np.isfinite(t['specdis_distance_kpc'])).sum()),flush=True)


def sample(ids, selected, maximum, salt):
    rows = np.flatnonzero(selected)
    order = np.argsort(stable_rank(ids[rows], C['sample_seed']+salt), kind='stable')
    return rows[order[:maximum]]


def evaluate_batch(rows):
    t = PREDICT_INPUT
    return frozen_qrf(MODEL, t['mag_grz'][rows], t['error_grz'][rows], t['identity'][rows])


def forecast():
    global MODEL, PREDICT_INPUT
    folder = OUT/'forecast'
    folder.mkdir(parents=True, exist_ok=True)
    inventory = OUT/'census/complete.json'
    receipt = json.loads(inventory.read_text())
    t = load(verified(inventory.parent/'qualified.npz', receipt['outputs']['qualified.npz']))
    models, _, sources = prepare()
    MODEL = models['qrf']
    model_path = CLASSIFIERS/'south/model.joblib'
    marker = model_path.parent/'complete.json'
    classifier = joblib.load(verified(model_path, json.loads(marker.read_text())['outputs']['model.joblib']))
    trained = np.isin(t['identity'], MODEL.training_identity) | np.isin(t['ls_id'], -MODEL.training_identity)
    t['qrf_fit_overlap'] = trained
    in_support = np.empty(len(trained), bool)
    for start in range(0, len(trained), 8192):
        stop = start+8192
        in_support[start:stop] = support(MODEL, np.pad(t['mag_grz'][start:stop],((0,0),(0,2))))
    t['support'] = in_support
    selected = np.zeros(len(trained), bool)
    applications = np.zeros(len(trained), bool)
    for name, _, _ in C['fields']:
        field = t['field']==name
        apps = sample(t['ls_id'], field, C['application_per_field'], 1)
        applications[apps] = True
        for k, source in enumerate(('gaia','specdis','spdist')):
            available = field & in_support & ~trained & np.isfinite(t[source+'_distance_kpc'])
            use = sample(t['ls_id'], available, C['reference_per_field_per_provider'], 10+k)
            t.setdefault(source+'_evaluation', np.zeros(len(trained),bool))[use] = True
            selected[use] = True
    t['application_sample'] = applications
    selected |= applications & in_support
    t['predicted'] = selected
    rows = np.flatnonzero(selected)
    PREDICT_INPUT = t
    chunks = np.array_split(rows, max(1, int(np.ceil(len(rows)/192))))
    assert len(rows), 'No supported frozen model predictions in the fixed fields'
    with ProcessPoolExecutor(max_workers=16, mp_context=multiprocessing.get_context('fork')) as pool:
        results = list(pool.map(evaluate_batch, chunks))
    q = {k:np.concatenate([r[k] for r in results]) for k in ('supported','mu_edges','mass','distance')}
    assert q['supported'].all()
    bounds = 10 + 5*np.log10([2.,10.])
    widths = np.diff(q['mu_edges'],axis=1)
    cumulative = [np.sum(np.clip((m-q['mu_edges'][:,:-1])/widths,0,1)*q['mass'], axis=1) for m in bounds]
    score = apply_temperature(classifier['model'].predict_proba(probe_features(subset(t,rows),'grz')),
                              classifier['temperature'])[:,0]
    atomic_npz(folder/'pdf.npz', **q, rows=rows, TYPE=t['TYPE'][rows], ls_id=t['ls_id'][rows],
               dwarf_score=score, mass_2_10=cumulative[1]-cumulative[0])
    fields = ('TYPE','ls_id','source_id','field','r','longitude','latitude','RA','DEC','mag_grz',
              'error_grz','EBV','FRACFLUX_R','PSFSIZE_R','PSFDEPTH_R','classification','gaia_proxy',
              'qrf_fit_overlap','support','predicted','application_sample','gaia_state','historical_overlap',
              'unique_G3','spectra_split','gaia_parallax','gaia_parallax_error')
    keys = list(fields)
    for source in ('gaia','specdis','spdist'):
        keys += [source+'_'+x for x in ('distance_kpc','sigma_mu','evaluation')]
    atomic_npz(folder/'roster.npz', **{k:t[k] for k in keys})
    inputs = {str(p):sha(p) for p in (*sources, inventory, marker, model_path, Path(__file__))}
    finish(folder, inputs)
    print('Precise rows',len(trained),'nominal support',in_support.sum(),'predicted',len(rows),flush=True)


if __name__=='__main__':
    require_compute()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('census','forecast'))
    args=parser.parse_args()
    {'census':census,'forecast':forecast}[args.stage]()
