#!/usr/bin/env python3
"""Score frozen near-dwarf forecasts using same-frame Gaia and spectrum evidence."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import fitsio
import numpy as np

from photdis.gaia import SOURCE_ID_HEALPIX_FACTOR, derive_gaia_labels, join_gaia
from photdis.stage1 import require_compute
from photdis.v3.metrics import distance_summary, histogram_parallax, observed_parallax
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_2_minesweeper_inventory import DIGEST, FILENAME
from v3_2_reference_evidence import (DIST_COLS, MINESWEEPER_COLS, STAR_COLS, channels,
                                    minesweeper_channels, projected)
from v3_2_reference_inventory import ARCHIVES, text
from v3_3_sed import verified
from v3_4_near_development import DISTANCE_EDGES
from v3_4_near_development_inputs import CONFIG as DOMAIN, cells
from v3_4_near_forecast import OUT as FORECAST
from v3_4_near_reference_support import separation
from v3_4_near_validation_inputs import CONFIG, OUT as INPUTS, receipt_payload

GAIA = Path('/global/cfs/cdirs/cosmo/data/gaia/dr3/healpix')
ARCHIVE = ROOT/'data/v3_2/reference_sources'
MINE = ARCHIVE/'dr20_minesweeper'/FILENAME
OUT = INPUTS.parent/'evidence'
FREEZE = OUT/'freeze.json'
SPECTRA = ('DR20_MINESweeper', 'DR20_APO', 'DR19_Astra')
WEAK_SIGNED_SNR = 5.


def load_closed(inputs):
    record = json.loads((FORECAST/'complete.json').read_text())
    verified(CONFIG, record['inputs'][str(CONFIG.relative_to(ROOT))])
    predicted = receipt_payload(FORECAST, 'forecasts.npz', inputs)
    names = ('photometry', 'application', 'fresh_gaia', *SPECTRA)
    data = {name: receipt_payload(INPUTS, name+'.npz', inputs) for name in names}
    for name, x in data.items():
        rows = x['prediction_row'] if name != 'photometry' else slice(None)
        for key in ('identity', 'ls_id', 'TYPE'):
            np.testing.assert_array_equal(predicted[key][rows], x[key])
        if not (x['TYPE'] == 'PSF').all() or len(np.unique(x['identity'])) != len(x['identity']):
            raise ValueError(f'Invalid native PSF roster: {name}')
    return predicted, data


def source_ids(data):
    a, f = (data[k] for k in ('application', 'fresh_gaia'))
    return np.unique(np.r_[a['source_id'][a['gaia_unique']], f['source_id']])


def tile_inventory(ids):
    paths = [GAIA/f'healpix-{n:05d}.fits' for n in np.unique(ids//SOURCE_ID_HEALPIX_FACTOR)]
    status = {str(p):p.stat() if p.exists() else None for p in paths}
    return {p:[s.st_size, s.st_mtime_ns] if s is not None else None for p,s in status.items()}


def binding(data, hashes):
    acquisition = ROOT/'manifests/v3_2/reference_sources/dr19_acquisition.json'
    paths = [Path(__file__), CONFIG, DOMAIN, acquisition, *(ROOT/p for p in (
        'src/photdis/gaia.py', 'src/photdis/v3/metrics.py', 'scripts/v3_2_reference_evidence.py',
        'scripts/v3_4_near_development_inputs.py', 'scripts/v3_4_near_reference_support.py',
        'scripts/v3_4_near_validation_inputs.py', 'scripts/v3_4_near_development.py'))]
    hashes.update({str(p.relative_to(ROOT)): sha(p) for p in paths})
    astra = json.loads(acquisition.read_text())['archives']
    archives = {MINE:DIGEST, **{ARCHIVE/'dr19_astra_0.6.0'/name:astra[name]['sha256']
                              for name in ARCHIVES.values()}}
    for p, digest in archives.items():
        verified(p, digest)
        hashes[str(p.relative_to(ROOT))] = digest
    return dict(hashes=hashes, gaia_tiles=tile_inventory(source_ids(data)))


def geometry(p, err, eligible, cfg):
    strong = eligible & (p >= cfg['gaia_geometry_minimum_snr']*err)
    # Invert only the reliably positive high-S/N interval.
    center, low, high = (np.full(len(p), np.nan) for _ in range(3))
    sigma = cfg['geometry_sigma_multiplier']*err[strong]
    center[strong] = 1/p[strong]
    low[strong] = 1/(p[strong]+sigma)
    high[strong] = 1/(p[strong]-sigma)
    return center, low, high


def read_targeted(name, index):
    if name == 'DR19_Astra':
        base = ARCHIVE/'dr19_astra_0.6.0'
        changed = dict(index, identity=index['source_id'])
        dist = projected(base/ARCHIVES['distance'], changed, index['distance_row'], DIST_COLS)
        star = projected(base/ARCHIVES['stellar'], changed, index['stellar_row'], STAR_COLS)
        return channels(dist, star)
    obs = 'APO' if name == 'DR20_APO' else 'LCO'
    rows = index['reference_row'] if obs == 'APO' else index['distance_row']
    order = np.argsort(rows)
    with fitsio.FITS(str(MINE)) as source:
        names = {key.lower():key for key in source[1].get_colnames()}
        values = source[1].read(rows=rows[order], columns=[names[k] for k in MINESWEEPER_COLS])
    restore = np.argsort(order)
    table = {k:values[names[k]][restore] for k in MINESWEEPER_COLS}
    if table['source_id'].dtype.kind not in 'iu' or np.any(text(table['obs']) != obs):
        raise ValueError('DR20 source rows changed')
    np.testing.assert_array_equal(table['source_id'], index['source_id'])
    return minesweeper_channels(table)


def summarize_targeted(name, index, predicted, cfg):
    value = read_targeted(name, index)
    rows = index['prediction_row']
    label = value['class_label']
    d, sigma = (value[k] for k in ('distance_kpc', 'distance_error_kpc'))
    close = value['distance_precise'] & (d+cfg['geometry_sigma_multiplier']*sigma < cfg['near_kpc'])
    far = value['distance_precise'] & (d-cfg['geometry_sigma_multiplier']*sigma > cfg['near_kpc'])
    past = index.get('history_exposed', index.get('historical_evidence'))
    summary, distance = [], []
    for selection in ('primary', 'strict'):
        for exposure, include in [('all', np.ones(len(rows), bool)), ('outside_history', ~past)]:
            use = predicted[selection][rows] & include
            dwarf, giant = use & (label == 0), use & (label == 1)
            summary.append(dict(source=name, selection=selection, history=exposure, matched=int(use.sum()),
                dwarf=int(dwarf.sum()), giant=int(giant.sum()), class_unresolved=int((use & (label < 0)).sum()),
                giant_near=int((giant & close).sum()), giant_far=int((giant & far).sum()),
                giant_distance_unresolved=int((giant & ~(close | far)).sum())))
            mask = dwarf & value['distance_precise']
            q = predicted['qrf_quantiles_kpc'][rows][mask]
            distance.append(dict(source=name, selection=selection, history=exposure,
                **distance_summary(10+5*np.log10(q), 10+5*np.log10(d[mask]))))
    return value, summary, distance


def parallax_scores(predicted, rows, p, err, eligible):
    score = np.full((len(rows), 2), np.nan)
    for i in np.flatnonzero(eligible):
        r = rows[i]
        score[i, 0] = histogram_parallax(predicted['qrf_mu_edges'][r], predicted['qrf_mu_mass'][r],
                                        p[i], err[i], order=32)['log_score']
        score[i, 1] = observed_parallax(predicted['nominal_mu'], predicted['nominal_logp'][r],
                                       p[i], err[i])['log_score']
    return score


def summarize_gaia(name, index, predicted, photo, ids, gaia, labels, cfg):
    rows = index['prediction_row']
    native_id = index['source_id']
    lookup = np.clip(np.searchsorted(ids, native_id), 0, len(ids)-1)
    unique = index.get('gaia_unique', np.ones(len(rows), bool))
    exact = unique & (ids[lookup] == native_id) & gaia['gaia_match'][lookup]
    precise_position = np.zeros(len(rows), bool)
    precise_position[exact] = separation(index['RA'][exact], index['DEC'][exact],
                                         gaia['ra'][lookup[exact]], gaia['dec'][lookup[exact]])
    eligible = precise_position & labels['astro_quality'][lookup] & labels['zpt_valid'][lookup]
    p = labels['p_corr'][lookup].astype(float)
    err = gaia['parallax_error'][lookup].astype(float)
    center, low, high = geometry(p, err, eligible, cfg)
    near, far = high < cfg['near_kpc'], low > cfg['near_kpc']
    weights = index['design_weight']
    r, color = index['observed_r'], index['dereddened_g_r']
    weak = eligible & (p < WEAK_SIGNED_SNR*err)
    scores = parallax_scores(predicted, rows, p, err, eligible & predicted['primary'][rows])
    if np.any(~np.isfinite(scores[eligible & predicted['primary'][rows]])):
        raise ValueError('Non-finite score for eligible source')
    stratum, measured_r, measured_color = cells({k:v[rows] for k,v in photo.items()},
                                                json.loads(DOMAIN.read_text()))
    np.testing.assert_array_equal(stratum, index['stratum'])
    np.testing.assert_allclose(measured_r, r)
    np.testing.assert_allclose(measured_color, color)
    w1 = photo['band_state'][rows, 4]
    groups = {'all':np.ones(len(rows), bool), 'weak_signed_gaia':weak,
              'negative_signed_gaia':eligible & (p < 0)}
    groups.update({f'r_{a}_{b}':(r >= a) & (r < b) for a,b in [(0,14),(14,16),(16,18),(18,20)]})
    groups.update({f'field_{f:02d}':index['field'] == f for f in np.unique(index['field'])})
    groups.update({f'color_{a}_{b}':(color >= a) & (color < b)
                   for a,b in zip([-10,.2,.5,.8,1.1,1.4], [.2,.5,.8,1.1,1.4,10])})
    groups.update({f'w1_state_{i}':w1 == i for i in np.unique(w1)})
    summary, distributions, distance = [], [], []
    bins = predicted['qrf_bins'][rows]
    for selection in ('primary', 'strict'):
        selected = predicted[selection][rows]
        for group, include in groups.items():
            use = selected & include
            if not use.any():
                continue
            total = weights[use].sum()
            scored = use & eligible
            q = predicted['qrf_quantiles_kpc'][rows]
            reliable = use & np.isfinite(center)
            score = np.average(scores[scored], weights=weights[scored], axis=0) if scored.any() else (np.nan, np.nan)
            summary.append(dict(cohort=name, selection=selection, group=group, sample_n=int(use.sum()),
                estimated_n=float(total), gaia_exact_n=int((use & exact).sum()),
                signed_n=int(scored.sum()), weak_signed_n=int((use & weak).sum()),
                negative_n=int((scored & (p < 0)).sum()), geometry_n=int(reliable.sum()),
                near_n=int((use & near).sum()), far_n=int((use & far).sum()),
                geometric_unresolved_n=int((use & ~(near | far)).sum()),
                qrf_log_score=float(score[0]), nominal_log_score=float(score[1]),
                far_fraction_lower=float(weights[use & far].sum()/total),
                far_fraction_upper=float(1-weights[use & near].sum()/total)))
            distance.append(dict(cohort=name, selection=selection, group=group,
                **distance_summary(10+5*np.log10(q[reliable]), 10+5*np.log10(center[reliable]))))
            if group not in ('all', 'r_16_18', 'r_18_20'):
                continue
            for j,(lo, hi) in enumerate(zip(DISTANCE_EDGES[:-1], DISTANCE_EDGES[1:])):
                inside = (low >= lo) & (high < hi)
                outside = (high < lo) | (low >= hi)
                distributions.append(dict(cohort=name, selection=selection, group=group, lower_kpc=lo, upper_kpc=hi,
                    model_total=float(weights[use]@bins[use,j]), model_excluding_confirmed_far=float(
                        weights[use & ~far]@bins[use & ~far,j]), geometry_lower=float(weights[use & inside].sum()),
                    geometry_upper=float(weights[use & ~outside].sum()), denominator=float(total)))
    evidence = dict(identity=index['identity'], ls_id=index['ls_id'], TYPE=index['TYPE'],
        source_id=native_id, gaia_exact=exact, gaia_positional_agreement=precise_position, signed_eligible=eligible,
        p_corr=np.where(eligible,p,np.nan), parallax_error=np.where(eligible,err,np.nan), distance_kpc=center,
        lower_kpc=low, upper_kpc=high, near=near, far=far, qrf_log_score=scores[:,0], nominal_log_score=scores[:,1])
    return evidence, summary, distributions, distance


def main():
    require_compute()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze','score'))
    args = parser.parse_args()
    hashes = {}
    predicted, data = load_closed(hashes)
    frozen = binding(data, hashes)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == 'freeze':
        with FREEZE.open('x') as stream:
            json.dump(frozen, stream, indent=2, sort_keys=True)
        return
    if json.loads(FREEZE.read_text()) != frozen:
        raise ValueError('Evidence, evaluator or forecasts changed since freeze')
    started = OUT/'started.json'
    complete = OUT/'complete.json'
    if complete.exists():
        raise FileExistsError('Scored outcomes already recorded')
    if started.exists():
        if json.loads(started.read_text())['freeze_sha256'] != sha(FREEZE):
            raise ValueError('Cannot retry with different frozen inputs')
    else:
        with started.open('x') as stream:
            json.dump(dict(freeze_sha256=sha(FREEZE), utc=datetime.now(timezone.utc).isoformat(),
                           job=os.getenv('SLURM_JOB_ID')), stream, indent=2)
    cfg = json.loads(CONFIG.read_text())
    ids = source_ids(data)
    gaia = join_gaia(ids, GAIA)
    labels = derive_gaia_labels(gaia)
    if tile_inventory(ids) != frozen['gaia_tiles']:
        raise ValueError('Gaia source tiles changed during the exact join')
    all_gaia, distributions, distances, spectra, spectra_distance = [], [], [], [], []
    for name in ('application', 'fresh_gaia'):
        out, metrics, dist, scatter = summarize_gaia(name, data[name], predicted, data['photometry'],
                                                      ids, gaia, labels, cfg)
        atomic_npz(OUT/f'{name}.npz', **out)
        all_gaia.extend(metrics)
        distributions.extend(dist)
        distances.extend(scatter)
    app = data['application']
    spec_flags = {k:np.zeros(len(predicted['identity']), bool) for k in ('dwarf','giant','giant_near','giant_far')}
    for name in SPECTRA:
        values, composition, scatter = summarize_targeted(name, data[name], predicted, cfg)
        index = data[name]
        atomic_npz(OUT/f'{name}.npz', **{k:index[k] for k in ('identity','ls_id','TYPE','source_id')}, **values)
        rows = index['prediction_row']
        giant = values['class_label'] == 1
        d, error = values['distance_kpc'], values['distance_error_kpc']
        spec_flags['dwarf'][rows] |= values['class_label'] == 0
        spec_flags['giant'][rows] |= giant
        core = giant & values['distance_precise']
        spec_flags['giant_near'][rows] |= core & (d+cfg['geometry_sigma_multiplier']*error < cfg['near_kpc'])
        spec_flags['giant_far'][rows] |= core & (d-cfg['geometry_sigma_multiplier']*error > cfg['near_kpc'])
        spectra.extend(composition)
        spectra_distance.extend(scatter)
    rows, wt = app['prediction_row'], app['design_weight']
    a = {k:v[rows] for k,v in spec_flags.items()}
    contradiction = (a['dwarf'] & a['giant']) | (a['giant_near'] & a['giant_far'])
    dwarfs, giants = a['dwarf'] & ~contradiction, a['giant'] & ~contradiction
    bounds = []
    for cut in ('primary','strict'):
        use = predicted[cut][rows]
        total = wt[use].sum()
        row = dict(selection=cut, sampled=int(use.sum()), estimated_total=float(total),
                   confirmed_dwarf=int((use & dwarfs).sum()), confirmed_giant=int((use & giants).sum()),
                   conflict=int((use & contradiction).sum()), class_unresolved=int((use & ~(dwarfs | giants)).sum()))
        for key, excluded in [('giant', dwarfs), ('giant_near', dwarfs | (giants & a['giant_far'])),
                              ('giant_far', dwarfs | (giants & a['giant_near']))]:
            certain = giants & a[key]
            row[key+'_lower'] = float(wt[use & certain].sum()/total)
            row[key+'_upper'] = float(1-wt[use & excluded].sum()/total)
        bounds.append(row)
    for key, rows in [('giant_bounds', bounds), ('gaia_summary', all_gaia), ('distances', distances), ('distributions', distributions),
                      ('spectral_classes', spectra), ('spectral_dwarfs', spectra_distance)]:
        csv_rows(OUT/f'{key}.csv', rows)
    save_json(complete, dict(job=os.getenv('SLURM_JOB_ID'), freeze_sha256=sha(FREEZE), all_PSF=True,
        new_outcomes_read=True, outputs={p.name:sha(p) for p in sorted(OUT.iterdir())}))


if __name__ == '__main__':
    main()
