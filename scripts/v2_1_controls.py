#!/usr/bin/env python3
"""Assess existing PSF nonstellar controls using DR1 primary redshift quality."""
import json
import os
import concurrent.futures
import multiprocessing as mp
import joblib
import fitsio
import numpy as np
from v2_1_near import ROOT, DEST, TABLES, load, subset, csv_rows, save_json, sha, atomic_npz
from v2_1_information import sample_indices
from baseline_v1_4_experiments import arrays
from photdis.stage1 import require_compute
from photdis.reference_v1_4 import unique_join
from photdis.baseline_v1 import integrate
from photdis.probability_v2_1 import near_mass

ZCAT = '/global/cfs/cdirs/desi/public/dr1/spectro/redux/iron/zcatalog/v1/zall-pix-iron.fits'
COLS = ['TARGETID', 'ZCAT_PRIMARY', 'OBJTYPE', 'SPECTYPE', 'Z', 'ZERR', 'ZWARN',
        'DELTACHI2', 'COADD_FIBERSTATUS', 'TARGET_RA', 'TARGET_DEC']
_T = _MODEL = None


def predict(rows):
    m, e, ids = arrays(_T, rows)
    p = integrate(_MODEL, m, e, ids, draws=128)['wide']
    return dict(rows=rows, identity=ids, mu=p['mu'],
                qnear=near_mass(p['pdf'], _MODEL.edges, m[:, 0]))


def main():
    global _T, _MODEL
    require_compute()
    t = load(DEST/'nonstellar_unassessed_psf.npz')
    assert (t['TYPE'] == 'PSF').all()
    with np.load(ROOT/'data/baseline_v1_4/references.npz') as z:
        j = unique_join(t['identity'], z['identity'])
        assert (j >= 0).all()
        for key in ('targetid', 'RA', 'DEC'):
            t[key] = z[key][j]
    ids, count = np.unique(t['targetid'], return_counts=True)
    unique_ids = ids[(count == 1) & (ids > 0)]
    folder = DEST/'redrock_chunks'
    folder.mkdir(exist_ok=True)
    stat = os.stat(ZCAT)
    fingerprint = dict(path=ZCAT, size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                       query_sha256=sha(DEST/'nonstellar_unassessed_psf.npz'),
                       code_sha256=sha(__file__), columns=COLS, block=500000)
    freeze = folder/'freeze.json'
    if freeze.exists():
        assert json.loads(freeze.read_text()) == fingerprint
    else:
        save_json(freeze, fingerprint)
    parts = []
    with fitsio.FITS(ZCAT) as f:
        n = f[1].get_nrows()
        for start in range(0, n, 500000):
            path = folder/f'{start:09}.npz'
            marker = path.with_suffix('.json')
            if marker.exists():
                assert sha(path) == json.loads(marker.read_text())['sha256']
                parts.append(load(path))
                continue
            block = f[1].read(rows=np.arange(start, min(start+500000, n)), columns=['TARGETID', 'ZCAT_PRIMARY'])
            keep = np.isin(block['TARGETID'], unique_ids) & block['ZCAT_PRIMARY']
            rows = np.flatnonzero(keep)+start
            if len(rows):
                raw = f[1].read(rows=rows, columns=COLS)
            else:
                raw = f[1].read(rows=np.array([0]), columns=COLS)[:0]
            payload = {k: raw[k] for k in COLS}
            payload['source_row'] = rows
            atomic_npz(path, **payload)
            save_json(marker, dict(sha256=sha(path), rows=len(rows)))
            parts.append(payload)
            print('REDROCK', min(start+500000, n), '/', n, 'matched', sum(len(p['TARGETID']) for p in parts), flush=True)
    z = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    j = unique_join(t['targetid'], z['TARGETID'])
    matched = j >= 0
    jj = np.maximum(j, 0)
    spectype = np.char.strip(z['SPECTYPE'][jj].astype('U'))
    sep = 3600*np.hypot((z['TARGET_RA'][jj]-t['RA']+180)%360-180,
                       z['TARGET_DEC'][jj]-t['DEC'])
    # Use a true small-angle tangent-plane angular distance, including cos(dec).
    dra = ((z['TARGET_RA'][jj]-t['RA']+180)%360-180)*np.cos(np.deg2rad(t['DEC']))
    sep = 3600*np.hypot(dra, z['TARGET_DEC'][jj]-t['DEC'])
    base = (matched & (sep < 1) & np.isin(spectype, ['QSO', 'GALAXY'])
            & (np.char.strip(z['OBJTYPE'][jj].astype('U')) == 'TGT')
            & (z['ZWARN'][jj] == 0) & (z['COADD_FIBERSTATUS'][jj] == 0)
            & np.isfinite(z['DELTACHI2'][jj]) & (z['DELTACHI2'][jj] > 100)
            & np.isfinite(z['ZERR'][jj]) & (z['ZERR'][jj] > 0)
            & (z['Z'][jj] > .02) & (z['Z'][jj] > 5*z['ZERR'][jj]))
    train = load(ROOT/'data/v2_0/training_expanded_A.npz')['identity']
    base &= ~np.isin(t['identity'], train)
    for key in COLS:
        t['redrock_'+key] = z[key][jj]
    t['redrock_separation_arcsec'] = sep
    t['redrock_quality_assessed'] = base
    atomic_npz(DEST/'nonstellar_assessed_psf.npz', **subset(t, base))
    rows, sampling = [], []
    for split in (1, 2):
        for kind in ('GALAXY', 'QSO'):
            pool = np.flatnonzero(base & (t['split'] == split) & (spectype == kind))
            selected = pool[sample_indices(t['identity'][pool], 512)]
            rows.extend(selected)
            sampling.append(dict(split=split, spectype=kind, pool_n=len(pool), predicted_n=len(selected)))
    _T = subset(t, np.array(rows, int))
    _MODEL = joblib.load(ROOT/'data/baseline_v1_4/experiments/expanded_A_distance.joblib')
    tasks = [np.arange(i, min(i+64, len(rows))) for i in range(0, len(rows), 64)]
    if tasks:
        with concurrent.futures.ProcessPoolExecutor(max_workers=8, mp_context=mp.get_context('fork')) as pool:
            predictions = list(pool.map(predict, tasks))
        p = {k: np.concatenate([p[k] for p in predictions]) for k in predictions[0]}
        assert np.array_equal(p['identity'], _T['identity'])
        atomic_npz(DEST/'nonstellar_predictions.npz', **p, TYPE=_T['TYPE'], split=_T['split'],
                   spectype=_T['redrock_SPECTYPE'])
        scores = []
        for row in sampling:
            use = (_T['split'] == row['split']) & (np.char.strip(_T['redrock_SPECTYPE'].astype('U')) == row['spectype'])
            for threshold in [.5, .8, .95]:
                scores.append(dict(**row, threshold=threshold,
                    false_near=int((p['qnear'][use] > threshold).sum()),
                    false_near_fraction=float(np.mean(p['qnear'][use] > threshold)) if use.any() else None))
        csv_rows(TABLES/'nonstellar_near.csv', scores)
    stat_after = os.stat(ZCAT)
    assert (stat.st_size, stat.st_mtime_ns) == (stat_after.st_size, stat_after.st_mtime_ns)
    save_json(DEST/'nonstellar_complete.json', dict(job_id=os.getenv('SLURM_JOB_ID'),
        raw_psf=len(t['identity']), primary_unique_matched=int(matched.sum()),
        duplicate_query_target_excluded=int(np.isin(t['targetid'], ids[count != 1]).sum()),
        assessed_controls=int(base.sum()), sampled_for_prediction=len(rows), sampling=sampling,
        criteria='Primary TGT; SPECTYPE GALAXY/QSO; ZWARN0; fiberstatus0; DELTACHI2>100; Z>0.02 and5ZERR; position<1arcsec; verified DR11 PSF; no training identity',
        sources=['https://data.desi.lbl.gov/doc/releases/dr1/', 'https://data.desi.lbl.gov/doc/releases/dr1/known-issues/'],
        limitation='Conservative pipeline-quality controls, not visually confirmed labels or full nonstellar completeness; Redrock can be overconfident. DELTACHI2 and redshift guards are prespecified here, not universal DESI cuts.',
        WD='49 WD candidates remain unconfirmed and unused as labels', fingerprint=fingerprint))


if __name__ == '__main__':
    main()
