#!/usr/bin/env python3
"""Reference-separated far-shape evidence, retaining unknown luminosity support."""
import os
import json
import numpy as np
from v2_1_near import ROOT, DEST as OLD, load, subset, csv_rows, save_json, sha, atomic_npz, reference
from photdis.baseline_v1 import quantiles_interpolated
from photdis.baseline_v0 import distance_quantiles, metrics
from photdis.stage1 import require_compute

DEST = ROOT/'data/v2_2'


def color_cell(t):
    return np.digitize(t['g0']-t['r0'], [-.2, .3, .8, 1.3, 2])*6+np.digitize(t['r0']-t['z0'], [0, .3, .6, 1, 1.5])


def main():
    require_compute()
    t = load(OLD/'development.npz'); n = len(t['identity'])
    edges = load(OLD/'joint_grid.npz')['edges']; centers = (edges[1:]+edges[:-1])/2
    p = load(OLD/'probability.npz')
    refs = {kind: reference(t, kind) for kind in ('chosen', 'Gaia', 'specdis', 'spdist')}
    mu = np.full((n, 5), np.nan); mode = np.full(n, np.nan); ownmass = np.zeros(n)
    window = {kind: np.full((n, 6), np.nan) for kind in refs}
    sigma_window = {kind: np.full(n, np.nan) for kind in refs}
    chunks = []
    for path in sorted((OLD/'joint_chunks').glob('*.npz')):
        marker = json.loads(path.with_suffix('.json').read_text())
        assert sha(path) == marker['sha256']; chunks.append(marker['sha256'])
        q = load(path); rows = q['rows']; assert (q['TYPE'] == 'PSF').all()
        cl = np.where(t['class_label'][rows] >= 0, t['class_label'][rows], 3).astype(int)
        own = q['joint'][np.arange(len(rows)), cl].astype(float)
        total = own.sum(axis=1); good = total > 0; ownmass[rows] = total
        norm = own[good]/total[good, None]
        mu[rows[good]], _ = distance_quantiles(t['g0'][rows[good]], quantiles_interpolated(norm, edges))
        mode[rows[good]] = t['g0'][rows[good]]-centers[np.argmax(norm, axis=1)]
        for kind, (ref, sigma) in refs.items():
            central = t['g0'][rows]-ref[rows]
            overlap = np.maximum(0, np.minimum(edges[1:][None], central[:, None]+1)-np.maximum(edges[:-1][None], central[:, None]-1))/np.diff(edges)
            window[kind][rows] = np.einsum('ncb,nb->nc', q['joint'], overlap)
            overlap = np.maximum(0, np.minimum(edges[1:][None], central[:, None]+sigma[rows, None])-np.maximum(edges[:-1][None], central[:, None]-sigma[rows, None]))/np.diff(edges)
            sigma_window[kind][rows] = np.einsum('nb,nb->n', q['pdf'], overlap)
    atomic_npz(DEST/'far_shape.npz', identity=t['identity'], TYPE=t['TYPE'], oracle_mu=mu,
        oracle_mode=mode, oracle_class_contribution=ownmass,
        **{'window_'+k: v for k, v in window.items()}, **{'sigma_window_'+k: v for k, v in sigma_window.items()})
    results = []
    for kind, (ref, sigma) in refs.items():
        dist = 10**((ref-10)/5)
        for split, role in [(1, 'cal'), (2, 'development_test')]:
            for lo, hi in [(10, 20), (20, 50)]:
                for c, label in [(None, 'all'), (0, 'dwarf'), (1, 'evolved'), (2, 'other'), (-1, 'unknown')]:
                    use = (t['split'] == split) & (dist >= lo) & (dist < hi) & np.isfinite(ref)
                    if c is not None:
                        use &= t['class_label'] == c
                    conditional = use & (ownmass > 0)
                    row = dict(split=role, reference=kind, low_kpc=lo, high_kpc=hi, group=label,
                        n=int(use.sum()), conditional_n=int(conditional.sum()), independent_reference=False,
                        mean_qnear=float(p['qnear'][use].mean()) if use.any() else None,
                        mean_reference_window_mass=float(window[kind][use, :4].sum(axis=1).mean()) if use.any() else None,
                        mean_reference_sigma_mass=float(np.nanmean(sigma_window[kind][use])) if np.isfinite(sigma_window[kind][use]).any() else None,
                        mode_within_1mag=float((abs(mode[conditional]-ref[conditional]) < 1).mean()) if conditional.any() else None,
                        mean_own_class_mass=float(ownmass[use].mean()) if use.any() else None)
                    row.update({'conditional_'+k: v for k, v in metrics(mu[conditional], ref[conditional]).items()})
                    for j, channel in enumerate(('dwarf', 'evolved', 'other', 'unlabeled', 'Gaia_label', 'SpecDis_label')):
                        row['window_'+channel] = float(window[kind][use, j].mean()) if use.any() else None
                    results.append(row)
    csv_rows(DEST/'tables/far_shape.csv', results)
    tr = load(OLD/'frozen_training_audit.npz'); tr = subset(tr, tr['TYPE'] == 'PSF')
    app = load(DEST/'application_inputs.npz'); ap = load(DEST/'expanded_A_application.npz')
    select = []
    for name, catalog, weight, label, mass in [
        ('train_verified_PSF', tr, np.ones(len(tr['identity'])), tr['class_label'], None),
        ('cal', subset(t, t['split'] == 1), np.ones((t['split'] == 1).sum()), t['class_label'][t['split'] == 1], p['class_mass'][t['split'] == 1]),
        ('development', subset(t, t['split'] == 2), np.ones((t['split'] == 2).sum()), t['class_label'][t['split'] == 2], p['class_mass'][t['split'] == 2]),
        ('application', app, app['sample_weight'], np.full(len(app['identity']), -1), ap['weights'])]:
        cells = color_cell(catalog)
        for cell in np.unique(cells):
            use = cells == cell; w = weight[use]
            row = dict(population=name, color_cell=int(cell), n=int(use.sum()), weight=float(w.sum()))
            for k, cl in enumerate(('dwarf', 'evolved', 'other', 'unlabeled')):
                row['observed_'+cl] = None if name == 'application' else int((use & (label == (k if k < 3 else -1))).sum())
                row['model_'+cl] = float(np.average(mass[use, k], weights=w)) if mass is not None else None
            select.append(row)
    csv_rows(DEST/'tables/color_selection.csv', select)
    save_json(DEST/'far_complete.json', dict(job_id=os.getenv('SLURM_JOB_ID'), checked_chunks=len(chunks),
        reference_types=list(refs), class_conditioning='Oracle diagnostic only; unknown is missing-label channel, not a stellar species',
        independent_far_truth=False, reason='Existing Gaia/SpecDis/SPDist share training/reference provenance; per-source agreement does not certify independent branch truth',
        action='No branch reweighting or retraining; target class frequencies and independent distance-scale evidence remain missing'))
    print('FAR COMPLETE', flush=True)


if __name__ == '__main__':
    main()
