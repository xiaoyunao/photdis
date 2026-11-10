#!/usr/bin/env python3
"""Frozen full-PDF application comparison on the existing weighted PSF sample."""
import json
import os
import joblib
import numpy as np
import fitsio
from v2_1_near import ROOT, DEST as OLD, load, subset, csv_rows, save_json, sha, atomic_npz
from baseline_v1_common import arrays
from baseline_v1_predict import load_selected
from photdis.baseline_v0 import BANDS, ERR
from photdis.baseline_v1 import integrate
from photdis.probability_v2_1 import near_mass
from photdis.selection_v2_2 import canonical_probability, evidence_cells
from photdis.stage1 import require_compute

DEST = ROOT/'data/v2_2'


def conditions(t):
    groups = dict(all=np.ones(len(t['identity']), bool), no_W1=~t['B'], W1_only=t['B'] & ~t['C'], W1W2=t['C'])
    for key, values, edges in [('r', t['r'], [-np.inf, 18, 20, 22, 25, np.inf]),
                               ('error', t['optical_error'], [0, .03, .1, .2, np.inf]),
                               ('gr', t['g0']-t['r0'], [-np.inf, -.2, .3, .8, 1.3, 2, np.inf]),
                               ('rz', t['r0']-t['z0'], [-np.inf, 0, .3, .6, 1, 1.5, np.inf])]:
        for lo, hi in zip(edges[:-1], edges[1:]):
            groups[f'{key}_{lo}_{hi}'] = (values >= lo) & (values < hi)
    sky_key = 'field' if 'field' in t else 'cell'
    for f in np.unique(t[sky_key]):
        groups[f'{sky_key}_{f}'] = t[sky_key] == f
    return groups


def main():
    require_compute(); (DEST/'application_chunks').mkdir(parents=True, exist_ok=True)
    app = load(OLD/'application_psf.npz'); original = {k: v.copy() for k, v in app.items()}
    assert len(app['identity']) == 26000 and (app['TYPE'] == 'PSF').all()
    # Reproduce exact stored float inputs, not recomputed raw-flux roundoff.
    for field in np.unique(app['field']):
        use = app['field'] == field
        columns = ['identity', 'ls_id', *BANDS, *ERR, 'r', 'B', 'C', 'optical_error', 'mu', 'route', 'draws']
        with fitsio.FITS(str(ROOT/f'data/baseline_v1/catalog_{field:02}.fits')) as f:
            a = f[1].read(rows=app['parent_row'][use], columns=columns)
        assert np.array_equal(a['identity'], app['identity'][use]) and np.array_equal(a['ls_id'], app['ls_id'][use])
        for key in columns[2:]:
            if key not in app:
                app[key] = np.zeros((26000,)+a[key].shape[1:], dtype=a[key].dtype)
            app[key][use] = a[key]
    atomic_npz(DEST/'application_inputs.npz', **app)
    delta = {k: float(np.nanmax(abs(app[k]-original[k]))) for k in BANDS+ERR}
    lock, deployed = load_selected()
    assert all(lock['models'][k]['variant'] == 'wide' for k in 'ABC')
    assert all(lock['numerical'][k]['draws'] == 128 for k in ('low', 'high')) and (app['draws'] == 128).all()
    model_path = ROOT/'data/baseline_v1_4/experiments/expanded_A_distance.joblib'
    expanded = joblib.load(model_path)
    inputs = [OLD/'application_psf.npz', model_path, ROOT/'data/baseline_v1/selection.json',
              ROOT/'docs/v2_2_protocol.md', ROOT/'scripts/v2_2_application.py', ROOT/'src/photdis/selection_v2_2.py']
    inputs += [ROOT/f'data/baseline_v0/{name}.joblib' for name in 'ABC']
    freeze = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    frozen_path = DEST/'application_freeze.json'
    if frozen_path.exists():
        previous = json.loads(frozen_path.read_text())
        if previous != freeze:
            changed = [k for k in freeze if previous.get(k) != freeze[k]]
            assert changed == ['scripts/v2_2_application.py'], 'Explicit checkpoint migration required'
            initial = 'f06f0d8077c1c6c5333c0bcccaafc9ce7043eb0717c4d01bb6827f4235f09e2a'
            summary_writer = '7062d2a36f53fd901f5f3f72702efff37a2e1a5181d84ea0b20d50e9ef63b853'
            assert previous[changed[0]] in (initial, summary_writer), 'Unrecognized writer migration'
            if previous[changed[0]] == initial:
                assert not list((DEST/'application_chunks').glob('baseline_v1_*.npz'))
                migration_name = 'grid'
                reason = 'Zero-pad storage only; preserve native ABC grids and scientific outputs'
            else:
                assert len(list((DEST/'application_chunks').glob('*.npz'))) == 204
                migration_name = 'summary'
                reason = 'Development sky uses cell while application uses field; both full prediction caches reused unchanged'
            for path in (DEST/'application_chunks').glob('*.npz'):
                assert sha(path) == json.loads(path.with_suffix('.json').read_text())['sha256']
            save_json(DEST/('application_freeze_previous_'+previous[changed[0]][:8]+'.json'), previous)
            save_json(DEST/('application_'+migration_name+'_migration.json'), dict(changed=changed, previous=previous[changed[0]], current=freeze[changed[0]], reason=reason))
            save_json(frozen_path, freeze)
    else:
        save_json(frozen_path, freeze)
    atomic_npz(DEST/'application_grids.npz', expanded_A=expanded.edges, **{k: deployed[k].edges for k in 'ABC'})
    summaries = {}
    for name in ('expanded_A', 'baseline_v1'):
        pieces = []
        for start in range(0, 26000, 256):
            ix = np.arange(start, min(start+256, 26000))
            path = DEST/'application_chunks'/f'{name}_{start:06}.npz'
            meta = path.with_suffix('.json')
            if path.exists() and meta.exists():
                assert sha(path) == json.loads(meta.read_text())['sha256']
                p = load(path)
            else:
                m, e, ids = arrays(app, ix)
                route = np.zeros(len(ix), dtype=int) if name == 'expanded_A' else app['route'][ix]
                assert np.isin(route, [0, 1, 2]).all()
                p = dict(identity=ids, TYPE=app['TYPE'][ix], route=route)
                for code in np.unique(route):
                    use = route == code
                    model = expanded if name == 'expanded_A' else deployed['ABC'[code]]
                    value = integrate(model, m[use], e[use], ids[use], 128, keep_pdf=True)['wide']
                    value['qnear'] = canonical_probability(near_mass(value['pdf'], model.edges, m[use, 0]))
                    for k, v in value.items():
                        if k == 'pdf' and name == 'baseline_v1':
                            if k not in p:
                                p[k] = np.zeros((len(ix), max(len(x.centers) for x in deployed.values())), dtype=v.dtype)
                                p['pdf_bins'] = np.zeros(len(ix), dtype=int)
                            p[k][use, :v.shape[1]] = v
                            p['pdf_bins'][use] = v.shape[1]
                        else:
                            if k not in p:
                                p[k] = np.empty((len(ix),)+v.shape[1:], dtype=v.dtype)
                            p[k][use] = v
                atomic_npz(path, **p)
                save_json(meta, dict(sha256=sha(path), n=len(ix)))
            assert np.array_equal(p['identity'], app['identity'][ix]) and (p['TYPE'] == 'PSF').all()
            assert np.allclose(p['pdf'].sum(axis=1), 1, atol=1e-7)
            pieces.append({k: v for k, v in p.items() if k != 'pdf'})
            print(name, min(start+256, 26000), '/26000', flush=True)
        summary = {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
        atomic_npz(DEST/(name+'_application.npz'), **summary)
        summaries[name] = summary
    # Original catalog quantiles were serialized to float32.
    error = float(np.max(abs(summaries['baseline_v1']['mu']-app['mu'])))
    assert error < 5e-6, ('Catalog reproduction failed', error)
    dev = load(OLD/'development.npz'); probability = load(OLD/'probability.npz')
    devcell = evidence_cells(dev); appcell = evidence_cells(app)
    eligible = (dev['split'] == 2) & np.isfinite(dev['ref_chosen'])
    count = np.bincount(devcell[eligible], minlength=2160)
    evidence_n = count[appcell]
    rows, bins, cellrows = [], [], []
    for name, p in summaries.items():
        for threshold in (-1, .5, .8, .95):
            selected = np.ones(26000, bool) if threshold < 0 else p['qnear'] > threshold
            for group, mask in conditions(app).items():
                use = selected & mask; w = app['sample_weight'][use]; total = w.sum()
                avg = lambda x: float(np.average(x[use], weights=w)) if total else None
                rows.append(dict(model=name, population='application', threshold=threshold, group=group,
                    n=int(use.sum()), weighted_n=float(total), eligible_weight=float(app['sample_weight'][mask].sum()),
                    mean_q=avg(p['qnear']), mean_width68=avg(p['mu'][:, 3]-p['mu'][:, 1]),
                    mean_width95=avg(p['mu'][:, 4]-p['mu'][:, 0]), ambiguous_fraction=avg(p['ambiguous']),
                    mean_unlabeled_contribution=avg(p['weights'][:, 3]),
                    evidence_zero_weight=float(app['sample_weight'][use & (evidence_n == 0)].sum()),
                    evidence_sparse_weight=float(app['sample_weight'][use & (evidence_n > 0) & (evidence_n < 20)].sum()),
                    evidence_20plus_weight=float(app['sample_weight'][use & (evidence_n >= 20)].sum())))
            for key, value, edges in [('qnear', p['qnear'], np.linspace(0, 1, 21)),
                                      ('width68', p['mu'][:, 3]-p['mu'][:, 1], np.r_[np.arange(0, 13, .5), np.inf])]:
                hist = np.histogram(value[selected], bins=edges, weights=app['sample_weight'][selected])[0]
                assert np.isclose(hist.sum(), app['sample_weight'][selected].sum())
                bins.extend(dict(model=name, population='application', threshold=threshold, variable=key,
                    low=lo, high=hi, weight=float(n)) for lo, hi, n in zip(edges[:-1], edges[1:], hist))
        for key in np.unique(appcell):
            use = appcell == key
            cellrows.append(dict(model=name, observable_cell=int(key), development_chosen_n=int(count[key]),
                sample_n=int(use.sum()), weighted_n=float(app['sample_weight'][use].sum()),
                high_q_weight=float(app['sample_weight'][use & (p['qnear'] > .95)].sum())))
    # Same expanded_A and observable grids for development and application; no v1 performance substitution.
    for threshold in (-1, .5, .8, .95):
        selected = eligible & (True if threshold < 0 else probability['qnear'] > threshold)
        for group, mask in conditions(dev).items():
            use = selected & mask
            rows.append(dict(model='expanded_A', population='development', threshold=threshold, group=group,
                n=int(use.sum()), weighted_n=int(use.sum()), eligible_weight=int((eligible & mask).sum()),
                mean_q=float(probability['qnear'][use].mean()) if use.any() else None,
                mean_width68=float(np.mean(dev['mu'][use, 3]-dev['mu'][use, 1])) if use.any() else None,
                mean_width95=float(np.mean(dev['mu'][use, 4]-dev['mu'][use, 0])) if use.any() else None,
                ambiguous_fraction=float(dev['ambiguous'][use].mean()) if use.any() else None,
                mean_unlabeled_contribution=float(probability['class_mass'][use, 3].mean()) if use.any() else None))
    csv_rows(DEST/'tables/application_summary.csv', rows)
    csv_rows(DEST/'tables/application_distribution.csv', bins)
    csv_rows(DEST/'tables/application_evidence_cells.csv', cellrows)
    save_json(DEST/'application_complete.json', dict(job_id=os.getenv('SLURM_JOB_ID'), frozen_inputs=freeze,
        n=26000, parent_weight=float(app['sample_weight'].sum()), max_catalog_mu_difference=error,
        input_representation_delta=delta, routes={k: int((app['route'] == k).sum()) for k in range(3)},
        interpretation='Exact original catalog photometry for both models. expanded_A development/app is same model; baseline_v1 is an application control. Counts are predicted membership, not true population counts. Cell occupancy is evidence availability, not reliability certification.'))
    assert all(sha(ROOT/k) == v for k, v in freeze.items())
    print('APPLICATION COMPLETE', flush=True)


if __name__ == '__main__':
    main()
