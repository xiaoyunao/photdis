#!/usr/bin/env python3
"""CMDs of the exact cached distance-comparison identities; no model refitting."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import fitsio
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

SOURCE = Path('data/distance_comparison_v1')
OUT = Path('data/distance_comparison_cmd_v1')
FIG = OUT / 'figures'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def matched_photometry(path, identities):
    """Read only rows matching the fixed prediction cache, in cache order."""
    assert len(np.unique(identities)) == len(identities)
    with fitsio.FITS(str(path)) as handle:
        ids = handle[1].read(columns=['source_id'])['source_id']
        index = np.flatnonzero(np.isin(ids, identities))
        rows = handle[1].read(columns=['source_id', 'g0', 'gr0'], rows=index)
    assert len(rows) == len(identities), 'Missing or ambiguous source identity'
    order = np.argsort(rows['source_id'])
    rows = rows[order[np.searchsorted(rows['source_id'][order], identities)]]
    assert np.array_equal(rows['source_id'], identities)
    return rows


def absolute_g(g0, distance_kpc):
    return np.asarray(g0) - 10 - 5 * np.log10(distance_kpc)


def plot_group(samples, title, filename):
    """Each column shares identities and axes between reference and prediction."""
    fig, axes = plt.subplots(2, len(samples), figsize=(5 * len(samples), 8),
                             squeeze=False, sharex=True, sharey=True,
                             layout='constrained')
    colors = np.concatenate([s['gr0'] for s in samples])
    mags = np.concatenate([s[k] for s in samples for k in ('mg_ref', 'mg_phot')])
    assert np.all(np.isfinite(colors)) and np.all(np.isfinite(mags))
    xpad = max(.03, float(np.ptp(colors)) * .06)
    ypad = max(.1, float(np.ptp(mags)) * .06)
    limits = (colors.min() - xpad, colors.max() + xpad,
              mags.min() - ypad, mags.max() + ypad)
    artists = []
    metrics = []
    for col, sample in enumerate(samples):
        n = len(sample['gr0'])
        metrics.append({'label': sample['label'], 'n': n,
                        'gr0_range': [float(sample['gr0'].min()), float(sample['gr0'].max())],
                        'reference_mg_range': [float(sample['mg_ref'].min()), float(sample['mg_ref'].max())]})
        for row, key in enumerate(('mg_ref', 'mg_phot')):
            ax = axes[row, col]
            h = ax.hexbin(sample['gr0'], sample[key], gridsize=38,
                          extent=limits, mincnt=1, cmap='viridis', linewidths=0)
            artists.append(h)
            origin = sample['reference'] if row == 0 else 'phot-d (model-derived)'
            ax.set_title(f"{sample['label']} | N={n:,}\n$M_g$ from {origin}", fontsize=10)
            ax.set_xlim(limits[:2]); ax.set_ylim(limits[3], limits[2])
            ax.grid(alpha=.12)
            if col == 0: ax.set_ylabel(r'$M_g$ [mag]')
            if row == 1: ax.set_xlabel(r'$(g-r)_0$ [mag]')
    norm = LogNorm(vmin=1, vmax=max(2, max(float(h.get_array().max()) for h in artists)))
    for h in artists: h.set_norm(norm)
    fig.colorbar(artists[0], ax=axes.ravel().tolist(), label='Stars / hexagon (log scale)', shrink=.8, pad=.02)
    fig.suptitle(title + '\nSame sources as distance panels; no CMD-based reselection; brighter stars at top', fontsize=12)
    fig.savefig(FIG / filename, dpi=170)
    plt.close(fig)
    return metrics


def main():
    from photdis.stage1 import require_compute
    require_compute()
    if (OUT / 'summary.json').exists(): raise FileExistsError(OUT / 'summary.json')
    FIG.mkdir(parents=True, exist_ok=True)
    original = json.loads((SOURCE / 'summary.json').read_text())
    registrations = ['configs/cold_v4_native_transfer_fields.json',
                     'configs/cold_v4_native_transfer_north_final.json']
    fields = sum([json.loads(Path(p).read_text())['independent_final_fields'] for p in registrations], [])
    inputs = {str(SOURCE / 'summary.json'): sha(SOURCE / 'summary.json')}
    cold = []
    for j, item in enumerate(fields):
        assert item['name'] == original['cold'][j]['field']
        cache = SOURCE / f'cold_{j}.npz'
        task = json.loads(Path(item['field_config']).read_text())
        path = Path(task['output_dir']) / 'field_diagnostics.fits'
        assert sha(path) == original['cold'][j]['source_sha256']
        with np.load(cache) as data:
            phot = matched_photometry(path, data['source_id'])
            assert len(phot) == original['cold'][j]['n']
            cold.append({'label': item['name'].split('_dec')[0].replace('native_', ''),
                         'reference': 'Gaia-d', 'gr0': phot['gr0'],
                         'mg_ref': absolute_g(phot['g0'], data['gaia_kpc']),
                         'mg_phot': absolute_g(phot['g0'], data['phot_kpc'])})
        inputs[str(cache)] = sha(cache); inputs[str(path)] = sha(path)
    result = {'cold': plot_group(cold, 'Cold-v4 strict Gaia-anchor CMDs', '01_cold_cmd.png')}
    path = Path('data/giant_vac_v3/same_star.fits')
    assert sha(path) == original['warm']['source_sha256']
    with np.load(SOURCE / 'warm.npz') as data:
        phot = matched_photometry(path, data['source_id'])
        for supported, filename in [(True, '02_warm_supported_cmd.png'), (False, '03_warm_all_cmd.png')]:
            samples = []
            for key, label in [('spdist', 'SPDist'), ('specdis', 'SPECDIS')]:
                take = np.ones(len(phot), bool) if key == 'spdist' else data['specdis_valid'].copy()
                if supported: take &= data['supported']
                expected = original['warm'][key if supported else key + '_all']['n']
                assert int(take.sum()) == expected
                assert np.all(np.isfinite(data[key + '_kpc'][take]) & (data[key + '_kpc'][take] > 0))
                samples.append({'label': label + (' | supported' if supported else ' | all reserved'),
                                'reference': label + '-d', 'gr0': phot['gr0'][take],
                                'mg_ref': absolute_g(phot['g0'][take], data[key + '_kpc'][take]),
                                'mg_phot': absolute_g(phot['g0'][take], data['phot_kpc'][take])})
            result['warm_supported' if supported else 'warm_all'] = plot_group(
                samples, 'Warm-giant CMDs (5000–7000 K) | spectral distances are teacher references', filename)
    inputs[str(path)] = sha(path); inputs[str(SOURCE / 'warm.npz')] = sha(SOURCE / 'warm.npz')
    result.update(input_sha256=inputs, script_sha256=sha(__file__),
                  selection='Exact cached distance-panel identities and masks; no new cuts.',
                  magnitude='Mg = g0 - 10 - 5 log10(distance/kpc)',
                  interpretation='Phot-d CMDs are model-derived, not independent phase confirmation.',
                  figure_sha256={str(p): sha(p) for p in sorted(FIG.glob('*.png'))})
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key in ('cold', 'warm_supported', 'warm_all')}))


if __name__ == '__main__':
    main()
