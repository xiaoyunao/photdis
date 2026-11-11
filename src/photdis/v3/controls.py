"""Fixed same-identity controls and observable-only information probes."""
import numpy as np
from scipy.special import softmax
from scipy.optimize import minimize_scalar


def optical_magnitudes(table):
    indices = [0, 1, 3]
    f = np.asarray(table['flux'])[:, indices]
    e = np.asarray(table['flux_error'])[:, indices]
    transmission = np.asarray(table['transmission'])[:, indices]
    if not (np.asarray(table['band_mask'])[:, indices].all() and (f > 0).all()):
        raise ValueError('Native clean grz observations required')
    return 22.5-2.5*np.log10(f/transmission), 2.5/np.log(10)*e/f


def probe_features(table, view):
    """No target class, reference M, distance, FeH or Gaia enters these features.

    All views share apparent r, sky, dust, measured-error and availability terms.
    Optional signed fluxes are represented relative to positive observed r flux;
    absent/contaminated values remain NaN, handled natively by the fixed learner.
    """
    mag, _ = optical_magnitudes(table)
    # Normalize instrumental flux errors by r only: g/z SNR would itself encode colors.
    errors = np.asarray(table['flux_error'])[:, [0, 1, 3]]/np.asarray(table['flux'])[:, 1, None]
    ra = np.deg2rad(table['RA']); dec = np.deg2rad(table['DEC'])
    mask = np.asarray(table['band_mask'])
    base = np.c_[mag[:, 1], np.sin(ra), np.cos(ra), np.sin(dec), table['EBV'],
                 np.log(np.maximum(errors, 1e-9)), mask.astype(float)]
    if view == 'no_color': return base
    color = np.c_[mag[:, 0]-mag[:, 1], mag[:, 1]-mag[:, 2]]
    if view == 'grz': return np.c_[base, color]
    indices = [2] if view == 'griz' else [2, 4, 5] if view == 'multiband' else None
    if indices is None: raise ValueError('Unknown fixed probe view')
    flux = np.asarray(table['flux'])/np.asarray(table['transmission'])
    optional = flux[:, indices]/flux[:, 1, None]
    optional = np.where(mask[:, indices], optional, np.nan)
    return np.c_[base, color, optional]


def calibrate_temperature(probability, labels):
    probability = np.asarray(probability)
    logp = np.log(np.maximum(probability, 1e-12)); labels = np.asarray(labels, int)
    def objective(log_temperature):
        p = softmax(logp/np.exp(log_temperature), axis=1)
        return -np.log(np.maximum(p[np.arange(len(p)), labels], 1e-300)).mean()
    result = minimize_scalar(objective, bounds=(np.log(.25), np.log(4.)), method='bounded')
    if not result.success: raise RuntimeError('Temperature fit failed')
    return float(np.exp(result.x))


def apply_temperature(probability, temperature):
    if temperature <= 0: raise ValueError('Positive temperature required')
    return softmax(np.log(np.maximum(probability, 1e-12))/temperature, axis=1)


def probability_metrics(probability, labels):
    p = np.asarray(probability); labels = np.asarray(labels, int)
    if not len(p): return dict(n=0)
    if not np.isin(labels, [0, 1, 2]).all(): raise ValueError('Missing labels are not physical classes')
    if (p < 0).any() or not np.allclose(p.sum(axis=1), 1): raise ValueError('Normalized class probabilities required')
    onehot = np.eye(3)[labels]
    return dict(n=len(p), log_score=float(np.log(np.maximum(p[np.arange(len(p)), labels], 1e-300)).mean()),
                brier=float(((p-onehot)**2).sum(axis=1).mean()),
                accuracy=float((p.argmax(axis=1) == labels).mean()))


def reliability_rows(probability, labels, bins=10):
    p = np.asarray(probability); labels = np.asarray(labels)
    rows = []
    for c in range(3):
        group = np.minimum((np.clip(p[:, c], 0, 1)*bins).astype(int), bins-1)
        for b in range(bins):
            m = group == b
            rows.append(dict(class_index=c, bin=b, n=int(m.sum()),
                probability_mean=float(p[m, c].mean()) if m.any() else None,
                observed_fraction=float((labels[m] == c).mean()) if m.any() else None))
        assert sum(r['n'] for r in rows if r['class_index'] == c) == len(p)
    return rows
