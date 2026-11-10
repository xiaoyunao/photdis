"""Selection diagnostics with explicit probability-endpoint conservation."""
import numpy as np


def canonical_probability(q, tolerance=1e-7):
    q = np.asarray(q, dtype=float)
    if not np.isfinite(q).all() or np.any((q < -tolerance) | (q > 1+tolerance)):
        raise ValueError('Nonfinite or materially out-of-range probability')
    return np.clip(q, 0., 1.)


def reliability_bins(q, event, bins=10):
    q = canonical_probability(q)
    event = np.asarray(event, bool)
    if q.shape != event.shape:
        raise ValueError('Probability/event shape mismatch')
    index = np.minimum((q*bins).astype(int), bins-1)
    rows = []
    for i in range(bins):
        use = index == i
        rows.append(dict(low=i/bins, high=(i+1)/bins, n=int(use.sum()),
                         mean_q=float(q[use].mean()) if use.any() else None,
                         observed_near=float(event[use].mean()) if use.any() else None))
    assert sum(r['n'] for r in rows) == len(q)
    return rows


def wilson(success, total, z=1.959963984540054):
    if not total:
        return None, None
    p = success/total
    denominator = 1+z*z/total
    middle = (p+z*z/(2*total))/denominator
    half = z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/denominator
    return float(middle-half), float(middle+half)


def evidence_cells(t):
    """Prespecified observed conditions, never labels, true distance or FeH."""
    r = np.digitize(t['r'], [18., 20., 22., 25.])
    gr = np.digitize(t['g0']-t['r0'], [-.2, .3, .8, 1.3, 2.])
    rz = np.digitize(t['r0']-t['z0'], [0., .3, .6, 1., 1.5])
    err = np.digitize(t['optical_error'], [.03, .1, .2])
    wise = t['B'].astype(int)+t['C'].astype(int)
    return ((((r*6+gr)*6+rz)*4+err)*3+wise).astype(int)
