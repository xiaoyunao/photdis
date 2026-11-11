"""Shared target-to-reference photometric/dust sensitivity draws.

A zero point common to both reference SEDs and targets is a change of units and
cancels. The declared offsets here are TARGET-RELATIVE to the fitted reference
system. They must not be interpreted as independently measured absolute errors.
"""
import numpy as np


def calibration_vectors(table, draws, draw_id):
    matches = np.flatnonzero(draws['draw_id'] == draw_id)
    if len(matches) != 1: raise ValueError('Unique shared draw ID required')
    j = matches[0]; field = np.asarray(table['field']); cell = np.asarray(table['cell'])
    dust = np.empty(len(field))
    known = {int(f): float(d) for f, d in zip(draws['field_ids'], draws['field_dust_offset'][j])}
    # Existing named fields use the original stored shared draw. Development and
    # newly frozen fields use deterministic sky-cell draws, independent of chunk
    # boundaries, processing order, object IDs, and observed reference residuals.
    units = np.where(field >= 0, field, 100000+cell)
    for unit in np.unique(units):
        if int(unit) in known: value = known[int(unit)]
        else: value = np.random.default_rng(np.random.SeedSequence([302811, int(draw_id), int(unit)])).normal(0., .003)
        dust[units == unit] = value
    trans = np.asarray(table['transmission']); ebv = np.asarray(table['EBV'])
    valid = np.isfinite(trans) & (trans > 0) & (ebv[:, None] > 0)
    coefficient = np.zeros_like(trans)
    with np.errstate(divide='ignore', invalid='ignore'):
        coefficient[valid] = (-2.5*np.log10(trans)/ebv[:, None])[valid]
    if ((ebv <= 0) & np.asarray(table['band_mask']).any(axis=1)).any():
        raise ValueError('Explicit native extinction coefficients required at zero EBV')
    calibration = draws['global_band_zero_point_mag'][j][None]+coefficient*dust[:, None]
    return calibration, dust
