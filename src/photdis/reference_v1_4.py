"""Versioned new-sky geometry and optional-band inputs; old interfaces untouched."""
from __future__ import annotations

import hashlib
import numpy as np

FEATURE_VERSION = "griz_v1"
FEATURE_BANDS = ("G", "R", "I", "Z")


def spatial_split(ra, dec, legacy_bounds, seed=20261104):
    """Whole 10-degree cells; any legacy overlap quarantines the entire cell.

    70/15/15 hash allocation is fixed without labels or residuals. Invalid
    positions are excluded (3). This is geometry, not equal-area sampling.
    """
    ra, dec = np.asarray(ra, float), np.asarray(dec, float)
    valid = np.isfinite(ra) & np.isfinite(dec) & (ra >= 0) & (ra < 360) & (dec >= -90) & (dec < 90)
    r = np.floor(np.where(valid, ra, 0) / 10).astype(int) * 10
    d = np.floor(np.where(valid, dec, 0) / 10).astype(int) * 10
    cell = (r // 10) * 18 + (d + 90) // 10
    split = np.full(len(ra), 3, dtype="i1")
    for c in np.unique(cell[valid]):
        ix = valid & (cell == c)
        r0, d0 = r[ix][0], d[ix][0]
        old = any(r0 < b[2] and r0 + 10 > b[0] and d0 < b[3] and d0 + 10 > b[1]
                  for b in legacy_bounds)
        if not old:
            u = int.from_bytes(hashlib.sha256(f"{seed}:{c}".encode()).digest()[:8], "big") / 2**64
            split[ix] = 0 if u < .70 else (1 if u < .85 else 2)
    return cell.astype("i2"), split


def unique_join(query, catalog):
    """Exact scalar or structured-key join; duplicated catalog keys are refused."""
    order = np.argsort(catalog, kind="stable")
    key = catalog[order]
    lo, hi = np.searchsorted(key, query, "left"), np.searchsorted(key, query, "right")
    out = np.full(len(query), -1, dtype="i8")
    ok = hi - lo == 1
    out[ok] = order[lo[ok]]
    return out


def pair_keys(target, source):
    out = np.empty(len(target), dtype=[("target", "i8"), ("source", "i8")])
    out["target"], out["source"] = target, source
    return out


def optional_photometry(raw):
    """Independent i quality and exhaustive WISE measurement states.

    WISE state: 0 no observation, 1 invalid measurement, 2 quality failure,
    3 low S/N (including zero/negative flux), 4 usable S/N>=5 color.
    Negative flux remains a valid measurement, never a Gaussian magnitude.
    """
    out = {}
    for band in ("I", "W1", "W2"):
        f, v, t = [np.asarray(raw[f"{p}_{band}"], float)
                   for p in ("FLUX", "FLUX_IVAR", "MW_TRANSMISSION")]
        observed = raw[f"NOBS_{band}"] > 0
        measured = observed & np.isfinite(f) & np.isfinite(v) & (v > 0)
        snr = np.full(len(f), np.nan)
        snr[measured] = f[measured] * np.sqrt(v[measured])
        trans = np.isfinite(t) & (t > 0)
        if band == "I":
            clean = ((raw["FRACMASKED_I"] < .4) & (raw["FRACIN_I"] > .7)
                     & (raw["FRACFLUX_I"] < .5)
                     & ((raw["MASKBITS"].astype("i8") & ((1 << 14) | (1 << 15) | (1 << 10))) == 0))
        else:
            clean = (raw[f"FRACFLUX_{band}"] < 1) & (raw[f"WISEMASK_{band}"] == 0)
        usable = measured & trans & clean & (snr >= 5)
        k = band.lower()
        out[k + "_usable"] = usable
        if band == "I":
            out["i0"] = np.full(len(f), np.nan, dtype="f4")
            out["sigma_i"] = np.full(len(f), np.nan, dtype="f4")
            out["i0"][usable] = 22.5 - 2.5 * np.log10(f[usable] / t[usable])
            out["sigma_i"][usable] = 2.5 / np.log(10) / snr[usable]
        state = np.zeros(len(f), "i1")
        state[observed] = 1
        state[measured & trans] = 2
        state[measured & trans & clean] = 3
        state[usable] = 4
        out[k + "_measurement_state"] = state
    return out


def optical_features(magnitudes, variant):
    """Input has explicit [g,r,i,z] order for BOTH matched controls."""
    m = np.asarray(magnitudes)
    if m.ndim != 2 or m.shape[1] != 4:
        raise ValueError("griz_v1 requires explicit g,r,i,z columns")
    if variant == "grz":
        return np.column_stack((m[:, 0] - m[:, 1], m[:, 1] - m[:, 3]))
    if variant == "griz":
        return m[:, :-1] - m[:, 1:]
    raise ValueError(variant)
