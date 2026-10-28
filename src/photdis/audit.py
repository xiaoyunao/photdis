"""Photometric audit quantities for the fixed ten-degree pilot."""
from __future__ import annotations
import numpy as np

GROUP_NAMES = ("strict optical pass", "SATUR only", "FRACMASKED only", "SATUR + FRACMASKED",
               "ALLMASK / BAILOUT", "no valid measured grz", "other optical cut")
GROUP_COLORS = ("#276a9b", "#b77b26", "#6b9644", "#c5513e", "#865c9b", "#555555", "#888888")
PS1_COEFFICIENTS = {
    "g": (0.00062, 0.03604, 0.01028, -0.00613),
    "r": (0.00495, -0.08435, 0.03222, -0.01140),
    "z": (0.02583, -0.07690, 0.02824, -0.00898),
}


def checked_gaia_color(data):
    """Recompute BP-RP using observed flux/NOBS validity, retaining real zero colors."""
    good = np.asarray(data["gaia_match"], bool).copy()
    sigma2 = np.zeros(len(good), dtype=float)
    for b in ("bp", "rp"):
        flux = np.asarray(data[f"phot_{b}_mean_flux"], float)
        error = np.asarray(data[f"phot_{b}_mean_flux_error"], float)
        mag = np.asarray(data[f"phot_{b}_mean_mag"], float)
        good &= (np.isfinite(flux) & (flux > 0) & np.isfinite(error) & (error > 0)
                 & (data[f"phot_{b}_n_obs"] > 0) & np.isfinite(mag))
        sigma2 += np.divide(error, flux, out=np.full(len(good), np.inf), where=flux > 0)**2
    color = np.full(len(good), np.nan)
    color[good] = data["phot_bp_mean_mag"][good].astype(float)-data["phot_rp_mean_mag"][good].astype(float)
    error = 2.5/np.log(10)*np.sqrt(sigma2)
    return {"bp_rp_valid": good, "bp_rp_checked": color, "bp_rp_error": error,
            "bp_rp_precise": good & (error < .1),
            "bp_rp_catalog_disagrees": good & (np.abs(color-data["bp_rp"]) > 1e-3)}


def rvs_metallicity(feh, teff, logg):
    """DESI DR1 author correction (feh_correct.py, RVS); lower bounds inclusive."""
    feh, teff, logg = np.broadcast_arrays(np.asarray(feh, float), np.asarray(teff, float), np.asarray(logg, float))
    valid = np.isfinite(feh) & (teff >= 4200) & (teff < 6600) & (logg >= 1.8) & (logg < 5.5)
    result = np.full(feh.shape, np.nan)
    for subset, coeff in ((valid & (logg < 4.3), (.197, -.809, .441)),
                          (valid & (logg >= 4.3), (.039, -.284, .079))):
        x = np.log10(teff[subset]/5000)/.1
        result[subset] = feh[subset] - np.polynomial.polynomial.polyval(x, coeff)
    return result, valid


def ps1_decam(median, band):
    gi = median[:,0]-median[:,2]
    return median[:,{"g":0,"r":1,"z":3}[band]] + np.polynomial.polynomial.polyval(gi, PS1_COEFFICIENTS[band])


def failure_groups(raw, phot):
    """Partition records; ALLMASK and missing measurements take precedence."""
    size = len(raw)
    measured = np.ones(size, bool)
    fmfail = np.zeros(size, bool)
    infail = np.zeros(size, bool)
    fffail = np.zeros(size, bool)
    for b in "GRZ":
        for p in ("FLUX", "FLUX_IVAR", "MW_TRANSMISSION"):
            value = raw[f"{p}_{b}"]
            measured &= np.isfinite(value) & (value > 0)
        measured &= raw[f"NOBS_{b}"] >= 1
        fmfail |= ~np.isfinite(raw[f"FRACMASKED_{b}"]) | (raw[f"FRACMASKED_{b}"] >= .4)
        infail |= ~np.isfinite(raw[f"FRACIN_{b}"]) | (raw[f"FRACIN_{b}"] <= .7)
        fffail |= ~np.isfinite(raw[f"FRACFLUX_{b}"]) | (raw[f"FRACFLUX_{b}"] >= .5)
    mask = raw["MASKBITS"].astype("i8")
    satur = (mask & sum(1<<b for b in (2,3,4))) != 0
    allmask = (mask & sum(1<<b for b in (5,6,7,10))) != 0
    other = infail | fffail | ~phot["grz_base"] | ~phot["grz_precise"]
    group = np.zeros(size, dtype="i2")
    group[satur & ~fmfail] = 1
    group[~satur & fmfail] = 2
    group[satur & fmfail] = 3
    group[other] = 6
    group[allmask] = 4
    group[~measured] = 5
    if not np.array_equal(group == 0, phot["grz_high_quality"]):
        raise ValueError("Failure groups do not reproduce strict optical selection")
    return {"group": group, "satur": satur, "fracmasked_fail": fmfail,
            "allmask_bailout": allmask, "measured": measured, "other": other}


def stats(value):
    value = np.asarray(value)
    value = value[np.isfinite(value)]
    if not len(value):
        return {"n": 0}
    med = np.median(value)
    return {"n": len(value), "median": float(med), "nmad": float(1.4826*np.median(np.abs(value-med))),
            "p16": float(np.percentile(value,16)), "p84": float(np.percentile(value,84)),
            "abs_gt_0p1": int((np.abs(value) > .1).sum()), "abs_gt_0p3": int((np.abs(value) > .3).sum())}
