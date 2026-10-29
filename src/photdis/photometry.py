"""Measured Legacy photometry and explicit quality tiers for the sample audit."""

from __future__ import annotations

import numpy as np


BANDS = ("G", "R", "Z", "W1", "W2")
MAG_ERROR_FACTOR = 2.5 / np.log(10.0)
# DR11 SATUR_G/R/Z, ALLMASK_G/R/Z, and BAILOUT. These affect the clean
# subset, while the measured parent retains them for coverage diagnostics.
CLEAN_REJECT_MASKBITS = (2, 3, 4, 5, 6, 7, 10)


def derive_photometry(
    data,
    *,
    min_snr=5.0,
    nobs_min=1,
    fracmasked_max=0.4,
    fracin_min=0.7,
    fracflux_max=0.5,
    wise_fracflux_max=1.0,
    w1_snr_min=5.0,
    w2_snr_min=3.0,
    optical_sigma_max=0.03,
    w1_sigma_max=0.05,
    w2_sigma_max=0.10,
    clean_reject_maskbits=CLEAN_REJECT_MASKBITS,
    require_wisemask_zero=True,
):
    """Return same-length columns and flags; never filter or mutate input rows.

    ``grz_base`` uses only positive finite measured flux/inverse variance,
    positive finite MW transmission, observations and optical S/N. Morphology,
    sky selection and Gaia quality belong to the caller. ``grz_clean`` adds
    explicit image/deblend cuts; ``grz_precise`` applies the optical error cut
    without those clean cuts so their effect can be measured separately.

    Optional-IR flags use WISE masks and the configurable contamination limit.
    Missing/negative W2 measurements cannot remove a row from ``grz_base`` or
    ``grz_w1``. Raw fluxes, masks and fit statistics remain caller-owned.
    Magnitude errors are first-order flux-error propagation, without an added
    calibration floor or dust uncertainty. Colors share band errors; the caller
    must preserve this covariance when using them in later inference.
    """
    names = set(data.dtype.names or ())
    required = {"MASKBITS"}
    for band in BANDS:
        required.update(f"{prefix}_{band}" for prefix in
                        ("FLUX", "FLUX_IVAR", "MW_TRANSMISSION", "NOBS"))
    for band in ("G", "R", "Z"):
        required.update(f"{prefix}_{band}" for prefix in
                        ("FRACMASKED", "FRACIN", "FRACFLUX"))
    for band in ("W1", "W2"):
        required.add(f"FRACFLUX_{band}")
        if require_wisemask_zero:
            required.add(f"WISEMASK_{band}")
    missing = sorted(required - names)
    if missing:
        raise ValueError("Missing Legacy photometry fields: " + ", ".join(missing))

    size = len(data)
    out = {}
    for band in BANDS:
        key = band.lower()
        flux = np.asarray(data[f"FLUX_{band}"], dtype=np.float64)
        ivar = np.asarray(data[f"FLUX_IVAR_{band}"], dtype=np.float64)
        trans = np.asarray(data[f"MW_TRANSMISSION_{band}"], dtype=np.float64)
        flux_ok = np.isfinite(flux) & (flux > 0)
        ivar_ok = np.isfinite(ivar) & (ivar > 0)
        trans_ok = np.isfinite(trans) & (trans > 0)
        valid = (flux_ok & ivar_ok & trans_ok
                 & (np.asarray(data[f"NOBS_{band}"]) >= nobs_min))

        snr = np.full(size, np.nan, dtype=np.float32)
        snr_ok = np.isfinite(flux) & ivar_ok
        snr[snr_ok] = flux[snr_ok] * np.sqrt(ivar[snr_ok])
        valid &= np.isfinite(snr)

        mag = np.full(size, np.nan, dtype=np.float32)
        mag0 = np.full(size, np.nan, dtype=np.float32)
        mag[flux_ok] = 22.5 - 2.5 * np.log10(flux[flux_ok])
        mag_ok = flux_ok & trans_ok
        mag0[mag_ok] = (22.5 - 2.5 * np.log10(flux[mag_ok])
                       + 2.5 * np.log10(trans[mag_ok]))

        sigma = np.full(size, np.nan, dtype=np.float32)
        err_ok = flux_ok & ivar_ok & np.isfinite(snr) & (snr > 0)
        sigma[err_ok] = MAG_ERROR_FACTOR / snr[err_ok]
        out[key] = mag
        out[f"{key}0"] = mag0
        out[f"sigma_{key}"] = sigma
        out[f"snr_{key}"] = snr
        out[f"valid_{key}"] = valid

    base = np.ones(size, dtype=bool)
    fraction_clean = np.ones(size, dtype=bool)
    gr_base = np.ones(size, dtype=bool)
    gr_fraction_clean = np.ones(size, dtype=bool)
    precise = np.ones(size, dtype=bool)
    for band in ("G", "R", "Z"):
        key = band.lower()
        base &= out[f"valid_{key}"] & (out[f"snr_{key}"] >= min_snr)
        fm = np.asarray(data[f"FRACMASKED_{band}"])
        fi = np.asarray(data[f"FRACIN_{band}"])
        ff = np.asarray(data[f"FRACFLUX_{band}"])
        fraction_clean &= (np.isfinite(fm) & (fm < fracmasked_max)
                           & np.isfinite(fi) & (fi > fracin_min)
                           & np.isfinite(ff) & (ff < fracflux_max))
        if band in ("G", "R"):
            gr_base &= out[f"valid_{key}"] & (out[f"snr_{key}"] >= min_snr)
            gr_fraction_clean &= (np.isfinite(fm) & (fm < fracmasked_max)
                                  & np.isfinite(fi) & (fi > fracin_min)
                                  & np.isfinite(ff) & (ff < fracflux_max))
        precise &= out[f"sigma_{key}"] < optical_sigma_max

    reject = sum(1 << int(bit) for bit in set(clean_reject_maskbits))
    mask_clean = (np.asarray(data["MASKBITS"], dtype=np.int64) & reject) == 0
    gr_reject = sum(1 << int(bit) for bit in set(clean_reject_maskbits)
                    if bit not in (4, 7))
    gr_mask_clean = (np.asarray(data["MASKBITS"], dtype=np.int64) & gr_reject) == 0
    out["gr_base"] = gr_base
    out["gr_fraction_clean"] = gr_fraction_clean
    out["gr_mask_clean"] = gr_mask_clean
    out["gr_clean"] = gr_base & gr_fraction_clean & gr_mask_clean
    out["grz_base"] = base
    out["grz_fraction_clean"] = fraction_clean
    out["grz_mask_clean"] = mask_clean
    out["grz_clean"] = base & fraction_clean & mask_clean
    out["grz_precise"] = base & precise
    out["grz_high_quality"] = out["grz_clean"] & precise

    for band, snr_min in (("W1", w1_snr_min), ("W2", w2_snr_min)):
        key = band.lower()
        ff = np.asarray(data[f"FRACFLUX_{band}"])
        clean = (out[f"valid_{key}"] & (out[f"snr_{key}"] >= snr_min)
                 & np.isfinite(ff) & (ff < wise_fracflux_max))
        if require_wisemask_zero:
            clean &= np.asarray(data[f"WISEMASK_{band}"]) == 0
        out[f"{key}_clean"] = clean

    out["grz_w1"] = base & out["w1_clean"]
    out["grz_w1_w2"] = out["grz_w1"] & out["w2_clean"]
    out["grz_w1_high_quality"] = (out["grz_high_quality"] & out["w1_clean"]
                                  & (out["sigma_w1"] < w1_sigma_max))
    out["full_high_quality"] = (out["grz_w1_high_quality"] & out["w2_clean"]
                                & (out["sigma_w2"] < w2_sigma_max))
    tier = np.full(size, -1, dtype=np.int8)
    tier[base] = 0
    tier[out["grz_w1"]] = 1
    tier[out["grz_w1_w2"]] = 2
    out["photometry_tier"] = tier
    for color, first, second in (("gr0", "g0", "r0"), ("rz0", "r0", "z0"),
                                  ("zw10", "z0", "w10"),
                                  ("w1w20", "w10", "w20")):
        out[color] = out[first] - out[second]
    return out
