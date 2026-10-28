"""Exact Gaia DR3 identifier joins and signed-parallax quality flags."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np


SOURCE_ID_HEALPIX_FACTOR = 2**49
NPIX = 12 * 32**2
FLOAT64_FIELDS = {"RA", "DEC", "ECL_LAT"}
INTEGER_FIELDS = {
    "SOURCE_ID": np.int64,
    "ASTROMETRIC_PARAMS_SOLVED": np.int16,
    "VISIBILITY_PERIODS_USED": np.int16,
    "NON_SINGLE_STAR": np.int16,
}
BOOLEAN_FIELDS = {"DUPLICATED_SOURCE", "IN_QSO_CANDIDATES", "IN_GALAXY_CANDIDATES"}
GAIA_COLUMNS = (
    "SOURCE_ID", "RA", "DEC", "REF_EPOCH", "PARALLAX", "PARALLAX_ERROR",
    "PMRA", "PMRA_ERROR", "PMDEC", "PMDEC_ERROR", "RUWE",
    "VISIBILITY_PERIODS_USED", "ASTROMETRIC_PARAMS_SOLVED",
    "PHOT_G_MEAN_MAG", "PHOT_BP_MEAN_MAG", "PHOT_RP_MEAN_MAG", "BP_RP",
    "NU_EFF_USED_IN_ASTROMETRY", "PSEUDOCOLOUR", "ECL_LAT",
    "DUPLICATED_SOURCE", "IN_QSO_CANDIDATES", "IN_GALAXY_CANDIDATES",
    "NON_SINGLE_STAR", "ASTROMETRIC_EXCESS_NOISE", "IPD_FRAC_MULTI_PEAK",
    "CLASSPROB_DSC_COMBMOD_QUASAR", "CLASSPROB_DSC_COMBMOD_GALAXY",
    "CLASSPROB_DSC_COMBMOD_STAR",
)


def _decode_boolean(values, field):
    """Decode both FITS logical and Gaia's string-valued candidate flags."""
    values = np.asarray(values)
    if values.dtype.kind == "b":
        return values
    if values.dtype.kind in "SU":
        normalized = np.char.lower(np.char.strip(values.astype("U")))
        yes = np.isin(normalized, ("true", "t", "1"))
        no = np.isin(normalized, ("false", "f", "0"))
    elif values.dtype.kind in "iu":
        yes, no = values == 1, values == 0
    else:
        raise ValueError(f"Unsupported Gaia boolean dtype for {field}: {values.dtype}")
    if not np.all(yes | no):
        unknown = np.unique(values[~(yes | no)])[:5]
        raise ValueError(f"Unknown Gaia boolean tokens for {field}: {unknown!r}")
    return yes


def _empty_columns(size):
    result = {}
    for field in GAIA_COLUMNS:
        if field in INTEGER_FIELDS:
            result[field.lower()] = np.full(size, -1, dtype=INTEGER_FIELDS[field])
        elif field in BOOLEAN_FIELDS:
            result[field.lower()] = np.zeros(size, dtype=bool)
        else:
            dtype = np.float64 if field in FLOAT64_FIELDS else np.float32
            result[field.lower()] = np.full(size, np.nan, dtype=dtype)
    result["gaia_match"] = np.zeros(size, dtype=bool)
    result["gaia_file_missing"] = np.zeros(size, dtype=bool)
    return result


@lru_cache(maxsize=4)
def _read_healpix(root, pixel):
    """Cache four projected, ID-sorted tiles per process; never load full schemas."""
    import fitsio

    path = Path(root) / f"healpix-{pixel:05d}.fits"
    if not path.exists():
        return None
    with fitsio.FITS(str(path)) as hdus:
        hdu = hdus[1]
        missing = sorted(set(GAIA_COLUMNS) - set(hdu.get_colnames()))
        if missing:
            raise ValueError(f"Gaia schema mismatch in {path}: {missing}")
        header = hdu.read_header()
        if header.get("HPXNSIDE") != 32 or not header.get("HPXNEST"):
            raise ValueError(f"Unexpected Gaia HEALPix convention in {path}")
        table = hdu.read(columns=list(GAIA_COLUMNS))
    order = np.argsort(table["SOURCE_ID"], kind="stable")
    table = table[order]
    if len(table) > 1 and np.any(table["SOURCE_ID"][1:] == table["SOURCE_ID"][:-1]):
        raise ValueError(f"Duplicate Gaia SOURCE_ID in {path}")
    table.flags.writeable = False
    return table


def join_gaia(source_ids, root):
    """Return same-length lowercase Gaia fields joined by exact int64 source ID.

    The caller supplies DR3 IDs (Legacy DR11 REF_CAT=G3), using zero for unknown
    identities. NSIDE=32 nested tile numbers are SOURCE_ID // 2**49. Input order
    and repeated IDs are preserved. Missing matches have NaN floats, -1 integers,
    false boolean fields and gaia_match=False. Missing files have an additional
    flag; schema/corruption errors raise rather than masquerade as absent matches.
    """
    source_ids = np.asarray(source_ids)
    if source_ids.ndim != 1 or source_ids.dtype.kind not in "iu":
        raise ValueError("Gaia SOURCE_ID must be a one-dimensional integer array")
    if source_ids.dtype.kind == "u" and np.any(source_ids > np.iinfo(np.int64).max):
        raise ValueError("Gaia SOURCE_ID exceeds signed int64 range")
    source_ids = source_ids.astype(np.int64, copy=False)
    root = str(Path(root).resolve())
    if not Path(root).is_dir():
        raise FileNotFoundError(f"Gaia HEALPix directory does not exist: {root}")
    result = _empty_columns(len(source_ids))
    valid = (source_ids > 0) & (source_ids < NPIX * SOURCE_ID_HEALPIX_FACTOR)
    rows = np.flatnonzero(valid)
    if not len(rows):
        return result
    pixels = source_ids[rows] // SOURCE_ID_HEALPIX_FACTOR
    order = np.argsort(pixels, kind="stable")
    rows, pixels = rows[order], pixels[order]
    boundaries = np.r_[0, np.flatnonzero(pixels[1:] != pixels[:-1]) + 1, len(rows)]
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        target_rows = rows[start:stop]
        table = _read_healpix(root, int(pixels[start]))
        if table is None:
            result["gaia_file_missing"][target_rows] = True
            continue
        if not len(table):
            continue
        position = np.searchsorted(table["SOURCE_ID"], source_ids[target_rows])
        inside = position < len(table)
        inside[inside] &= (table["SOURCE_ID"][position[inside]]
                           == source_ids[target_rows[inside]])
        destination = target_rows[inside]
        position = position[inside]
        result["gaia_match"][destination] = True
        for field in GAIA_COLUMNS:
            values = table[field][position]
            if field in BOOLEAN_FIELDS:
                values = _decode_boolean(values, field)
            result[field.lower()][destination] = values
    return result


@lru_cache(maxsize=1)
def _get_zero_point():
    try:
        from zero_point import zpt
    except ImportError as exc:
        raise RuntimeError("Install gaiadr3-zeropoint for Gaia label preparation") from exc
    zpt.load_tables()
    return zpt


def derive_gaia_labels(gaia, *, ruwe_max=1.4, visibility_min=9, anchor_snr_min=20.0):
    """Derive flags without dropping weak/negative parallaxes or changing inputs.

    finite_parallax includes a finite positive uncertainty. Zero points are
    evaluated only inside the stated calibration ranges: 6<G<21, five-parameter
    nu_eff in [1.1,1.9] or six-parameter pseudocolour in [1.24,1.72]. Thus an
    out-of-domain object retains raw parallax and quality information but has
    no extrapolated corrected-parallax label. The strict astrometric flag does
    not itself impose a parallax sign, S/N threshold or contaminant exclusion.
    """
    match = np.asarray(gaia["gaia_match"], dtype=bool)
    parallax = np.asarray(gaia["parallax"], dtype=np.float64)
    error = np.asarray(gaia["parallax_error"], dtype=np.float64)
    ruwe = np.asarray(gaia["ruwe"], dtype=np.float64)
    visibility = np.asarray(gaia["visibility_periods_used"])
    solved = np.asarray(gaia["astrometric_params_solved"], dtype=np.int16)
    gmag = np.asarray(gaia["phot_g_mean_mag"], dtype=np.float64)
    nu_eff = np.asarray(gaia["nu_eff_used_in_astrometry"], dtype=np.float64)
    pseudocolour = np.asarray(gaia["pseudocolour"], dtype=np.float64)
    ecl_lat = np.asarray(gaia["ecl_lat"], dtype=np.float64)
    duplicate = np.asarray(gaia["duplicated_source"], dtype=bool)

    finite = match & np.isfinite(parallax) & np.isfinite(error) & (error > 0)
    solution = match & ((solved == 31) | (solved == 95))
    before_duplicate = (finite & solution & np.isfinite(ruwe) & (ruwe > 0)
                        & (ruwe < ruwe_max) & (visibility >= visibility_min))
    quality = before_duplicate & ~duplicate
    color_range = (((solved == 31) & np.isfinite(nu_eff)
                    & (nu_eff >= 1.1) & (nu_eff <= 1.9))
                   | ((solved == 95) & np.isfinite(pseudocolour)
                      & (pseudocolour >= 1.24) & (pseudocolour <= 1.72)))
    in_range = (solution & np.isfinite(gmag) & (gmag > 6) & (gmag < 21)
                & np.isfinite(ecl_lat) & (np.abs(ecl_lat) <= 90) & color_range)
    zpt_mas = np.full(len(match), np.nan, dtype=np.float32)
    if np.any(in_range):
        zpt = _get_zero_point()
        zpt_mas[in_range] = zpt.get_zpt(
            gmag[in_range], nu_eff[in_range], pseudocolour[in_range],
            ecl_lat[in_range], solved[in_range], _warnings=False,
        )
    zpt_valid = in_range & np.isfinite(zpt_mas)
    corrected = np.full(len(match), np.nan, dtype=np.float32)
    correction_ok = zpt_valid & np.isfinite(parallax)
    corrected[correction_ok] = parallax[correction_ok] - zpt_mas[correction_ok]
    raw_snr = np.full(len(match), np.nan, dtype=np.float32)
    corrected_snr = np.full(len(match), np.nan, dtype=np.float32)
    raw_snr[finite] = parallax[finite] / error[finite]
    snr_ok = finite & zpt_valid
    corrected_snr[snr_ok] = corrected[snr_ok] / error[snr_ok]
    contaminant = match & (np.asarray(gaia["in_qso_candidates"], dtype=bool)
                           | np.asarray(gaia["in_galaxy_candidates"], dtype=bool))
    return {
        "gaia_match": match.copy(),
        "finite_parallax": finite,
        "astrometric_solution_valid": solution,
        "astro_quality_before_duplicate": before_duplicate,
        "astro_quality": quality,
        "zpt_inrange": in_range,
        "zpt_valid": zpt_valid,
        "zpt_mas": zpt_mas,
        "p_corr": corrected,
        "snr_raw": raw_snr,
        "snr_corr": corrected_snr,
        "negative_raw_parallax": match & np.isfinite(parallax) & (parallax < 0),
        "negative_corrected_parallax": correction_ok & (corrected < 0),
        "anchor_astrometry": quality & zpt_valid & (corrected_snr >= anchor_snr_min),
        "contaminant_candidate": contaminant,
    }
