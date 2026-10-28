"""Project stellar spectroscopy for coverage and source-ID joins, never training.

``build_spectroscopy(config, output_dir)`` must run on allocated compute. It
returns ``{'desi': FITS_path, 'apogee': FITS_path, 'audit': {...}}``. Input paths
are ``config['desi_path']`` and ``config['apogee_path']``. The FITS pools include
rejected measurements and quality flags, so quality cuts remain auditable.

DESI quality follows Koposov et al. (2026), arXiv:2505.14787, section 4:
PRIMARY, RR_SPECTYPE=STAR, RVS_WARN=0, SN_R>=10, VSINI<30 km/s. Finite
parameters and positive uncertainties are additionally required here.
APOGEE bit definitions: https://www.sdss4.org/dr17/irspec/apogee-bitmasks/.
The selected bad bits below are explicit project choices, not STARFLAG==0.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


DESI_COLUMNS = [
    "TARGETID", "PRIMARY", "RR_SPECTYPE", "RVS_WARN", "SN_R", "VSINI",
    "TEFF", "TEFF_ERR", "LOGG", "LOGG_ERR", "FEH", "FEH_ERR", "SUCCESS",
    "SURVEY", "PROGRAM",
]
FIBERMAP_COLUMNS = [
    "TARGETID", "TARGET_RA", "TARGET_DEC", "RELEASE", "MORPHTYPE",
    "COADD_FIBERSTATUS",
]
APOGEE_COLUMNS = [
    "APOGEE_ID", "GAIAEDR3_SOURCE_ID", "RA", "DEC", "SNR", "TEFF",
    "TEFF_ERR", "LOGG", "LOGG_ERR", "FE_H", "FE_H_ERR", "FE_H_FLAG",
    "ASPCAPFLAG", "STARFLAG", "EXTRATARG",
]

# STAR_BAD summarizes bad atmospheric parameters, chi-square, rotation, S/N
# and grid edges. Explicit missing-result/unsuitable-target bits add protection.
APOGEE_ASPCAP_BAD_BITS = (23, 24, 31, 32, 33, 36, 40, 41)
# STARFLAG is an OR across visits; warning bits alone do not invalidate a star.
APOGEE_STAR_BAD_BITS = (0, 3, 18, 22)

QUALITY_REASON = {
    "not_stellar_classification": 0,
    "pipeline_warning_or_bad_flag": 1,
    "low_or_invalid_snr": 2,
    "large_or_invalid_vsini": 3,
    "invalid_parameter_or_error": 4,
    "nonprimary": 5,
    "iron_abundance_flag": 6,
    "invalid_coordinates": 7,
}

_COMMON_DTYPE = [
    ("source_id", "i8"), ("ra", "f8"), ("dec", "f8"),
    ("survey", "S8"), ("object_id", "S32"),
    ("quality", "?"), ("quality_reason", "u2"),
    ("is_stellar", "?"), ("known_contaminant", "?"),
    ("spectype", "S8"), ("primary", "?"),
    ("snr", "f4"), ("teff", "f4"), ("teff_err", "f4"),
    ("logg", "f4"), ("logg_err", "f4"), ("feh", "f4"),
    ("feh_err", "f4"), ("label_dwarf", "?"), ("label_giant", "?"),
]


def _column(data: Any, name: str) -> np.ndarray:
    """Accept a structured array or a dictionary with case-insensitive keys."""
    names = data.dtype.names if getattr(data, "dtype", None) is not None else data.keys()
    actual = {key.upper(): key for key in names}[name.upper()]
    return np.asarray(data[actual])


def _strings(values: np.ndarray) -> np.ndarray:
    return np.char.strip(np.asarray(values).astype("U"))


def _bitmask(bits: tuple[int, ...]) -> int:
    return sum(1 << bit for bit in bits)


def _valid_parameters(data: Any, iron_name: str) -> np.ndarray:
    """Do not treat FITS sentinel -9999 or zero errors as physical labels."""
    valid = np.ones(len(_column(data, "TEFF")), dtype=bool)
    for name in ("TEFF", "LOGG", iron_name):
        value = _column(data, name)
        error = _column(data, name + "_ERR")
        valid &= np.isfinite(value) & (value > -9000)
        valid &= np.isfinite(error) & (error > 0)
    return valid & (_column(data, "TEFF") > 0)


def _reasons(conditions: dict[str, np.ndarray]) -> np.ndarray:
    result = np.zeros(len(next(iter(conditions.values()))), dtype=np.uint16)
    for name, passed in conditions.items():
        result[~passed] |= np.uint16(1 << QUALITY_REASON[name])
    return result


def desi_quality_reasons(data: Any, snr_min: float = 10.0) -> np.ndarray:
    """Return independent rejection bits for RVS parameter quality."""
    snr = _column(data, "SN_R")
    vsini = _column(data, "VSINI")
    return _reasons({
        "not_stellar_classification": _strings(_column(data, "RR_SPECTYPE")) == "STAR",
        "pipeline_warning_or_bad_flag": (_column(data, "RVS_WARN") == 0)
        & _column(data, "SUCCESS").astype(bool),
        "low_or_invalid_snr": np.isfinite(snr) & (snr >= snr_min),
        "large_or_invalid_vsini": np.isfinite(vsini) & (vsini >= 0) & (vsini < 30),
        "invalid_parameter_or_error": _valid_parameters(data, "FEH"),
        "nonprimary": _column(data, "PRIMARY").astype(bool),
    })


def desi_quality_mask(data: Any, snr_min: float = 10.0) -> np.ndarray:
    return desi_quality_reasons(data, snr_min) == 0


def apogee_quality_reasons(data: Any, snr_min: float = 50.0) -> np.ndarray:
    """Calibrated DR17 atmospheric labels with selected bad flags removed."""
    snr = _column(data, "SNR")
    aspcap = _column(data, "ASPCAPFLAG").astype(np.int64)
    star = _column(data, "STARFLAG").astype(np.int64)
    return _reasons({
        "pipeline_warning_or_bad_flag": (aspcap & _bitmask(APOGEE_ASPCAP_BAD_BITS) == 0)
        & (star & _bitmask(APOGEE_STAR_BAD_BITS) == 0),
        "low_or_invalid_snr": np.isfinite(snr) & (snr >= snr_min),
        "invalid_parameter_or_error": _valid_parameters(data, "FE_H"),
        "iron_abundance_flag": _column(data, "FE_H_FLAG") == 0,
    })


def apogee_quality_mask(data: Any, snr_min: float = 50.0) -> np.ndarray:
    return apogee_quality_reasons(data, snr_min) == 0


def normalize_source_ids(values: np.ndarray, *, desi: bool = False) -> np.ndarray:
    """Keep Gaia IDs integer exact; MWS uses 999999 for an absent Gaia match."""
    values = np.asarray(values)
    if values.dtype.kind not in "iu":
        raise TypeError("Gaia source IDs must be integer arrays, never floating point")
    result = values.astype(np.int64, copy=True)
    result[result <= 0] = 0
    if desi:
        result[result == 999999] = 0
    return result


def _base_projection(size: int, extras: list[tuple[str, str]]) -> np.ndarray:
    result = np.zeros(size, dtype=_COMMON_DTYPE + extras)
    for name in ("ra", "dec", "snr", "teff", "teff_err", "logg", "logg_err", "feh", "feh_err"):
        result[name] = np.nan
    return result


def _finish_quality(result: np.ndarray) -> None:
    coords_ok = np.isfinite(result["ra"]) & np.isfinite(result["dec"])
    coords_ok &= (result["ra"] >= 0) & (result["ra"] < 360)
    coords_ok &= (result["dec"] >= -90) & (result["dec"] <= 90)
    result["quality_reason"][~coords_ok] |= 1 << QUALITY_REASON["invalid_coordinates"]
    result["quality"] = result["quality_reason"] == 0
    # Conservative branch tags only; one sigma must stay on the same side.
    # Hot stars/BHB/WD require their own labels, not these cool-star thresholds.
    cool = result["quality"] & (result["teff"] < 7000)
    result["label_dwarf"] = cool & (result["logg"] - result["logg_err"] > 4.2)
    result["label_giant"] = cool & (result["logg"] + result["logg_err"] < 3.2)


def project_desi(rv: np.ndarray, fibermap: np.ndarray, gaia: np.ndarray,
                 snr_min: float = 10.0) -> np.ndarray:
    """Project row-aligned HDUs from the same MWS file, retaining nonstars."""
    if len(rv) != len(fibermap) or len(rv) != len(gaia):
        raise ValueError("MWS HDU row counts differ")
    if not np.array_equal(_column(rv, "TARGETID"), _column(fibermap, "TARGETID")):
        raise ValueError("RVTAB and FIBERMAP TARGETID order differs")
    result = _base_projection(len(rv), [
        ("targetid", "i8"), ("rvs_warn", "i8"), ("vsini", "f4"),
        ("success", "?"), ("desi_survey", "S8"), ("desi_program", "S8"),
        ("imaging_release", "i4"), ("imaging_morphtype", "S4"),
        ("coadd_fiberstatus", "i8"),
    ])
    result["source_id"] = normalize_source_ids(_column(gaia, "SOURCE_ID"), desi=True)
    result["ra"], result["dec"] = _column(fibermap, "TARGET_RA"), _column(fibermap, "TARGET_DEC")
    result["survey"] = b"DESI"
    result["targetid"] = _column(rv, "TARGETID")
    result["object_id"] = result["targetid"].astype("S32")
    result["spectype"] = _strings(_column(rv, "RR_SPECTYPE")).astype("S8")
    result["is_stellar"] = result["spectype"] == b"STAR"
    # This is Redrock's classification, whose uncertainty rises at low S/N.
    result["known_contaminant"] = np.isin(result["spectype"], [b"QSO", b"GALAXY"])
    result["primary"] = _column(rv, "PRIMARY")
    result["quality_reason"] = desi_quality_reasons(rv, snr_min)
    for name in ("TEFF", "TEFF_ERR", "LOGG", "LOGG_ERR", "FEH", "FEH_ERR", "VSINI", "RVS_WARN", "SUCCESS"):
        result[name.lower()] = _column(rv, name)
    result["snr"] = _column(rv, "SN_R")
    result["desi_survey"] = _column(rv, "SURVEY").astype("S8")
    result["desi_program"] = _column(rv, "PROGRAM").astype("S8")
    result["imaging_release"] = _column(fibermap, "RELEASE")
    result["imaging_morphtype"] = _column(fibermap, "MORPHTYPE").astype("S4")
    result["coadd_fiberstatus"] = _column(fibermap, "COADD_FIBERSTATUS")
    _finish_quality(result)
    return result


def project_apogee(data: np.ndarray, snr_min: float = 50.0) -> np.ndarray:
    result = _base_projection(len(data), [
        ("aspcapflag", "i8"), ("starflag", "i8"), ("fe_h_flag", "i8"),
        ("extratarg", "i8"), ("native_duplicate", "?"),
    ])
    result["source_id"] = normalize_source_ids(_column(data, "GAIAEDR3_SOURCE_ID"))
    result["ra"], result["dec"] = _column(data, "RA"), _column(data, "DEC")
    result["survey"] = b"APOGEE"
    result["object_id"] = _column(data, "APOGEE_ID").astype("S32")
    result["spectype"] = b"STAR"
    result["is_stellar"] = True
    result["quality_reason"] = apogee_quality_reasons(data, snr_min)
    for name in ("TEFF", "TEFF_ERR", "LOGG", "LOGG_ERR", "SNR", "ASPCAPFLAG", "STARFLAG", "FE_H_FLAG", "EXTRATARG"):
        result["snr" if name == "SNR" else name.lower()] = _column(data, name)
    result["feh"], result["feh_err"] = _column(data, "FE_H"), _column(data, "FE_H_ERR")
    result["native_duplicate"] = (result["extratarg"] & (1 << 4)) != 0
    result["primary"] = ~result["native_duplicate"]
    _finish_quality(result)
    return result


def apogee_best_rows(data: np.ndarray) -> np.ndarray:
    """One highest-S/N row per Gaia ID, falling back to APOGEE_ID when absent.

    Do not discard native duplicates first: EXTRATARG bit 4 refers to repeats
    across fields, whereas this grouping explicitly resolves source identity.
    Stable input row order breaks exact S/N ties.
    """
    ids = normalize_source_ids(_column(data, "GAIAEDR3_SOURCE_ID"))
    keys = np.char.add("G:", ids.astype("U20"))
    missing = ids == 0
    keys = keys.astype("U40")
    keys[missing] = np.char.add("A:", _strings(_column(data, "APOGEE_ID"))[missing])
    snr = np.asarray(_column(data, "SNR"), dtype=float)
    snr = np.where(np.isfinite(snr), snr, -np.inf)
    order = np.lexsort((np.arange(len(data)), -snr, keys))
    first = np.ones(len(order), dtype=bool)
    if len(order) > 1:
        first[1:] = keys[order[1:]] != keys[order[:-1]]
    return np.sort(order[first])


def _audit(data: np.ndarray) -> dict[str, Any]:
    ids = data["source_id"]
    positive = ids > 0
    return {
        "rows": int(len(data)), "quality_rows": int(data["quality"].sum()),
        "stellar_rows": int(data["is_stellar"].sum()),
        "spectral_nonstar_rows": int(data["known_contaminant"].sum()),
        "gaia_id_missing_rows": int((~positive).sum()),
        "unique_gaia_ids": int(len(np.unique(ids[positive]))),
        "duplicate_gaia_id_rows": int(positive.sum() - len(np.unique(ids[positive]))),
        "rejection_counts_nonexclusive": {
            name: int(((data["quality_reason"] & (1 << bit)) != 0).sum())
            for name, bit in QUALITY_REASON.items()
        },
    }


def build_spectroscopy(config: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    """Read projected full catalogs on compute and write compact audited pools.

    DESI retains PRIMARY rows even when Redrock labels them QSO/GALAXY. APOGEE
    retains the globally highest-S/N repeat per Gaia ID or APOGEE_ID, including
    rows that fail parameter quality. Neither output is an independent distance
    truth catalog. Original flags and rejected-parameter values are preserved.
    """
    import fitsio

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    desi_out = output_dir / "desi_mws_pool.fits"
    apogee_out = output_dir / "apogee_pool.fits"
    desi_snr = float(config.get("desi_snr_min", 10.0))
    apogee_snr = float(config.get("apogee_snr_min", 50.0))
    audit: dict[str, Any] = {"quality_reason_bits": QUALITY_REASON.copy()}

    with fitsio.FITS(str(config["desi_path"])) as hdus:
        total = hdus["RVTAB"].get_nrows()
        if any(hdus[ext].get_nrows() != total for ext in ("FIBERMAP", "GAIA")):
            raise ValueError("DESI catalog HDUs are not row aligned")
        primary = hdus["RVTAB"].read(columns=["PRIMARY"])["PRIMARY"].astype(bool)
        rows = np.flatnonzero(primary)
        rv = hdus["RVTAB"].read(columns=DESI_COLUMNS, rows=rows)
        fibermap = hdus["FIBERMAP"].read(columns=FIBERMAP_COLUMNS, rows=rows)
        gaia = hdus["GAIA"].read(columns=["SOURCE_ID"], rows=rows)
    desi = project_desi(rv, fibermap, gaia, snr_min=desi_snr)
    del rv, fibermap, gaia, primary, rows
    audit["desi"] = _audit(desi)
    audit["desi"].update({"input_rows": total, "snr_min": desi_snr,
                           "source_path": str(config["desi_path"]),
                           "output_path": str(desi_out)})
    fitsio.write(str(desi_out), desi, extname="SPECTROSCOPY", clobber=True,
                 header={"CATALOG": "DESI DR1 MWS v1.0", "SNRCUT": desi_snr,
                         "SAMPLE": "PRIMARY audit pool; QUALITY selects labels"})
    del desi

    data = fitsio.read(str(config["apogee_path"]), ext=1, columns=APOGEE_COLUMNS)
    total = len(data)
    best = apogee_best_rows(data)
    apogee = project_apogee(data[best], snr_min=apogee_snr)
    del data, best
    audit["apogee"] = _audit(apogee)
    audit["apogee"].update({"input_rows": total,
                             "repeat_rows_removed": total - len(apogee),
                             "snr_min": apogee_snr,
                             "aspcap_bad_bits": list(APOGEE_ASPCAP_BAD_BITS),
                             "star_bad_bits": list(APOGEE_STAR_BAD_BITS),
                             "source_path": str(config["apogee_path"]),
                             "output_path": str(apogee_out)})
    fitsio.write(str(apogee_out), apogee, extname="SPECTROSCOPY", clobber=True,
                 header={"CATALOG": "APOGEE DR17 synspec_rev1", "SNRCUT": apogee_snr,
                         "SAMPLE": "Best-SNR identity pool; QUALITY selects labels"})
    return {"desi": str(desi_out), "apogee": str(apogee_out), "audit": audit}
