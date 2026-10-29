#!/usr/bin/env python3
"""Inspect the discordant blue anchor using DR11, Gaia, DESI and PS1 facts."""
from __future__ import annotations

import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".deps")]
import fitsio
import healpy as hp
import numpy as np

from photdis.audit import ps1_decam
from photdis.stage1 import require_compute, tile_bounds, write_json

SOURCE_ID = 3601173023628559616
FIELD = Path("data/field_targeted_ra180_190_dec_m010_p000_v1/field_diagnostics.fits")
CONFIG = Path("configs/field_targeted_ra180_190_dec_m010_p000.json")
OUT = Path("manifests/blue_outlier_source_3601173023628559616.json")
PS1_ROOT = Path("/global/cfs/cdirs/cosmo/work/ps1/cats/chunks-qz-star-v3")
PS1_COLUMNS = ["OBJ_ID", "RA_OK", "DEC_OK", "EPOCHMEAN_OK", "MEDIAN",
               "NMAG_OK", "STDEV"]
DIAGNOSTIC = ["ls_id", "source_id", "ra", "dec", "g", "r", "z", "g0",
              "r0", "z0", "gr0", "rz0", "sigma_g", "sigma_r", "sigma_z",
              "grz_clean", "desi_match", "desi_match_ambiguous",
              "desi_source_id", "desi_object_id", "desi_match_method",
              "desi_separation_arcsec", "desi_teff", "desi_logg", "desi_feh",
              "gaia_ra", "gaia_dec", "gaia_ref_epoch", "gaia_pmra",
              "gaia_pmdec", "gaia_phot_g_mean_mag", "gaia_bp_rp", "gaia_ruwe",
              "gaia_duplicated_source", "gaia_ipd_frac_multi_peak"]
RAW = ["RELEASE", "BRICKID", "OBJID", "TYPE", "RA", "DEC", "REF_CAT",
       "REF_ID", "MASKBITS", "FITBITS", "EBV"]
RAW += [f"{base}_{band}" for band in "GRZ" for base in
        ("FLUX", "FLUX_IVAR", "MW_TRANSMISSION", "NOBS", "RCHISQ",
         "FRACMASKED", "FRACIN", "FRACFLUX", "ANYMASK", "ALLMASK",
         "PSFSIZE", "PSFDEPTH")]


def scalar(value):
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode().strip()
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        x = float(value)
        return x if np.isfinite(x) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return str(value).strip()


def compact(row, names):
    return {name.lower(): scalar(row[name]) for name in names}


def arcsec(ra, dec, ra0, dec0):
    dra = (np.asarray(ra, float) - ra0) * np.cos(np.deg2rad(dec0))
    ddec = np.asarray(dec, float) - dec0
    return np.hypot(dra, ddec) * 3600


def ps1_near(gaia):
    ra0, dec0 = float(gaia["gaia_ra"]), float(gaia["gaia_dec"])
    pixels = hp.query_disc(32, hp.ang2vec(ra0, dec0, lonlat=True),
                           np.deg2rad(.02), inclusive=True, nest=False)
    nearby = []
    for pixel in pixels:
        path = PS1_ROOT / f"ps1-{pixel:05d}.fits"
        if not path.exists():
            continue
        with fitsio.FITS(str(path)) as file:
            coordinates = file[1].read(columns=["RA_OK", "DEC_OK"])
            initial = np.flatnonzero(arcsec(coordinates["RA_OK"],
                coordinates["DEC_OK"], ra0, dec0) < 30)
            if len(initial):
                nearby.append(file[1].read(columns=PS1_COLUMNS, rows=initial))
    if not nearby:
        return {"nearby": [], "matched_within_1_arcsec": 0}
    rows = np.concatenate(nearby)
    _, unique = np.unique(rows["OBJ_ID"], return_index=True)
    rows = rows[unique]
    epoch = 2000 + (rows["EPOCHMEAN_OK"].astype(float) - 51544.5) / 365.25
    dt = epoch - float(gaia["gaia_ref_epoch"])
    pmra, pmdec = float(gaia["gaia_pmra"]), float(gaia["gaia_pmdec"])
    propagated_ra = ra0 + pmra * dt / (3.6e6 * np.cos(np.deg2rad(dec0)))
    propagated_dec = dec0 + pmdec * dt / 3.6e6
    separation = arcsec(rows["RA_OK"], rows["DEC_OK"],
                        propagated_ra, propagated_dec)
    results = []
    for i in np.argsort(separation)[:8]:
        phot = np.asarray(rows["MEDIAN"][i], float)
        transformed = {band: float(ps1_decam(phot[None, :], band)[0])
                       for band in "grz"}
        results.append({"obj_id": int(rows["OBJ_ID"][i]),
                        "separation_arcsec": float(separation[i]),
                        "epoch_mjd": float(rows["EPOCHMEAN_OK"][i]),
                        "median_griz": phot[:4].tolist(),
                        "nmag_griz": rows["NMAG_OK"][i][:4].astype(int).tolist(),
                        "stdev_griz": rows["STDEV"][i][:4].astype(float).tolist(),
                        "decam_grz": transformed})
    return {"nearby": results,
            "matched_within_1_arcsec": int(np.sum(separation < 1))}


def main():
    require_compute()
    if OUT.exists():
        raise FileExistsError(OUT)
    with fitsio.FITS(str(FIELD)) as file:
        ids = file[1].read(columns=["source_id"])["source_id"]
        found = np.flatnonzero(ids == SOURCE_ID)
        if len(found) != 1:
            raise ValueError(f"Expected one source, found {len(found)}")
        row = file[1].read(columns=DIAGNOSTIC, rows=found)[0]
        position = file[1].read(columns=["ra", "dec", "r", "ls_id", "source_id"])
    distance = arcsec(position["ra"], position["dec"],
                      float(row["ra"]), float(row["dec"]))
    near = np.flatnonzero((distance > 0) & (distance < 10))
    neighbors = [dict(compact(position[i], position.dtype.names),
                      separation_arcsec=float(distance[i]))
                 for i in near[np.argsort(distance[near])[:10]]]
    config = json.loads(CONFIG.read_text())
    root = Path(json.loads(Path(config["base_config"]).read_text())["sweep_root"])
    ra, dec = float(row["ra"]), float(row["dec"])
    file_paths = []
    for filename in config["sweep_files"]:
        path = root / filename
        r0, d0, r1, d1 = tile_bounds(path)
        if r0 <= ra < r1 and d0 <= dec < d1:
            file_paths.append(path)
    if len(file_paths) != 1:
        raise ValueError(f"Expected one sweep for {ra}, {dec}: {file_paths}")
    sweep = file_paths[0]
    identity = int(row["ls_id"])
    with fitsio.FITS(str(sweep)) as file:
        ids = file[1].read(columns=["RELEASE", "BRICKID", "OBJID"])
        packed = ((ids["RELEASE"].astype(np.int64) << 40)
                  | (ids["BRICKID"].astype(np.int64) << 20)
                  | ids["OBJID"].astype(np.int64))
        found_raw = np.flatnonzero(packed == identity)
        if len(found_raw) != 1:
            raise ValueError(f"Expected one sweep row, found {len(found_raw)}")
        raw = file[1].read(columns=RAW, rows=found_raw)[0]
    result = {"gaia_source_id": SOURCE_ID,
              "field_row": compact(row, DIAGNOSTIC),
              "sweep_file": str(sweep), "sweep_row": compact(raw, RAW),
              "neighbors_within_10_arcsec": neighbors,
              "ps1": ps1_near(row), "compute_host": socket.gethostname()}
    ps1 = result["ps1"]["nearby"]
    if ps1 and ps1[0]["separation_arcsec"] < 1:
        result["legacy_minus_ps1_decam"] = {
            band: float(row[band]) - ps1[0]["decam_grz"][band]
            for band in "grz"}
    write_json(OUT, result)
    print(json.dumps({"gaia_source_id": SOURCE_ID, "sweep": str(sweep),
                      "neighbors": len(neighbors), "ps1_near": len(ps1)}), flush=True)


if __name__ == "__main__":
    main()
