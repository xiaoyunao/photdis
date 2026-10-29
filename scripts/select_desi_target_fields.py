#!/usr/bin/env python3
"""Rank DR11 10° fields from qualified metal-poor DESI spectra first."""
from __future__ import annotations

import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".deps")]
import fitsio
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

from photdis.audit import rvs_metallicity
from photdis.stage1 import require_compute, write_json

POOL = Path("data/stage1_v1/spectroscopy/desi_mws_pool.fits")
MWS = Path("/global/cfs/cdirs/desi/public/dr1/vac/dr1/mws/iron/v1.0/mwsall-pix-iron.fits")
SWEEPS = Path("/global/cfs/projectdirs/cosmo/data/legacysurvey/dr11/south/sweep/11.0")
OUTPUT = Path("manifests/desi_target_field_ranking.json")


def coord(dec):
    return ("p" if dec >= 0 else "m") + f"{abs(dec):03d}"


def sweep_names(ra0, dec0):
    return [f"sweep-{ra:03d}{coord(dec)}-{ra+5:03d}{coord(dec+5)}.fits"
            for ra in (ra0, ra0 + 5) for dec in (dec0, dec0 + 5)]


def main():
    require_compute()
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    with fitsio.FITS(str(POOL)) as f:
        pool = f[1].read(columns=["ra", "dec", "source_id", "quality",
            "teff", "logg", "logg_err", "feh", "feh_err", "object_id"])
    cal, valid = rvs_metallicity(pool["feh"], pool["teff"], pool["logg"])
    target = (pool["quality"] & valid & (pool["teff"] >= 4200)
        & (pool["teff"] < 5000) & np.isfinite(pool["logg_err"])
        & (pool["logg_err"] > 0) & (pool["logg"] - pool["logg_err"] > 4.2)
        & np.isfinite(pool["feh_err"]) & (pool["feh_err"] > 0)
        & (cal < -1.5) & (pool["source_id"] > 0)
        & np.isfinite(pool["ra"]) & np.isfinite(pool["dec"])
        & (pool["dec"] >= -40) & (pool["dec"] < 35))
    indices = np.flatnonzero(target)
    # The projected pool contains only PRIMARY MWS rows in their original order.
    # Reconstruct their original row indices before sampling aligned HDUs.
    with fitsio.FITS(str(MWS)) as f:
        primary = f["RVTAB"].read(columns=["PRIMARY"])["PRIMARY"].astype(bool)
        original_rows = np.flatnonzero(primary)
        if len(original_rows) != len(pool):
            raise ValueError("Projected pool length differs from MWS PRIMARY rows")
        chosen = original_rows[indices]
        embedded = f["GAIA"].read(columns=["SOURCE_ID", "PARALLAX_OVER_ERROR"], rows=chosen)
        fibermap = f["FIBERMAP"].read(columns=["EBV"], rows=chosen)
    if not np.array_equal(embedded["SOURCE_ID"], pool["source_id"][indices]):
        raise ValueError("Projected DESI pool and MWS Gaia rows are not aligned")
    ra = pool["ra"][indices].astype(float)
    dec = pool["dec"][indices].astype(float)
    b = SkyCoord(ra=ra*u.deg, dec=dec*u.deg).galactic.b.deg
    dust = fibermap["EBV"].astype(float)
    snr = embedded["PARALLAX_OVER_ERROR"].astype(float)
    suitable = (np.abs(b) >= 30) & np.isfinite(dust) & (dust < .03)
    ra0 = (np.floor(ra / 10).astype(int) * 10) % 360
    dec0 = np.floor(dec / 10).astype(int) * 10
    old = set()
    for path in Path("configs").glob("field_*.json"):
        config = json.loads(path.read_text())
        bounds = config.get("bounds_deg", [])
        if len(bounds) == 4 and bounds[2] - bounds[0] == 10 and bounds[3] - bounds[1] == 10:
            old.add((int(bounds[0]), int(bounds[1])))
    candidates = []
    for r, d in sorted(set(zip(ra0.tolist(), dec0.tolist()))):
        cell = (ra0 == r) & (dec0 == d)
        names = sweep_names(r, d)
        available = all((SWEEPS / name).exists() for name in names)
        if not available:
            continue
        candidates.append({"bounds_deg": [r, d, r + 10, d + 10],
            "sweep_files": names, "already_configured": (r, d) in old,
            "qualified_poor_core": int(cell.sum()),
            "high_lat_low_dust": int(np.sum(cell & suitable)),
            "high_lat_low_dust_raw_parallax_snr10": int(np.sum(cell & suitable & (snr >= 10))),
            "high_lat_low_dust_raw_parallax_snr20": int(np.sum(cell & suitable & (snr >= 20)))})
    candidates.sort(key=lambda x: (
        x["already_configured"], -x["high_lat_low_dust_raw_parallax_snr20"],
        -x["high_lat_low_dust_raw_parallax_snr10"], -x["high_lat_low_dust"],
        -x["qualified_poor_core"]))
    report = {"selection": "DESI RVS quality, 4200<=Teff<5000, logg-error>4.2, calibrated FeH<-1.5, Gaia source ID>0",
        "ranking_domain": "complete four-sweep DR11 10x10 cells; prioritize |b|>=30 and MWS EBV<0.03",
        "qualified_poor_core_all_dec_minus40_plus35": len(indices),
        "high_lat_low_dust": int(suitable.sum()),
        "high_lat_low_dust_raw_parallax_snr20": int(np.sum(suitable & (snr >= 20))),
        "candidates": candidates[:50], "compute_host": socket.gethostname()}
    write_json(OUTPUT, report)
    print(json.dumps({"targets": len(indices), "top": candidates[:8]}), flush=True)


if __name__ == "__main__":
    main()
