#!/usr/bin/env python3
"""Rank untouched DR11 fields by qualified cool DESI giants before extraction."""
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

from photdis.stage1 import require_compute, write_json
from select_desi_target_fields import sweep_names

POOL = Path("data/stage1_v1/spectroscopy/desi_mws_pool.fits")
MWS = Path("/global/cfs/cdirs/desi/public/dr1/vac/dr1/mws/iron/v1.0/mwsall-pix-iron.fits")
DR11 = Path("/global/cfs/projectdirs/cosmo/data/legacysurvey/dr11")
OUT = Path("manifests/desi_cool_giant_field_ranking_v1.json")


def main():
    require_compute()
    if OUT.exists():
        raise FileExistsError(OUT)
    pool = fitsio.read(str(POOL), ext=1, columns=[
        "ra", "dec", "source_id", "quality", "teff", "logg", "logg_err"])
    giant = (pool["quality"] & (pool["source_id"] > 0)
             & np.isfinite(pool["teff"]) & (pool["teff"] >= 3500)
             & (pool["teff"] < 5200) & np.isfinite(pool["logg"])
             & np.isfinite(pool["logg_err"]) & (pool["logg_err"] > 0)
             & (pool["logg"] + pool["logg_err"] < 3.2)
             & np.isfinite(pool["ra"]) & np.isfinite(pool["dec"])
             & (pool["dec"] >= -40) & (pool["dec"] < 80))
    indices = np.flatnonzero(giant)
    _, first = np.unique(pool["source_id"][indices], return_index=True)
    indices = indices[first]
    with fitsio.FITS(str(MWS)) as f:
        primary = f["RVTAB"].read(columns=["PRIMARY"])["PRIMARY"].astype(bool)
        original = np.flatnonzero(primary)
        if len(original) != len(pool):
            raise ValueError("MWS projected pool is not aligned with primary rows")
        ebv = f["FIBERMAP"].read(columns=["EBV"], rows=original[indices])["EBV"]
    ra, dec = pool["ra"][indices].astype(float), pool["dec"][indices].astype(float)
    b = SkyCoord(ra=ra*u.deg, dec=dec*u.deg).galactic.b.deg
    low_dust = (np.abs(b) >= 30) & np.isfinite(ebv) & (ebv < .03)
    k_range = (pool["teff"][indices] >= 4200) & (pool["teff"][indices] < 5000)
    very_cool = pool["teff"][indices] < 4200
    ra0 = (np.floor(ra / 10).astype(int) * 10) % 360
    dec0 = np.floor(dec / 10).astype(int) * 10
    configured = set()
    for path in Path("configs").glob("field_*.json"):
        config = json.loads(path.read_text())
        bounds = config.get("bounds_deg", [])
        if len(bounds) == 4 and bounds[2] - bounds[0] == 10 and bounds[3] - bounds[1] == 10:
            configured.add((int(bounds[0]), int(bounds[1])))
    cells = []
    for r, d in sorted(set(zip(ra0.tolist(), dec0.tolist()))):
        tile = (ra0 == r) & (dec0 == d)
        if not np.any(tile & low_dust & k_range):
            continue
        names = sweep_names(r, d)
        region = next((name for name in ("south", "north")
                       if all((DR11 / name / "sweep/11.0" / item).exists()
                              for item in names)), None)
        if region is None:
            continue
        cells.append({"bounds_deg": [r, d, r+10, d+10],
                      "dr11_region": region, "sweep_files": names,
                      "already_configured": (r, d) in configured,
                      "qualified_cool_lt5200": int(tile.sum()),
                      "high_lat_low_dust_k4200_5000": int(np.sum(tile & low_dust & k_range)),
                      "high_lat_low_dust_lt4200": int(np.sum(tile & low_dust & very_cool)),
                      "high_lat_low_dust_all_cool": int(np.sum(tile & low_dust))})
    cells.sort(key=lambda x: (x["already_configured"],
                              -x["high_lat_low_dust_k4200_5000"],
                              -x["high_lat_low_dust_lt4200"],
                              -x["high_lat_low_dust_all_cool"]))
    result = {"selection": "MWS primary projected pool; RVS quality, 3500<=Teff<5200, logg+sigma<3.2, Gaia source ID; prioritize 4200-5000 K and |b|>=30, MWS EBV<0.03",
              "available_dr11": "complete 10x10 four-sweep cells in north or south",
              "qualified_all": len(indices), "high_lat_low_dust": int(low_dust.sum()),
              "candidates": cells[:40], "compute_host": socket.gethostname()}
    write_json(OUT, result)
    print(json.dumps({"qualified_all": len(indices),
                      "high_lat_low_dust": int(low_dust.sum()),
                      "top": cells[:8]}), flush=True)


if __name__ == "__main__":
    main()
