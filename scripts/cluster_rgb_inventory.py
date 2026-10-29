#!/usr/bin/env python3
"""Join whole-cluster Gaia membership lists to measured DR11 photometry."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import socket
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".deps")]
import fitsio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from photdis.photometry import derive_photometry
from photdis.stage1 import CORE_COLUMNS, PHOT_COLUMNS, legacy_ids, require_compute, structured, write_json

MEMBERS = Path("data/reference/vasiliev_baumgardt_2021_members.zip")
SYSTEMS = {
    "m3": {
        "role": "development_diagnostic",
        "member_file": "catalogues/NGC_5272_M_3.txt",
        "sweep_root": "/global/cfs/projectdirs/cosmo/data/legacysurvey/dr11/south/sweep/11.0",
        "ra": (200, 205), "dec": (25, 30), "center": (205.548, 28.377),
        "sweep_ra": (200, 205), "sweep_dec": (25, 30),
        "mu": 15.041, "mu_stat": 0.017, "mu_sys": 0.036,
        "mu_reference": "https://arxiv.org/abs/2008.11745",
    },
    "ngc2419": {
        "role": "development_diagnostic",
        "member_file": "catalogues/NGC_2419.txt",
        "sweep_root": "/global/cfs/projectdirs/cosmo/data/legacysurvey/dr11/north/sweep/11.0",
        "ra": (110, 115), "dec": (35, 40), "center": (114.535, 38.882),
        "sweep_ra": (110, 115), "sweep_dec": (35, 40),
        "mu": 19.71, "mu_stat": 0.08, "mu_sys": 0.0,
        "mu_reference": "https://arxiv.org/abs/1011.5398",
    },
    "m53": {
        "role": "development_bridge",
        "member_file": "catalogues/NGC_5024_M_53.txt",
        "sweep_root": "/global/cfs/projectdirs/cosmo/data/legacysurvey/dr11/south/sweep/11.0",
        "ra": (195, 200), "dec": (15, 20), "center": (198.23, 18.168),
        "sweep_ra": (195, 200), "sweep_dec": (15, 20),
        "mu": 16.403, "mu_stat": 0.024, "mu_sys": 0.033,
        "mu_reference": "https://arxiv.org/abs/2012.13495",
    },
    "ngc7006": {
        "role": "reserved_whole_system",
        "member_file": "catalogues/NGC_7006.txt",
        "sweep_root": "/global/cfs/projectdirs/cosmo/data/legacysurvey/dr11/south/sweep/11.0",
        "ra": (310, 315), "dec": (10, 15), "center": (315.373, 16.188),
        "sweep_ra": (310, 315), "sweep_dec": (10, 15),
        "mu": 18.07448608, "mu_stat": 0.07378790, "mu_sys": 0.0,
        "mu_reference": "https://arxiv.org/abs/2212.05021",
    },
}


def read_members(path: Path, member_file: str):
    """Parse IDs as integers, preserving all 64 bits before parsing floats."""
    records = []
    with zipfile.ZipFile(path) as archive:
        with archive.open(member_file) as stream:
            for line in io.TextIOWrapper(stream):
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                records.append((int(parts[0]), float(parts[12]), float(parts[13]),
                                int(parts[15]), float(parts[16])))
    out = np.array(records, dtype=[("source_id", "i8"), ("gaia_g", "f8"),
                                   ("bp_rp", "f8"), ("qflag", "i2"),
                                   ("pmember", "f8")])
    order = np.lexsort((-out["pmember"], out["source_id"]))
    out = out[order]
    _, first = np.unique(out["source_id"], return_index=True)
    return out[first]


def sweep_paths(config):
    root = Path(config["sweep_root"])
    for ra in config["sweep_ra"]:
        for dec in config["sweep_dec"]:
            path = root / f"sweep-{ra:03d}p{dec:03d}-{ra+5:03d}p{dec+5:03d}.fits"
            if not path.exists():
                raise FileNotFoundError(path)
            yield path


def matched_sweeps(config, source_ids, *, prefer="grz"):
    parts = []
    input_rows = 0
    for path in sweep_paths(config):
        small = fitsio.read(str(path), ext=1, columns=["REF_ID", "REF_CAT", "RA", "DEC"])
        ra, dec = config["ra"], config["dec"]
        valid = ((small["RA"] >= ra[0]) & (small["RA"] < ra[1] + 5)
                 & (small["DEC"] >= dec[0]) & (small["DEC"] < dec[1] + 5)
                 & (np.char.strip(small["REF_CAT"].astype("U")) == "G3")
                 & np.isin(small["REF_ID"], source_ids))
        selected = np.flatnonzero(valid)
        input_rows += len(selected)
        if len(selected):
            parts.append(fitsio.read(str(path), ext=1,
                                     columns=CORE_COLUMNS + PHOT_COLUMNS,
                                     rows=selected))
    if not parts:
        raise ValueError("No Gaia-linked DR11 member photometry")
    rows = np.concatenate(parts)
    photo = derive_photometry(rows)
    sid = rows["REF_ID"].astype(np.int64)
    if prefer not in ("gr", "grz"):
        raise ValueError(prefer)
    score = (1000 * photo[f"{prefer}_clean"] + 100 * photo[f"{prefer}_base"]
             + np.nan_to_num(photo["snr_g"], nan=0, posinf=0) / 10000)
    order = np.lexsort((-score, sid))
    _, first = np.unique(sid[order], return_index=True)
    keep = order[first]
    return rows[keep], {key: value[keep] for key, value in photo.items()}, input_rows


def separation_arcmin(ra, dec, center):
    ra0, dec0 = np.deg2rad(center)
    rar, decr = np.deg2rad(ra), np.deg2rad(dec)
    cosang = np.sin(decr)*np.sin(dec0) + np.cos(decr)*np.cos(dec0)*np.cos(rar-ra0)
    return np.rad2deg(np.arccos(np.clip(cosang, -1, 1))) * 60


def make_figure(path, tab, mu, system, *, reserved=False):
    high = tab["pmember"] > .9
    clean = high & tab["grz_clean"]
    fig, axes = plt.subplots(1, 1 if reserved else 3,
                             figsize=(5.6 if reserved else 15, 4.7), constrained_layout=True)
    if reserved:
        axes = [axes]
    gaia_ok = high & np.isfinite(tab["bp_rp"]) & np.isfinite(tab["gaia_g"])
    axes[0].scatter(tab["bp_rp"][gaia_ok], tab["gaia_g"][gaia_ok]-mu,
                    s=2, alpha=.15, color="0.5", label="P>0.9")
    axes[0].scatter(tab["bp_rp"][clean & gaia_ok], tab["gaia_g"][clean & gaia_ok]-mu,
                    s=2, alpha=.4, color="tab:blue", label="clean grz")
    axes[0].set(xlabel="Gaia BP−RP", ylabel="Gaia G − cluster μ", title="Independent Gaia CMD")
    axes[0].legend(markerscale=3, fontsize=8)
    if not reserved:
        for ax, color, label in ((axes[1], "gr0", "(g−r)₀"),
                                 (axes[2], "rz0", "(r−z)₀")):
            good = clean & np.isfinite(tab[color]) & np.isfinite(tab["g0"])
            ax.scatter(tab[color][good], tab["g0"][good]-mu,
                       s=3, alpha=.3, color="tab:orange")
            ax.set(xlabel=label, ylabel="Legacy g₀ − cluster μ")
    for ax in axes:
        ax.set_ylim(9, -4)
        ax.grid(alpha=.15)
    fig.suptitle(f"{system}: member CMD and clean coverage" if reserved else
                 f"{system}: all likely members; phase selection pending")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    require_compute()
    parser = argparse.ArgumentParser()
    parser.add_argument("system", choices=SYSTEMS)
    args = parser.parse_args()
    cfg = SYSTEMS[args.system]
    outdir = Path("data") / f"cluster_rgb_{args.system}_v1"
    out = outdir / "members.fits"
    manifest = Path("manifests") / f"cluster_rgb_{args.system}_inventory_v1.json"
    figure = Path("figures/cluster_rgb") / f"01_member_cmd_{args.system}.png"
    if out.exists() or manifest.exists():
        raise FileExistsError(out if out.exists() else manifest)
    members = read_members(MEMBERS, cfg["member_file"])
    eligible = members[members["pmember"] >= .5]
    rows, photo, raw_rows = matched_sweeps(cfg, eligible["source_id"])
    source = rows["REF_ID"].astype(np.int64)
    index = np.searchsorted(eligible["source_id"], source)
    assert np.all(eligible["source_id"][index] == source)
    member = eligible[index]
    table = structured({
        "source_id": source, "ls_id": legacy_ids(rows),
        "ra": rows["RA"], "dec": rows["DEC"],
        "radius_arcmin": separation_arcmin(rows["RA"], rows["DEC"], cfg["center"]),
        "pmember": member["pmember"], "gaia_g": member["gaia_g"],
        "bp_rp": member["bp_rp"], "gaia_qflag": member["qflag"],
        "type": rows["TYPE"], "ebv": rows["EBV"],
        "maskbits": rows["MASKBITS"], "fitbits": rows["FITBITS"],
        **{name.lower(): rows[name] for name in rows.dtype.names
           if name.startswith(("FRACMASKED_", "FRACIN_", "FRACFLUX_", "RCHISQ_"))},
        **{key: value for key, value in photo.items() if key in (
            "g", "r", "z", "g0", "r0", "z0", "w10", "w20", "gr0", "rz0",
            "zw10", "w1w20", "sigma_g", "sigma_r", "sigma_z", "snr_w1",
            "snr_w2", "grz_base", "grz_clean", "grz_fraction_clean",
            "grz_mask_clean", "w1_clean", "w2_clean")},
    })
    outdir.mkdir(parents=True, exist_ok=True)
    fitsio.write(str(out), table)
    make_figure(figure, table, cfg["mu"], args.system,
                reserved=cfg["role"] == "reserved_whole_system")
    high = table["pmember"] > .9
    info = {"system": args.system, "role": cfg["role"],
            "member_source": str(MEMBERS),
            "membership_reference": "https://zenodo.org/records/4891252",
            "sweep_root": cfg["sweep_root"], "external_modulus": {
                "mu_mag": cfg["mu"], "stat_sigma_mag": cfg["mu_stat"],
                "systematic_sigma_mag": cfg["mu_sys"],
                "url": cfg["mu_reference"]},
            "counts": {"member_catalog_all": len(members),
                       "member_catalog_p_ge_0p5": len(eligible),
                       "member_catalog_p_gt_0p9": int((eligible["pmember"] > .9).sum()),
                       "dr11_matching_rows": raw_rows,
                       "dr11_unique_members": len(table),
                       "dr11_p_gt_0p9": int(high.sum()),
                       "grz_base_p_gt_0p9": int((high & table["grz_base"]).sum()),
                       "grz_clean_p_gt_0p9": int((high & table["grz_clean"]).sum()),
                       "grz_clean_p_0p5_to_0p9": int((~high & table["grz_clean"]).sum()),
                       "w1_clean_grz_p_gt_0p9": int((high & table["grz_clean"] & table["w1_clean"]).sum()),
                       "w2_clean_grz_p_gt_0p9": int((high & table["grz_clean"] & table["w2_clean"]).sum())},
            "data_file": str(out), "figure": str(figure),
            "compute_host": socket.gethostname()}
    write_json(manifest, info)
    print(json.dumps(info["counts"]), flush=True)


if __name__ == "__main__":
    main()
