#!/usr/bin/env python3
"""Compare frozen native cold-v4 anchors with matched PS1 optical photometry."""
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
from scipy.spatial import cKDTree

from photdis.audit import ps1_decam
from photdis.followup import identity_hash
from photdis.holdouts import load_reference_catalogs
from photdis.stage1 import require_compute, unit_vectors, write_json
from cold_v4 import PREPARED, configured_field, sha
from cold_v4_dust_sensitivity import score
from cold_v4_native_transfer import LOCK, REGISTRATION, robust, spatial_bootstrap
from cold_v4_policy_noise import frozen_models


PS1_ROOT = Path("/global/cfs/cdirs/cosmo/work/ps1/cats/chunks-qz-star-v3")
NORTH = Path("configs/cold_v4_native_transfer_north_final.json")
CONFIG = Path("configs/cold_v4_ps1_relative_audit.json")
OUT = Path("data/cold_v4_ps1_relative_audit/final.json")
PS1_COLUMNS = ("OBJ_ID", "RA_OK", "DEC_OK", "EPOCHMEAN_OK", "MEDIAN",
               "NMAG_OK", "STDEV")
EXTRA = ("source_id", "gaia_ra", "gaia_dec", "gaia_ref_epoch", "gaia_pmra",
         "gaia_pmdec", "mw_transmission_g", "mw_transmission_r",
         "mw_transmission_z", "ebv", "grz_clean", "snr_w1", "snr_w2")


def ps1_catalog(bounds):
    ra_lo, dec_lo, ra_hi, dec_hi = bounds
    ra_center = (ra_lo + ra_hi) / 2
    dec_center = (dec_lo + dec_hi) / 2
    radius = np.deg2rad(8)
    pixels = hp.query_disc(32, hp.ang2vec(ra_center, dec_center, lonlat=True),
                           radius, inclusive=True, nest=False)
    chunks = []
    missing = 0
    for pixel in pixels:
        path = PS1_ROOT / f"ps1-{pixel:05d}.fits"
        if not path.exists():
            missing += 1
            continue
        rows = fitsio.read(str(path), ext=1, columns=list(PS1_COLUMNS))
        ra = np.asarray(rows["RA_OK"], float)
        dec = np.asarray(rows["DEC_OK"], float)
        use = ((ra >= ra_lo - .1) & (ra < ra_hi + .1) &
               (dec >= dec_lo - .1) & (dec < dec_hi + .1) &
               np.isfinite(rows["EPOCHMEAN_OK"]) &
               (rows["EPOCHMEAN_OK"] > 50000) &
               (rows["EPOCHMEAN_OK"] < 60000))
        if use.any():
            chunks.append(rows[use])
    if not chunks:
        raise ValueError(f"No PS1 sources in {bounds}")
    catalog = np.concatenate(chunks)
    _, unique = np.unique(catalog["OBJ_ID"], return_index=True)
    return catalog[unique], {"queried_nside32_pixels": len(pixels),
                             "missing_ps1_pixels": missing,
                             "sources_in_margin": len(catalog),
                             "unique_sources_in_margin": len(unique)}


def match_ps1(gaia, catalog):
    """Match Gaia positions propagated to each candidate PS1 mean epoch."""
    size = len(gaia["gaia_ra"])
    finite_pm = np.isfinite(gaia["gaia_pmra"]) & np.isfinite(gaia["gaia_pmdec"])
    ref_epoch = np.asarray(gaia["gaia_ref_epoch"], float)
    finite_pm &= np.isfinite(ref_epoch)
    epoch = 2000 + (catalog["EPOCHMEAN_OK"].astype(float) - 51544.5) / 365.25

    def position(target_epoch):
        dt = target_epoch - ref_epoch
        ra = np.asarray(gaia["gaia_ra"], float)
        dec = np.asarray(gaia["gaia_dec"], float)
        pmra = np.asarray(gaia["gaia_pmra"], float)
        pmdec = np.asarray(gaia["gaia_pmdec"], float)
        return ((ra + pmra * dt / (3.6e6 * np.cos(np.deg2rad(dec)))) % 360,
                dec + pmdec * dt / 3.6e6)

    mid_ra, mid_dec = position(float(np.median(epoch)))
    mid_ra = np.where(finite_pm, mid_ra, 0)
    mid_dec = np.where(finite_pm, mid_dec, 0)
    tree = cKDTree(unit_vectors(catalog["RA_OK"], catalog["DEC_OK"]))
    limit = 2 * np.sin(np.deg2rad(5 / 3600) / 2)
    distance, candidate = tree.query(unit_vectors(mid_ra, mid_dec), k=3,
                                     distance_upper_bound=limit)
    safe = np.minimum(candidate, len(catalog) - 1)
    separations = np.full((size, 3), np.inf)
    for j in range(3):
        pra, pdec = position(epoch[safe[:, j]])
        xyz = unit_vectors(pra, pdec)
        other = unit_vectors(catalog["RA_OK"][safe[:, j]],
                             catalog["DEC_OK"][safe[:, j]])
        chord = np.linalg.norm(xyz - other, axis=1)
        angle = np.rad2deg(2 * np.arcsin(np.minimum(chord / 2, 1))) * 3600
        separations[:, j] = np.where(np.isfinite(distance[:, j]) & finite_pm,
                                     angle, np.inf)
    nearest = np.argmin(separations, axis=1)
    index = safe[np.arange(size), nearest]
    separation = separations[np.arange(size), nearest]
    matched = separation < 1
    ambiguous = (separations < 1).sum(axis=1) > 1
    ids, count = np.unique(catalog["OBJ_ID"][index[matched]], return_counts=True)
    ambiguous |= matched & np.isin(catalog["OBJ_ID"][index], ids[count > 1])
    phot = catalog["MEDIAN"][index].astype(float)
    nobs = catalog["NMAG_OK"][index]
    stdev = catalog["STDEV"][index].astype(float)
    gi = phot[:, 0] - phot[:, 2]
    quality = matched & ~ambiguous & (gi > .4) & (gi < 2.7)
    for j in (0, 1, 2, 3):
        quality &= ((nobs[:, j] >= 3) & np.isfinite(stdev[:, j]) &
                    (stdev[:, j] > 0) & (stdev[:, j] < .03) &
                    np.isfinite(phot[:, j]) & (phot[:, j] > 15) &
                    (phot[:, j] < 20))
    transformed = {band: ps1_decam(phot, band) for band in "grz"}
    return quality, transformed, {"gaia_with_finite_pm": int(finite_pm.sum()),
                                  "matched_1arcsec": int(matched.sum()),
                                  "ambiguous": int((matched & ambiguous).sum()),
                                  "quality_all_griz": int(quality.sum()),
                                  "median_separation_arcsec": float(
                                      np.median(separation[quality])) if quality.any() else None}


def substitute_grz(table, quality, transformed):
    """Replace optical measurements while preserving catalog extinction."""
    output = dict(table)
    g0 = np.asarray(table["g0"], float)
    r0 = g0 - np.asarray(table["gr0"], float)
    z0 = r0 - np.asarray(table["rz0"], float)
    old = {"g": g0, "r": r0, "z": z0}
    delta = {}
    for band in "grz":
        observed = old[band] - 2.5 * np.log10(
            np.asarray(table[f"mw_transmission_{band}"], float))
        delta[band] = np.where(quality, transformed[band] - observed, 0)
    output["g0"] = g0 + delta["g"]
    output["r0"] = r0 + delta["r"]
    output["gr0"] = np.asarray(table["gr0"], float) + delta["g"] - delta["r"]
    output["rz0"] = np.asarray(table["rz0"], float) + delta["r"] - delta["z"]
    output["zw10"] = np.asarray(table["zw10"], float) + delta["z"]
    return output, {band: -delta[band] for band in "grz"}


def field_result(item, references, route, forests, policy, expected, seed):
    table, cold, _, metadata = configured_field(item["field_config"], references)
    with fitsio.FITS(metadata["path"]) as file:
        anchor_rows = np.flatnonzero(file[1].read(columns=["anchor"])["anchor"])
        extras = file[1].read(columns=list(EXTRA), rows=anchor_rows)
    if not np.array_equal(extras["source_id"], table["source_id"]):
        raise ValueError("Anchor order differs from PS1 extras")
    table.update({name: extras[name] for name in EXTRA if name != "source_id"})
    base_level, baseline, baseline_pred = score(table, cold, route, forests, policy)
    if identity_hash(table["source_id"][baseline]) != expected["selected_source_id_sha256"]:
        raise ValueError("Frozen selected identity differs from final score")
    select_rows = np.flatnonzero(baseline)
    gaia = {name: np.asarray(table[name])[select_rows] for name in EXTRA
            if name.startswith("gaia_")}
    catalog, inventory = ps1_catalog(item["bounds_deg"])
    quality_selected, transformed_selected, matching = match_ps1(gaia, catalog)
    quality = np.zeros(len(baseline), bool)
    quality[select_rows] = quality_selected
    transformed = {band: np.zeros(len(baseline)) for band in "grz"}
    for band in "grz":
        transformed[band][select_rows] = transformed_selected[band]
    shifted, legacy_minus_ps1 = substitute_grz(table, quality, transformed)
    new_level, new_selected, new_pred = score(shifted, cold, route, forests, policy)
    common = baseline & quality & new_selected
    p = np.asarray(table["p_corr"], float)
    near = common & (p >= 2)

    def summarize(mask):
        delta = new_pred[mask] - baseline_pred[mask]
        mu_ref = 10 - 5 * np.log10(p[mask])
        ra = np.asarray(table["ra"][mask], float)
        dec = np.asarray(table["dec"][mask], float)
        return {"n": int(mask.sum()),
                "paired_prediction_shift": robust(delta),
                "paired_prediction_shift_sky_block": spatial_bootstrap(
                    delta, ra, dec, seed),
                "legacy_residual": robust(baseline_pred[mask] - mu_ref),
                "ps1_replacement_residual": robust(new_pred[mask] - mu_ref),
                "tier_changes": int((base_level[mask] != new_level[mask]).sum())}

    quality_mask = baseline & quality
    optical = {}
    for band in "grz":
        values = legacy_minus_ps1[band][quality_mask]
        optical[band] = {"legacy_minus_transformed_ps1_mag": robust(values),
                         "sky_block": spatial_bootstrap(values,
                             np.asarray(table["ra"][quality_mask], float),
                             np.asarray(table["dec"][quality_mask], float),
                             seed + ord(band))}
    return {"field": metadata["name"], "baseline_selected_count": int(baseline.sum()),
            "ps1_catalog": inventory, "matching": matching,
            "quality_on_baseline_count": int(quality_mask.sum()),
            "replacement_common_count": int(common.sum()),
            "replacement_retention_fraction": float(common.sum() / max(quality_mask.sum(), 1)),
            "optical": optical, "replacement_all": summarize(common),
            "replacement_parallax_ge_2_mas": summarize(near)}


def main():
    require_compute()
    if OUT.exists():
        raise FileExistsError(OUT)
    config = json.loads(CONFIG.read_text())
    south = json.loads(REGISTRATION.read_text())
    north = json.loads(NORTH.read_text())
    if (south["prepared_sha256"] != sha(PREPARED) or
            south["policy_lock_sha256"] != sha(LOCK) or
            north["parent_registration_sha256"] != sha(REGISTRATION)):
        raise ValueError("Frozen model or registered fields differ")
    scored_south = json.loads(Path("manifests/cold_v4_native_transfer_final.json").read_text())
    scored_north = json.loads(Path("manifests/cold_v4_native_transfer_north_final.json").read_text())
    expected = {field["field"]: field for field in
                scored_south["fields"] + scored_north["fields"]}
    items = south["independent_final_fields"] + north["independent_final_fields"]
    if [item["name"] for item in items] != config["fields"]:
        raise ValueError("PS1 audit field registration differs")
    route, forests = frozen_models()
    references = load_reference_catalogs("data/reference")
    policy = json.loads(LOCK.read_text())["chosen_policy"]
    results = [field_result(item, references, route, forests, policy,
                            expected[item["name"]], 20261230 + 100 * index)
               for index, item in enumerate(items)]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_json(OUT, {"registration_sha256": sha(CONFIG),
                     "prepared_sha256": sha(PREPARED),
                     "policy_lock_sha256": sha(LOCK),
                     "compute_host": socket.gethostname(),
                     "method": config["reference"],
                     "fields": results})


if __name__ == "__main__":
    main()
