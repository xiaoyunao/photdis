#!/usr/bin/env python3
"""Score registered new fields with the unchanged cold-v4 WISE20 route."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".deps")]
import fitsio
import healpy as hp
import numpy as np

from photdis.followup import identity_hash
from photdis.holdouts import load_reference_catalogs
from photdis.route import LEVEL, feature_matrix, route_rows
from photdis.route_policy import choose_policy
from photdis.stage1 import require_compute, write_json
from cold_v4 import PREPARED, configured_field, sha
from cold_v4_policy_noise import frozen_models

REGISTRATION = Path("configs/cold_v4_native_transfer_fields.json")
LOCK = Path("configs/cold_v4_deep_policy.json")
OUT = Path("data/cold_v4_native_transfer")
EXTRA = ("source_id", "r0", "ebv", "mw_transmission_g",
         "grz_clean", "snr_w1", "snr_w2")


def robust(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0}
    center = float(np.median(values))
    return {"n": len(values), "median": center,
            "nmad": float(1.4826 * np.median(abs(values - center))),
            "p05": float(np.quantile(values, .05)),
            "p95": float(np.quantile(values, .95))}


def spatial_bootstrap(residual, ra, dec, seed, repeats=400):
    pixels = hp.ang2pix(64, ra, dec, nest=True, lonlat=True)
    groups = [np.flatnonzero(pixels == pixel) for pixel in np.unique(pixels)]
    if len(groups) < 5:
        return None
    rng = np.random.default_rng(seed)
    medians, nmads = [], []
    for _ in range(repeats):
        sample = residual[np.concatenate([groups[index] for index in
            rng.integers(len(groups), size=len(groups))])]
        center = np.median(sample)
        medians.append(center)
        nmads.append(1.4826 * np.median(abs(sample - center)))
    return {"n_nside64_blocks": len(groups),
            "median_95pct": np.quantile(medians, [.025, .975]).tolist(),
            "nmad_95pct": np.quantile(nmads, [.025, .975]).tolist()}


def field_result(path, references, route, forests, policy, seed):
    table, cold, _, metadata = configured_field(path, references)
    with fitsio.FITS(metadata["path"]) as file:
        anchor_rows = np.flatnonzero(file[1].read(columns=["anchor"])["anchor"])
        extras = file[1].read(columns=list(EXTRA), rows=anchor_rows)
    if not np.array_equal(extras["source_id"], table["source_id"]):
        raise ValueError("Anchor extras differ from configured-field order")
    table.update({name: extras[name] for name in EXTRA if name != "source_id"})
    routed = route_rows(table, route)
    level = choose_policy(table, routed, policy)
    selected = cold & (level > 0)
    prediction = np.full(len(selected), np.nan)
    for tier, number in LEVEL.items():
        use = selected & (level == number)
        if use.any():
            quantiles = forests[tier].predict(feature_matrix(table, tier)[use])
            prediction[use] = np.asarray(table["g0"][use], float) - quantiles[:, 2]
    true_mu = 10 - 5 * np.log10(np.asarray(table["p_corr"][selected], float))
    residual = prediction[selected] - true_mu
    if not np.isfinite(residual).all():
        raise ValueError("Nonfinite selected native residual")
    bp_rp = np.asarray(table["bp_rp_checked"], float)
    p = np.asarray(table["p_corr"], float)
    r0 = np.asarray(table["r0"], float)
    ag = -2.5 * np.log10(np.asarray(table["mw_transmission_g"], float))
    groups = {
        "parallax_1_to_2_mas": selected & (p >= 1) & (p < 2),
        "parallax_ge_2_mas": selected & (p >= 2),
        "bp_rp_1p3_to_1p8": selected & (bp_rp >= 1.3) & (bp_rp < 1.8),
        "bp_rp_1p8_to_2p3": selected & (bp_rp >= 1.8) & (bp_rp < 2.3),
        "bp_rp_2p3_to_2p8": selected & (bp_rp >= 2.3) & (bp_rp < 2.8),
        "bp_rp_2p8_to_3p5": selected & (bp_rp >= 2.8) & (bp_rp < 3.5),
        "r0_lt_17": selected & (r0 < 17),
        "r0_17_to_18": selected & (r0 >= 17) & (r0 < 18),
        "r0_ge_18": selected & (r0 >= 18),
        "ag_le_0p05": selected & (ag <= .05),
        "ag_gt_0p05": selected & (ag > .05),
    }
    for tier, number in LEVEL.items():
        groups[tier] = selected & (level == number)
    by_group = {name: robust((prediction[use] -
                             (10 - 5 * np.log10(p[use]))))
                for name, use in groups.items()}
    parallax_block_ci = {}
    for name in ("parallax_1_to_2_mas", "parallax_ge_2_mas"):
        use = groups[name]
        parallax_block_ci[name] = spatial_bootstrap(
            prediction[use] - (10 - 5 * np.log10(p[use])),
            np.asarray(table["ra"][use], float),
            np.asarray(table["dec"][use], float), seed + len(name))
    return {"field": metadata["name"], "bounds_deg": metadata["bounds_deg"],
            "strict_anchor_count": metadata["strict_anchor_rows"],
            "strict_cold_count": int(cold.sum()),
            "selected_count": int(selected.sum()),
            "acceptance_of_strict_cold": float(selected.sum() / max(cold.sum(), 1)),
            "selected_source_id_sha256": identity_hash(table["source_id"][selected]),
            "tier_counts": {tier: int(np.sum(selected & (level == number)))
                            for tier, number in LEVEL.items()},
            "distance_modulus_residual": robust(residual),
            "distance_relative_residual": robust(10**(residual / 5) - 1),
            "absolute_modulus_error_gt_0p5": int(np.sum(abs(residual) > .5)),
            "sky_block_bootstrap": spatial_bootstrap(residual,
                np.asarray(table["ra"][selected], float),
                np.asarray(table["dec"][selected], float), seed),
            "selected_properties": {
                "p_corr_mas": robust(p[selected]),
                "gaia_bp_rp": robust(bp_rp[selected]),
                "r0": robust(r0[selected]), "source_ag": robust(ag[selected])},
            "by_group": by_group,
            "parallax_group_sky_block_bootstrap": parallax_block_ci}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("development", "final"))
    args = parser.parse_args()
    require_compute()
    out = OUT / f"{args.role}.json"
    if out.exists():
        raise FileExistsError(out)
    registration = json.loads(REGISTRATION.read_text())
    if registration["prepared_sha256"] != sha(PREPARED) or (
            registration["policy_lock_sha256"] != sha(LOCK)):
        raise ValueError("Frozen model or policy differs from field registration")
    lock = json.loads(LOCK.read_text())
    route, forests = frozen_models()
    references = load_reference_catalogs("data/reference")
    key = "development_fields" if args.role == "development" else "independent_final_fields"
    fields = registration[key]
    result = {"role": args.role, "registration_sha256": sha(REGISTRATION),
              "prepared_sha256": sha(PREPARED), "policy_lock_sha256": sha(LOCK),
              "compute_host": socket.gethostname(),
              "frozen_policy": lock["chosen_policy"],
              "fields": [field_result(item["field_config"], references, route, forests,
                                      lock["chosen_policy"], 20261110 + index)
                         for index, item in enumerate(fields)]}
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(out, result)


if __name__ == "__main__":
    main()
