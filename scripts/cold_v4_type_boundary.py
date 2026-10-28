#!/usr/bin/env python3
"""Inspect accepted DESI cold giants and color-rejected metal-poor dwarfs."""
from __future__ import annotations

import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".deps")]
import fitsio
import joblib
import numpy as np

from photdis.audit import rvs_metallicity
from photdis.route import LEVEL, feature_matrix, route_rows
from photdis.route_policy import choose_policy
from photdis.stage1 import require_compute, write_json
from cold_v4 import CAL_CONFIG, FINAL_CONFIGS, OUT, PREPARED, PREPARED_TRACKED, sha
from cold_v4_type_audit import COLS, class_masks, unique_spectra

LOCK = Path("configs/cold_v4_deep_policy.json")
RESULT = Path("data/cold_v4_type/boundary_v2.json")
EXTRA = ["mg", "gaia_bp_rp", "p_corr", "gaia_parallax_error", "ebv",
         "grz_fraction_clean", "grz_mask_clean", "snr_w1", "snr_w2"]
READ = sorted(set(COLS + EXTRA))


def finite(value):
    number = float(value)
    return number if np.isfinite(number) else None


def failed_edges(gr, rz, low, high):
    result = []
    for name, value, lower, upper in zip(("gr0", "rz0"), (gr, rz), low, high):
        if not np.isfinite(value):
            result.append(f"{name}_not_finite")
        elif value < lower:
            result.append(f"{name}_below_low")
        elif value > upper:
            result.append(f"{name}_above_high")
    return result


def support_ratios(table, route, index):
    result = {}
    for tier in LEVEL:
        colors = feature_matrix(table, tier)[index:index+1]
        if not np.isfinite(colors).all():
            result[tier] = None
            continue
        model = route["tiers"][tier]
        distance = model["tree"].query(colors / model["scale"],
                                         k=route["neighbors"])[0][0, -1]
        result[tier] = finite(distance / model["radius"])
    return result


def object_record(table, i, route, routed, chosen, calibrated_feh, feh_valid):
    gr, rz = float(table["gr0"][i]), float(table["rz0"][i])
    return {"source_id": int(table["source_id"][i]),
            "ls_id": int(table["ls_id"][i]),
            "desi_object_id": str(table["desi_object_id"][i].decode(errors="replace")
                if isinstance(table["desi_object_id"][i], bytes)
                else table["desi_object_id"][i]).strip(),
            "ra_deg": finite(table["ra"][i]), "dec_deg": finite(table["dec"][i]),
            "r": finite(table["r"][i]), "ebv": finite(table["ebv"][i]),
            "gr0": finite(gr), "rz0": finite(rz),
            "zw10": finite(table["zw10"][i]),
            "w1w20": finite(table["w1w20"][i]),
            "legacy_mg_from_gaia_anchor": finite(table["mg"][i]),
            "gaia_bp_rp": finite(table["gaia_bp_rp"][i]),
            "corrected_parallax_mas": finite(table["p_corr"][i]),
            "parallax_error_mas": finite(table["gaia_parallax_error"][i]),
            "desi_snr": finite(table["desi_snr"][i]),
            "desi_teff_k": finite(table["desi_teff"][i]),
            "desi_logg": finite(table["desi_logg"][i]),
            "desi_feh_raw": finite(table["desi_feh"][i]),
            "desi_feh_error": finite(table["desi_feh_err"][i]),
            "desi_feh_calibrated": finite(calibrated_feh[i]) if feh_valid[i] else None,
            "desi_feh_calibration_domain": bool(feh_valid[i]),
            "desi_quality": bool(table["desi_quality"][i]),
            "desi_match_ambiguous": bool(table["desi_match_ambiguous"][i]),
            "grz_clean": bool(table["grz_clean"][i]),
            "grz_fraction_clean": bool(table["grz_fraction_clean"][i]),
            "grz_mask_clean": bool(table["grz_mask_clean"][i]),
            "grz_high_quality": bool(table["grz_high_quality"][i]),
            "sigma_g": finite(table["sigma_g"][i]),
            "sigma_r": finite(table["sigma_r"][i]),
            "sigma_z": finite(table["sigma_z"][i]),
            "snr_w1": finite(table["snr_w1"][i]),
            "snr_w2": finite(table["snr_w2"][i]),
            "color_box": bool(routed["color_box"][i]),
            "failed_optical_edges": failed_edges(gr, rz, route["optical_low"],
                                                   route["optical_high"]),
            "support_radius_ratio": support_ratios(table, route, i),
            "baseline_tier": int(routed["tier"][i]),
            "wise20_tier": int(chosen[i])}


def field_result(config, route, policy):
    task = json.loads(Path(config).read_text())
    raw = fitsio.read(Path(task["output_dir"]) / "field_diagnostics.fits",
                      columns=READ)
    table = {name: raw[name] for name in raw.dtype.names}
    unique, dedup = unique_spectra(table)
    labels = class_masks(table, unique & table["grz_base"])
    routed = route_rows(table, route)
    chosen = choose_policy(table, routed, policy)
    feh, valid = rvs_metallicity(table["desi_feh"], table["desi_teff"],
                                 table["desi_logg"])
    poor = labels["cool_dwarf"] & valid & np.isfinite(table["desi_feh_err"]) & (
        table["desi_feh_err"] > 0) & (feh < -1.5)
    giant_selected = labels["cool_giant"] & (chosen > 0)
    poor_box_rejected = poor & ~routed["color_box"]
    rng = np.random.default_rng(20261028)
    background = {}
    for name, mask in (("cold_dwarf", labels["cool_dwarf"]),
                       ("cold_giant", labels["cool_giant"])):
        ids = np.flatnonzero(mask)
        if len(ids) > 600:
            ids = np.sort(rng.choice(ids, 600, replace=False))
        background[name] = [{"gr0": finite(table["gr0"][i]),
                             "rz0": finite(table["rz0"][i]),
                             "mg": finite(table["mg"][i])} for i in ids]
    return {"field": task["name"], "deduplication": dedup,
            "qualified_cold_giants": int(labels["cool_giant"].sum()),
            "selected_cold_giants": int(giant_selected.sum()),
            "poor_cold_dwarfs_in_feh_domain": int(poor.sum()),
            "poor_box_rejected": int(poor_box_rejected.sum()),
            "accepted_cold_giant_objects": [object_record(table, i, route, routed,
                chosen, feh, valid) for i in np.flatnonzero(giant_selected)],
            "optical_box_rejected_poor_dwarf_objects": [object_record(
                table, i, route, routed, chosen, feh, valid)
                for i in np.flatnonzero(poor_box_rejected)],
            "background_sample": background}


def main():
    require_compute()
    if RESULT.exists():
        raise FileExistsError(RESULT)
    if sha(PREPARED) != sha(PREPARED_TRACKED):
        raise ValueError("Frozen preparation changed")
    lock = json.loads(LOCK.read_text())
    if lock["prepared_sha256"] != sha(PREPARED):
        raise ValueError("WISE20 policy differs from frozen model")
    prepared = json.loads(PREPARED.read_text())
    route_path = OUT / "route.joblib"
    if sha(route_path) != prepared["route"]["route_sha256"]:
        raise ValueError("Frozen route changed")
    route = joblib.load(route_path)
    result = {"prepared_sha256": sha(PREPARED),
              "policy_lock_sha256": sha(LOCK), "compute_host": socket.gethostname(),
              "label_scope": "unique quality-qualified DESI spectra in grz_base; class-conditional inspection only",
              "optical_box_low": route["optical_low"].tolist(),
              "optical_box_high": route["optical_high"].tolist(), "fields": {}}
    for config in (CAL_CONFIG, *FINAL_CONFIGS):
        field = field_result(config, route, lock["chosen_policy"])
        result["fields"][field.pop("field")] = field
        print(json.dumps({"field": json.loads(Path(config).read_text())["name"],
                          "giants_selected": field["selected_cold_giants"],
                          "poor_box_rejected": field["poor_box_rejected"]}), flush=True)
    write_json(RESULT, result)


if __name__ == "__main__":
    main()
