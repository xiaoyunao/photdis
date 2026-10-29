#!/usr/bin/env python3
"""Export frozen-v4 photometric candidates with per-source route diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".deps")]
import fitsio
import joblib
import numpy as np

from photdis.class_evidence import desi_class_evidence
from photdis.route import FEATURES, LAYER, LEVEL, feature_matrix, route_rows
from photdis.route_audit import higher_tier_reasons
from photdis.route_policy import choose_policy
from photdis.stage1 import require_compute, write_json
from cold_v4 import OUT, PREPARED, PREPARED_TRACKED, sha

LOCK = Path("configs/cold_v4_deep_policy.json")
FIELDS = ("configs/field_cold_v4_ra000_010.json",
          "configs/field_cold_v4_ra150_160.json",
          "configs/field_cold_v4_ra220_230.json")
COLUMNS = sorted(set(("ls_id", "source_id", "ra", "dec", "g0", "r", "ebv",
                      "grz_clean", "grz_high_quality", "grz_w1_high_quality",
                      "full_high_quality", "valid_w1", "valid_w2",
                      "w1_clean", "w2_clean", "snr_w1", "snr_w2",
                      "desi_match", "desi_match_ambiguous", "desi_snr",
                      "desi_quality", "desi_quality_reason", "desi_is_stellar",
                      "desi_known_contaminant", "desi_teff", "desi_logg",
                      "desi_logg_err",
                      *LAYER.values(),
                      *{name for names in FEATURES.values() for name in names})))
QUALITY = {"A_grz": "grz_high_quality", "B_grzW1": "grz_w1_high_quality",
           "C_grzW1W2": "full_high_quality"}


def support_ratios(table, route, selected):
    """Return each tier's actual tenth-neighbor distance divided by its limit."""
    result = {}
    for tier in FEATURES:
        ratio = np.full(len(selected), np.nan, np.float32)
        model = route["tiers"][tier]
        colors = feature_matrix(table, tier)
        use = selected & np.isfinite(colors).all(axis=1)
        if use.any():
            distance = model["tree"].query(colors[use] / model["scale"],
                    k=route["neighbors"])[0][:, -1]
            ratio[use] = distance / model["radius"]
        result[tier] = ratio
    return result


def output_rows(table, route, policy, forests):
    routed = route_rows(table, route)
    chosen = choose_policy(table, routed, policy)
    use = chosen > 0
    if not use.any():
        return None
    reasons = higher_tier_reasons(table, routed, chosen)
    evidence = desi_class_evidence(table)
    ratios = support_ratios(table, route, use)
    selected = np.flatnonzero(use)
    dtype = [("ls_id", "i8"), ("source_id", "i8"), ("ra", "f8"),
             ("dec", "f8"), ("g0", "f4"), ("r", "f4"), ("ebv", "f4"),
             ("tier", "u1"), ("reason_b", "S24"), ("reason_c", "S24"),
             ("support_ratio_a", "f4"), ("support_ratio_b", "f4"),
             ("support_ratio_c", "f4"), ("infrared_fallback_conflict", "?"),
             ("used_tier_high_quality", "?"),
             ("cold_dwarf_hypothesis", "?"),
             ("class_evidence_available", "?"),
             ("class_evidence", "S24"),
             ("class_evidence_source", "S12"),
             ("class_evidence_quality", "S28"),
             ("class_conflicts_cold_dwarf", "?"),
             ("depth_state", "S24"), ("mg_median", "f4"),
             ("mu_median", "f4"), ("distance_pc_median", "f4")]
    out = np.empty(len(selected), dtype=dtype)
    for name in ("ls_id", "source_id", "ra", "dec", "g0", "r", "ebv"):
        out[name] = table[name][selected]
    out["tier"] = chosen[selected]
    out["reason_b"] = np.char.encode(reasons["B"][selected], "ascii")
    out["reason_c"] = np.char.encode(reasons["C"][selected], "ascii")
    for tier, suffix in (("A_grz", "a"), ("B_grzW1", "b"),
                         ("C_grzW1W2", "c")):
        out[f"support_ratio_{suffix}"] = ratios[tier][selected]
    out["infrared_fallback_conflict"] = reasons["higher_support_conflict"][selected]
    out["cold_dwarf_hypothesis"] = True
    for name, values in evidence.items():
        out[name] = values[selected]
    out["used_tier_high_quality"] = False
    out["mg_median"] = np.nan
    for tier, level in LEVEL.items():
        local = out["tier"] == level
        if local.any():
            out["used_tier_high_quality"][local] = table[QUALITY[tier]][selected[local]]
            out["mg_median"][local] = forests[tier].predict(
                feature_matrix(table, tier)[selected[local]])[:, 2]
    out["mu_median"] = out["g0"] - out["mg_median"]
    out["distance_pc_median"] = 10 ** ((out["mu_median"] + 5) / 5)
    out["depth_state"] = b"r_lt_18"
    out["depth_state"][out["r"] >= 18] = b"r18_22_stress_tested"
    out["depth_state"][out["r"] >= 22] = b"r_ge_22_unvalidated"
    if not np.isfinite(out["distance_pc_median"]).all():
        raise ValueError("Nonfinite candidate median distance")
    if np.any(out["infrared_fallback_conflict"] & (out["tier"] == 3)):
        raise ValueError("C-tier source has higher-tier fallback conflict")
    return out


def append_rows(path, rows):
    """Append to the single catalog table, never create another FITS HDU."""
    if not path.exists():
        fitsio.write(str(path), rows)
    else:
        with fitsio.FITS(str(path), "rw") as catalog:
            if len(catalog) != 2:
                raise ValueError("Candidate catalog has multiple table extensions")
            catalog[1].append(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", required=True, choices=[
        json.loads(Path(path).read_text())["name"] for path in FIELDS])
    parser.add_argument("--chunk", type=int, default=100000)
    args = parser.parse_args()
    require_compute()
    if args.chunk < 1:
        parser.error("--chunk must be positive")
    if sha(PREPARED) != sha(PREPARED_TRACKED):
        raise ValueError("Frozen v4 preparation changed")
    lock = json.loads(LOCK.read_text())
    if lock["chosen_policy"] != "clean_wise_snr_20" or lock["prepared_sha256"] != sha(PREPARED):
        raise ValueError("Locked WISE20 policy changed")
    prepared = json.loads(PREPARED.read_text())
    route_path = OUT / "route.joblib"
    if sha(route_path) != prepared["route"]["route_sha256"]:
        raise ValueError("Frozen route changed")
    route = joblib.load(route_path)
    forests = {}
    for tier in FEATURES:
        model_path = OUT / f"{tier}.joblib"
        if sha(model_path) != prepared["tier_training"][tier]["model_sha256"]:
            raise ValueError(f"Frozen forest changed: {tier}")
        forests[tier] = joblib.load(model_path)
    config_path = next(path for path in FIELDS if
                       json.loads(Path(path).read_text())["name"] == args.field)
    field = json.loads(Path(config_path).read_text())
    source = Path(field["output_dir"]) / "field_diagnostics.fits"
    target = Path("data/cold_v4_candidates_v2") / f"{args.field}.fits"
    manifest_path = Path("manifests") / f"cold_v4_candidates_v2_{args.field}.json"
    if target.exists() or manifest_path.exists():
        raise FileExistsError("Candidate export already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    counts = {tier: 0 for tier in FEATURES}
    total = conflict = 0
    evidence_counts = {name: 0 for name in ("unknown", "other_star_or_transition",
        "cool_main_sequence", "cool_giant", "nonstellar")}
    with fitsio.FITS(str(source)) as file:
        nrows = file[1].get_nrows()
        for start in range(0, nrows, args.chunk):
            raw = file[1].read(columns=COLUMNS,
                               rows=np.arange(start, min(start + args.chunk, nrows)))
            table = {name: raw[name] for name in raw.dtype.names}
            out = output_rows(table, route, lock["chosen_policy"], forests)
            if out is None:
                continue
            append_rows(target, out)
            total += len(out)
            conflict += int(out["infrared_fallback_conflict"].sum())
            for name in evidence_counts:
                evidence_counts[name] += int(np.sum(out["class_evidence"] == name.encode()))
            for tier, level in LEVEL.items():
                counts[tier] += int(np.sum(out["tier"] == level))
            print(json.dumps({"field": args.field, "read_to": start+len(raw),
                              "selected_so_far": total}), flush=True)
    if sum(counts.values()) != total:
        raise ValueError("Tier counts do not sum to candidate count")
    if sum(evidence_counts.values()) != total:
        raise ValueError("Class-evidence counts do not sum to candidate count")
    if not total:
        raise ValueError("No candidates were exported")
    with fitsio.FITS(str(target)) as catalog:
        if len(catalog) != 2 or catalog[1].get_nrows() != total:
            raise ValueError("Candidate FITS table row count differs from manifest")
    manifest = {"field": args.field, "source": str(source),
                "output": str(target), "output_sha256": sha(target),
                "input_rows": nrows, "selected_rows": total,
                "tier_counts": counts, "infrared_fallback_conflict": conflict,
                "class_evidence_counts": evidence_counts,
                "policy_lock_sha256": sha(LOCK),
                "prepared_sha256": sha(PREPARED),
                "distance_semantics": "median conditional on cold-dwarf hypothesis; no calibrated type probability or released deep interval",
                "depth_state_semantics": "measured r bin only; r18_22 has diagnostic stress tests with undercoverage, r>=22 is beyond them",
                "compute_host": socket.gethostname()}
    write_json(manifest_path, manifest)
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
