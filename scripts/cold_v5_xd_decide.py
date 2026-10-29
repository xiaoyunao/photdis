#!/usr/bin/env python3
"""Apply the preregistered development gate to the one selected XD model."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".deps")]
from photdis.stage1 import write_json
from cold_v4 import sha

CONFIG = Path("configs/cold_v5_xd_experiment.json")
SELECTION = Path("manifests/cold_v5_xd_selection.json")
COMPARISON = Path("manifests/cold_v5_xd_comparison.json")
OUTPUT = Path("manifests/cold_v5_xd_decision.json")


def evaluate_field(field, gate):
    """Return each locked inequality on the same accepted source set."""
    forest, candidate = field["qrf"], field["xd"]
    if forest["n"] != candidate["n"] or forest["n"] != field["n_same_selected"]:
        raise ValueError("v4 and XD counts do not match")
    checks = {
        "center": abs(candidate["center_mag"]) - abs(forest["center_mag"])
                  <= gate["absolute_center_noninferiority_mag"],
        "nmad": candidate["nmad_mag"] - forest["nmad_mag"]
                <= gate["point_nmad_noninferiority_mag"],
        "tail": candidate["abs_gt_0p5"] - forest["abs_gt_0p5"]
                <= gate["tail_abs_residual_gt_0p5_noninferiority_fraction"],
    }
    retained = 1 - gate["median_g_and_r_perturbation_abs_response_reduction_fraction"]
    for band in ("g", "r"):
        response = field["perturbation_plus_0p01_mag"][band]
        checks[f"response_{band}"] = (
            response["xd"]["median_abs_mag"] <=
            retained * response["qrf"]["median_abs_mag"])
    return {"n": int(forest["n"]), "checks": checks,
            "passes": bool(all(checks.values()))}


def decide(config, selection, comparison, names):
    if comparison["components"] != selection["selected_components"]:
        raise ValueError("Comparison did not use the selected model")
    if comparison["configuration_sha256"] != selection["configuration_sha256"]:
        raise ValueError("Configuration differs between selection and comparison")
    fields = {name: evaluate_field(comparison["fields"][name],
                                  config["candidate_gate"]) for name in names}
    passed = bool(all(row["passes"] for row in fields.values()))
    return {"decision": "lock_candidate_for_independent_validation" if passed
                        else "retain_frozen_v4",
            "selected_components": int(selection["selected_components"]),
            "development_fields": fields,
            "gate": config["candidate_gate"]}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    config = json.loads(CONFIG.read_text())
    selection = json.loads(SELECTION.read_text())
    comparison = json.loads(COMPARISON.read_text())
    if selection["configuration_sha256"] != sha(CONFIG):
        raise ValueError("Registered configuration changed")
    if comparison["selection_manifest_sha256"] != sha(SELECTION):
        raise ValueError("Comparison used another size selection")
    names = [json.loads(Path(path).read_text())["name"]
             for path in config["configuration_development_fields"]]
    result = decide(config, selection, comparison, names)
    result.update({"configuration_sha256": sha(CONFIG),
                   "selection_manifest_sha256": sha(SELECTION),
                   "comparison_manifest_sha256": sha(COMPARISON)})
    write_json(OUTPUT, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
