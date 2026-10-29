#!/usr/bin/env python3
"""Select one XD size from registered held-out development blocks."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".deps")]
from photdis.stage1 import write_json
from cold_v4 import sha

CONFIG = Path("configs/cold_v5_xd_experiment.json")
OUTPUT = Path("manifests/cold_v5_xd_selection.json")


def choose(config, manifests, names):
    scores = {}
    for size in config["components"]:
        fit = manifests[size]
        if fit["configuration_sha256"] != sha(CONFIG):
            raise ValueError(f"Configuration differs for size {size}")
        if not all(entry["converged"] for entry in fit["tiers"].values()):
            raise ValueError(f"XD optimization unfinished for size {size}")
        fields = {}
        for name in names:
            weighted = count = 0
            for tier, entry in fit["tiers"].items():
                row = entry["development"][name]
                if row["n"]:
                    weighted += row["n"] * row["mean_negative_conditional_log_density"]
                    count += row["n"]
            if not count:
                raise ValueError(f"No development predictions in {name}")
            fields[name] = {"n": count, "mean_negative_conditional_log_density":
                            weighted/count}
        scores[size] = {"equal_field_mean_nll": sum(
            row["mean_negative_conditional_log_density"] for row in fields.values())/len(fields),
            "fields": fields}
    chosen = min(scores, key=lambda size: (scores[size]["equal_field_mean_nll"], size))
    return chosen, scores


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    config = json.loads(CONFIG.read_text())
    names = [json.loads(Path(path).read_text())["name"]
             for path in config["configuration_development_fields"]]
    manifests = {size: json.loads((Path("manifests") /
        f"cold_v5_xd_k{size:02d}.json").read_text()) for size in config["components"]}
    chosen, scores = choose(config, manifests, names)
    result = {"configuration_sha256": sha(CONFIG),
              "selection_rule": config["configuration_score"],
              "selected_components": chosen,
              "scores": {str(size): row for size, row in scores.items()},
              "fit_manifest_sha256": {str(size): sha(Path("manifests") /
                  f"cold_v5_xd_k{size:02d}.json") for size in config["components"]}}
    write_json(OUTPUT, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
