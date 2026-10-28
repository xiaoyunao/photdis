#!/usr/bin/env python3
"""Score the preregistered supplemental development field without refitting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".deps")]

from photdis.holdouts import load_reference_catalogs
from photdis.stage1 import require_compute, write_json
from cold_v4 import PREPARED, sha
from cold_v4_policy_noise import frozen_models
from cold_v4_native_transfer import field_result, LOCK, REGISTRATION, OUT

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", default="configs/cold_v4_native_transfer_supplement.json")
    parser.add_argument("--output", default="supplemental_development")
    args = parser.parse_args()
    require_compute()
    supplement = Path(args.registration)
    destination = OUT / f"{args.output}.json"
    if destination.exists():
        raise FileExistsError(destination)
    config = json.loads(supplement.read_text())
    parent = json.loads(REGISTRATION.read_text())
    if config["parent_registration_sha256"] != sha(REGISTRATION):
        raise ValueError("Supplement does not match frozen field registration")
    if parent["prepared_sha256"] != sha(PREPARED) or (
            parent["policy_lock_sha256"] != sha(LOCK)):
        raise ValueError("Frozen model or policy differs from field registration")
    policy = json.loads(LOCK.read_text())["chosen_policy"]
    route, forests = frozen_models()
    references = load_reference_catalogs("data/reference")
    fields = [field_result(item["field_config"], references, route, forests,
                           policy, 20261120 + index)
              for index, item in enumerate(config["supplemental_development_fields"])]
    result = {"role": args.output, "registration_sha256": sha(supplement),
              "parent_registration_sha256": sha(REGISTRATION),
              "prepared_sha256": sha(PREPARED), "policy_lock_sha256": sha(LOCK),
              "compute_host": socket.gethostname(), "frozen_policy": policy,
              "fields": fields}
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(destination, result)


if __name__ == "__main__":
    main()
