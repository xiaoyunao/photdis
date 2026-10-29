#!/usr/bin/env python3
"""Version joint evidence after an independent band-level outlier audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".deps")]
import fitsio
import numpy as np

from photdis.stage1 import require_compute, write_json
from cold_v4 import sha

AUDIT = Path("manifests/blue_outlier_source_3601173023628559616.json")


def audit_decision(audit):
    delta = audit["legacy_minus_ps1_decam"]
    raw = audit["sweep_row"]
    same_identity = raw["ref_id"] == audit["gaia_source_id"]
    independent_bad_r = (abs(delta["r"]) > 0.3 and abs(delta["g"]) < 0.05
                         and abs(delta["z"]) < 0.05 and raw["rchisq_r"] > 1000)
    if not same_identity or not independent_bad_r or audit["ps1"]["matched_within_1_arcsec"] != 1:
        raise ValueError("The registered independent photometry evidence is not reproduced")
    return "r_band_discordant"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", choices=("v1", "targeted_v1"), required=True)
    args = parser.parse_args()
    require_compute()
    source = Path(f"data/joint_calibration_{args.sample}_evidence_v2/same_star.fits")
    output = Path(f"data/joint_calibration_{args.sample}_evidence_v3/same_star.fits")
    manifest = Path(f"manifests/joint_calibration_{args.sample}_evidence_v3.json")
    if output.exists() or manifest.exists():
        raise FileExistsError("Versioned joint evidence already exists")
    audit = json.loads(AUDIT.read_text())
    decision = audit_decision(audit)
    rows = fitsio.read(str(source))
    changed = rows["source_id"] == audit["gaia_source_id"]
    result = np.empty(len(rows), dtype=rows.dtype.descr + [("photometry_audit_status", "S24")])
    for name in rows.dtype.names:
        result[name] = rows[name]
    result["photometry_audit_status"] = b"none"
    result["photometry_audit_status"][changed] = decision.encode()
    result["trusted_blue_core"][changed] = False
    output.parent.mkdir(parents=True, exist_ok=True)
    fitsio.write(str(output), result, extname="JOINT", clobber=False)
    entry = {"sample": args.sample, "source": str(source), "source_sha256": sha(source),
             "audit": str(AUDIT), "audit_sha256": sha(AUDIT),
             "output": str(output), "output_sha256": sha(output),
             "rows": len(result), "independently_flagged": int(changed.sum()),
             "trusted_blue_before": int(rows["trusted_blue_core"].sum()),
             "trusted_blue_after": int(result["trusted_blue_core"].sum()),
             "compute_host": socket.gethostname()}
    write_json(manifest, entry)
    print(json.dumps(entry), flush=True)


if __name__ == "__main__":
    main()
