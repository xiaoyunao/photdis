#!/usr/bin/env python3
"""Run the first-stage sample scan/finalization on allocated NERSC CPU resources."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / ".deps"))
from photdis.stage1 import scan, finalize

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("action", choices=("scan", "finalize"))
parser.add_argument("--config", default="configs/stage1.json")
parser.add_argument("--workers", type=int, default=16)
parser.add_argument("--limit", type=int, help="Explicit small I/O smoke only; recorded in manifest")
args = parser.parse_args()
config = json.loads(Path(args.config).read_text())
if args.action == "scan":
    scan(config, args.workers, args.limit)
else:
    finalize(config)
