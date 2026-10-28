#!/usr/bin/env python3
"""Project spectroscopy auxiliary pools on an allocated NERSC compute node."""
import argparse
import json
from pathlib import Path
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".deps")]
from photdis.spectroscopy import build_spectroscopy
from photdis.stage1 import require_compute, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage1.json")
    args = parser.parse_args()
    require_compute()
    config = json.loads(Path(args.config).read_text())
    output = Path(config["output_dir"]) / "spectroscopy"
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Spectroscopy output exists; inspect it or choose a fresh run directory")
    start = time.time()
    index = build_spectroscopy(config, output)
    index.update(compute_host=socket.gethostname(), elapsed_seconds=time.time() - start)
    write_json(output / "index.json", index)
    print(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
