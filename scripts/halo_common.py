"""Version-bound inputs and outputs for the halo response experiments."""
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_6_vac_distances import read_columns
from v3_10_figures import check_templates
from v3_8_direct_figures import BLUE, ORANGE, GREEN, plt, save
from paper_v1_figures import PAPER

sys.path.insert(0, str(Path(__file__).parent))
CONFIG = ROOT / 'configs/halo_response_20261118.json'
C = json.loads(CONFIG.read_text())
OUT = ROOT / 'data/halo_response_20261118'
MU = np.round(np.arange(14, 20.001, .1), 1)
DISTANCE = 10 ** ((MU - 10) / 5)
RNG = np.random.default_rng(C['seed'])


def folder(stage):
    result = OUT / stage
    result.mkdir(parents=True, exist_ok=True)
    return result


def complete(path, inputs, **metadata):
    save_json(path / 'complete.json', dict(
        job=os.getenv('SLURM_JOB_ID'), config=sha(CONFIG), inputs=inputs,
        code={p.name:sha(p) for p in Path(__file__).parent.glob('halo_*.py')},
        style=check_templates(), **metadata,
        outputs={p.name:sha(p) for p in path.iterdir() if p.is_file() and p.name!='complete.json'}))


def read(path, fields, inputs):
    record = json.loads((path / 'complete.json').read_text())
    inputs[str(path.relative_to(ROOT))] = sha(path / 'complete.json')
    name, *columns = fields
    return read_columns(verified(path / name, record['outputs'][name]), columns)


def figure(path, name, fig, rows, arrays=None):
    save(fig, path, name, rows, arrays)
