"""Fixed five-control sequence experiment, separate from all frozen products."""
from sky6_common import ROOT,sha,verified,read_columns,save_json,csv_rows,PAPER,plt,STREAM,HALO
from stream_common import npz,read,figure
from pathlib import Path
import json,csv,os,sys
import numpy as np
from v3_10_figures import check_templates

CONFIG=ROOT/'configs/sequence_mechanisms_20261119.json';C=json.loads(CONFIG.read_text());CH=sha(CONFIG)
OUT=ROOT/'data/sequence_mechanisms_20261119';OLD=ROOT/'data/sky_sequence_20261119'
METHODS=('A','B','C','D','E','QRF')


def folder(stage):
    p=OUT/stage;p.mkdir(parents=True,exist_ok=True);return p


def complete(d,inputs,**meta):
    save_json(d/'complete.json',dict(config=CH,job=os.getenv('SLURM_JOB_ID'),inputs=inputs,code={p.name:sha(p) for p in Path(__file__).parent.glob('seq7_*.py')},style=check_templates(),**meta,outputs={p.name:sha(p) for p in d.iterdir() if p.is_file() and p.name!='complete.json'}))
