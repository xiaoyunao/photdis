"""Frozen two-parameter native luminosity experiment, separate from production."""
from seq7_common import ROOT,sha,verified,read_columns,save_json,csv_rows,PAPER,plt,STREAM,HALO
from stream_common import npz,read,figure
from pathlib import Path
import numpy as np
import json,csv,os,sys
from v3_10_figures import check_templates

CONFIG=ROOT/'configs/native_luminosity_20261119.json';C=json.loads(CONFIG.read_text());CH=sha(CONFIG)
OUT=ROOT/'data/native_luminosity_20261119';OLD=ROOT/'data/sequence_mechanisms_20261119'


def load(p):
    with np.load(p) as f:return dict(f)


def folder(stage):
    p=OUT/stage;p.mkdir(parents=True,exist_ok=True);return p


def complete(d,inputs,**meta):
    save_json(d/'complete.json',dict(config=CH,job=os.getenv('SLURM_JOB_ID'),inputs=inputs,code={p.name:sha(p) for p in Path(__file__).parent.glob('nat8_*.py')},style=check_templates(),**meta,outputs={p.name:sha(p) for p in d.iterdir() if p.is_file() and p.name!='complete.json'}))


def old(stage,name,inputs):
    marker=OLD/stage/'complete.json';r=json.loads(marker.read_text());inputs[str(marker.relative_to(ROOT))]=sha(marker);p=verified(OLD/stage/name,r['outputs'][name]);return load(p)


def parameter_map(means,x,a):return means-a[0]-a[1]*x
