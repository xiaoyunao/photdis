"""Independent all-sky display and native-ridge science; frozen v5 inputs."""
import csv,json,os,sys
from pathlib import Path
import numpy as np
from halo_common import ROOT,sha,verified,read_columns,save_json,csv_rows,PAPER,plt
from stream_common import npz,read,figure
from v3_10_figures import check_templates

CONFIG=ROOT/'configs/sky_sequence_20261119.json'
C=json.loads(CONFIG.read_text());CH=sha(CONFIG)
OUT=ROOT/'data/sky_sequence_20261119'
STREAM=ROOT/'data/stream_response_20261119'
HALO=ROOT/'data/halo_response_20261118'


def folder(name):
    p=OUT/name;p.mkdir(parents=True,exist_ok=True);return p


def complete(p,inputs,**extra):
    save_json(p/'complete.json',dict(config=CH,job=os.getenv('SLURM_JOB_ID'),inputs=inputs,
        code={q.name:sha(q) for q in Path(__file__).parent.glob('sky6_*.py')},style=check_templates(),**extra,
        outputs={q.name:sha(q) for q in p.iterdir() if q.is_file() and q.name!='complete.json'}))


def stages():return ('inventory','ridge_inputs','identities','counts','area/south','area/north','atlas','rgb','old_domain','coverage')
