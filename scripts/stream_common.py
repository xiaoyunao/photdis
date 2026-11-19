"""Frozen inputs and independent outputs for the19November stream experiments."""
import csv,json,os,sys
from pathlib import Path
import numpy as np
from halo_common import ROOT,sha,verified,read_columns,save_json,csv_rows,PAPER,plt
from v3_10_figures import check_templates
sys.path.insert(0,str(Path(__file__).parent))
CONFIG=ROOT/'configs/stream_response_20261119.json'
C=json.loads(CONFIG.read_text());CONFIG_SHA=sha(CONFIG)
OUT=ROOT/'data/stream_response_20261119';OLD=ROOT/'data/halo_response_20261118'
RNG=np.random.default_rng(C['seed'])
CODE={p.name:sha(p) for p in Path(__file__).parent.glob('stream_*.py')}


def folder(name):
    p=OUT/name;p.mkdir(parents=True,exist_ok=True);return p


def npz(path,**data):
    with path.with_suffix('.tmp').open('wb') as f:np.savez_compressed(f,**data)
    path.with_suffix('.tmp').replace(path)


def read(path,fields,inputs):
    r=json.loads((path/'complete.json').read_text());inputs[str(path.relative_to(ROOT))]=sha(path/'complete.json')
    name,*cols=fields;return read_columns(verified(path/name,r['outputs'][name]),cols)


def complete(path,inputs,**metadata):
    save_json(path/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),config=CONFIG_SHA,code=CODE,inputs=inputs,
       style=check_templates(),**metadata,outputs={p.name:sha(p) for p in path.iterdir() if p.is_file() and p.name!='complete.json'}))


def figure(d,name,f,rows,arrays=None):
    for ext in ('pdf','png'):f.savefig(d/(name+'.'+ext),dpi=300)
    plt.close(f);csv_rows(d/(name+'.csv'),rows)
    if arrays:npz(d/(name+'.npz'),**arrays)


def track(name):
    from astropy.table import Table
    filename={'Pal5.ibata2024':'track.st.Pal5.ibata2024.ecsv','Sagittarius.antoja2020':'track.st.Sagittarius.antoja2020.ecsv'}[name]
    p=ROOT/'configs/stream_inputs'/filename
    a=Table.read(p);return {k:np.asarray(a[k]) for k in a.colnames},p


def profile(name,ra,dec):
    a,_=track(C['local'][name]['track']);lo,hi=C['local'][name]['ra_range']
    # Restrict wrap-separated source track before interpolating sky geometry.
    use=(a['ra']>=lo-3)&(a['ra']<=hi+3)
    order=np.argsort(a['ra'][use]);r=a['ra'][use][order];d=a['dec'][use][order]
    slope=np.gradient(d,r);at=np.interp(ra,r,slope)
    offset=(dec-np.interp(ra,r,d))/np.sqrt(1+(at/np.cos(np.deg2rad(dec)))**2)
    along=(ra-lo)/(hi-lo)
    return offset,along
