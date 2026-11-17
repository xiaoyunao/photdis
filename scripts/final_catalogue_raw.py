#!/usr/bin/env python3
"""Refine the frozen native exclusion ledger using only geometry, reddening, and r."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

from astropy.io import fits
import numpy as np

from photdis.stage1 import galactic, require_compute
from v2_1_near import ROOT, csv_rows, save_json, sha
from v3_0_native import science_domain
from v3_3_sed import verified
from v3_9_catalog import CONFIG, LEDGER, OUT, settings

DEST = ROOT/'data/final_catalogue'
GROUPS = 48
BAD = ('latitude','extinction','magnitude')


def read_raw(path,n):
    with fits.open(path,memmap=True) as h:
        a=h[1]
        dt=a.data.dtype
        start=a.fileinfo()['datLoc']
        assert a.header['NAXIS1']==dt.itemsize and a.header['NAXIS2']==n
        for k in ('RA','DEC','EBV','FLUX_R'):
            i=dt.names.index(k)+1
            if a.header.get(f'TSCAL{i}',1)!=1 or a.header.get(f'TZERO{i}',0)!=0:
                raise ValueError(f'Unrecognized scaling in {path} for {k}')
    return np.memmap(path,dtype=dt,mode='r',offset=start,shape=(n,))


def outside_details(source,side,cfg):
    path=Path(cfg['sweep_root'])/source['filename']
    st=path.stat()
    assert (st.st_size,st.st_mtime_ns)==(source['size'],source['mtime_ns'])
    folder=OUT/side/'photo'/path.stem
    marker=folder/'complete.json'
    record=json.loads(marker.read_text())
    file='parent_state.npz'
    with np.load(verified(folder/file,record['outputs'][file]),allow_pickle=False) as z:
        state=z['state']
    assert len(state)==source['n_rows']==record['rows']
    exact=np.bincount(state,minlength=len(LEDGER))
    assert np.array_equal(exact,[record['counts'][s] for s in LEDGER])
    a=read_raw(path,len(state))
    reasons=np.zeros(8,'i8')
    legacy=0
    admitted=0
    for i in range(0,len(a),200000):
        stop=min(i+200000,len(a))
        y=state[i:stop]
        x=a[i:stop]
        use=y>=4
        _,b=galactic(x['RA'][use],x['DEC'][use])
        ebv=x['EBV'][use]
        with np.errstate(divide='ignore',invalid='ignore'):
            r=(22.5-2.5*np.log10(x['FLUX_R'][use].astype('f8'))).astype('f4')
        hi=abs(b)>30
        valid_b=np.isfinite(b)
        lat=valid_b&(hi if side=='north' else True)
        dust=np.isfinite(ebv)&(ebv>=0)&(ebv<.03)
        mag=np.isfinite(r)&(r<np.where((side=='south')&~hi,20.,23.))
        old=(side=='south')&valid_b&hi&dust&np.isfinite(r)&(r<20)
        ok=np.logical_or.reduce([science_domain(r,b,ebv,**v) for v in cfg['domains'].values()])
        np.testing.assert_array_equal(ok,y[y>=4]>=5)
        assert np.array_equal(ok,(lat&dust&mag)&~old)
        bits=(~lat).astype('u1')+2*(~dust).astype('u1')+4*(~mag).astype('u1')
        z=bits[~ok&~old]
        assert np.all(z>0)
        reasons+=np.bincount(z,minlength=8)
        admitted+=int(ok.sum())
        legacy+=int(old.sum())
    assert reasons.sum()+legacy==exact[4] and admitted==exact[5:].sum()
    values=dict(zip(LEDGER,map(int,exact)))
    union={f'fail_mask_{i}':int(n) for i,n in enumerate(reasons)}
    return dict(side=side,sweep=path.name,raw=len(state),**values,legacy_bright_region=legacy,**union)


def main():
    require_compute()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('side',choices=('south','north'))
    a=p.parse_args()
    side=a.side
    k=int(os.environ['SLURM_ARRAY_TASK_ID'])
    marker=OUT/side/'inventory.json'
    inv=json.loads(marker.read_text())
    assert len(inv['groups'])==GROUPS
    cfg=settings()[side]
    out=DEST/'raw'/side
    out.mkdir(exist_ok=True,parents=True)
    with ThreadPoolExecutor(max_workers=int(os.getenv('SLURM_CPUS_PER_TASK','1'))) as pool:
        rows=list(pool.map(lambda source:outside_details(source,side,cfg),inv['groups'][k]))
    path=out/f'{k:02}.csv'
    csv_rows(path,rows)
    save_json(path.with_suffix('.json'),dict(job=os.getenv('SLURM_JOB_ID'),rows=len(rows),
        inputs=dict(inventory=sha(marker),source_config=sha(CONFIG),script=sha(Path(__file__))),
        outputs={path.name:sha(path)}))
    print(side,k,'sweeps',len(rows),'raw',sum(x['raw'] for x in rows),flush=True)


if __name__=='__main__':
    main()
