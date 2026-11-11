#!/usr/bin/env python3
"""Same unopened reservation; sequential FITS I/O and per-field checkpoints."""
import json
import os
from pathlib import Path
import numpy as np


def native_keys(path):
    """Read fixed-width rows sequentially; retain exact native integer/string fields."""
    from astropy.io import fits
    names=['RELEASE','BRICKID','OBJID','TYPE']
    with fits.open(path,memmap=True) as h:
        table=h[1];dtype=table.data.dtype;header=table.header
        assert dtype.itemsize==header['NAXIS1']
        assert not any(dtype[k].hasobject for k in dtype.names)
        for key in names:
            column=list(dtype.names).index(key)+1
            assert header.get(f'TSCAL{column}',1)==1 and header.get(f'TZERO{column}',0)==0
        offset=table.fileinfo()['datLoc'];n=header['NAXIS2']
    raw=np.fromfile(path,dtype=dtype,count=n,offset=offset)
    assert len(raw)==n
    return {k:raw[k].copy() for k in names}


def main():
    import fitsio
    import joblib
    from v2_4_population import ROOT,DEST,TABLES,OLD,load,subset,sha,atomic_npz,save_json,csv_rows,observable,legacy_ids,unique_join,psf_admission,require_compute
    require_compute()
    freeze={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'docs/v2_4_reservation_io_amendment.md',ROOT/'scripts/v2_4_population.py',OLD/'control.joblib']}
    marker=DEST/'reservation_io_freeze.json'
    if marker.exists():assert json.loads(marker.read_text())==freeze
    else:save_json(marker,freeze)
    cache=DEST/'reservation_fields';cache.mkdir(exist_ok=True)
    used=[]
    for path in [ROOT/'data/baseline_v0/references.npz',ROOT/'data/baseline_v1_3/references.npz',ROOT/'data/baseline_v1_4/references.npz',ROOT/'data/v2_2/application_inputs.npz']:
        with np.load(path) as z:used.append(z['identity'])
    used=np.unique(np.concatenate(used));model=joblib.load(OLD/'control.joblib')
    fields=json.loads((ROOT/'data/baseline_v0/prepared.json').read_text())['fields'];pieces=[];counts=[];native_sources={}
    for f in fields[:13]:
        field=f['index'];checkpoint=cache/f'{field:02}.npz';meta=checkpoint.with_suffix('.json')
        if meta.exists():
            record=json.loads(meta.read_text());assert sha(checkpoint)==record['sha256'];pieces.append(load(checkpoint));counts.append(record['counts']);native_sources.update(record['sources']);continue
        path=ROOT/f'data/baseline_v1/catalog_{field:02}.fits'
        cols=['identity','ls_id','r','g0','r0','z0','w10','w20','sigma_g','sigma_r','sigma_z','sigma_w1','sigma_w2','optical_error','ra','dec','ebv','B','C']
        with fitsio.FITS(str(path)) as h:a=h[1].read(columns=cols)
        t={k:a[k] for k in cols};u=(t['r']<20)&(t['optical_error']<.1)&~np.isin(t['identity'],used);ix=np.flatnonzero(u)
        with np.errstate(over='ignore'):rank=t['identity'][ix].astype('u8')*np.uint64(11400714819323198485)
        ix=ix[np.argsort(rank,kind='stable')[:2500]];t=subset(t,ix);u=observable(t,model)
        types=np.full(len(ix),'','U4');task=json.loads((ROOT/f['config']).read_text());cfg=json.loads((ROOT/task['base_config']).read_text());sources={}
        for sweep in task['sweep_files']:
            source=os.path.join(task.get('sweep_root',cfg['sweep_root']),sweep)
            a=native_keys(source);j=unique_join(t['ls_id'],legacy_ids(a));found=j>=0;types[found]=np.char.strip(a['TYPE'][j[found]].astype('U'))
            # Independent CFITSIO reader verifies integer packing, byte order and TYPE.
            rows=np.unique(np.r_[np.linspace(0,len(a['OBJID'])-1,16,dtype=int),j[found][:16]])
            with fitsio.FITS(source) as h:check=h[1].read(columns=list(a),rows=rows)
            for key in a:
                left=a[key][rows].astype(check[key].dtype);right=check[key]
                if key=='TYPE':left=np.char.strip(left);right=np.char.strip(right)
                np.testing.assert_array_equal(left,right)
            stat=os.stat(source);sources[source]=dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns,independent_checked_rows=len(rows))
            del a
        good,audit=psf_admission(types);take=np.flatnonzero(good&u)[:250];t['TYPE']=types;t['field']=np.full(len(ix),field)
        piece=subset(t,take);count=dict(field=field,candidates=len(ix),observable=int(u.sum()),reserved=len(take),**audit)
        atomic_npz(checkpoint,**piece);save_json(meta,dict(sha256=sha(checkpoint),counts=count,sources=sources))
        pieces.append(piece);counts.append(count);native_sources.update(sources);print('RESERVE FIELD',field,len(take),flush=True)
    out={k:np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
    assert len(np.unique(out['identity']))==len(out['identity']) and not np.intersect1d(used,out['identity']).size and (out['TYPE']=='PSF').all()
    atomic_npz(DEST/'prospective_identity_reservation.npz',**out);csv_rows(TABLES/'reservation_counts.csv',counts)
    save_json(DEST/'reservation.json',dict(n=len(out['identity']),sha256=sha(DEST/'prospective_identity_reservation.npz'),native_sources=native_sources,
        reference_outcomes_opened=False,predictions_generated=False,known_historical_overlap=0,
        limitations='Same existing application footprints; not new sky. System membership and independent references still required.',
        io_amendment='Sequential fixed-width native rows; exact original admission/ranking; independent CFITSIO spot checks and field checkpoints.',job=os.getenv('SLURM_JOB_ID')))


if __name__=='__main__':main()
