#!/usr/bin/env python3
"""Recover exact native raw photometry for the small positional candidate set."""
from baseline_v1_4_native import *


def read_brick(task):
    path, indices, objids, expected = task
    with fitsio.FITS(str(path)) as f:
        names = [k.upper() for k in f[1].get_colnames()]
        columns = RAW_COLUMNS + [k for k in ['MJD_MIN', 'MJD_MAX', 'BRICK_PRIMARY'] if k in names]
        raw = uppercase_fields(f[1].read(columns=columns, rows=objids))
        actual = (raw['RELEASE'].astype('i8') << 40) | (raw['BRICKID'].astype('i8') << 20) | raw['OBJID'].astype('i8')
        assert np.array_equal(actual, expected), str(path)
    return indices, raw


def main():
    require_compute()
    marker = OUT/'zero_candidates_raw.json'
    output = OUT/'zero_candidates_raw.npz'
    if marker.exists():
        assert sha(output) == json.loads(marker.read_text())['sha256']
        return
    with np.load(OUT/'zero_photometry.npz') as z:
        c = {k:z[k] for k in ['BRICKID','OBJID','ls_id','grz_base','REF_CAT']}
    assert sha(OUT/'zero_photometry.npz') == json.loads((OUT/'zero_match.json').read_text())['photometry_sha256']
    bricks = uppercase_fields(fitsio.read(str(SOUTH/'survey-bricks-dr11-south.fits.gz'), columns=['BRICKID','BRICKNAME']))
    names = {int(b):str(n.decode() if isinstance(n,bytes) else n).strip() for b,n in zip(bricks['BRICKID'],bricks['BRICKNAME'])}
    tasks=[]
    for b in np.unique(c['BRICKID']):
        ix=np.flatnonzero(c['BRICKID']==b);name=names[int(b)]
        tasks.append((SOUTH/'tractor'/name[:3]/('tractor-'+name+'.fits'),ix,c['OBJID'][ix],c['ls_id'][ix]))
    result={}
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:
        for count,(ix,raw) in enumerate(pool.map(read_brick,tasks)):
            if not result:
                result={k:np.empty((len(c['ls_id']),)+raw[k].shape[1:],raw[k].dtype) for k in raw.dtype.names}
            for k in result:result[k][ix]=raw[k]
            if count%250==0:print('raw candidate bricks',count,'/',len(tasks),flush=True)
    result['ls_id']=c['ls_id'];atomic_npz(output,**result)
    report=dict(rows=len(c['ls_id']),bricks=len(tasks),sha256=sha(output),
        candidate_sha256=sha(OUT/'zero_photometry.npz'),
        type_counts={str(k):int(v) for k,v in zip(*np.unique(result['TYPE'].astype('U'),return_counts=True))},
        type_grz_base={str(k):int(((result['TYPE'].astype('U')==k)&c['grz_base']).sum()) for k in np.unique(result['TYPE'].astype('U'))})
    write_json(marker,report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
