#!/usr/bin/env python3
"""Fixed residual-independent parent sample; preserve original v0 artifacts."""
from baseline_v1_common import *
import fitsio
from photdis.stage1 import require_compute

def main():
    require_compute();OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'sample.npz').exists():raise FileExistsError('Fixed sample exists')
    rng=np.random.default_rng(CFG['seed']);parts=[];counts={}
    cols=['identity','ls_id','g0','r0','z0','w10','w20',*ERR,'r','B','C','supported','ambiguous','optical_error','field']
    for f in range(13):
        t=fitsio.read(str(OLD/f'catalog_{f:02}.fits'),columns=cols)
        e=np.searchsorted([.03,.06,.12],t['optical_error']);tier=t['B'].astype(int)+t['C'].astype(int)
        cell=e*12+tier*4+t['supported']*2+t['ambiguous'];idx=[]
        for k in np.unique(cell):
            pool=np.flatnonzero(cell==k);n=min(len(pool),CFG['convergence_per_cell'])
            idx.extend(rng.choice(pool,n,replace=False));counts[f'{f}:{k}']=len(pool)
        parts.append(t[np.sort(idx)])
    t=np.concatenate(parts)
    if len(t)>3000:t=t[np.sort(rng.choice(len(t),3000,replace=False))]
    np.savez(OUT/'sample.npz',**{k:t[k] for k in t.dtype.names})
    write_json(OUT/'sample.json',dict(n=len(t),identity_sha256=identity_hash(t['identity']),
        parent_cell_counts=counts,config_sha256=sha(ROOT/'configs/baseline_v1.json'),
        v0_models={k:sha(OLD/(k+'.joblib')) for k in MODELS}))
    print('Fixed parent sample',len(t),flush=True)
if __name__=='__main__':main()
