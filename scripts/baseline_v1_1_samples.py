#!/usr/bin/env python3
"""Freeze probability samples before their predictions; no full-catalog update."""
from baseline_v1_1_common import *
import fitsio
from photdis.stage1 import require_compute

# Fixed before scores: sample uniformly within field / observed-SNR / WISE cells.
PARALLAX_CELL_CAP=200
SNR_EDGES=np.array([-np.inf,0,5,10,20,np.inf])

def main():
    require_compute();OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'parent_sample.npz').exists() or (OUT/'parallax_sample.npz').exists():
        raise FileExistsError('Frozen samples exist')
    rng=np.random.default_rng(CFG['seed']);parts=[];counts={}
    columns=['identity','g0','r0','z0','w10','w20',*ERR,'r','B','C','optical_error','field']
    for field in range(13):
        with fitsio.FITS(str(OLD/f'catalog_{field:02}.fits')) as f:
            size=f[1].get_nrows();rows=np.sort(rng.choice(size,min(size,CFG['parent_sample_per_field']),replace=False))
            raw=f[1].read(rows=rows,columns=columns)
        counts[str(field)]=size;parts.append({**{k:raw[k] for k in columns},'parent_row':rows,'sample_weight':np.full(len(rows),size/len(rows))})
    parent=concat(parts);np.savez_compressed(OUT/'parent_sample.npz',**parent)
    t=load(OLD/'references.npz')
    good=(t['split']==2)&t['astro_quality']&t['zpt_valid']&np.isfinite(t['p_corr'])&np.isfinite(t['gaia_parallax_error'])&(t['gaia_parallax_error']>0)
    snr=t['p_corr']/t['gaia_parallax_error'];bins=np.searchsorted(SNR_EDGES[1:-1],snr,side='right')
    tier=t['B'].astype(int)+t['C'].astype(int);selected=[];weights=[];cells={}
    for field in (5,6,10):
        for b in range(5):
            for w in range(3):
                pool=np.flatnonzero(good&(t['field']==field)&(bins==b)&(tier==w))
                if not len(pool):continue
                rows=np.sort(rng.choice(pool,min(len(pool),PARALLAX_CELL_CAP),replace=False))
                selected.extend(rows);weights.extend([len(pool)/len(rows)]*len(rows));cells[f'{field}:{b}:{w}']=dict(n_parent=len(pool),n_sample=len(rows))
    ix=np.argsort(selected);rows=np.asarray(selected)[ix]
    par={k:v[rows] for k,v in t.items()};par.update(reference_row=rows,sample_weight=np.asarray(weights)[ix],snr_bin=bins[rows],snr_lo=SNR_EDGES[bins[rows]],snr_hi=SNR_EDGES[bins[rows]+1])
    np.savez_compressed(OUT/'parallax_sample.npz',**par)
    write_json(OUT/'samples.json',dict(parent=dict(n=len(parent['identity']),counts=counts,identity_sha256=identity_hash(parent['identity'])),
        parallax=dict(n=len(rows),eligible=int(good.sum()),negative=int((par['p_corr']<0).sum()),cells=cells,identity_sha256=identity_hash(par['identity']),
            admission='test fields; Gaia quality and valid zero point, finite measured parallax, positive error; no parallax SNR cut; conditional within SNR cells',cell_cap=PARALLAX_CELL_CAP),
        parent_sha256=sha(OUT/'parent_sample.npz'),parallax_sha256=sha(OUT/'parallax_sample.npz')))
    print('Frozen parent/parallax',len(parent['identity']),len(rows),flush=True)

if __name__=='__main__':main()
