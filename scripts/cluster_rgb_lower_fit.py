#!/usr/bin/env python3
"""Fit scope/shape controls; never reads NGC 5466."""
import copy
import hashlib
import json
from pathlib import Path
import socket
import numpy as np
from cluster_rgb_conditional_fit import load_training, joined, sha_ids, sha256
from photdis.cluster_rgb_lower import fit, mean, slope
from photdis.stage1 import require_compute, write_json

OUT=Path('manifests/cluster_rgb_lower_model_v2.json')


def main():
    require_compute()
    if OUT.exists(): raise FileExistsError(OUT)
    cfg=json.loads(Path('configs/cluster_rgb_conditional_v1.json').read_text())
    cfg['development']={k:v for k,v in cfg['development'].items() if k in ('m3','m53')}
    for v in cfg['development'].values(): v['phases']=['lower_rgb']
    parts=load_training(cfg,'grz_common')
    cells=[]
    for lo in np.arange(.5,3,.5):
        counts=[int(np.sum((p['mg']>=lo)&(p['mg']<lo+.5))) for p in parts]
        if min(counts)>=5: cells.append([float(lo),float(lo+.5),*counts])
    for p in parts:
        good=np.logical_or.reduce([(p['mg']>=lo)&(p['mg']<hi) for lo,hi,*_ in cells])
        for key in ('tab','stage','mg'): p[key]=p[key][good]
    arrays=[]
    seeds=[]
    # Identical joined arrays are reused by every fit, including gr vs grz.
    for i in range(48):
        seed=730230+i
        seeds.append(seed)
        arrays.append(joined(parts,np.random.default_rng(seed),.01,perturb=i>0))
    nominal=arrays[0]
    draws=[{k:a[k].tolist() for k in ('mg','feh','color')} for a in arrays]
    drawhash=hashlib.sha256(json.dumps(draws,sort_keys=True).encode()).hexdigest()
    results={}
    for kind in ('quadratic','monotone'):
        for nc in (1,2):
            key=f'{kind}_{"gr" if nc==1 else "grz"}'
            models=[fit(a,nc,kind) for a in arrays]
            r=nominal['color'][:,:nc]-mean(models[0],nominal['mg'],nominal['feh'])
            _,ix,cnt=np.unique(nominal['cluster'],return_inverse=True,return_counts=True)
            energy=np.sum(r*r,axis=1)/cnt[ix]
            top=max(1,int(np.ceil(len(r)*.01)))
            audit={}
            for name in ('m3','m53'):
                audit[name]={}
                for lo,hi,*_ in cells:
                    mask=(nominal['cluster']==name)&(nominal['mg']>=lo)&(nominal['mg']<hi)
                    audit[name][f'{lo}_{hi}']={'n':int(mask.sum()),'median':np.median(r[mask],axis=0).tolist(),
                        'nmad':(1.4826*np.median(abs(r[mask]-np.median(r[mask],axis=0)),axis=0)).tolist()}
            results[key]={'models':models,'development_residuals':audit,
                'top_one_percent_trace_fraction':float(np.sort(energy)[-top:].sum()/energy.sum()),
                'positive_slope_draw_count':np.sum([np.any(slope(m,np.linspace(.5,3,251))>1e-7,axis=0)
                                                   for m in models],axis=0).tolist()}
            print(key,results[key]['top_one_percent_trace_fraction'],flush=True)
    # Large per-draw input arrays stay on NERSC; tracked manifest records hashes.
    path=Path('data/cluster_rgb_lower_v2');path.mkdir(exist_ok=True,parents=True)
    (path/'paired_training_draws.json').write_text(json.dumps(draws))
    np.savez(path/'development.npz',**nominal)
    write_json(OUT,{'source_revision':'73b023a','role':'development_only',
        'protocol_sha256':sha256(Path('docs/cluster_rgb_lower_protocol_v2.md')),
        'source_ids_sha256':sha_ids(None,parts),'paired_draws_sha256':drawhash,
        'paired_seeds':seeds,'support':cells,'counts':{p['name']:len(p['mg']) for p in parts},
        'input_sha256':{p['name']:sha256(Path(f"data/cluster_rgb_gr_v2/{p['name']}.fits")) for p in parts},
        'results':results,'compute_host':socket.gethostname()})

if __name__=='__main__':main()
