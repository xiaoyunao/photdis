#!/usr/bin/env python3
"""Triggered development-only robust control, frozen before diagnosis."""
import hashlib,json,socket
from pathlib import Path
import numpy as np
from photdis.cluster_rgb_lower import fit_contamination
from photdis.stage1 import require_compute,write_json


def main():
    require_compute()
    model=Path('manifests/cluster_rgb_lower_model_v2.json')
    out=Path('manifests/cluster_rgb_lower_robust_v2.json')
    if out.exists():raise FileExistsError(out)
    frozen=json.loads(model.read_text())
    trigger=frozen['results']['monotone_grz']['top_one_percent_trace_fraction']>.2
    results={}
    if trigger:
        path=Path('data/cluster_rgb_lower_v2')
        nominal=dict(np.load(path/'development.npz'))
        draws=json.loads((path/'paired_training_draws.json').read_text())
        if hashlib.sha256(json.dumps(draws,sort_keys=True).encode()).hexdigest()!=frozen['paired_draws_sha256']:
            raise ValueError('Paired draw hash changed')
        for nc in (1,2):
            models=[]
            for draw in draws:
                a=dict(nominal);a.update({k:np.array(v) for k,v in draw.items()})
                models.append(fit_contamination(a,nc))
            key='robust_'+('gr' if nc==1 else 'grz')
            results[key]={'models':models};print(key,'complete',flush=True)
    write_json(out,{'parent_sha256':hashlib.sha256(model.read_bytes()).hexdigest(),
                   'triggered':trigger,'results':results,'compute_host':socket.gethostname()})

if __name__=='__main__':main()
