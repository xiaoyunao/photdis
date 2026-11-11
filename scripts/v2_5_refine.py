#!/usr/bin/env python3
"""Numerical-only reevaluation of the frozen four-parameter likelihood."""
import json
import os
import shutil
import numpy as np
import v2_5_population as population
from v2_1_near import ROOT,load,sha,save_json
from photdis.stage1 import require_compute
from photdis.population_v2_5 import posterior_grid
D=ROOT/'data/v2_5'


def refined_grid(counts,response,base,power=14,seed=20261111):
    return posterior_grid(counts,response,base,power={14:18,13:17,12:14}[power],seed=seed)


def main():
    require_compute();archive=D/'numerical_initial'
    paths=[ROOT/'scripts/v2_5_refine.py',ROOT/'docs/v2_5_numerical_amendment.md',ROOT/'scripts/v2_5_population.py',ROOT/'src/photdis/population_v2_5.py',D/'catalogs.npz',D/'cal_templates.npz',D/'test_templates.npz']
    freeze={str(p.relative_to(ROOT)):sha(p) for p in paths}
    marker=D/'numerical_freeze.json'
    if marker.exists():assert json.loads(marker.read_text())==freeze
    else:save_json(marker,freeze)
    if not archive.exists():
        assert json.loads((D/'validation.json').read_text())['all_passed']
        archive.mkdir();shutil.copytree(D/'tables',archive/'tables')
        for name in ('posterior_regime0.npz','posterior_regime1.npz','population_complete.json','validation.json'):
            shutil.copy2(D/name,archive/name)
        save_json(archive/'archive.json',dict(job=os.getenv('SLURM_JOB_ID'),reason='Numeric integration resolution only',hashes={str(p.relative_to(archive)):sha(p) for p in archive.rglob('*') if p.is_file()}))
    c=load(D/'cal_templates.npz');test=load(D/'test_templates.npz');a=load(D/'catalogs.npz');meta=json.loads((D/'templates.json').read_text())
    pieces=[]
    for path in sorted((D/'qrf_chunks').glob('*.npz')):
        assert sha(path)==json.loads(path.with_suffix('.json').read_text())['sha256'];pieces.append(load(path))
    p={k:np.concatenate([x[k] for x in pieces]) for k in pieces[0]};assert np.array_equal(p['identity'],a['identity'])
    population.posterior_grid=refined_grid
    population.evaluate(c,test,np.array(meta['base']),np.array(meta['noise_bins']),np.array(meta['noise_curve']),a,p)
    save_json(D/'numerical_complete.json',dict(job=os.getenv('SLURM_JOB_ID'),main_power=18,comparison_power=17,sensitivity_power=14,unchanged_catalog=sha(D/'catalogs.npz')==freeze['data/v2_5/catalogs.npz'],model_refitted=False))
    print('NUMERICAL REFINEMENT COMPLETE',flush=True)


if __name__=='__main__':main()
