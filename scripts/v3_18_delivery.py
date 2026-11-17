#!/usr/bin/env python3
"""Validate matched GD-1 member and passband deliverables."""
import os

import numpy as np

from photdis.stage1 import require_compute
from v2_1_near import save_json, sha
from v3_10_figures import check_templates
from v3_13_delivery import bind, figure_check
from v3_17_sample import DEST as V317
from v3_18_members import C, CONFIG, DEST, read


def run():
    inputs={'config':sha(CONFIG),'style':check_templates()}
    for name in ('members','bands'):
        path=DEST/name
        r=bind(path/'complete.json',inputs)
        assert r['inputs']['style']==inputs['style'] and r['inputs']['config']==inputs['config']
        for p in path.glob('*.png'):
            figure_check(p)
    t=read(DEST/'members'/'roster.csv')
    support={int(x['ls_id']) for x in t if x['science']=='True'}
    assert len(t)==291 and len({x['source_id'] for x in t})==291
    assert all(x['TYPE']=='PSF' and not(x['train_cal']=='True' and x['science']=='True') for x in t)
    with np.load(DEST/'members'/'posteriors.npz',allow_pickle=False) as z:
        assert np.all(z['TYPE']=='PSF') and set(z['ls_id'][z['science']])==support
        assert np.all(np.diff(z['mu_edges'],axis=1)>0) and np.all(z['mass']>=0)
        np.testing.assert_allclose(z['mass'].sum(axis=1),1,rtol=0,atol=2e-7)
    with np.load(DEST/'bands'/'paired_offsets.npz',allow_pickle=False) as z:
        assert len(z['TYPE'])==6000 and np.all(z['TYPE']=='PSF')
        k=(z['south_color']>=C['color_edges'][0])&(z['south_color']<C['color_edges'][-1])
        np.testing.assert_allclose(np.median(z['difference'][k],axis=0),z['applied_median'])
    old=bind(V317/'distance'/'complete.json',inputs)
    reference=read(V317/'distance'/'scores.csv')
    control={(int(x['segment']),x['method']):float(x['distance_kpc']) for x in reference
             if x['cohort']=='same_qrf_source' and x['method'] in ('M13 CMD','M92 CMD')}
    rows=read(DEST/'bands'/'cmd_sensitivity.csv')
    actual={(int(x['segment']),x['method']+' CMD'):float(x['distance_kpc']) for x in rows
            if x['variant']=='Published ridge' and x['depth']=='r < 22'}
    np.testing.assert_equal(actual,control)
    out=DEST/'delivery'
    out.mkdir(parents=True,exist_ok=True)
    save_json(out/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,
               publications=old['job'],member_science_rows=len(support),frozen_baseline_replayed=len(actual),outputs={}))


if __name__=='__main__':
    require_compute()
    run()
