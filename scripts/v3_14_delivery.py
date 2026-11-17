#!/usr/bin/env python3
"""Verify mask, population and structure artifacts without replacing frozen products."""
import json
import os

import numpy as np

from photdis.stage1 import require_compute
from v2_1_near import ROOT, save_json, sha
from v3_6_vac_distances import read_columns
from v3_10_figures import check_templates
from v3_13_delivery import figure_check
from v3_14_audit import DEST as AUDIT, MASK, redraw
from v3_14_population import DEST as POPULATION, MIN_MASS
from v3_14_structure import DEST as STRUCTURE

DEST = ROOT/'data/v3_14/delivery'


def validate(folder,inputs,lock):
    path = folder/'complete.json'
    receipt = json.loads(path.read_text())
    if receipt['inputs'].get('style',lock)!=lock:
        raise ValueError('Shared plot style changed')
    inputs[str(path.relative_to(ROOT))] = sha(path)
    for name,digest in receipt['outputs'].items():
        f = folder/name
        if sha(f)!=digest:
            raise ValueError(f'Changed artifact {f}')
        if f.suffix=='.png':
            figure_check(f)
    return receipt


def identity(t,n):
    if len(np.unique(t['ls_id']))!=n or len(t['ls_id'])!=n or not np.all(t['TYPE']=='PSF'):
        raise ValueError('Unexpected native row roster')


def run():
    lock = check_templates()
    inputs = {}
    for name in ('M92','M13','M30','M5'):
        folder = AUDIT/name
        receipt = validate(folder,inputs,lock)
        t = read_columns(folder/'native_join.npz',('TYPE','ls_id','MASKBITS','cluster_mask','member','qrf_kpc'))
        identity(t,receipt['rows'])
        np.testing.assert_array_equal(t['cluster_mask'],(t['MASKBITS'].astype('i8')&MASK)!=0)
        assert int(t['cluster_mask'].sum())==receipt['mask_rows']
    for side in ('north','south'):
        folder = POPULATION/side
        receipt = validate(folder,inputs,lock)
        t = read_columns(folder/'heldout.npz',('TYPE','ls_id','mu','usable_training_mass','branch_weights'))
        identity(t,receipt['held_out'])
        s = np.isfinite(t['mu'][2]).all(axis=1)
        assert np.all(t['usable_training_mass'][s]>=MIN_MASS)
        np.testing.assert_allclose(t['branch_weights'][s].sum(axis=1),1.,atol=1e-12,rtol=0)
        assert np.all(np.diff(t['mu'][2,s],axis=1)>=0)
    receipt = validate(STRUCTURE,inputs,lock)
    t = read_columns(STRUCTURE/'field.npz',('TYPE','ls_id','training_or_calibration','qrf_mass'))
    identity(t,receipt['rows'])
    mass = t['qrf_mass'].sum(axis=1,dtype=float)
    assert not (mass[t['training_or_calibration']]>0).any()
    np.testing.assert_allclose(mass[mass>0],1.,atol=2e-7,rtol=0)
    DEST.mkdir(parents=True,exist_ok=True)
    save_json(DEST/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),style=lock,inputs=inputs,outputs={}))


if __name__=='__main__':
    require_compute()
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--redraw',action='store_true')
    if parser.parse_args().redraw:
        for name in ('M92','M13','M30','M5'):
            redraw(name)
    run()
