#!/usr/bin/env python3
"""Validate paired identities, posterior arrays, resamples and product checksums."""
import csv
import json
import os

import numpy as np

from photdis.reference_v1_4 import unique_join
from photdis.stage1 import require_compute
from photdis.v3.data import SIGNIFICANT
from v2_1_near import ROOT, csv_rows, save_json, sha
from v3_11_layers import read_receipt
from v3_14_delivery import identity, validate
from v3_14_population import DEST as V314
from v3_15_bands import BRANCH, DEST as BANDS, MIN_MASS
from v3_15_transfer import AUDIT, DEST as TRANSFER, DRAW, M5_D, D68, SEED, SYSTEMS, cohort
from v3_10_figures import check_templates

DEST = ROOT/'data/v3_15/delivery'


def validate_transfer(inputs, lock):
    r = validate(TRANSFER,inputs,lock)
    assert r['inputs']['style']==lock
    with np.load(TRANSFER/'rosters.npz',allow_pickle=False) as d:
        assert len(np.unique(d['template_id']))==99
        np.testing.assert_array_equal(d['draw_index'],
            np.random.default_rng(SEED).integers(99,size=(DRAW,99)))
        with (TRANSFER/'comparison.csv').open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows)==8
        for name in ('M5','M68'):
            row, = (x for x in rows if (x['cluster'],x['sample'],x['template'])==(name,'strict','strict'))
            target = d[name+'_target']
            identity({k:d[name+'_'+k] for k in ('ls_id','TYPE')},
                     int(row['target'])+int(row['background']))
            assert target.sum()==int(row['target'])
    with np.load(TRANSFER/'matched_transfer.npz',allow_pickle=False) as d:
        for row in rows:
            key = '_'.join(row[k] for k in ('cluster','sample','template'))
            assert np.all(np.isfinite(d[key]))
            assert d['distance_kpc'][d[key].argmax()]==float(row['cmd_peak'])
        for cluster in ('M5','M68'):
            for sample in ('strict','all'):
                assert d[cluster+'_'+sample].shape==(DRAW,)
    return r['job']


def validate_bands(side,inputs,lock):
    out = BANDS/side
    r = validate(out,inputs,lock)
    old = read_receipt(V314/side,'complete.json','heldout.npz',('TYPE','ls_id','branch'),inputs)
    identity(old,r['heldout'])
    counts = {}
    with (out/'roster.csv').open() as f:
        rows = list(csv.DictReader(f))
    for name,cols in (('grzi',[0,1,2,3]),('grzW1W2',[0,1,3,4,5])):
        capture = [x for x in rows if x['pair']==name]
        assert len(capture)==len(BRANCH)
        file = out/(name+'_paired.npz')
        if file.name not in r['outputs']:
            assert not any(int(x['test_pair']) for x in capture) or not any(int(x['train_pair']) for x in capture)
            counts[name] = 0
            continue
        with np.load(file,allow_pickle=False) as d:
            n = len(d['ls_id'])
            identity(d,n)
            identity({k:d['train_'+k] for k in ('ls_id','TYPE')},
                     sum(int(x['train_pair']) for x in capture))
            j = unique_join(d['ls_id'],old['ls_id'])
            assert (j>=0).all() and not np.intersect1d(d['ls_id'],d['train_ls_id']).size
            assert (d['bands'][:,cols]==SIGNIFICANT).all()
            np.testing.assert_array_equal(d['true_branch'],old['branch'][j])
            for k,x in enumerate(capture):
                assert BRANCH[k]==x['branch'] and (d['true_branch']==k).sum()==int(x['test_pair'])
            for view in ('grz',name):
                p,m,q = (d[view+'_'+k] for k in ('probability','usable_mass','mu'))
                assert ((p>=0)&(p<=1)).all()
                np.testing.assert_allclose(p.sum(axis=1),1.,atol=1e-12,rtol=0)
                use = np.isfinite(q).all(axis=1)
                assert (m[use]>=MIN_MASS).all() and np.all(np.diff(q[use],axis=1)>=0)
            counts[name] = n
    return dict(job=r['job'],matched_tests=counts)


def compare_member_cmd(inputs):
    """Measure color at fixed reference absolute magnitude on the shared CMD range."""
    template = read_receipt(AUDIT/'M30','complete.json','native_join.npz',
                            ('TYPE','ls_id','g0','r0'),inputs)
    with np.load(TRANSFER/'rosters.npz',allow_pickle=False) as roster:
        ix = unique_join(roster['template_id'],template['ls_id'])
        assert (ix>=0).all() and (template['TYPE'][ix]=='PSF').all()
    samples = [('M30',template,ix,SYSTEMS['M30'][1])]
    for name,d in (('M5',M5_D),('M68',D68)):
        p,v = cohort(name,inputs)
        use = v['strict']['used']&p['target']&p['member']
        samples.append((name,p,use,d))
    rows = []
    for name,p,ix,d in samples:
        color = (p['g0']-p['r0'])[ix]
        absolute = p['r0'][ix]-10-5*np.log10(d)
        for lo in np.arange(2.5,5.5,.5):
            x = color[(absolute>=lo)&(absolute<lo+.5)]
            row = dict(cluster=name,Mr_lo=lo,Mr_hi=lo+.5,n=len(x))
            if len(x):
                q = np.quantile(x,(.16,.5,.84))
                row.update(gr_p16=q[0],gr_p50=q[1],gr_p84=q[2])
            rows.append(row)
    return rows


def run():
    lock = check_templates()
    inputs = {}
    transfer = validate_transfer(inputs,lock)
    bands = {side:validate_bands(side,inputs,lock) for side in ('north','south')}
    DEST.mkdir(parents=True,exist_ok=True)
    path = DEST/'member_cmd.csv'
    csv_rows(path,compare_member_cmd(inputs))
    save_json(DEST/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,style=lock,
              transfer=transfer,bands=bands,outputs={path.name:sha(path)}))


if __name__=='__main__':
    require_compute()
    run()
