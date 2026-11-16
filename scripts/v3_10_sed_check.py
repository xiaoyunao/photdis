#!/usr/bin/env python3
"""Evaluate the nominal SED gate on targeted held-out photometric controls."""
import argparse
import concurrent.futures
import json
import multiprocessing
import os

import numpy as np

from photdis.reference_v1_4 import unique_join
from photdis.stage1 import require_compute
from photdis.v3.metrics import distance_summary
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_6_vac_distances import read_columns
from v3_9_catalog import OUT as EXPANSION
from v3_9_references import choose, join, subset
from v3_9_validate import nominal_control, nominal_setup
from v3_10_evidence import DEST as EVIDENCE
from v3_10_redrock import DEST as REDROCK
from v3_10_selection import DEST as SELECTION, ROUTES

DEST = ROOT/'data/v3_10/selection_sed'
CONTEXT = None


def product(root, filename, keys, inputs):
    mark = root/'complete.json'
    r = json.loads(mark.read_text())
    inputs[str(mark.relative_to(ROOT))] = sha(mark)
    return read_columns(verified(root/filename,r['outputs'][filename]),keys)


def controls(side, inputs):
    score = product(SELECTION/side,'scores.npz',
                    ('TYPE','ls_id','score','score_original','held_out'),inputs)
    rr = product(REDROCK,'qualified.npz',('TYPE','ls_id','qualified'),inputs)
    parts = []
    for region in ROUTES[side]:
        t = product(EVIDENCE/region,'references.npz',('TYPE','ls_id','r','class_label','gaia_state',
             'qrf_quantiles_kpc','reference_kpc','specdis_kpc','spdist_kpc','specdis_quality','spdist_quality'),inputs)
        j = unique_join(score['ls_id'],t['ls_id'])
        use = score['held_out'] & np.isfinite(score['score_original']) & np.isfinite(score['score']) & (j>=0)
        t = subset(t,j[use])
        np.testing.assert_array_equal(score['TYPE'][use],t['TYPE'])
        q = unique_join(t['ls_id'],rr['ls_id'])
        control = np.full(len(q),'',dtype='U8')
        hit = q>=0
        control[hit] = rr['qualified'][q[hit]]
        r = t['r']
        d = np.where(t['specdis_quality'],t['specdis_kpc'],
                     np.where(t['spdist_quality'],t['spdist_kpc'],t['reference_kpc']))
        c = np.where(control=='',t['class_label'],-1)
        groups = (('dwarf_2_5',(c==0)&(d>=2)&(d<5)),
                  ('dwarf_5_10',(c==0)&(d>=5)&(d<10)),
                  ('giant_5_10',(c==1)&(d>=5)&(d<10)),
                  ('giant_beyond10',(c==1)&(d>=10)),
                  ('other_2_10',(c==2)&(d>=2)&(d<10)),
                  ('qso',control=='QSO'),('galaxy',control=='GALAXY'))
        for lo,hi in ((-np.inf,20),(20,21),(21,22),(22,23)):
            mag = (r>=lo)&(r<hi)
            for i,(name,k) in enumerate(groups):
                j = choose(t['ls_id'],mag&k,48,31051+i)
                if not len(j):
                    continue
                s = {key:t[key][j] for key in ('TYPE','ls_id','qrf_quantiles_kpc')}
                choose_source = np.flatnonzero(use)[j]
                parts.append(dict(**s,reference_kpc=d[j],r=r[j],
                    score_original=score['score_original'][choose_source],score_new=score['score'][choose_source],
                    stratum=np.full(len(j),name),region=np.full(len(j),region),
                    sampling_weight=np.full(len(j),float((mag&k).sum()/len(j)))))
    selected = join(parts)
    keys = selected['ls_id']
    if len(np.unique(keys))!=len(keys) or not np.all(selected['TYPE']=='PSF'):
        raise ValueError('Duplicate or invalid native PSF in targeted gate controls')
    return selected


def photometry(side, selected, inputs):
    directory = EXPANSION/side/'reference'
    keys = ('TYPE','ls_id','flux','flux_error','transmission','band_mask','EBV','region')
    parts = []
    for path in sorted(directory.glob('[0-9][0-9].json')):
        s = json.loads(path.read_text())
        for name,h in s['outputs'].items():
            p = read_columns(verified(directory/name,h),keys)
            take = np.isin(p['ls_id'],selected['ls_id'])
            if take.any():
                parts.append(subset(p,take))
        inputs[str(path.relative_to(ROOT))] = sha(path)
    p = join(parts)
    j = unique_join(selected['ls_id'],p['ls_id'])
    if np.any(j<0):
        raise ValueError('Targeted SED controls lack original photometry')
    np.testing.assert_array_equal(selected['TYPE'],p['TYPE'][j])
    return subset(p,j)


def evaluate(rows):
    p,model = CONTEXT
    return nominal_control(p,model,rows)


def run(side):
    global CONTEXT
    inputs = {}
    t = controls(side,inputs)
    p = photometry(side,t,inputs)
    model = nominal_setup(side)
    CONTEXT = p,model
    n = len(t['ls_id'])
    cpus = int(os.environ.get('SLURM_CPUS_PER_TASK',1))
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(cpus,16),
            mp_context=multiprocessing.get_context('fork')) as pool:
        chunks = np.split(np.arange(n),np.arange(12,n,12))
        outcomes = list(pool.map(evaluate,chunks))
    CONTEXT = None
    q,flag = [np.concatenate(a) for a in zip(*outcomes)]
    if not np.all(np.isfinite(q)&(q>=0)&(q<=1)):
        raise ValueError('Invalid targeted SED gate probabilities')
    rows = []
    for region in ROUTES[side]:
        for lo,hi in ((-np.inf,20),(20,21),(21,22),(22,23)):
            for label in np.unique(t['stratum']):
                j = (t['region']==region)&(t['r']>=lo)&(t['r']<hi)&(t['stratum']==label)
                if not j.any():
                    continue
                base = j&~flag
                for cut in (.65,.80):
                    for method in ('score_original','score_new'):
                        a = base&(t[method]>=cut)
                        b = a&(q>=.90)
                        row = dict(region=region,r_min=lo,r_max=hi,group=label,threshold=cut,method=method,
                                   drawn=int(j.sum()),evaluated=int(base.sum()),score_pass=int(a.sum()),
                                   both_pass=int(b.sum()),invalid=int((j&flag).sum()))
                        for stage,x in (('score',a),('joint',b)):
                            keep = x & (t['reference_kpc']>0) & (label not in ('qso','galaxy'))
                            out = distance_summary(10+5*np.log10(t['qrf_quantiles_kpc'][keep]),
                                                   10+5*np.log10(t['reference_kpc'][keep]))
                            row.update({stage+'_'+k:v for k,v in out.items()})
                        rows.append(row)
    dest = DEST/side
    dest.mkdir(parents=True,exist_ok=True)
    csv_rows(dest/'sed_gate_comparisons.csv',rows)
    atomic_npz(dest/'sed_gate_controls.npz',**t,nominal_qnear=q,numerical_flag=flag)
    save_json(dest/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,rows=n,
        valid=int((~flag).sum()),outputs={f.name:sha(f) for f in dest.iterdir() if f.name!='complete.json'}))


if __name__=='__main__':
    require_compute()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('side',choices=tuple(ROUTES))
    run(parser.parse_args().side)
