#!/usr/bin/env python3
"""Compare a small registered selection grid while preserving the released PDFs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path

import joblib
import numpy as np

from photdis.stage1 import require_compute
from photdis.v3.metrics import distance_summary
from photdis.v3.model import inference_model
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_3_catalog import FREEZE as MODEL_FREEZE, read_frozen
from v3_3_sed import verified
import v3_4_near_forecast as production
from v3_5_dr11_inventory import CONFIG
from v3_5_dr11_validate import join_parts
from v3_6_catalog_projection import OUT as PROJECTION
from v3_6_distance_evidence import OUT as EVIDENCE, SHELLS
from v3_6_lf_sensitivity import OUT as LF, raw_photo
from v3_6_reference_index import OUT as INDEX
from v3_6_vac_distances import DISTANCE_EDGES, read_columns
from v3_7_reference_comparison import SOURCES, load_source
from v3_7_weak_gaia import PROTOCOL


OUT = ROOT/'data/v3_7/selection'
POINTER = ('catalog_row','ls_id','TYPE','parent_row','sweep_number','band_state')
SELECTION = ('dwarf_selection_score','nominal_qnear','observed_r','dereddened_g_r',
             'primary','strict','qrf_quantiles_kpc','qrf_bins')
INPUTS = (PROTOCOL,EVIDENCE/'complete.json',LF/'complete.json',PROJECTION/'complete.json',
          INDEX/'complete.json',CONFIG,MODEL_FREEZE)


def probabilities(p,partial):
    x = np.asarray(p['nominal_qnear']).copy()
    ids = p['catalog_row']
    loc = np.searchsorted(partial['catalog_row'],ids)
    use = np.flatnonzero(loc<len(partial['catalog_row']))
    use = use[partial['catalog_row'][loc[use]]==ids[use]]
    if np.isfinite(x[use]).any():
        raise ValueError('Override would replace a published probability')
    np.testing.assert_array_equal(partial['ls_id'][loc[use]],p['ls_id'][use])
    x[use] = partial['nominal_qnear'][loc[use]]
    return x


def workpoints(p,protocol,prob):
    score = p['dwarf_selection_score']
    r = p['observed_r']
    for name,(dwarf,near) in protocol['workpoints'].items():
        use = score>=dwarf
        if name.endswith('_bright'):
            use &= r<protocol['bright_r_limit']
        if not np.isfinite(prob[use]).all():
            raise ValueError(f'{name} has a missing required nominal probability')
        use &= prob>=near
        if name in ('primary','strict'):
            np.testing.assert_array_equal(use,p[name],err_msg=f'{name} anchor replay')
        yield name,use


def prepare(shards):
    marker = OUT/'prepare.json'
    if marker.exists():
        raise FileExistsError(marker)
    protocol,evidence,lf,projection = (json.loads(p.read_text()) for p in INPUTS[:4])
    minimum = min(v[0] for v in protocol['workpoints'].values())
    ids = np.empty(0,dtype='i8')
    excluded = {}
    for source in SOURCES:
        a,excluded[source] = load_source(source,evidence)
        ids = np.union1d(ids,a['catalog_row'])
    sample = read_columns(verified(LF/'predictions.npz',lf['outputs']['predictions.npz']),
                          ('catalog_row','ls_id','TYPE','weight'))
    if not (sample['TYPE']=='PSF').all() or len(np.unique(sample['catalog_row']))!=len(sample['ls_id']):
        raise ValueError('The stratified draw requires unique native PSFs')
    ids = np.union1d(ids,sample['catalog_row'])
    needed,draw = [],[]
    for name,digest in projection['child_receipts'].items():
        mark = verified(ROOT/name,digest)
        child = json.loads(mark.read_text())
        path = mark.with_name('rows.npz')
        p = read_columns(verified(path,child['outputs'][path.name]),(*POINTER,*SELECTION))
        k = p['catalog_row']
        eligible = p['TYPE']=='PSF'
        if not eligible.all():
            raise ValueError(f'Projection is not a native PSF parent: {name}')
        use = np.isin(k,ids)&~np.isfinite(p['nominal_qnear'])&(p['dwarf_selection_score']>=minimum)
        if use.any():
            needed.append({f:p[f][use] for f in POINTER})
        use = np.isin(k,sample['catalog_row'])
        if use.any():
            draw.append({f:p[f][use] for f in (*POINTER,*SELECTION)})
    draw = join_parts(draw)
    a,b = np.argsort(draw['catalog_row']),np.argsort(sample['catalog_row'])
    for k in ('catalog_row','ls_id','TYPE'):
        np.testing.assert_array_equal(draw[k][a],sample[k][b])
    draw['weight'] = np.empty(len(a))
    draw['weight'][a] = sample['weight'][b]
    need = join_parts(needed) if needed else {k:draw[k][:0] for k in POINTER}
    if len(np.unique(need['catalog_row']))!=len(need['catalog_row']):
        raise ValueError('Duplicate forecast tasks')
    OUT.mkdir(parents=True,exist_ok=True)
    outputs = {}
    for shard in range(shards):
        # Keep each native sweep on a single reader for the missing-gate pass.
        k = need['sweep_number']%shards==shard
        name = f'need_{shard:02d}.npz'
        atomic_npz(OUT/name,**{f:v[k] for f,v in need.items()})
        outputs[name] = sha(OUT/name)
    atomic_npz(OUT/'draw.npz',**draw)
    outputs['draw.npz'] = sha(OUT/'draw.npz')
    save_json(marker,dict(inputs={str(x.relative_to(ROOT)):sha(x) for x in (*INPUTS,Path(__file__).resolve())},
                          shards=shards,need=len(need['ls_id']),draw=len(a),
                          excluded_morphology=excluded,outputs=outputs))
    print(f"Missing production gates: {len(need['ls_id']):,}; weighted-draw identities: {len(a):,}",flush=True)


def replay(shard,receipt):
    dest = OUT/f'replay_{shard:02d}.json'
    if dest.exists():
        raise FileExistsError(dest)
    source = f'need_{shard:02d}.npz'
    p = read_columns(verified(OUT/source,receipt['outputs'][source]),POINTER)
    if not (p['TYPE']=='PSF').all():
        raise ValueError('Nominal replay requires verified native PSF')
    path = INPUTS[4]
    inventory = json.loads(path.read_text())
    names = read_columns(verified(INDEX/'catalog_rows.npz',inventory['outputs']['catalog_rows.npz']),('sweep',))['sweep']
    cfg = json.loads(CONFIG.read_text())
    entry = read_frozen()['models']['nominal']
    nominal_path = verified(ROOT/entry['path'],entry['sha256'])
    result = np.empty(0)
    flags = np.empty(0,bool)
    if len(p['ls_id']):
        photo = raw_photo(p,names,Path(cfg['sweep_root']))
        model = joblib.load(nominal_path)
        production._STATE = photo,inference_model(model),model.weights@model.classes,False
        with ProcessPoolExecutor(max_workers=int(os.getenv('SLURM_CPUS_PER_TASK','1')),
                                mp_context=mp.get_context('fork')) as pool:
            result,flags,_ = zip(*pool.map(production.nominal_row,range(len(p['ls_id'])),chunksize=12))
        production._STATE = None
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite nominal probabilities in selection replay')
    target = OUT/f'replay_{shard:02d}.npz'
    atomic_npz(target,ls_id=p['ls_id'],catalog_row=p['catalog_row'],TYPE=p['TYPE'],
               nominal_qnear=np.asarray(result),numerical_flag=np.asarray(flags,bool))
    save_json(dest,dict(rows=len(p['ls_id']),need_sha256=receipt['outputs'][source],
                        nominal_sha256=entry['sha256'],outputs={target.name:sha(target)}))


def metrics_reference(source,p,prob,protocol):
    d = p['reference_kpc']
    mu = 10+5*np.log10(p['qrf_quantiles_kpc'])
    ref = 10+5*np.log10(d)
    r,c = p['observed_r'],p['dereddened_g_r']
    bright = r<protocol['bright_r_limit']
    every = np.ones(len(r),bool)
    for name,selected in workpoints(p,protocol,prob):
        for dom,keep in (('all',every),('bright',bright),('faint',~bright),
                         ('faint_blue',(~bright)&(c>=.5)&(c<.8))):
            for cat,cl in (('all',every),('dwarf',p['consensus_class']==0),('giant',p['consensus_class']==1),
                           ('other',p['consensus_class']==2),('unknown',p['consensus_class']<0)):
                for history,role in (('all',every),('outside_recorded_history',~p['recorded_history'])):
                    for i,(lo,hi) in enumerate(zip(DISTANCE_EDGES[:-1],DISTANCE_EDGES[1:])):
                        denominator = keep&cl&role&(d>=lo)&(d<hi)
                        include = selected&denominator
                        entry = dict(source=source,selection=name,domain=dom,reference_class=cat,history=history,
                                     reference_shell=SHELLS[i],reference_denominator=int(denominator.sum()))
                        stats = distance_summary(mu[include],ref[include])
                        hard = np.histogram(p['qrf_quantiles_kpc'][include,2],bins=DISTANCE_EDGES)[0]
                        soft = p['qrf_bins'][include].sum(axis=0)
                        yield dict(**entry,**stats,**{f'hard_{j}':int(v) for j,v in enumerate(hard)},
                                   **{f'pdf_{j}':v for j,v in enumerate(soft)})


def summarize(receipt):
    final = OUT/'complete.json'
    if final.exists():
        raise FileExistsError(final)
    partial,markers = [],[]
    for shard in range(receipt['shards']):
        mark = OUT/f'replay_{shard:02d}.json'
        markers.append(mark)
        r = json.loads(mark.read_text())
        if r['need_sha256']!=receipt['outputs'][f'need_{shard:02d}.npz']:
            raise ValueError(f'Shard input mismatch: {shard}')
        target = mark.with_suffix('.npz')
        partial.append(read_columns(verified(target,r['outputs'][target.name]),
                                    ('catalog_row','ls_id','TYPE','nominal_qnear','numerical_flag')))
    p = join_parts(partial)
    order = np.argsort(p['catalog_row'])
    if len(order)!=receipt['need'] or (np.diff(p['catalog_row'][order])<=0).any() or not (p['TYPE']=='PSF').all():
        raise ValueError('Invalid completed selection replay')
    p = {k:v[order] for k,v in p.items()}
    cfg = json.loads(PROTOCOL.read_text())
    evidence = json.loads(INPUTS[1].read_text())
    def reference():
        for name in SOURCES:
            a,_ = load_source(name,evidence)
            yield from metrics_reference(name,a,probabilities(a,p),cfg)
    csv_rows(OUT/'reference.csv',reference())
    t = read_columns(verified(OUT/'draw.npz',receipt['outputs']['draw.npz']),(*POINTER,*SELECTION,'weight'))
    rows = []
    for name,use in workpoints(t,cfg,probabilities(t,p)):
        d = t['qrf_quantiles_kpc'][:,2]
        w = t['weight']*use
        for j,(lo,hi) in enumerate(zip(DISTANCE_EDGES[:-1],DISTANCE_EDGES[1:])):
            hard = w*((d>=lo)&(d<hi))
            soft = w*t['qrf_bins'][:,j]
            rows.append(dict(selection=name,shell=SHELLS[j],draw_n=int(use.sum()),weight_total=float(w.sum()),
                             hard_count=float(hard.sum()),pdf_count=float(soft.sum()),
                             hard_effective_n=float(hard.sum()**2/(hard@hard)) if hard.any() else None,
                             pdf_effective_n=float(soft.sum()**2/(soft@soft)) if soft.any() else None))
    csv_rows(OUT/'catalog.csv',rows)
    save_json(final,dict(inputs={str(x.relative_to(ROOT)):sha(x) for x in (OUT/'prepare.json',*markers)},
        nominal_replayed=len(order),nominal_numerical_flags=int(p['numerical_flag'].sum()),
        outputs={n:sha(OUT/n) for n in ('reference.csv','catalog.csv')}))


def main():
    require_compute()
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('stage',choices=('prepare','replay','report'))
    cli.add_argument('--shards',type=int,default=12)
    cli.add_argument('--shard',type=int)
    arg = cli.parse_args()
    if arg.stage=='prepare':
        if arg.shards<1:
            cli.error('--shards must be positive')
        prepare(arg.shards)
    else:
        receipt = json.loads((OUT/'prepare.json').read_text())
        for path,digest in receipt['inputs'].items():
            verified(ROOT/path,digest)
        if arg.stage=='replay':
            if arg.shard not in range(receipt['shards']):
                cli.error('--shard is outside the frozen range')
            replay(arg.shard,receipt)
        else:
            summarize(receipt)


if __name__=='__main__':
    main()
