#!/usr/bin/env python3
"""Compare archived expanded_A predictions to frozen DR11 predictions on the same stars."""
import csv
import json
import os
from pathlib import Path

import numpy as np

from photdis.reference_v1_4 import unique_join
from photdis.stage1 import require_compute
from photdis.v3.metrics import distance_summary
from v2_1_near import ROOT, atomic_npz, csv_rows, load, save_json, sha
from v3_3_sed import verified
from v3_4_near_reference_support import separation
from v3_5_dr11_spatial import DELIVERY
from v3_5_dr11_validate import take_rows
from v3_6_reference_index import OUT as INDEX
from v3_6_vac_distances import ARCHIVE, DISTANCE_EDGES, read_columns
from v3_6_vac_flow import gather_forecasts


OUT = ROOT/'data/v3_6/qrf_cache_comparison'
OLD = ROOT/'data/v2_0'
REFS = ('Gaia','specdis','spdist')
SPLITS = {1:'calibration',2:'test'}


def compare(t):
    r,c = t['observed_r'],t['dereddened_g_r']
    groups = {'all':np.ones(len(r),bool), 'blue':(c>=.5)&(c<.8),
              'faint_blue':(r>=18)&(c>=.5)&(c<.8)}
    groups.update({lab:t['class_label']==val for val,lab in
                   ((-1,'unknown_class'),(0,'dwarf'),(1,'giant'),(2,'other_star'))})
    support = {'all_current_support':groups['all'], 'both_supported':t['expanded_supported'],
               'current_only_support':~t['expanded_supported']}
    cuts = {'before_gates':groups['all'],'primary':t['primary'],'strict':t['strict']}
    q = {'expanded_A':t['expanded_mu'],'current_grz':10+5*np.log10(t['qrf_quantiles_kpc'])}
    valid_pair = np.logical_and.reduce([np.isfinite(mu).all(axis=1) for mu in q.values()])
    rows = []
    for source in REFS:
        ref = t[f'ref_{source}']
        d = 10**((ref-10)/5)
        for lo,hi in zip(DISTANCE_EDGES[:-1],DISTANCE_EDGES[1:]):
            shell = np.isfinite(ref)&(d>=lo)&(d<hi)
            for split,split_label in SPLITS.items():
                for exposed in (False,True):
                    cohort = shell&(t['split']==split)&(t['current_qrf_training']==exposed)
                    for scope,available in support.items():
                        for selection,selected in cuts.items():
                            for group,group_mask in groups.items():
                                use = cohort&available&selected&group_mask
                                n = int(use.sum())
                                if not n:
                                    continue
                                # Report paired performance with both quantile vectors present.
                                use &= valid_pair
                                for model,mu in q.items():
                                    rows.append(dict(source=source,lower_kpc=lo,upper_kpc=hi,
                                        old_split=split_label,current_qrf_training=exposed,
                                        support=scope,selection=selection,group=group,model=model,
                                        cohort_n=n,**distance_summary(mu[use],ref[use])))
    return rows


def open_old_cache():
    snapshot = OLD/'manifests/snapshot.json'
    completion = json.loads((OLD/'collection_complete.json').read_text())
    snapshots = json.loads(verified(snapshot,completion['snapshot_sha256']).read_text())
    expanded = next(m for m in snapshots['models'] if m['model']=='expanded_A')
    predictions = verified(ROOT/expanded['chosen_view'],expanded['chosen_view_sha256'])
    train = verified(ROOT/expanded['training_identity_file'],expanded['training_file_sha256'])
    fields = ('identity','ls_id','source_id','RA','DEC','TYPE','split','mu','supported',
              'in_model_training','class_label','class_eligible','ref_Gaia','ref_specdis',
              'ref_spdist','specdis_sigma_mu','spdist_sigma_mu','p_corr','gaia_parallax_error')
    old,train_ids = read_columns(predictions,fields),read_columns(train,('identity',))['identity']
    if old['in_model_training'].any() or np.isin(old['identity'],train_ids).any():
        raise ValueError('Archived expanded_A evaluation overlaps its training identities')
    return old,expanded,[snapshot,predictions,train]


def main():
    require_compute()
    old,expanded,inputs = open_old_cache()
    index_path = INDEX/'complete.json'
    index = json.loads(index_path.read_text())
    catalog = load(verified(INDEX/'catalog_rows.npz',index['outputs']['catalog_rows.npz']))
    native = load(verified(INDEX/'baseline_v1_4.npz',index['outputs']['baseline_v1_4.npz']))
    parent = ARCHIVE/'references.npz'
    a = read_columns(verified(parent,index['inputs'][str(parent.relative_to(ROOT))]),
                     ('identity','ls_id','source_id'))
    original = unique_join(old['identity'],a['identity'])
    native_order = native['catalog_row'][original.clip(min=0)].copy()
    consistent_parent = (original>=0)&(a['ls_id'][original]==old['ls_id'])
    native_order[~consistent_parent] = -1
    candidate = consistent_parent&(old['TYPE']=='PSF')&(native_order>=0)
    delivery_path = DELIVERY/'complete.json'
    delivery = json.loads(delivery_path.read_text())
    with verified(DELIVERY/'sweeps.csv',delivery['outputs']['sweeps.csv']).open() as stream:
        receipts = {row['sweep']:row['receipt_sha256'] for row in csv.DictReader(stream)}
    p = gather_forecasts(native_order[candidate],catalog,receipts)
    q = take_rows(old,candidate)
    checked_key = ((q['ls_id']==p['ls_id'])|((q['source_id']>0)&(q['source_id']==p['source_id'])))
    checked_pos = separation(q['RA'],q['DEC'],p['RA'],p['DEC'])
    keep = checked_key&checked_pos
    p,q = take_rows(p,keep),take_rows(q,keep)
    if len(np.unique(p['ls_id']))!=len(p['ls_id']) or not (q['TYPE']=='PSF').all():
        raise ValueError('Comparison requires unique native PSF model rows')
    controls = ROOT/'manifests/v3_0/controls/complete.json'
    control = json.loads(controls.read_text())
    training = read_columns(verified(ROOT/'data/v3_0/controls/training.npz',
                            control['outputs']['training.npz']),('identity',))['identity']
    p.update(expanded_identity=q['identity'],expanded_ls_id=q['ls_id'],
             expanded_mu=q['mu'],expanded_supported=q['supported'],split=q['split'],
             class_label=np.where(q['class_eligible'],q['class_label'],-1),
             current_qrf_training=np.isin(p['identity'],training)|np.isin(p['source_id'],training)
                  |np.isin(q['identity'],training),
             **{key:q[key] for key in ('ref_Gaia','ref_specdis','ref_spdist','specdis_sigma_mu',
                                       'spdist_sigma_mu','p_corr','gaia_parallax_error')})
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'complete.json').exists():
        raise FileExistsError(OUT/'complete.json')
    tally = dict(archived_cache=len(old['identity']),archived_non_psf=int((old['TYPE']!='PSF').sum()),
                 psf_parent_unmatched=int(((old['TYPE']=='PSF')&~consistent_parent).sum()),
                 psf_without_current_supported_row=int(((old['TYPE']=='PSF')&consistent_parent&(native_order<0)).sum()),
                 final_key_or_position_rejected=int((~keep).sum()),paired=int(keep.sum()))
    if sum(list(tally.values())[1:])!=tally['archived_cache']:
        raise ValueError('Archived model cache accounting did not reconcile')
    atomic_npz(OUT/'paired.npz',**p)
    csv_rows(OUT/'performance.csv',compare(p))
    csv_rows(OUT/'association.csv',[tally])
    save_json(OUT/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),**tally,
        expanded_model_sha256=expanded['model_sha256'],
        inputs={str(x.relative_to(ROOT)):sha(x) for x in
                (*inputs,index_path,parent,delivery_path,controls,Path(__file__))},
        scope='Archived evaluation cache only. Predictions use the original archived and current measurement versions. Current selection membership is kept identical across both models. Reported reference intervals are point-center agreements.',
        outputs={name:sha(OUT/name) for name in ('paired.npz','performance.csv','association.csv')}))
    print(tally,flush=True)


if __name__=='__main__':
    main()
