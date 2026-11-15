#!/usr/bin/env python3
"""Measure selection retention and distance performance for legacy VAC matches."""
import argparse
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from photdis.stage1 import require_compute
from photdis.v3.metrics import distance_summary
from v2_1_near import ROOT, atomic_npz, csv_rows, load, save_json, sha
from v3_3_sed import verified
from v3_4_near_reference_support import separation
from v3_5_dr11_production import CUTS, PRODUCTION
from v3_5_dr11_spatial import DELIVERY
from v3_5_dr11_validate import join_parts, take_rows
from v3_6_reference_index import OUT as INDEX, workers
from v3_6_vac_distances import ARCHIVE, DISTANCE_EDGES, OUT as VAC, read_columns


OUT = ROOT/'data/v3_6/vac_flow'
FORECAST = ('ls_id','source_id','identity','TYPE','RA','DEC','observed_r','dereddened_g_r',
            'dwarf_selection_score','nominal_qnear','qrf_qnear','qrf_bins','qrf_quantiles_kpc',
            'primary','strict','band_state','parent_row')
ARCHIVED = ('class_label','class_eligible','split','historical_identity','rvs_parameter_quality',
            'rvs_logg','rvs_teff','rvs_feh','rvs_logg_err','rvs_teff_err','rvs_feh_err')


def gather_forecasts(rows,catalog,receipts):
    offsets = catalog['sweep_offset']
    owner = np.searchsorted(offsets[1:],rows,side='right')
    order = np.argsort(owner,kind='stable')
    boundaries = np.r_[0,np.flatnonzero(np.diff(owner[order]))+1,len(rows)]
    groups = [order[a:b] for a,b in zip(boundaries[:-1],boundaries[1:]) if a<b]

    def read(j):
        tile = owner[j[0]]
        name = catalog['sweep'][tile]
        work = PRODUCTION/Path(name).stem
        marker = verified(work/'complete.json',receipts[name])
        record = json.loads(marker.read_text())
        k = rows[j]-offsets[tile]
        offset = 0
        chunks, slots = [],[]
        for path,digest in sorted(record['outputs'].items()):
            if not path.startswith('part_'):
                continue
            with np.load(work/path,allow_pickle=False) as z:
                ids = z['ls_id']
                take = (k>=offset)&(k<offset+len(ids))
                if take.any():
                    verified(work/path,digest)
                    inner = k[take]-offset
                    p = {field:z[field][inner] for field in FORECAST if field!='ls_id'}
                    p.update(ls_id=ids[inner],sweep=np.full(len(inner),name),part=np.full(len(inner),path),
                             part_row=inner)
                    chunks.append(p)
                    slots.append(j[take])
                offset += len(ids)
        if offset!=offsets[tile+1]-offsets[tile]:
            raise ValueError(f'Part offsets differ from the delivery index: {name}')
        return join_parts(chunks),np.concatenate(slots)

    with ThreadPoolExecutor(max_workers=workers()) as pool:
        products = list(pool.map(read,groups))
    position = np.concatenate([p[1] for p in products])
    np.testing.assert_array_equal(np.sort(position),np.arange(len(rows)))
    data = take_rows(join_parts([p[0] for p in products]),np.argsort(position))
    np.testing.assert_array_equal(data['ls_id'],catalog['ls_id'][rows])
    if not (data['TYPE']=='PSF').all():
        raise ValueError('New VAC matches require native PSF')
    return data


def selection_stages(p,config):
    steps = {}
    for cut in ('primary','strict'):
        gate = config[cut]
        score = p['dwarf_selection_score']>=gate['minimum_dwarf_score']
        near = p['nominal_qnear']>=gate['minimum_near_probability']
        np.testing.assert_array_equal(score&near,p[cut])
        steps[f'{cut}_score_pass'] = score
        steps[f'{cut}_near_pass_after_score'] = score&near
    return steps


def flow(vac,linked,predictions):
    p = predictions
    bins = np.searchsorted(DISTANCE_EDGES,vac['distance_kpc'],side='right')-1
    bins[~vac['finite_distance']] = -1
    rows = []
    for i in range(-1,len(DISTANCE_EDGES)-1):
        all_rows = bins==i
        detected = all_rows[p['vac_row']]
        row = dict(reference_distance_lo_kpc=DISTANCE_EDGES[i] if i>=0 else None,
                   reference_distance_hi_kpc=DISTANCE_EDGES[i+1] if i>=0 else None,
                   raw_rows=int(all_rows.sum()),with_distance_error=int((all_rows&vac['usable_uncertainty']).sum()),
                   historical_quality=int((all_rows&vac['historical_quality']).sum()),
                   no_supported_key=int((all_rows&(vac['match_status']==0)).sum()),
                   ambiguous=int((all_rows&(vac['match_status']==1)).sum()),
                   no_position=int((all_rows&(vac['match_status']==2)).sum()),
                   position_rejected=int((all_rows&(vac['catalog_row']>=0)&~linked).sum()),
                   verified_model_supported=int((all_rows&linked).sum()))
        for cut in ('primary','strict'):
            row[f'{cut}_score_pass'] = int((detected&p[f'{cut}_score_pass']).sum())
            row[f'{cut}_near_pass_after_score'] = int((detected&p[cut]).sum())
            row[f'{cut}_historical_quality'] = int((detected&p[cut]&vac['historical_quality'][p['vac_row']]).sum())
        exclusive = ('no_supported_key','ambiguous','no_position','position_rejected','verified_model_supported')
        if sum(row[k] for k in exclusive)!=row['raw_rows']:
            raise ValueError('VAC raw rows did not reconcile against the supported index')
        rows.append(row)
    return rows


def metrics(p):
    ref = p['distance_kpc']
    with np.errstate(divide='ignore',invalid='ignore'):
        mu = 10+5*np.log10(ref)
        predicted = 10+5*np.log10(p['qrf_quantiles_kpc'])
    r = p['observed_r']; color = p['dereddened_g_r']
    ranges = [('all',np.ones(len(r),bool)),('faint',r>=18),('blue',(color>=.5)&(color<.8)),
              ('faint_blue',(r>=18)&(color>=.5)&(color<.8))]
    result = []
    for name,section in ranges:
        for cut,selected in [('supported',np.ones(len(r),bool)),('primary',p['primary']),('strict',p['strict'])]:
            for cls in ('all',-1,0,1,2):
                domain = section&selected
                if cls!='all':
                    domain &= p['class_label']==cls
                for quality,qmask in [('legacy_quality',p['historical_quality']),
                                      ('any_finite_error',p['usable_uncertainty'])]:
                    for lo,hi in zip(DISTANCE_EDGES[:-1],DISTANCE_EDGES[1:]):
                        use = domain&qmask&(ref>=lo)&(ref<hi)
                        result.append(dict(measurement_domain=name,selection=cut,class_code=cls,
                            reference_quality=quality,distance_lo_kpc=lo,distance_hi_kpc=hi,
                            qrf_training_n=int((use&p['qrf_training']).sum()),
                            **distance_summary(predicted[use],mu[use])))
                        outside = use&~p['qrf_training']
                        result[-1].update({f'outside_qrf_train_{k}':v for k,v in
                                           distance_summary(predicted[outside],mu[outside]).items()})
    return result


def main():
    require_compute()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind',choices=('specdis','spdist'))
    kind = parser.parse_args().kind
    source = VAC/kind/'complete.json'
    m = json.loads(source.read_text())
    vac = load(verified(source.parent/'vac.npz',m['outputs']['vac.npz']))
    index = INDEX/'complete.json'
    rec = json.loads(index.read_text())
    catalog = read_columns(verified(INDEX/'catalog_rows.npz',rec['outputs']['catalog_rows.npz']),
                           ('ls_id','sweep_offset','sweep'))
    receipt = DELIVERY/'complete.json'
    d = json.loads(receipt.read_text())
    with verified(DELIVERY/'sweeps.csv',d['outputs']['sweeps.csv']).open() as stream:
        files = {r['sweep']:r['receipt_sha256'] for r in csv.DictReader(stream)}
    eligible = np.flatnonzero(vac['catalog_row']>=0)
    forecasts = gather_forecasts(vac['catalog_row'][eligible],catalog,files)
    ra,dec = [vac[f'reference_{k}'][eligible] for k in ('ra','dec')]
    use = vac['match_status'][eligible]==4
    use |= separation(ra,dec,forecasts['RA'],forecasts['DEC'])
    matched = eligible[use]
    linked = np.zeros(len(vac['source_id']),bool)
    linked[matched] = True
    p = take_rows(forecasts,use)
    p.update(vac_row=matched, **{k:vac[k][matched] for k in (
        'targetid','catalog_row','archive_row','match_status','distance_kpc','sigma_mu',
        'historical_quality','usable_uncertainty')})
    archive = ARCHIVE/'references.npz'
    a = read_columns(verified(archive,rec['inputs'][str(archive.relative_to(ROOT))]),ARCHIVED)
    old = p['archive_row']>=0
    for key,v in a.items():
        fill = False if v.dtype.kind=='b' else np.nan if v.dtype.kind=='f' else -1
        out = np.full(len(matched),fill,dtype=bool if v.dtype.kind=='b' else float if v.dtype.kind=='f' else 'i8')
        out[old] = v[p['archive_row'][old]]
        p[key] = out
    p['class_label'][~p['class_eligible']] = -1
    p.update(selection_stages(p,json.loads(CUTS.read_text())))
    controls = ROOT/'manifests/v3_0/controls/complete.json'
    control = json.loads(controls.read_text())
    training = read_columns(verified(ROOT/'data/v3_0/controls/training.npz',
                            control['outputs']['training.npz']),('identity','class_label'))
    p['qrf_training'] = np.isin(p['identity'],training['identity'])|np.isin(p['source_id'],training['identity'])
    class_train = training['identity'][training['class_label']>=0]
    p['selector_training'] = np.isin(p['identity'],class_train)|np.isin(p['source_id'],class_train)
    dest = OUT/kind
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():
        raise FileExistsError(dest/'complete.json')
    (n,s,f) = (dest/x for x in ('matched.npz','performance.csv','flow.csv'))
    atomic_npz(n,**p)
    csv_rows(s,metrics(p))
    csv_rows(f,flow(vac,linked,p))
    save_json(dest/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),reference=kind,
              matched=len(matched),primary=int(p['primary'].sum()),strict=int(p['strict'].sum()),
              inputs={str(x.relative_to(ROOT)):sha(x) for x in (Path(__file__),source,index,receipt,CUTS,controls)},
              interpretation='Raw VAC to verified supported catalog and selection. Missing supported keys still need the native-parent audit. Reference-selected data outside QRF training may include historical or calibration exposures.',
              outputs={x.name:sha(x) for x in (n,s,f)}))
    print(kind,len(matched),int(p['primary'].sum()),int(p['strict'].sum()),flush=True)


if __name__ == '__main__':
    main()
