#!/usr/bin/env python3
"""Complete frozen nominal gates and derive calibrated full distance PDFs."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from photdis.stage1 import require_compute
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_4_near_forecast import qrf_bins
from v3_5_dr11_production import PRODUCTION
from v3_5_dr11_spatial import DELIVERY
from v3_5_dr11_validate import join_parts
from v3_6_catalog_projection import OUT as PROJECTION
from v3_6_vac_distances import read_columns
import v3_7_selection_replay as replay
from v3_8_candidate import read_cache
from v3_8_candidate_index import OUT as MEMBERS
from v3_8_redrock import OUT as REDROCK
from v3_8_reliability import OUT as RELIABILITY
from v3_9_audit import verify_pdf

OUT = ROOT/'data/v3_9/existing'
GATES = OUT/'gates'
PDF = OUT/'pdf'


def prepare(shards):
    GATES.mkdir(parents=True,exist_ok=True)
    mark = GATES/'prepare.json'
    if mark.exists():
        old = json.loads(mark.read_text())
        if old['shards']!=shards:
            raise ValueError('The partition count is frozen')
        return
    m = read_columns(read_cache(MEMBERS,'membership.npz'),('catalog_row','ls_id','status','TYPE','qrf_median_kpc'))
    k = (m['status']==3)&(m['qrf_median_kpc']>=5)&(m['qrf_median_kpc']<10)
    target,lsid = m['catalog_row'][k],m['ls_id'][k]
    if len(target)!=336678 or len(np.unique(target))!=len(target) or not (m['TYPE'][k]=='PSF').all():
        raise ValueError('Frozen unresolved count, ownership or morphology changed')
    r = read_columns(read_cache(REDROCK,'controls.npz'),('ls_id','TYPE','nominal','primary','candidate'))
    rr = np.isfinite(r['nominal']) & np.isin(r['ls_id'],lsid)
    if not (r['TYPE'][rr]=='PSF').all():
        raise ValueError('Invalid reused Redrock gate')
    rr_ids = r['ls_id'][rr]
    cache = GATES/'reused.npz'
    rid = np.argsort(lsid)
    loc = rid[np.searchsorted(lsid[rid],rr_ids)]
    atomic_npz(cache,ls_id=rr_ids,catalog_row=target[loc],TYPE=r['TYPE'][rr],nominal_qnear=r['nominal'][rr])
    need = target[~np.isin(lsid,rr_ids)]
    paths = json.loads((PROJECTION/'complete.json').read_text())['child_receipts']
    pieces = []
    for name,h in paths.items():
        parent = verified(ROOT/name,h).parent
        a = read_columns(read_cache(parent,'rows.npz'),replay.POINTER)
        use = np.isin(a['catalog_row'],need)
        if use.any():
            pieces.append({key:val[use] for key,val in a.items()})
    p = join_parts(pieces)
    np.testing.assert_array_equal(np.sort(p['catalog_row']),np.sort(need))
    i = np.argsort(target)
    j = i[np.searchsorted(target[i],p['catalog_row'])]
    np.testing.assert_array_equal(p['ls_id'],lsid[j])
    outputs = {cache.name:sha(cache)}
    for shard in range(shards):
        path = GATES/f'need_{shard:02d}.npz'
        take = p['sweep_number']%shards==shard
        atomic_npz(path,**{key:v[take] for key,v in p.items()})
        outputs[path.name] = sha(path)
    sources = (MEMBERS/'complete.json',REDROCK/'complete.json',PROJECTION/'complete.json',Path(__file__).resolve())
    save_json(mark,dict(shards=shards,target=len(target),reused=len(rr_ids),inputs={
        str(q.relative_to(ROOT)):sha(q) for q in sources},outputs=outputs))
    print('Frozen unresolved targets',len(target),'reused',len(rr_ids),flush=True)


def gate(shard):
    receipt = json.loads((GATES/'prepare.json').read_text())
    if not 0<=shard<receipt['shards']:
        raise ValueError('Shard outside frozen manifest')
    replay.OUT = GATES
    if not (GATES/f'replay_{shard:02d}.json').exists():
        replay.replay(shard,receipt)


def finish_gates():
    meta = json.loads((GATES/'prepare.json').read_text())
    r = read_columns(verified(GATES/'reused.npz',meta['outputs']['reused.npz']),
                     ('ls_id','catalog_row','TYPE','nominal_qnear'))
    r['numerical_flag'] = np.zeros(len(r['ls_id']),bool)
    r['from_redrock'] = np.ones(len(r['ls_id']),bool)
    r['numerical_flag_known'] = np.zeros(len(r['ls_id']),bool)
    parts = [r]
    for i in range(meta['shards']):
        name = f'replay_{i:02d}'
        check = json.loads((GATES/f'{name}.json').read_text())
        if check['need_sha256']!=meta['outputs'][f'need_{i:02d}.npz']:
            raise ValueError('Nominal replay does not match its source partition')
        part = read_columns(read_cache(GATES,name+'.npz',name+'.json'),
                            ('ls_id','catalog_row','TYPE','nominal_qnear','numerical_flag'))
        part['from_redrock'] = np.zeros(len(part['ls_id']),bool)
        part['numerical_flag_known'] = np.ones(len(part['ls_id']),bool)
        parts.append(part)
    p = join_parts(parts)
    m = read_columns(read_cache(MEMBERS,'membership.npz'),('catalog_row','ls_id','status','qrf_median_kpc'))
    wanted = (m['status']==3)&(m['qrf_median_kpc']>=5)&(m['qrf_median_kpc']<10)
    i,j = np.argsort(p['catalog_row']),np.argsort(m['catalog_row'][wanted])
    for key in ('catalog_row','ls_id'):
        np.testing.assert_array_equal(p[key][i],m[key][wanted][j])
    if not np.isfinite(p['nominal_qnear']).all() or not (p['TYPE']=='PSF').all():
        raise ValueError('Completed nominal gates must be finite native PSFs')
    p['candidate'] = p['nominal_qnear']>=.90
    file = GATES/'completed_gates.npz'
    atomic_npz(file,**{key:val[i] for key,val in p.items()})
    save_json(GATES/'complete.json',dict(rows=len(i),passes=int(p['candidate'].sum()),
        from_redrock=int(r['from_redrock'].sum()),numerical_flags=int(p['numerical_flag'].sum()),
        outputs={file.name:sha(file)},prepare_sha256=sha(GATES/'prepare.json')))


def transform_pdf(edges,mass,quantile,scale):
    if np.any(scale<=0):
        raise ValueError('PDF scale must be positive')
    mu50 = 10+5*np.log10(quantile[:,2])
    new_edges = mu50[:,None]+(edges-mu50[:,None])*scale[:,None]
    new_quantile = quantile[:,2,None]*(quantile/quantile[:,2,None])**scale[:,None]
    np.testing.assert_array_equal(new_quantile[:,2],quantile[:,2])
    if not np.all(np.diff(new_edges,axis=1)>0):
        raise ValueError('Transformed PDF edges must remain ordered')
    result = qrf_bins(new_edges,mass)
    np.testing.assert_allclose(result.sum(axis=1),1.,atol=1e-12,rtol=0)
    return new_edges,new_quantile,result


def calibrated(shard,count):
    dest = PDF/f'{shard:02d}'
    dest.mkdir(parents=True,exist_ok=True)
    marker = dest/'complete.json'
    if marker.exists():
        if json.loads(marker.read_text())['shards']!=count:
            raise ValueError('PDF shards use the original partition count')
        return
    table = read_cache(RELIABILITY,'interval_calibration.csv')
    with table.open() as stream:
        rules = [r for r in csv.DictReader(stream) if r['status']=='evaluated']
    if len(rules)!=4:
        raise ValueError('Expected four validated brightness/color/WISE cells')
    with read_cache(DELIVERY,'sweeps.csv').open() as stream:
        sweeps = list(csv.DictReader(stream))
    rows,outputs = [],{}
    for item in sweeps[shard::count]:
        directory = PRODUCTION/Path(item['sweep']).stem
        source = verified(directory/'complete.json',item['receipt_sha256'])
        parts = json.loads(source.read_text())['outputs']
        for name,h in parts.items():
            if not name.startswith('part_'):
                continue
            with np.load(verified(directory/name,h),allow_pickle=False) as z:
                sample = z['primary']&~z['nominal_numerical_flag']&(z['observed_r']>=16)&(z['observed_r']<18)
                quantile = z['qrf_quantiles_kpc']
                sample &= (quantile[:,2]>0)&(quantile[:,2]<2)
                color,w = z['dereddened_g_r'],z['band_state'][:,4]
                scale = np.zeros(len(sample))
                for rule in rules:
                    use = (sample&(color>=float(rule['color_lo']))&(color<float(rule['color_hi']))&
                           (w==int(rule['w1_state'])))
                    if np.any(scale[use]>0):
                        raise ValueError('Overlapping calibrated cells')
                    scale[use] = float(rule['scale'])
                take = scale>0
                if not take.any():
                    continue
                if not (z['TYPE'][take]=='PSF').all():
                    raise ValueError('Calibrated rows require verified native PSF')
                mass = z['qrf_mu_mass'][take]
                edges,qq,bins = transform_pdf(z['qrf_mu_edges'][take],mass,quantile[take],scale[take])
                cdf_error,shell_error = verify_pdf(edges,mass,qq,bins)
                path = dest/f'{directory.name}_{name}'
                atomic_npz(path,ls_id=z['ls_id'][take],TYPE=z['TYPE'][take],scale=scale[take],
                           qrf_mu_edges=edges,qrf_mu_mass=mass,qrf_bins=bins,qrf_quantiles_kpc=qq)
                outputs[path.name] = sha(path)
                old = z['qrf_bins'][take].sum(axis=0)
                row = dict(sweep=item['sweep'],part=name,rows=int(take.sum()),source_sha256=h,
                           quantile_cdf_error=cdf_error,shell_probability_error=shell_error)
                for i,(a,b) in enumerate(zip(old,bins.sum(axis=0))):
                    row.update({f'original_{i}':float(a),f'calibrated_{i}':float(b)})
                rows.append(row)
    path = dest/'shell_counts.csv'
    csv_rows(path,rows)
    outputs[path.name] = sha(path)
    save_json(marker,dict(shards=count,shard=shard,rows=sum(r['rows'] for r in rows),
        calibration_sha256=sha(table),outputs=outputs))


def main():
    require_compute()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('prepare','gate','finish','pdf'))
    p.add_argument('--shard',type=int,default=0)
    p.add_argument('--count',type=int,default=24)
    a = p.parse_args()
    if a.stage=='prepare':
        prepare(a.count)
    elif a.stage=='gate':
        gate(a.shard)
    elif a.stage=='finish':
        finish_gates()
    else:
        calibrated(a.shard,a.count)


if __name__=='__main__':
    main()
