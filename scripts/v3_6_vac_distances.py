#!/usr/bin/env python3
"""Recover both historical distance VACs, preserving the raw inventory and errors."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from photdis.gaia import _decode_boolean
from photdis.reference_v1_4 import pair_keys, unique_join
from photdis.stage1 import require_compute
from v2_1_near import ROOT, atomic_npz, csv_rows, load, save_json, sha
from v3_3_sed import verified
from v3_6_reference_index import OUT as INDEX


ARCHIVE = ROOT/'data/baseline_v1_4'
OUT = ROOT/'data/v3_6/vac_distances'
KEYS = ('source_id','targetid')
SPECDIS = ('ra','dec','dist','disterr','binary_flag','binary_possibility',
           'parallax','parallax_err','parallax_zpc','pmra','pmdec')
SPDIST = ('flag_good','qflag_input','g0','e_gmag','mg_50','mg_16','mg_84','mg_5','mg_95',
          'teff_cor','logg_cor')
DISTANCE_EDGES = np.array([0.,2.,5.,10.,20.,50.,np.inf])
ASSOC = ('no_supported_key','ambiguous_or_duplicate','no_source_position',
         'requires_native_position_check','verified_historical_position')


def read_columns(path, keys):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k] for k in keys}


def distance_quality(payload, kind):
    """Raw measurable quantities, independent of the later quality decision."""
    with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
        if kind=='specdis':
            d,e = (payload[k].astype('f8') for k in ('dist','disterr'))
            sigma = 5/np.log(10)*np.sqrt(np.log1p((e/d)**2))
            valid = np.isfinite(e)&(e>0)&(e<d)&(payload['binary_flag']==1)
        else:
            g,m,lo,hi,e = (payload[k].astype('f8') for k in ('g0','mg_50','mg_16','mg_84','e_gmag'))
            d = 10**((g-m-10)/5)
            sigma = np.sqrt(e**2+((hi-lo)/2)**2+.167**2)
            valid = (_decode_boolean(payload['flag_good'],'flag_good') &
                     _decode_boolean(payload['qflag_input'],'qflag_input') & (e>=0)&(g>10)&(g<30)&
                     (m>-8)&(m<15)&(hi>=lo)&(hi-lo<4))
    center = np.isfinite(d)&(d>0)
    uncertainty = np.isfinite(sigma)&(sigma>0)
    if kind=='specdis':
        uncertainty &= e>0
    return dict(distance_kpc=np.where(center,d,np.nan),sigma_mu=np.where(center&uncertainty,sigma,np.nan),
                finite_distance=center,usable_uncertainty=center&uncertainty,
                historical_quality=center&uncertainty&valid)


def positions(payload,mws,kind):
    if kind=='specdis':
        return payload['ra'],payload['dec']
    # SPDist carries no sky coordinates. Reuse the matching DESI pair positions
    # from the source archive, keeping unmatched/varying visits separate.
    pairs = pair_keys(mws['rvs_targetid'],mws['source_id'])
    primary = np.flatnonzero(_decode_boolean(mws['rvs_primary'],'rvs_primary'))
    m = unique_join(pair_keys(payload['targetid'],payload['source_id']),pairs[primary])
    accepted = (m>=0)&(payload['source_id']>0)&(payload['targetid']>0)
    coords = []
    for k in ('ra','dec'):
        x = np.full(len(m),np.nan)
        x[accepted] = mws[k][primary[m[accepted]]]
        coords.append(x)
    return coords


def bind_matches(payload, archive, attached, catalog, mws, kind):
    n = len(payload['source_id'])
    # The reference row index is upstream source provenance, not a distance
    # selection. Reuse it even when the old distance quality cut failed.
    refs = np.flatnonzero(archive[f'{kind}_row']>=0)
    back = unique_join(np.arange(n),archive[f'{kind}_row'][refs])
    old_row = np.full(n,-1,'i8')
    has_old = back>=0
    old_row[has_old] = refs[back[has_old]]
    old_catalog = np.full(n,-1,'i8')
    old_catalog[has_old] = attached['catalog_row'][old_row[has_old]]
    raw_ids = payload['source_id']
    positive = raw_ids>0
    ra,dec = positions(payload,mws,kind)
    has_position = np.isfinite(ra)&np.isfinite(dec)&(abs(dec)<=90)
    native_rows = np.flatnonzero(catalog['source_id']>0)
    j = unique_join(raw_ids,catalog['source_id'][native_rows])
    catalog_row = np.full(n,-1,'i8')
    found = (j>=0)&positive
    catalog_row[found] = native_rows[j[found]]
    status = np.zeros(n,'u1')
    status[(catalog_row>=0)&~has_position] = 2
    status[(catalog_row>=0)&has_position] = 3
    good_old = old_catalog>=0
    conflicts = good_old&(catalog_row>=0)&(old_catalog!=catalog_row)
    catalog_row[good_old] = old_catalog[good_old]
    status[good_old] = 4
    # Keep a verified historical parent as the canonical comparison row when
    # multiple VAC rows use the same Gaia ID; all raw rows remain in the output.
    duplicate_gaia = positive&(unique_join(raw_ids,raw_ids)<0)
    duplicate_link = (old_catalog>=0)&(unique_join(old_catalog,old_catalog)<0)
    status[((catalog_row>=0)&duplicate_gaia&~good_old)|duplicate_link|conflicts] = 1
    return dict(archive_row=old_row,catalog_row=np.where(status>=3,catalog_row,-1),
                match_status=status,reference_ra=ra,reference_dec=dec,duplicate_source_id=duplicate_gaia)


def inventory(data):
    edges = np.r_[-1.,DISTANCE_EDGES]
    # -1 to 0 holds missing reference distances without merging it with 0–2.
    bin_id = np.searchsorted(edges,np.nan_to_num(data['distance_kpc'],nan=-.5),side='right')-1
    rows = []
    for i,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
        m = bin_id==i
        rows.append(dict(reference_distance_lo_kpc=lo,reference_distance_hi_kpc=hi,raw_rows=int(m.sum()),
                         with_uncertainty=int((m&data['usable_uncertainty']).sum()),
                         historical_quality=int((m&data['historical_quality']).sum()),
                         **{name:int((m&(data['match_status']==j)).sum()) for j,name in enumerate(ASSOC)}))
    if sum(r['raw_rows'] for r in rows)!=len(bin_id):
        raise ValueError('VAC inventory lost a reference row')
    return rows


def main():
    require_compute()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind',choices=('specdis','spdist'))
    kind = parser.parse_args().kind
    rec_path = INDEX/'complete.json'
    rec = json.loads(rec_path.read_text())
    attached = load(verified(INDEX/'baseline_v1_4.npz',rec['outputs']['baseline_v1_4.npz']))
    catalog = read_columns(verified(INDEX/'catalog_rows.npz',rec['outputs']['catalog_rows.npz']),('ls_id','source_id'))
    archive_path = ARCHIVE/'references.npz'
    archive = read_columns(verified(archive_path,rec['inputs'][str(archive_path.relative_to(ROOT))]),
                           (f'{kind}_row',))
    raw_marker = ARCHIVE/f'{kind}_raw.json'
    raw_hash = json.loads(raw_marker.read_text())['sha256']
    cols = SPECDIS if kind=='specdis' else SPDIST
    payload = read_columns(verified(ARCHIVE/f'{kind}_raw.npz',raw_hash),KEYS+cols)
    inputs = {str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),rec_path,raw_marker)}
    mws = {}
    if kind=='spdist':
        mws_marker = ARCHIVE/'sources.json'
        wanted = ('rvs_targetid','rvs_primary','source_id','ra','dec')
        mws = read_columns(verified(ARCHIVE/'sources.npz',json.loads(mws_marker.read_text())['source_sha256']),wanted)
        inputs[str(mws_marker.relative_to(ROOT))] = sha(mws_marker)
    payload.update(distance_quality(payload,kind),**bind_matches(payload,archive,attached,catalog,mws,kind))
    dest = OUT/kind
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():
        raise FileExistsError(dest/'complete.json')
    p = dest/'vac.npz'
    atomic_npz(p,**payload)
    table = dest/'inventory.csv'
    counts = inventory(payload)
    csv_rows(table,counts)
    save_json(dest/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),reference=kind,inputs=inputs,
              raw_sha256=raw_hash,inventory_rows=len(payload['source_id']),statuses=ASSOC,
              interpretation='Raw distance rows and candidate joins. Status 3 still requires native position verification.',
              outputs={x.name:sha(x) for x in (p,table)}))
    print(kind,len(payload['source_id']),counts,flush=True)


if __name__=='__main__':
    main()
