#!/usr/bin/env python3
"""Find physical duplicates around the Legacy Surveys ownership boundary."""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from photdis.reference_v1_4 import unique_join
from photdis.stage1 import galactic, tile_bounds
from v2_1_near import atomic_npz, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_5_dr11_production import PRODUCTION
from v3_5_dr11_spatial import DELIVERY
from v3_6_vac_distances import read_columns
from v3_9_catalog import OUT
from v3_9_references import join
from v3_8_candidate import read_cache

BOUNDARY = 32.375
HALF_WIDTH = 2.
KEYS = ('ls_id','TYPE','source_id','RA','DEC')


def within_strip(bounds):
    _,lo,_,hi = bounds
    return lo<BOUNDARY+HALF_WIDTH and hi>BOUNDARY-HALF_WIDTH


def take(path):
    a = read_columns(path,KEYS)
    keep = (np.abs(a['DEC']-BOUNDARY)<HALF_WIDTH)&(galactic(a['RA'],a['DEC'])[1]>30)
    if np.any(a['TYPE'][keep]!='PSF'):
        raise ValueError('Seam matching requires verified native PSF on both sides')
    return {k:v[keep] for k,v in a.items()}


def native_rows(side):
    items = OUT/side/'inventory.json'
    sources = [s for group in json.loads(items.read_text())['groups'] for s in group
               if within_strip(s['bounds_deg'])]
    parts = []
    for s in sources:
        stem = Path(s['filename']).stem
        p = OUT/side/'photo'/stem
        q = OUT/side/'predict'/stem
        photo = json.loads((p/'complete.json').read_text())
        prediction = json.loads((q/'complete.json').read_text())
        if sha(p/'complete.json') != prediction['binding']['photo']:
            raise ValueError(f'Parent mismatch for seam slab {stem}')
        for name,h in photo['outputs'].items():
            if name=='parent_state.npz':
                continue
            a = take(verified(p/name,h))
            if not len(a['ls_id']):
                continue
            flags = read_columns(verified(q/('state_'+name),prediction['outputs']['state_'+name]),
                                 ('ls_id','state'))
            order = np.argsort(flags['ls_id'])
            ix = np.searchsorted(flags['ls_id'],a['ls_id'],sorter=order)
            np.testing.assert_array_equal(flags['ls_id'][order[ix]],a['ls_id'])
            state = flags['state'][order[ix]]
            parts.append(dict(**a,prediction_state=state,version=np.full(len(state),'v3.9')))
    if side=='south':
        with read_cache(DELIVERY,'sweeps.csv').open() as stream:
            for s in csv.DictReader(stream):
                if not within_strip(tile_bounds(s['sweep'])):
                    continue
                p = PRODUCTION/Path(s['sweep']).stem
                photo = json.loads(verified(p/'complete.json',s['receipt_sha256']).read_text())
                for name,h in photo['outputs'].items():
                    if not name.startswith('part_'):
                        continue
                    a = take(verified(p/name,h))
                    n = len(a['ls_id'])
                    if n:
                        parts.append(dict(**a,prediction_state=np.full(n,3,'u1'),version=np.full(n,'v3.5')))
    return join(parts)


def on_sphere(ra,dec):
    a,b = np.deg2rad(ra),np.deg2rad(dec)
    return np.column_stack((np.cos(a)*np.cos(b),np.sin(a)*np.cos(b),np.sin(b)))


def match(n,s):
    # Admit G3 keys only if unique in both surveys and angularly consistent.
    # The geometric route also works for sources measured without Gaia.
    nxyz,sxyz = on_sphere(n['RA'],n['DEC']),on_sphere(s['RA'],s['DEC'])
    ns,ss = n['source_id'],s['source_id']
    i = np.flatnonzero((ns>0)&(unique_join(ns,ns)>=0))
    j = unique_join(ns[i],ss)
    found = j>=0
    i,j = i[found],j[found]
    close = np.linalg.norm(nxyz[i]-sxyz[j],axis=1)<np.deg2rad(1.5/3600)
    g1,g2 = i[close],j[close]
    ni,si = np.ones(len(ns),bool),np.ones(len(ss),bool)
    ni[g1] = False
    si[g2] = False
    ni,si = np.flatnonzero(ni),np.flatnonzero(si)
    candidates = cKDTree(nxyz[ni]).sparse_distance_matrix(cKDTree(sxyz[si]),
                      np.deg2rad(.7/3600),output_type='coo_matrix')
    # An incompatible G3 identity cannot be overridden by a positional match.
    c1,c2 = ni[candidates.row],si[candidates.col]
    credible = (ns[c1]<=0)|(ss[c2]<=0)|(ns[c1]==ss[c2])
    c1,c2 = c1[credible],c2[credible]
    unique = (np.bincount(c1,minlength=len(ns))[c1]==1)&(np.bincount(c2,minlength=len(ss))[c2]==1)
    a,b = np.r_[g1,c1[unique]],np.r_[g2,c2[unique]]
    method = np.repeat(['gaia_and_position','unique_position'],[len(g1),unique.sum()])
    if len(np.unique(a))!=len(a) or len(np.unique(b))!=len(b):
        raise ValueError('North–south counterpart has multiple assignments')
    return a,b,method,int((~unique).sum()),int((~credible).sum()),int((~close).sum())


def run(root):
    marker = root/'complete.json'
    sources = (OUT/'north/inventory.json',OUT/'south/inventory.json',DELIVERY/'complete.json',Path(__file__))
    bound = {p.name if p==Path(__file__) else str(p.parent.name+'/'+p.name):sha(p) for p in sources}
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved['inputs']!=bound:
            raise ValueError('Seam data are already bound to a different inventory')
        for name,h in saved['outputs'].items():
            verified(root/name,h)
        return
    n,s = native_rows('north'),native_rows('south')
    for label,a in (('north',n),('south',s)):
        if not a or len(np.unique(a['ls_id']))!=len(a['ls_id']):
            raise ValueError(f'Missing or non-unique {label} native sweep rows')
    i,j,method,ambiguous,conflict,bad_astrometry = match(n,s)
    if not len(i):
        raise ValueError('No verified native overlap pairs found at the ownership seam')
    north,south = n['ls_id'][i],s['ls_id'][j]
    keep_north = (n['DEC'][i]+s['DEC'][j])/2>=BOUNDARY
    before_n,before_s = n['DEC'][i]>=BOUNDARY,s['DEC'][j]<BOUNDARY
    count_before = before_n.astype('u1')+before_s
    # The published merge first applies the geometric ownership, then replaces
    # the owner bit for rows listed here. Native sources and forecasts stay accessible.
    changes = (before_n!=keep_north)|(before_s==keep_north)
    root.mkdir(parents=True,exist_ok=True)
    atomic_npz(root/'paired_sources.npz',north_ls_id=north,south_ls_id=south,TYPE=n['TYPE'][i],
               north_source_id=n['source_id'][i],south_source_id=s['source_id'][j],
               north_dec=n['DEC'][i],south_dec=s['DEC'][j],method=method,
               primary_ls_id=np.where(keep_north,north,south),
               secondary_ls_id=np.where(keep_north,south,north),
               north_prediction_state=n['prediction_state'][i],south_prediction_state=s['prediction_state'][j],
               south_version=s['version'][j])
    override = np.r_[north[changes],south[changes]]
    new_owner = np.r_[keep_north[changes],~keep_north[changes]]
    order = np.argsort(override)
    atomic_npz(root/'ownership_overrides.npz',ls_id=override[order],TYPE=np.full(len(order),'PSF'),
               primary_survey_owner=new_owner[order])
    columns = dict(north_rows=len(n['ls_id']),south_rows=len(s['ls_id']),paired=len(i),
                   gaia_pairs=int(np.sum(method=='gaia_and_position')),
                   position_pairs=int(np.sum(method=='unique_position')),
                   ambiguous_positional_edges=ambiguous,conflicting_positional_edges=conflict,
                   gaia_position_failures=bad_astrometry,
                   formerly_two_owners=int(np.sum(count_before==2)),
                   formerly_no_owner=int(np.sum(count_before==0)),override_rows=len(order))
    csv_rows(root/'counts.csv',[columns])
    save_json(marker,dict(**columns,inputs=bound,outputs={p.name:sha(p) for p in root.iterdir()}))
