#!/usr/bin/env python3
"""Inventory faint signed parallaxes and independent distance references in fixed apertures."""
import csv
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import zipfile

from astropy.coordinates import SkyCoord
import numpy as np

import environment_apogee as apogee
import environment_pilot as pilot
import environment_results as stats
from environment_transfer import C, FROZEN, OLD, OUT, ROOT, TABLES, configure
from photdis.stage1 import require_compute, tile_bounds
from v2_1_near import atomic_npz, csv_rows, load, sha
from v3_9_references import subset


def astro_inventory(t):
    receipt = apogee.EXTERNAL/'complete.json'
    path = apogee.EXTERNAL/'DR17_astroNN_inventory.npz'
    digest = json.loads(receipt.read_text())['outputs'][path.name]
    a = apogee.read_columns(apogee.verified(path,digest),
                               ('source_id','RA','DEC','distance_precise','class_label','distance_kpc'))
    cat = dict(_gaia=apogee.sorted_key(a['source_id'],np.flatnonzero(a['source_id']>0)),
               reference_ra=a['RA'],reference_dec=a['DEC'])
    ix = apogee.matched(t,cat)
    matched = (ix>=0)&t['unique_G3']
    precise = matched.copy()
    precise[matched] = a['distance_precise'][ix[matched]]
    dist, label = np.full(len(ix),np.nan),np.full(len(ix),-1,'i1')
    dist[precise] = a['distance_kpc'][ix[precise]]
    label[matched] = a['class_label'][ix[matched]]
    return dict(ls_id=t['ls_id'],TYPE=t['TYPE'],astro_distance_kpc=dist,astro_classification=label),digest


def faint_field(item):
    field, files = item
    _,l,b = field
    center = SkyCoord(l,b,frame='galactic',unit='deg').icrs
    cfg = dict(pilot.settings()['south'], maximum_grz_error_mag=C['max_grz_error_mag'],
               domains={'faint':dict(r_min=20,r_max=23)})
    parts = []
    for basename in files:
        path = Path(cfg['sweep_root'])/basename
        raw = pilot.read_native(path)
        with np.errstate(divide='ignore',invalid='ignore'):
            r = (22.5-2.5*np.log10(raw['FLUX_R'].astype('f8'))).astype('f4')
        rect = (r>=20)&(r<23)&(abs(raw['DEC']-center.dec.deg)<C['field_radius_deg'])
        raw = raw[rect]
        delta = np.deg2rad((raw['RA']-center.ra.deg+180)%360-180)
        decl, dec = np.deg2rad(raw['DEC']),np.deg2rad(center.dec.deg)
        cos = np.sin(decl)*np.sin(dec)+np.cos(decl)*np.cos(dec)*np.cos(delta)
        for part,_ in pilot.scan_chunk(raw[cos>=np.cos(np.deg2rad(C['field_radius_deg']))],tile_bounds(path),cfg,0):
            if part:
                parts.append(subset(part,~part['north_primary_region']))
    t = pilot.join(parts)
    assert len(np.unique(t['ls_id']))==len(t['ls_id']) and (t['TYPE']=='PSF').all()
    return field[0],{k:t[k] for k in ('TYPE','ls_id','source_id','RA','DEC','r','row_state')}


def signed_inventory(faint, rosters, field_defs, clean):
    rows = []
    for name,t in faint:
        r = rosters[name]
        precision = t['row_state']==7
        optical = t['row_state']>=6
        ix = pilot.unique_join(t['ls_id'],r['ls_id'])
        np.testing.assert_array_equal(precision,ix>=0)
        support = precision.copy()
        support[precision] = r['support'][ix[precision]]
        nofit = np.zeros(len(ix),bool)
        nofit[precision] = clean[name][ix[precision]]
        ids = t['source_id']
        gaia_valid = (ids>0)&(ids<pilot.NPIX*pilot.SOURCE_ID_HEALPIX_FACTOR)
        gaia_valid &= pilot.unique_join(ids,ids)>=0
        g = pilot.join_gaia(np.where(gaia_valid,ids,0),pilot.GAIA)
        flags = pilot.derive_gaia_labels(g)
        state,_,_,_ = pilot.distances(t,gaia_valid,flags,g,json.loads(pilot.CUTS.read_text()))
        negative = flags['p_corr']<0
        t['longitude'],t['latitude'] = pilot.galactic(t['RA'],t['DEC'])
        blocks = stats.covariance_blocks(t)
        for lo in (20,21,22):
            k = (t['r']>=lo)&(t['r']<lo+1)
            row = dict(field=name,r_lo=lo,r_hi=lo+1,native_psf=int(k.sum()))
            for stage,base in [('all',np.ones(len(k),bool)),('grz',optical),('precise',precision),
                               ('qrf_support',support),('qrf_history_clean',support&nofit)]:
                m = k&base
                weak = m&(state==4)
                row.update({stage+'_n':int(m.sum()),stage+'_signed_usable':int((m&(state>=4)).sum()),
                            stage+'_weak':int(weak.sum()),stage+'_negative':int((m&(state>=4)&negative).sum()),
                            stage+'_strong_geometry':int((m&(state==6)).sum()),
                            stage+'_weak_blocks':len(np.unique(blocks[weak]))})
            row['joint_weak_inventory_floor'] = row['qrf_history_clean_weak']>=200 and row['qrf_history_clean_weak_blocks']>=8
            rows.append(row)
    return rows


def distance_inventory(rosters, clean):
    rows = []
    for name,t in rosters.items():
        tags = stats.tag(t)
        for source in ('gaia','specdis','spdist','astro'):
            d = t[source+'_distance_kpc']
            for lo,hi in ((2,5),(5,10)):
                k = (d>=lo)&(d<hi)
                s = k&t['support']
                c = s&clean[name]
                counts = {f'n_{label}':int((c&(tags==label)).sum()) for label in sorted(set(tags))}
                nblocks = len(np.unique(stats.covariance_blocks(t)[c]))
                provenance = {'gaia':'signed zero-point-corrected DR3 geometry',
                    'specdis':'spectro-photometric teacher, ancestry overlaps QRF',
                    'spdist':'spectro-photometric teacher, ancestry overlaps QRF',
                    'astro':'APOGEE astroNN; method shares stellar information and reference scale'}[source]
                rows.append(dict(field=name,source=source,lo_kpc=lo,hi_kpc=hi,qualified=int(k.sum()),
                                 qrf_support=int(s.sum()),fit_overlap=int((k&t['qrf_fit_overlap']).sum()),
                                 history_overlap=int((k&t['historical_overlap']).sum()),
                                 history_clean_qrf=int(c.sum()),blocks=nblocks,r20plus=int((c&(t['r']>=20)).sum()),
                                 median_r=float(np.median(t['r'][c])) if c.any() else '',
                                 provenance=provenance,**counts))
    return rows


def cluster_inventory(fields, rosters, clean):
    from nat8_audit import members
    source = ROOT/'data/reference/vasiliev_baumgardt_2021_clusters.txt'
    archive = ROOT/'data/reference/vasiliev_baumgardt_2021_members.zip'
    records = np.genfromtxt(source,names=True,dtype=None,encoding='utf8')
    clusters = SkyCoord(records['ra'],records['dec'],frame='icrs',unit='deg')
    radii = records['rmax']/60
    rows = []
    with zipfile.ZipFile(archive) as z:
        for name,l,b in fields:
            center = SkyCoord(l,b,frame='galactic',unit='deg').icrs
            sep = center.separation(clusters).deg
            t = rosters[name]
            for lo,hi in ((2,5),(5,10)):
                dist = (records['dist']>=lo)&(records['dist']<hi)
                possible = np.flatnonzero(dist & (sep<=C['field_radius_deg']+radii))
                if not len(possible):
                    rows.append(dict(field=name,lo_kpc=lo,hi_kpc=hi,possible_systems=0,
                                     nearest_known_center_deg=float(sep[dist].min()),system='',
                                     strong_members=0,supported_history_clean=0))
                for j in possible:
                    star_name = str(records['Name'][j])
                    _, member = members(star_name,z)
                    ids = [int(row[0]) for row in member if float(row[-1])>.9]
                    use = np.isin(t['source_id'],ids)&t['unique_G3']
                    rows.append(dict(field=name,lo_kpc=lo,hi_kpc=hi,possible_systems=len(possible),
                                     nearest_known_center_deg=float(sep[j]),system=star_name,
                                     reference_kpc=records['dist'][j],reference_error_kpc=records['disterr'][j],
                                     strong_members=int(use.sum()),supported_history_clean=int((use&clean[name]&t['support']).sum())))
    return rows, {str(p):sha(p) for p in (source,archive)}


def main():
    require_compute()
    configure()
    TABLES.mkdir(parents=True,exist_ok=True)
    frozen = json.loads(FROZEN.read_text())
    files = {}
    fields, catalogs, clean = [],{},{}
    for root in (OLD,OUT):
        t = load(root/'forecast/roster.npz')
        if root==OUT:
            a,digest = astro_inventory(t)
            dest = OUT/'apogee'
            dest.mkdir(exist_ok=True)
            atomic_npz(dest/'references.npz',**a)
            pilot.finish(dest,dict(archive_sha256=digest,forecast=sha(OUT/'forecast/complete.json')))
            field_defs = [f for f in C['fields'] if f[0] in frozen['selected']]
            m = load(FROZEN.parent/'identity_masks.npz')
            np.testing.assert_array_equal(m['ls_id'],t['ls_id'])
            independent = m['clean']
        else:
            a = load(root/'apogee/references.npz')
            field_defs = json.loads((ROOT/'configs/environment_20261120.json').read_text())['fields']
            independent = ~(t['qrf_fit_overlap']|t['historical_overlap'])
        np.testing.assert_array_equal(t['ls_id'],a['ls_id'])
        assert (t['TYPE']=='PSF').all()
        t.update(a)
        with (root/'census/raw_admission.csv').open() as f:
            ledger = list(csv.DictReader(f))
        for name,l,b in field_defs:
            k = t['field']==name
            catalogs[name] = subset(t,k)
            clean[name] = independent[k]
            files[name] = [r['sweep'] for r in ledger if r['field']==name]
            fields.append([name,l,b])
    rows = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        for t in ex.map(faint_field,[(f,files[f[0]]) for f in fields]):
            rows.extend(signed_inventory([t],catalogs,fields,clean))
            print('Faint signed Gaia inventory',t[0],flush=True)
    groups, inputs = cluster_inventory(fields,catalogs,clean)
    outputs = dict(faint_signed_parallax=rows, distant_reference=distance_inventory(catalogs,clean),
                   cluster_inventory=groups)
    for name,values in outputs.items():
        csv_rows(TABLES/(name+'.csv'),values)
    pilot.save_json(TABLES/'inventory_receipt.json',dict(freeze=sha(FROZEN),
         inputs=dict(inputs,old=sha(OLD/'forecast/complete.json'),new=sha(OUT/'forecast/complete.json')),
         outputs={name+'.csv':sha(TABLES/(name+'.csv')) for name in outputs}))


if __name__=='__main__':
    main()
