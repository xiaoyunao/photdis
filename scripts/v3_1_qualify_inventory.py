#!/usr/bin/env python3
"""Audit cached native PSF evidence availability; never inspect model residuals.

This is a feasibility census, not an independent-test role assignment. Exclusion
uses historical reference cells, identities and every recorded field footprint.
"""
import json
import os
import numpy as np
from v2_1_near import ROOT, sha, save_json, csv_rows
from v3_1_inventory import projected
from photdis.stage1 import require_compute, galactic
from photdis.assembly_v1_4 import representative_rows


def main():
    require_compute()
    old = ROOT/'data/baseline_v1_4'; out = ROOT/'data/v3_1/qualified_inventory'
    out.mkdir(parents=True, exist_ok=True)
    paths = [old/'sources.npz', old/'source_evidence.npz', old/'references.npz',
             ROOT/'data/v3_0/references.npz', ROOT/'scripts/v3_1_qualify_inventory.py']
    configs = sorted((ROOT/'configs').glob('field*.json'))
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in paths+configs}
    marker = out/'complete.json'
    if marker.exists():
        record = json.loads(marker.read_text()); assert record['inputs'] == inputs
        for name, digest in record['outputs'].items(): assert sha(out/name) == digest
        return
    raw = projected(paths[0], ['identity','rvs_primary','rvs_sn_r'])
    evidence = projected(paths[1], ['identity','historical_reference_identity',
        'class_identity_conflict','target_identity_conflict','class_label',
        'specdis_valid','specdis_distance','spdist_valid','spdist_distance',
        'astro_quality','zpt_valid','finite_parallax','p_corr','gaia_parallax_error'])
    np.testing.assert_array_equal(raw['identity'], evidence['identity'])
    rep = representative_rows(raw['identity'],raw['rvs_primary'],raw['rvs_sn_r'])
    rep = rep[np.argsort(raw['identity'][rep])]
    evidence = {k:v[rep] for k,v in evidence.items()}; ids = evidence['identity']; del raw
    assert np.all(np.diff(ids) > 0)
    history = [projected(p, ['identity','cell']) for p in paths[2:4]]
    oldids = np.unique(np.concatenate([h['identity'] for h in history]))
    oldcells = np.unique(np.concatenate([h['cell'] for h in history]))
    bounds = [json.loads(p.read_text()).get('bounds_deg') for p in configs]
    bounds = [b for b in bounds if b]
    # Include the v3 once-opened new-sky fields even if absent from field configs.
    bounds += [[10,-45,15,-40],[350,-45,355,-40]]
    pieces = []; source_records = []; missing_type = 0; unmatched = 0
    columns = ['REF_ID','REF_CAT','TYPE','RELEASE','RA','DEC','EBV',
               'grz_base','grz_clean','r','sigma_g','sigma_r','sigma_z']
    for n, meta in enumerate(sorted((old/'photometry').glob('*.json'))):
        record = json.loads(meta.read_text()); path = meta.with_suffix('.npz')
        assert sha(path) == record['sha256'], path
        source_records.append(dict(path=str(path.relative_to(ROOT)), sha256=record['sha256']))
        if not record['selected_rows']: continue
        with np.load(path) as z:
            if 'TYPE' not in z.files:
                missing_type += len(z['REF_ID']); continue
            t = {k:z[k] for k in columns}
        types = np.char.strip(t['TYPE'].astype('U'))
        source = np.where(np.char.strip(t['REF_CAT'].astype('U')) == 'G3', t['REF_ID'], 0)
        j = np.searchsorted(ids, source); safe = np.minimum(j,len(ids)-1)
        matched = (source > 0) & (j < len(ids)) & (ids[safe] == source)
        unmatched += int((~matched).sum()); t = {k:v[matched] for k,v in t.items()}
        types = types[matched]; source = source[matched]; j = j[matched]
        cell = np.floor(t['RA']/10).astype('i8')*18+np.floor((t['DEC']+90)/10).astype('i8')
        footprint = np.zeros(len(j),bool)
        for rlo,dlo,rhi,dhi in bounds:
            ra_in = ((t['RA'] >= rlo) & (t['RA'] <= rhi)) if rlo <= rhi else ((t['RA'] >= rlo) | (t['RA'] <= rhi))
            footprint |= ra_in & (t['DEC'] >= dlo) & (t['DEC'] <= dhi)
        b = galactic(t['RA'],t['DEC'])[1]
        photo = t['grz_base'] & t['grz_clean'] & (abs(b) > 30) & (t['EBV'] >= 0) & (t['EBV'] < .03) & (t['RELEASE'] == 11010)
        conflict = evidence['class_identity_conflict'][j] | evidence['target_identity_conflict'][j]
        freshid = ~np.isin(source,oldids) & ~evidence['historical_reference_identity'][j]
        freshsky = ~np.isin(cell,oldcells) & ~footprint
        pieces.append(dict(identity=source, cell=cell, source_row=j, TYPE=types,
            photo=photo, freshid=freshid, freshsky=freshsky, conflict=conflict,
            bright=(t['r'] < 20) & (np.maximum.reduce([t['sigma_g'],t['sigma_r'],t['sigma_z']]) < .1)))
        if n%200 == 0: print('NATIVE CACHE VERIFIED',n,flush=True)
    assert len(source_records) == 1600, len(source_records)
    t = {k:np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
    _,inverse,counts = np.unique(t['identity'],return_inverse=True,return_counts=True)
    unique = counts[inverse] == 1; j = t['source_row']; types = t['TYPE']
    psf = types == 'PSF'; no_conflict = ~t['conflict']
    new = unique & t['freshid'] & t['freshsky'] & no_conflict
    masks = dict(native_source_matches=np.ones(len(j),bool), unique_identity=unique,
        outside_historical_identity=unique & t['freshid'],
        outside_historical_identity_and_sky=unique & t['freshid'] & t['freshsky'],
        outside_history_without_conflict=new, native_PSF=new & psf,
        native_PSF_photometric_domain=new & psf & t['photo'],
        native_PSF_bright_error_domain=new & psf & t['photo'] & t['bright'])
    rows = []; coverage = []
    for name, mask in masks.items():
        signed = evidence['astro_quality'][j] & evidence['zpt_valid'][j] & evidence['finite_parallax'][j]
        rows.append(dict(stage=name, native_rows=int(mask.sum()), identities=len(np.unique(t['identity'][mask])),
            sky_cells=len(np.unique(t['cell'][mask])), nonPSF=int((mask & (types != '') & ~psf).sum()),
            missing_TYPE=int((mask & (types == '')).sum()),
            class_labels=int((mask & (evidence['class_label'][j] >= 0)).sum()),
            evolved=int((mask & (evidence['class_label'][j] == 1)).sum()),
            signed_gaia=int((mask & signed).sum())))
        if not name.startswith('native_PSF_'): continue
        for source in ['specdis','spdist','gaia_precise']:
            if source == 'gaia_precise':
                available = signed & (evidence['p_corr'][j] > 10*evidence['gaia_parallax_error'][j])
                distance = np.full(len(j),np.nan); distance[available] = 1/evidence['p_corr'][j][available]
            else:
                available = evidence[source+'_valid'][j]; distance = evidence[source+'_distance'][j]
            for lo,hi in zip([0,2,5,10,20,50],[2,5,10,20,50,np.inf]):
                selected = mask & available & (distance >= lo) & (distance < hi)
                coverage.append(dict(stage=name,source=source,low_kpc=lo,high_kpc=None if np.isinf(hi) else hi,
                    n=int(selected.sum()),cells=len(np.unique(t['cell'][selected])),
                    evolved=int((selected & (evidence['class_label'][j] == 1)).sum())))
    csv_rows(out/'admission_funnel.csv',rows); csv_rows(out/'reference_coverage.csv',coverage)
    save_json(out/'native_sources.json',source_records)
    save_json(marker,dict(inputs=inputs, outputs={p.name:sha(p) for p in out.iterdir() if p.name != 'complete.json'},
        job=os.getenv('SLURM_JOB_ID'), native_files=len(source_records), missing_TYPE_unadmitted=missing_type,
        native_without_positive_exact_source_match=unmatched,
        scientific_role_assignments=0, predictions_read=False,
        exclusions='Native duplicate positive IDs excluded; historical reference IDs/cells and all recorded field footprints excluded. No positional/zero-Gaia recovery in this census.',
        limits='Feasibility only. All-history source/system ancestry and final role freeze still required. Bright/error domain does not yet impose QRF support. Cached source availability is not absolute-scale independence.'))
    print('V3.1 NATIVE QUALIFICATION CENSUS COMPLETE',flush=True)


if __name__ == '__main__': main()
