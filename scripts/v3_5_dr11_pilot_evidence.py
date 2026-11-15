#!/usr/bin/env python3
"""Check boundary transfer, Gaia eligibility and spectral support on fixed pilots."""
from collections import Counter
import json
import os
from pathlib import Path

import numpy as np

from photdis.gaia import derive_gaia_labels, join_gaia
from photdis.reference_v1_4 import unique_join
from photdis.stage1 import require_compute
from photdis.v3.metrics import distance_summary, histogram_parallax
from v2_1_near import ROOT, csv_rows, load, save_json, sha
from v3_3_sed import verified
from v3_4_near_recover_labels import archived_labels
from v3_4_near_reference_support import past_evidence, separation, spectroscopic_inventory
from v3_4_near_validate import GAIA, geometry, read_targeted, tile_inventory
from v3_5_dr11_inventory import CONFIG, OUT
from v3_5_dr11_pilot import CUTS, PILOT, in_tile

PRODUCT = OUT/'pilot_evidence'


def pilot_input(inv, hashes):
    parts, meta, rosters = [], [], []
    for name in inv['pilot']:
        folder = PILOT/Path(name).stem
        marker = folder/'complete.json'
        entry = json.loads(marker.read_text())
        if (entry['source']!=name or not entry['all_PSF'] or entry['new_outcomes_read']):
            raise ValueError(f'Unregistered pilot prediction source: {name}')
        hashes[str(marker.relative_to(ROOT))] = sha(marker)
        q = load(verified(folder/'supported_roster.npz',entry['outputs']['supported_roster.npz']))
        if not in_tile(q['RA'],q['DEC'],entry['bounds']).all():
            raise ValueError(f'Pilot straddles the native half-open boundary: {name}')
        rosters.append(q)
        if entry['sample_n']:
            p = load(verified(folder/'prediction.npz',entry['outputs']['prediction.npz']))
            np.testing.assert_array_equal(p['TYPE'],np.full(len(p['ls_id']),'PSF'))
            p['sweep'] = np.full(len(p['ls_id']),name)
            parts.append(p)
        meta.append(entry)
    ids = np.concatenate([r['ls_id'] for r in rosters])
    if len(np.unique(ids))!=len(ids):
        raise ValueError('Legacy ownership fails across pilot sweeps')
    return {k:np.concatenate([p[k] for p in parts]) for k in parts[0]}, meta, rosters


def groups(photo):
    r,color = photo['observed_r'],photo['dereddened_g_r']
    m = {'all':np.ones(len(r),bool), 'r18_20':(r>=18)&(r<20), 'color05_08':(color>=.5)&(color<.8)}
    m['r18_20_color05_08'] = m['r18_20']&m['color05_08']
    m.update({p:photo['sweep']==p for p in np.unique(photo['sweep'])})
    return m


def boundary_checks(photo, meta):
    result = []
    for a in meta:
        other = [b for b in meta if b['bounds'][0]==a['bounds'][2] and b['bounds'][1::2]==a['bounds'][1::2]]
        for b in other:
            edge = a['bounds'][2]
            for side,s in [('west',a),('east',b)]:
                delta = (photo['RA']-edge+180)%360-180
                selected = (photo['sweep']==s['source']) & (abs(delta)<1)
                w = photo['design_weight'][selected]
                result.append(dict(edge_ra=edge,side=side,sweep=s['source'],n=int(selected.sum()),
                    estimated_n=float(w.sum()),primary_fraction=float(np.average(photo['primary'][selected],weights=w))
                       if len(w) else np.nan))
    return result


def main():
    require_compute()
    inv_path = OUT/'inventory'/'complete.json'
    inv = json.loads(inv_path.read_text())
    hashes = {str(p.relative_to(ROOT)):sha(p) for p in (CONFIG,CUTS,inv_path,Path(__file__))}
    photo, meta, rosters = pilot_input(inv,hashes)
    alias = np.concatenate([r['source_id'] for r in rosters])
    source = np.where((photo['source_id']>0)&(unique_join(photo['source_id'],alias)>=0), photo['source_id'],0)
    photo['gaia_unique'] = source>0
    previous, legacy = past_evidence(hashes)
    fresh = ~np.isin(photo['identity'],previous)&~np.isin(source,previous[previous>0])&~np.isin(photo['ls_id'],legacy)
    tile_snapshot = tile_inventory(source[source>0])
    PRODUCT.mkdir(parents=True,exist_ok=True)
    freeze = PRODUCT/'freeze.json'
    record = dict(inputs=hashes.copy(),gaia_tiles=tile_snapshot,n=len(source),all_PSF=True,
                  availability_uses_gaia_catalog_fields=True)
    if freeze.exists():
        if json.loads(freeze.read_text())!=record:
            raise ValueError('Pilot evidence cannot be reopened with different inputs')
    else:
        save_json(freeze,record)
    gaia = join_gaia(source,GAIA)
    labels = derive_gaia_labels(gaia)
    p,err,matched = labels['p_corr'],gaia['parallax_error'],gaia['gaia_match']
    close = separation(photo['RA'],photo['DEC'],gaia['ra'],gaia['dec'])
    quality = labels['astro_quality'] & labels['finite_parallax']
    usable = matched & close & quality & labels['zpt_valid']
    if tile_inventory(source[source>0])!=tile_snapshot:
        raise ValueError('Gaia source changed')
    cfg = json.loads(CUTS.read_text())
    geom, _, _ = geometry(p,err,usable,cfg)
    score,pit = (np.full(len(p),np.nan) for _ in range(2))
    for i in np.flatnonzero(usable):
        s = histogram_parallax(photo['qrf_mu_edges'][i],photo['qrf_mu_mass'][i],p[i],err[i],order=32)
        score[i],pit[i] = s['log_score'],s['pit']
    arch, _, _, hist_hold = archived_labels(photo,hashes)
    specs = spectroscopic_inventory(hashes)
    spectra = []
    for name, s in specs.items():
        ix = unique_join(source,s['source_id'])
        keep = (source>0)&(ix>=0)
        keep &= separation(photo['RA'],photo['DEC'],s['RA'][ix],s['DEC'][ix])
        rows = np.flatnonzero(keep)
        if not len(rows):
            continue
        index = {k:v[ix[rows]] for k,v in s.items()}
        evidence = read_targeted(name,index)
        label = evidence['class_label']
        d,e = evidence['distance_kpc'],evidence['distance_error_kpc']
        interval = cfg['geometry_sigma_multiplier']*e
        near = evidence['distance_precise']&(d+interval<cfg['near_kpc'])
        far = evidence['distance_precise']&(d-interval>cfg['near_kpc'])
        for selection in ('primary','strict'):
            for name_group, group in groups(photo).items():
                chosen = group[rows]&photo[selection][rows]
                spectra.append(dict(survey=name,selection=selection,group=name_group,matched=int(chosen.sum()),
                    fresh=int((chosen&fresh[rows]).sum()),dwarf=int((chosen&(label==0)).sum()),
                    giant_near=int((chosen&(label==1)&near).sum()),giant_far=int((chosen&(label==1)&far).sum()),
                    giant_unresolved=int((chosen&(label==1)&~(near|far)).sum()),
                    no_label=int((chosen&(label<0)).sum())))
    result = []
    weights = photo['design_weight']
    q = photo['qrf_quantiles_kpc']
    wmean = lambda x, take: float(np.average(x[take],weights=weights[take])) if take.any() else np.nan
    for selection in ('primary','strict'):
        for name, m in groups(photo).items():
            m = m&photo[selection]
            agree = m&usable
            weak = agree&(p<5*err)
            base = dict(selection=selection,group=name,n=int(m.sum()),selected_weight=float(weights[m].sum()),
                gaia_match=int((m&matched).sum()),gaia_position=int((m&matched&close).sum()),
                gaia_quality=int((m&quality).sum()),gaia_zeropoint=int((m&labels['zpt_valid']).sum()),
                gaia_usable=int(agree.sum()),gaia_fresh=int((agree&fresh).sum()),weak=int(weak.sum()),
                signed_log_score=wmean(score,agree),weak_log_score=wmean(score,weak),
                signed_pit_outside_90=wmean((pit<.05)|(pit>.95),agree),
                archived_dwarf=int((m&(arch==0)).sum()),archived_giant=int((m&(arch==1)).sum()),
                archived_class_heldout=int((m&hist_hold).sum()))
            for j,(lo,hi) in enumerate(((0,2),(2,5),(5,10))):
                true = m&(geom>=lo)&(geom<hi)
                result.append(dict(**base,lower_kpc=lo,upper_kpc=hi,
                    qrf_expected=float(weights[m]@photo['qrf_bins'][m,j]),fresh_geometry=int((true&fresh).sum()),
                    **{f'geometry_{k}':v for k,v in distance_summary(
                        10+5*np.log10(q[true]),10+5*np.log10(geom[true])).items()}))
    csv_rows(PRODUCT/'gaia_and_distance.csv',result)
    csv_rows(PRODUCT/'targeted_spectra.csv',spectra)
    csv_rows(PRODUCT/'adjacent_boundary.csv',boundary_checks(photo,meta))
    save_json(PRODUCT/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=hashes,freeze_sha256=sha(freeze),
        pilot_supported=int(sum(len(x['ls_id']) for x in rosters)),sample_n=len(source),qualified_gaia=int(usable.sum()),
        fresh_qualified_gaia=int((usable&fresh).sum()),all_PSF=True,
        outputs={f.name:sha(f) for f in sorted(PRODUCT.glob('*.csv'))}))


if __name__=='__main__':
    main()
