#!/usr/bin/env python3
"""Build the de-duplicated north–south sky plates with the locked v3.8 style."""
import json
import os

import numpy as np

from photdis.stage1 import galactic, require_compute
from v2_1_near import ROOT, save_json, sha
from v3_3_sed import verified
from v3_6_vac_distances import read_columns
from v3_8_direct_figures import BLUE, GREEN, ORANGE, STYLE, aitoff, catalog, matched_roster, plt
from v3_9_delivery import DEST as RELEASE
from v3_9_seam import BOUNDARY
from v3_9_unified import owner_rows
from v3_9_figures import CHANNELS, read_inventory, select_channel
from v3_9_references import join, subset
from v3_10_figures import DEST, REGIONS, check_templates, load_references, sky_coord


def filter_native(z, side, overrides):
    z['north_primary_region'] = (z['DEC']>=BOUNDARY)&(galactic(z['RA'],z['DEC'])[1]>0)
    owner,_ = owner_rows(side,z,overrides)
    del z['north_primary_region']
    return owner


def collect(reference, result, selected):
    l,b = sky_coord(reference['RA'][selected],reference['DEC'][selected])
    keys = ('distance','class_label')
    result.append(dict(l=l,b=b,**{k:reference[k][selected] for k in keys}))


def main():
    require_compute()
    lock = check_templates()
    release = RELEASE/'complete.json'
    meta = json.loads(release.read_text())
    seam = RELEASE/'seam/complete.json'
    record = json.loads(verified(seam,meta['sources'][str(seam.relative_to(ROOT))]).read_text())
    override = read_columns(verified(seam.parent/'ownership_overrides.npz',
                                    record['outputs']['ownership_overrides.npz']),
                            ('ls_id','TYPE','primary_survey_owner'))
    science,native_ref = [],[]
    old = catalog()
    old['TYPE'] = np.full(len(old['ls_id']),'PSF')
    use,_ = owner_rows('south',old,override,frozen=True)
    if use.sum()!=meta['unified']['south_v35']['owned_rows']:
        raise ValueError('Frozen DECam projection has a different ownership count')
    reference,labels,_,_ = matched_roster(old)
    old.update(distance=old['qrf_quantiles_kpc'][:,2],class_label=labels)
    collect(old,science,use)
    old['distance'] = reference
    collect(old,native_ref,use&np.isfinite(reference))
    del old,reference
    for side in ('south','north'):
        _,parts,_,_ = read_inventory(side)
        for f in parts:
            p = read_columns(f,('ls_id','TYPE','RA','DEC','qrf_median_kpc','primary_survey_owner'))
            # The validation sky projection already contains the native geometric
            # decision. Undo the equal-side test only to apply owner overrides.
            p['north_primary_region'] = p.pop('primary_survey_owner')==(side=='north')
            valid,_ = owner_rows(side,p,override)
            p['distance'] = p['qrf_median_kpc']
            p['class_label'] = np.full(len(valid),-1,'i1')
            collect(p,science,valid)
    for region in REGIONS:
        t,_ = load_references(region)
        owner = filter_native(t,'north' if region=='north' else 'south',override)
        t['distance'] = np.full(len(owner),np.nan)
        for source in reversed(CHANNELS):
            d,q = select_channel(t,source)
            use = q&np.isfinite(d)&(d>0)
            t['distance'][use] = d[use]
        collect(t,native_ref,owner&np.isfinite(t['distance']))
    p,r = join(science),join(native_ref)
    del science,native_ref
    edge_l,edge_b = np.linspace(-np.pi,np.pi,145),np.linspace(-np.pi/2,np.pi/2,73)
    h = np.histogram2d(p['l'],p['b'],bins=(edge_l,edge_b))[0].T
    area = (.5*(edge_l[1:]+edge_l[:-1]),.5*(edge_b[1:]+edge_b[:-1]),h>0)
    d = DEST/'combined'
    d.mkdir(parents=True,exist_ok=True)
    with plt.rc_context(STYLE):
        for name,t,k,color in (
            ('catalog',p,np.isfinite(p['distance']),BLUE),
            ('references',r,np.isfinite(r['distance']),ORANGE),
            ('dwarfs',r,r['class_label']==0,GREEN),
            ('giants',r,r['class_label']==1,ORANGE)):
            aitoff(d,f'F04_{name}','North and south · '+name+' · Galactic longitude increases to the left',
                   t['distance'],t['l'],t['b'],k,area,color)
    save_json(d/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),style_lock_sha256=lock,
        release_sha256=sha(release),strict_photometric_rows=len(p['l']),qualified_reference_rows=len(r['l']),
        outputs={f.name:sha(f) for f in d.iterdir() if f.name!='complete.json'}))


if __name__=='__main__':
    main()
