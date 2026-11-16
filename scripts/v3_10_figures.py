#!/usr/bin/env python3
"""Render v3.9 expansion data with the v3.8 plate templates."""
import argparse
import json
import os

import numpy as np
from PIL import Image, ImageChops

from photdis.stage1 import galactic, require_compute
from v2_1_near import ROOT, save_json, sha
from v3_3_sed import verified
from v3_6_vac_distances import read_columns
from v3_8_direct_figures import (
    BLUE, GREEN, ORANGE, OUT as BASELINE, STYLE, aitoff, plt, reference_figures,
    render_reference_figures, spatial,
)
from v3_9_figures import CHANNELS, OUT, read_inventory, select_channel
from v3_9_references import join, subset

DEST = ROOT/'data/v3_10/figures'
LOCK = ROOT/'configs/v3_10_figure_style.json'
REGIONS = ('south_faint','south_low','north')


def check_templates():
    lock = json.loads(LOCK.read_text())
    for name,digest in lock['files'].items():
        verified(ROOT/name,digest)
    return sha(LOCK)


def regression(dest):
    reference_figures(dest)
    result = {}
    for plate in ('F02_direct_distances','F03_same_star_cmd'):
        png = plate+'.png'
        with Image.open(BASELINE/'reference'/png) as a, Image.open(dest/png) as b:
            if a.size!=b.size or ImageChops.difference(a.convert('RGB'),b.convert('RGB')).getbbox():
                raise ValueError(f'The shared template changed the published v3.8 image: {png}')
        result[png] = sha(BASELINE/'reference'/png)
    return result


def load_references(region):
    marker = OUT/'figures_labeled'/region/'complete.json'
    name = 'figure_reference_rows.npz'
    path = verified(marker.parent/name,json.loads(marker.read_text())['outputs'][name])
    keys = ('TYPE','ls_id','RA','DEC','g0','r0','class_label','reference_kpc','gaia_state',
            'specdis_kpc','specdis_quality','spdist_kpc','spdist_quality',
            'qrf_quantiles_kpc','evaluation_row','row_state')
    return read_columns(path,keys),marker


def sky_coord(ra, dec):
    lon,lat = galactic(ra,dec)
    return -np.deg2rad((lon+180)%360-180),np.deg2rad(lat)


def run(region, dest):
    t,marker = load_references(region)
    evaluation = t['evaluation_row']&(t['row_state']==7)
    q = t['qrf_quantiles_kpc']
    support = np.all(np.isfinite(q)&(q>0),axis=1)
    mag = np.column_stack((t['g0'],t['r0']))
    strata = ((None,'all'),(0,'dwarf'),(1,'giant'),(2,'other'))
    for cls,name in strata:
        root = dest/name
        root.mkdir(exist_ok=True)
        chosen = evaluation&support
        if cls is not None:
            chosen &= t['class_label']==cls
        cohorts = {}
        for source in CHANNELS:
            d,good = select_channel(t,source)
            j = chosen&good&np.isfinite(d)&(d>0)
            cohorts[source] = d[j],q[j],mag[j]
        render_reference_figures(root,cohorts,
            distance_title=f'Distance validation · strict photometry · {name} · {region.replace("_"," ")}')
    side = 'north' if region=='north' else 'south'
    _,sky,_,inputs = read_inventory(side)
    p = join([subset(z,(z['region']==region)&z['primary_survey_owner']) for path in sky
              if (z:=read_columns(path,('ls_id','TYPE','RA','DEC','region','qrf_median_kpc','primary_survey_owner')))])
    if not np.all(p['TYPE']=='PSF') or not np.all(t['TYPE']=='PSF'):
        raise ValueError('Only verified native PSF rows may be plotted')
    lp,bp = sky_coord(p['RA'],p['DEC'])
    lr,br = sky_coord(t['RA'],t['DEC'])
    le,be = np.linspace(-np.pi,np.pi,145),np.linspace(-np.pi/2,np.pi/2,73)
    sky_counts = np.histogram2d(lp,bp,bins=(le,be))[0].T
    footprint = (.5*(le[1:]+le[:-1]),.5*(be[1:]+be[:-1]),sky_counts>0)
    dp = p['qrf_median_kpc']
    ref = np.full(len(t['ls_id']),np.nan)
    for source in reversed(CHANNELS):
        d,good = select_channel(t,source)
        use = good&np.isfinite(d)&(d>0)
        ref[use] = d[use]
    known = np.isfinite(ref)
    aitoff(dest,'F04_catalog','Strict photometric predictions · Galactic longitude increases to the left',
           dp,lp,bp,np.isfinite(dp),footprint,BLUE)
    for mask,name,color in ((known,'references',ORANGE),(known&(t['class_label']==0),'dwarfs',GREEN),
                            (known&(t['class_label']==1),'giants',ORANGE)):
        aitoff(dest,f'F04_{name}',f'{name.title()} · reference distance · Galactic longitude increases to the left',
               ref,lr,br,mask,footprint,color)
    n,m = len(dp),len(ref)
    spatial(dest,dict(primary=np.r_[np.ones(n,bool),known]),
            np.r_[dp,np.full(m,np.nan)],np.r_[lp,lr],np.r_[bp,br],
            np.r_[np.full(n,np.nan),ref],np.r_[np.zeros(n,bool),known])
    inputs[str(marker.relative_to(ROOT))] = sha(marker)
    return inputs


def main():
    require_compute()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('region',choices=('regression',*REGIONS))
    region = parser.parse_args().region
    lock = check_templates()
    dest = DEST/region
    dest.mkdir(parents=True,exist_ok=True)
    marker = dest/'complete.json'
    if marker.exists():
        raise FileExistsError(marker)
    with plt.rc_context(STYLE):
        inputs = regression(dest) if region=='regression' else run(region,dest)
    save_json(marker,dict(style_lock_sha256=lock,inputs=inputs,job=os.getenv('SLURM_JOB_ID'),
                          outputs={str(p.relative_to(dest)):sha(p) for p in dest.rglob('*') if p.is_file()}))


if __name__=='__main__':
    main()
