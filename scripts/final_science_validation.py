#!/usr/bin/env python3
"""Check evidence, compare spectroscopy without good parallaxes, and bind figure assets."""
from collections import Counter
import csv
import json
import os

import numpy as np
from PIL import Image

from photdis.stage1 import require_compute
from photdis.v3.metrics import distance_summary
from v2_1_near import ROOT, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_6_vac_distances import read_columns
from v3_10_figures import check_templates
from final_science_catalog import CONFIG, DEST, REGIONS
from final_science_evidence import SHELLS, TIERS, SUMMARY_KEYS

REUSE = (
    ('v3_8/F01_catalog_overview.png','v3_8/direct_figures/overview/F01_catalog_overview.png'),
    ('v3_8/F02_direct_distances.png','v3_8/direct_figures/reference/F02_direct_distances.png'),
    ('v3_8/F03_same_star_cmd.png','v3_8/direct_figures/reference/F03_same_star_cmd.png'),
    ('v3_8/F04_catalog.png','v3_8/direct_figures/sky/F04_catalog.png'),
    ('v3_8/F04_references.png','v3_8/direct_figures/sky/F04_references.png'),
    ('v3_8/F04_dwarfs_reference.png','v3_8/direct_figures/sky/F04_dwarfs_reference.png'),
    ('v3_8/F04_giants_reference.png','v3_8/direct_figures/sky/F04_giants_reference.png'),
    ('v3_8/F05_heliocentric_projection.png','v3_8/direct_figures/overview/F05_heliocentric_projection.png'),
    ('v3_10/redraw/north_aitoff.png','v3_10/figures/north/F04_catalog.png'),
    ('v3_10/redraw/north_xyz.png','v3_10/figures/north/F05_heliocentric_projection.png'),
    ('v3_10/redraw/south_faint_aitoff.png','v3_10/figures/south_faint/F04_catalog.png'),
    ('v3_10/redraw/south_faint_distances.png','v3_10/figures/south_faint/all/F02_direct_distances.png'),
    ('v3_10/redraw/south_low_aitoff.png','v3_10/figures/south_low/F04_catalog.png'),
    ('v3_10/redraw/south_low_cmd.png','v3_10/figures/south_low/all/F03_same_star_cmd.png'),
)


def verify_receipt(marker, parents):
    r = json.loads(marker.read_text())
    parents[str(marker.relative_to(ROOT))] = sha(marker)
    for name,digest in r['outputs'].items():
        verified(marker.parent/name,digest)
    return r


def exact_references(t, dest):
    groups = (('all',np.ones(len(t['ls_id']),bool)),('reliable_geometry',t['gaia_state']==6),
              ('matched_below_geometry',np.isin(t['gaia_state'],(3,4,5))),
              ('unresolved_gaia',~np.isin(t['gaia_state'],(3,4,5,6))))
    spec = t['specdis_kpc']
    spd = t['spdist_kpc']
    heldout = t['heldout']
    dwarf = t['rvs_label']==0
    f = t['rvs_feh']
    fgood = (t['rvs_feh_err']>.0)&(t['rvs_feh_err']<=.3)
    q = 10+5*np.log10(t['q'])
    rows = []
    for ri,region in enumerate(REGIONS):
        for tier in np.unique(t['science'][t['region']==ri]):
            sample = (t['region']==ri)&(t['science']==tier)&heldout
            for check,group in groups:
                for label,pop in (('all',np.ones(len(f),bool)),('spectroscopic_dwarf',dwarf),
                                  ('dwarf_feh_below_-1.5',dwarf&fgood&(f<-1.5))):
                    frame = sample&group&pop
                    for name,ref,other in (('SpecDis',spec,spd),('SPDist',spd,spec)):
                        for a,b in zip(SHELLS[:3],SHELLS[1:4]):
                            subset = frame&np.isfinite(ref)&(ref>=a)&(ref<b)
                            metric = distance_summary(q[subset],10+5*np.log10(ref[subset]))
                            common = subset&np.isfinite(other)&(other>0)
                            bias = np.median(other[common]/ref[common]-1) if common.any() else None
                            rows.append(dict(region=region,tier=TIERS[tier],gaia=check,population=label,
                                source=name,ref_min=a,ref_max=b,common_n=int(common.sum()),
                                other_relative_bias=bias,**{k:metric.get(k) for k in SUMMARY_KEYS}))
    csv_rows(dest/'spectroscopic_increment.csv',rows)


def galactocentric(t,dest):
    # Galactic coordinates in the solar plane, with the positive X axis through the Sun.
    l,b = np.deg2rad(t['lon']),np.deg2rad(t['lat'])
    edges_r = np.linspace(0.,20.,81)
    edges_z = np.linspace(-10.,10.,81)
    rows = []
    for ri,region in enumerate(REGIONS):
        sel = (t['region']==ri)&(t['science']<=2)
        for src,z in (('Photdis',t['q'][:,2]),('Gaia same-footprint reference',t['gaia_kpc'])):
            j = sel&np.isfinite(z)&(z>0)&(z<10)
            x = 8.2-z[j]*np.cos(b[j])*np.cos(l[j])
            y = z[j]*np.cos(b[j])*np.sin(l[j])
            r,h = np.hypot(x,y),z[j]*np.sin(b[j])
            counts = np.histogram2d(r,h,bins=(edges_r,edges_z))[0].astype(int)
            if counts.sum()!=j.sum():
                raise ValueError('Galactocentric projection loses selected sources')
            for i,k in zip(*np.nonzero(counts)):
                rows.append(dict(region=region,source=src,R_lo=edges_r[i],R_hi=edges_r[i+1],
                                 Z_lo=edges_z[k],Z_hi=edges_z[k+1],n=counts[i,k]))
    csv_rows(dest/'galactocentric_RZ.csv',rows)


def main():
    require_compute()
    dest = DEST/'delivery'
    dest.mkdir(exist_ok=True,parents=True)
    parents = {'config':sha(CONFIG),'style':check_templates()}
    for a in ('area/north','area/south','evidence','figures'):
        verify_receipt(DEST/a/'complete.json',parents)
    evidence = DEST/'evidence'
    figures = DEST/'figures'
    names = ('ls_id','TYPE','science','region','bins','gaia_state','q','specdis_kpc','spdist_kpc',
             'gaia_kpc','heldout','rvs_label','rvs_feh','rvs_feh_err','lon','lat')
    t = read_columns(evidence/'selected.npz',names)
    if not np.all(t['TYPE']=='PSF') or len(np.unique(t['ls_id']))!=len(t['ls_id']):
        raise ValueError('Selected identities fail final PSF admission')
    np.testing.assert_allclose(t['bins'].sum(axis=1),1,atol=1e-7,rtol=0)
    with (evidence/'gaia_increment.csv').open() as f:
        n = Counter()
        for r in csv.DictReader(f):
            if r['magnitude_bin']=='-1':
                n[r['region'],r['tier']] += int(r['n'])
                if sum(int(r[f'hard_{j}']) for j in range(6))!=int(r['n']):
                    raise ValueError('Hard shell partition fails')
                np.testing.assert_allclose(sum(float(r[f'pdf_{j}']) for j in range(6)),int(r['n']),
                                           atol=2e-6,rtol=0)
    for (r,k),count in n.items():
        np.testing.assert_equal(count,((t['region']==REGIONS.index(r))&(t['science']==TIERS.index(k))).sum())
    np.testing.assert_equal(sum(n.values()),len(t['ls_id']))
    exact_references(t,dest)
    galactocentric(t,dest)
    old = []
    for repo,archive in REUSE:
        local = ROOT/'figures'/repo
        distant = ROOT/'data'/archive
        digest = sha(local)
        located = distant.parent
        if not (located/'complete.json').is_file():
            located = located.parent
        marker = located/'complete.json'
        r = json.loads(marker.read_text())
        parents[str(marker.relative_to(ROOT))] = sha(marker)
        for suffix in ('.png','.pdf','.csv','.npz'):
            path = distant.with_suffix(suffix)
            key = str(path.relative_to(located))
            if suffix!='.npz' or key in r['outputs']:
                verified(path,r['outputs'][key])
        with Image.open(local) as a, Image.open(distant) as b:
            exact = a.size==b.size and np.array_equal(np.asarray(a.convert('RGB')),np.asarray(b.convert('RGB')))
        if not exact:
            raise ValueError(f'Reused pixels do not match the original archived figure {repo}')
        old.append(dict(figure=str(local.relative_to(ROOT)),archive=str(distant.relative_to(ROOT)),
                        pixels_verified=True,sha256=digest))
    csv_rows(dest/'reused_figures.csv',old)
    new = []
    for image in sorted(figures.glob('*.png')):
        with Image.open(image) as im:
            im.verify()
        with Image.open(image) as im:
            dx,dy = im.info['dpi']
            if min(im.size)<900 or abs(dx-300)>.1 or abs(dy-300)>.1:
                raise ValueError(f'Unexpected export geometry: {image}')
            new.append(dict(figure=image.stem,width=im.width,height=im.height,dpi=dx))
        if not image.with_suffix('.pdf').read_bytes().startswith(b'%PDF'):
            raise ValueError(f'Missing vector-format figure: {image}')
    csv_rows(dest/'new_figures.csv',new)
    save_json(dest/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=parents,rows=len(t['ls_id']),
        new_figures=len(new),reused_figures=len(old),
        outputs={p.name:sha(p) for p in sorted(dest.glob('*.csv'))}))


if __name__=='__main__':
    main()
