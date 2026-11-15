#!/usr/bin/env python3
"""Count hard distance assignments and selection conflicts over the whole catalog."""
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import healpy as hp
import numpy as np

from photdis.stage1 import galactic, require_compute
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_6_catalog_projection import OUT as PROJECTION
from v3_6_vac_distances import DISTANCE_EDGES, read_columns


OUT = ROOT/'data/v3_6/catalog_observables'
LABELS = ('supported','primary','strict')
R_EDGES = np.array([0.,14.,16.,18.,19.,20.])
COLOR_EDGES = np.array([-np.inf,.2,.5,.8,1.1,1.4,np.inf])
Q_EDGES = np.linspace(0,1,51)
D_EDGES = np.geomspace(.01,1e4,97)
NSIDE = 16
CARTESIAN_EDGES = np.linspace(-2.,2.,101)
RED_DOMAIN = ROOT/'configs/v3_4_near_release_domain.json'
COLUMNS = ('RA','DEC','TYPE','primary','strict','observed_r','dereddened_g_r','band_state',
           'qrf_bins','qrf_qnear','nominal_qnear','dwarf_selection_score','qrf_quantiles_kpc')
COUNTS = ('rows','nominal_scored','qrf_d50_gt10','qrf_qnear_lt_half','either_conflict','both_conflicts',
          'median_exactly_10','qrf_qnear_exactly_half')


def red_cartesian(p,domain):
    r,c,w = p['observed_r'],p['dereddened_g_r'],p['band_state'][:,4]
    d = p['qrf_quantiles_kpc'][:,2]
    r0,r1 = domain['observed_r']
    c0,c1 = domain['dereddened_g_r']
    mask = ((r>=r0)&(r<r1)&(c>=c0)&(c<c1)&np.isin(w,domain['w1_states'])&(d>0)&(d<2))
    lon,lat = np.deg2rad(galactic(p['RA'][mask],p['DEC'][mask]))
    x = d[mask]*np.cos(lat)*np.cos(lon)
    y = d[mask]*np.cos(lat)*np.sin(lon)
    z = d[mask]*np.sin(lat)
    result = []
    for cut in (np.ones(len(x),bool),p['primary'][mask],p['strict'][mask]):
        variants = [np.histogram2d(x[cut],u[cut],bins=CARTESIAN_EDGES)[0].astype('i8') for u in (y,z)]
        np.testing.assert_equal([v.sum() for v in variants],np.repeat(cut.sum(),2))
        result.append(variants)
    return np.asarray(result)


def count(p,domain=None):
    r,c = p['observed_r'],p['dereddened_g_r']
    w = p['band_state'][:,4]
    d = p['qrf_quantiles_kpc'][:,2]
    q,v,m = (p[k] for k in ('qrf_qnear','nominal_qnear','dwarf_selection_score'))
    if not (p['TYPE']=='PSF').all() or np.any(p['strict']&~p['primary']):
        raise ValueError('Require native PSFs and nested frozen cuts')
    quality = np.isfinite(v)
    bad_distance = d>10.
    bad_near = q<.5
    groups = (np.ones(len(d),bool),quality,bad_distance,bad_near,bad_distance|bad_near,
              bad_distance&bad_near,d==10.,q==.5)
    cuts = (groups[0],p['primary'],p['strict'])
    rid = np.searchsorted(R_EDGES,r,side='right')-1
    cid = np.searchsorted(COLOR_EDGES,c,side='right')-1
    state = np.where(w==4,2,np.where(w==0,0,1))
    shape = (len(R_EDGES)-1,len(COLOR_EDGES)-1,3)
    cell = np.ravel_multi_index((rid,cid,state),shape)
    pixel = hp.ang2pix(NSIDE,p['RA'],p['DEC'],lonlat=True,nest=True)
    stat,profile,sky,hist = [],[],[],[]
    q_clip,v_clip,m_clip = [np.clip(a,0,1) for a in (q,v,m)]
    shell = np.searchsorted(DISTANCE_EDGES,d,side='right')-1
    for use in cuts:
        stat.append([np.bincount(cell[use&mask],minlength=int(np.prod(shape))).reshape(shape) for mask in groups])
        hard = np.bincount(shell[use],minlength=len(DISTANCE_EDGES)-1)
        profile.append(np.stack([hard,p['qrf_bins'][use].sum(axis=0)]))
        sky.append(np.bincount(pixel[use],minlength=hp.nside2npix(NSIDE)))
        hist.append((np.histogram2d(v_clip[use&quality],q_clip[use&quality],bins=(Q_EDGES,Q_EDGES))[0],
                     np.histogram2d(m_clip[use],q_clip[use],bins=(Q_EDGES,Q_EDGES))[0],
                     np.histogram2d(m_clip[use],d[use],bins=(Q_EDGES,D_EDGES))[0]))
    result = dict(cell_counts=np.asarray(stat),radial=np.asarray(profile),sky=np.stack(sky),
                  nominal_qrf=np.stack([h[0] for h in hist]),dwarf_qrf=np.stack([h[1] for h in hist]),
                  dwarf_d50=np.stack([h[2] for h in hist]))
    if domain is not None:
        result['bright_red_cartesian'] = red_cartesian(p,domain)
    return result


def write_csv(payload):
    rows = []
    for i,cut in enumerate(LABELS):
        for w,band in ((None,'all'),(0,'absent'),(1,'weak_or_rejected'),(2,'usable')):
            for j,(r0,r1) in enumerate(zip(R_EDGES[:-1],R_EDGES[1:])):
                for k,(c0,c1) in enumerate(zip(COLOR_EDGES[:-1],COLOR_EDGES[1:])):
                    a = payload['cell_counts'][i,:,j,k]
                    x = a.sum(axis=1) if w is None else a[:,w]
                    row = dict(zip(COUNTS,map(int,x)))
                    rows.append(dict(selection=cut,W1=band,r_lo=r0,r_hi=r1,gr_lo=c0,gr_hi=c1,
                        **row,conflict_fraction=x[4]/x[0] if x[0] else None))
    csv_rows(OUT/'conflicts.csv',rows)
    radial = []
    for i,cut in enumerate(LABELS):
        for k,(lo,hi) in enumerate(zip(DISTANCE_EDGES[:-1],DISTANCE_EDGES[1:])):
            hard,posterior = payload['radial'][i,:,k]
            radial.append(dict(selection=cut,lo_kpc=lo,hi_kpc=hi,median_count=int(hard),
                               posterior_count=float(posterior)))
    csv_rows(OUT/'radial_counts.csv',radial)


def main():
    require_compute()
    path = PROJECTION/'complete.json'
    delivery = json.loads(path.read_text())
    domain, = json.loads(RED_DOMAIN.read_text())['domains']
    files = []
    for name,h in delivery['child_receipts'].items():
        marker = verified(ROOT/name,h)
        receipt = json.loads(marker.read_text())
        files.append((marker.with_name('rows.npz'),receipt['outputs']['rows.npz']))
    def run(item):
        path,h = item
        return count(read_columns(verified(path,h),COLUMNS),domain)
    total = {}
    with ThreadPoolExecutor(max_workers=min(12,int(os.getenv('SLURM_CPUS_PER_TASK','1')))) as pool:
        for result in pool.map(run,files):
            for k,a in result.items():
                if k in total:
                    total[k] += a
                else:
                    total[k] = a
    n = total['sky'].sum(axis=1)
    cell = total['cell_counts'].sum(axis=(2,3,4))
    np.testing.assert_array_equal(n,cell[:,0])
    np.testing.assert_array_equal(n,total['radial'][:,0].sum(axis=1))
    np.testing.assert_allclose(n,total['radial'][:,1].sum(axis=1),rtol=0,atol=1e-6)
    np.testing.assert_array_equal(cell[:,1],total['nominal_qrf'].sum(axis=(1,2)))
    np.testing.assert_array_equal(n,total['dwarf_qrf'].sum(axis=(1,2)))
    if n[0]!=delivery['rows']:
        raise ValueError('Observable counts do not span the full supported denominator')
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'complete.json').exists():
        raise FileExistsError(OUT/'complete.json')
    write_csv(total)
    atomic_npz(OUT/'observables.npz',**total,r_edges=R_EDGES,color_edges=COLOR_EDGES,
               q_edges=Q_EDGES,d_edges=D_EDGES,shell_edges=DISTANCE_EDGES,cartesian_edges=CARTESIAN_EDGES)
    save_json(OUT/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),rows=n.tolist(),nside=NSIDE,
        inputs={str(p.relative_to(ROOT)):sha(p) for p in (path,RED_DOMAIN,Path(__file__))},
        conflict_by_selection={key:{k:int(v) for k,v in zip(COUNTS,counts)}
                               for key,counts in zip(LABELS,cell)},
        omitted_d50_display_rows=(n-total['dwarf_d50'].sum(axis=(1,2))).astype(int).tolist(),
        bright_red_d50_0_2=total['bright_red_cartesian'][:,0].sum(axis=(1,2)).tolist(),
        scope='Census of original DR11 PSF predictions. Conflicts are catalog assignments rather than measured contamination.',
        outputs={p.name:sha(p) for p in OUT.iterdir() if p.name!='complete.json'}))


if __name__=='__main__':
    main()
