#!/usr/bin/env python3
"""Compare the actual production PDFs of the frozen north–south matches."""
import argparse
import csv
import json
import os

import numpy as np
from scipy.ndimage import gaussian_filter1d

from photdis.stage1 import require_compute, tile_bounds
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_6_vac_distances import read_columns
from v3_9_delivery import DEST as RELEASE
from v3_9_seam import BOUNDARY, within_strip

DEST = ROOT/'data/v3_10/seam'
CUTS = np.array([1.,5.,10.])
PDF = ('qrf_quantiles_kpc','qrf_mu_edges','qrf_mu_mass')
PAIR_COUNT = 1951902


def compare_pdf(edges, mass):
    width = np.diff(edges,axis=1)
    if np.any(width<=0) or np.any(mass<0) or not np.allclose(mass.sum(axis=1),1.,atol=1e-11,rtol=0):
        raise ValueError('Invalid production PDF')
    cumulative = np.column_stack([np.sum(mass*np.clip((cut-edges[:,:-1])/width,0,1),axis=1)
                                  for cut in 10+5*np.log10(CUTS)])
    shells = np.diff(cumulative,axis=1,prepend=0.,append=1.)
    density = gaussian_filter1d(mass/width,1.,axis=1,mode='nearest')
    prev = np.column_stack([np.full(len(edges),-1.),density[:,:-1]])
    next_ = np.column_stack([density[:,1:],np.full(len(edges),-1.)])
    modes = ((density>prev)&(density>=next_)&(density>=.1*density.max(axis=1,keepdims=True))).sum(axis=1)
    return shells,modes


def lookup_values(native_ids, sorted_ids, order):
    position = np.searchsorted(sorted_ids,native_ids)
    local = np.flatnonzero(position<len(sorted_ids))
    local = local[sorted_ids[position[local]]==native_ids[local]]
    return local,order[position[local]]


def read_side(side, ids, published, parts, output):
    n = len(ids)
    order = np.argsort(ids)
    keys = ids[order]
    if n!=PAIR_COUNT or np.any(np.diff(keys)<=0):
        raise ValueError('Frozen seam IDs are duplicated or incomplete')
    r,gr = np.full((2,n),np.nan)
    q = np.full((n,5),np.nan)
    shells = np.full((n,4),np.nan)
    modes = np.zeros(n,'u1')
    states = np.full(n,255,'u1')
    for part in parts:
        if part['side']!=side or not within_strip(tile_bounds(part['sweep']+'.fits')):
            continue
        frozen = not part['state_file']
        columns = ('observed_r','dereddened_g_r') if frozen else ('r','g0','r0')
        p = read_columns(verified(ROOT/part['photo_file'],part['photo_sha256']),
                         ('ls_id','TYPE',*columns))
        j,k = lookup_values(p['ls_id'],keys,order)
        if not len(j):
            continue
        if np.any(states[k]!=255) or not np.all(p['TYPE'][j]=='PSF'):
            raise ValueError('Found duplicate or non-PSF native seam rows')
        if frozen:
            state = np.full(len(j),3,'u1')
            r[k],gr[k] = p['observed_r'][j],p['dereddened_g_r'][j]
        else:
            s = read_columns(verified(ROOT/part['state_file'],part['state_sha256']),('ls_id','TYPE','state'))
            np.testing.assert_array_equal(p['ls_id'][j],s['ls_id'][j])
            np.testing.assert_array_equal(p['TYPE'][j],s['TYPE'][j])
            state = s['state'][j]
            r[k],gr[k] = p['r'][j],p['g0'][j]-p['r0'][j]
        states[k] = state
        want = state==3
        if not want.any():
            continue
        with np.load(verified(ROOT/part['pdf_file'],part['pdf_sha256']),allow_pickle=False) as z:
            a,b = lookup_values(z['ls_id'],keys,order)
            np.testing.assert_array_equal(np.sort(b),np.sort(k[want]))
            if not np.all(z['TYPE'][a]=='PSF'):
                raise ValueError('Unverified morphology in paired PDF')
            q[b] = z[PDF[0]][a]
            data = [z[name][a] for name in PDF[1:]]
        for start in range(0,len(a),2048):
            j = slice(start,start+2048)
            shells[b[j]],modes[b[j]] = compare_pdf(*(column[j] for column in data))
    np.testing.assert_array_equal(states,published)
    np.testing.assert_array_equal(np.isfinite(q[:,2]),states==3)
    atomic_npz(output,ls_id=ids,TYPE=np.full(n,'PSF'),r=r,gr=gr,state=states,qrf_quantiles_kpc=q,
               distance_shell_mass=shells,native_pdf_modes=modes)


def compare(n, s, pairs, output):
    ns,ss = n['state']==3,s['state']==3
    nd,sd = n['qrf_quantiles_kpc'][:,2],s['qrf_quantiles_kpc'][:,2]
    dec = .5*(pairs['north_dec']+pairs['south_dec'])
    delta = nd/sd-1
    wn,ws = [5*np.log10(t['qrf_quantiles_kpc'][:,3]/t['qrf_quantiles_kpc'][:,1]) for t in (n,s)]
    north_shell,south_shell = np.digitize(nd,CUTS),np.digitize(sd,CUTS)
    rows,migration = [],[]
    regions = (('all',np.ones(len(nd),bool)),('south',dec<BOUNDARY),('north',dec>=BOUNDARY))
    for r0,r1 in ((-np.inf,np.inf),(-np.inf,18.),(18.,20.),(20.,21.),(21.,22.),(22.,23.)):
        for c0,c1 in ((-np.inf,np.inf),(-np.inf,.5),(.5,.8),(.8,1.4),(1.4,np.inf)):
            in_r = (s['r']>=r0)&(s['r']<r1) if np.isfinite([r0,r1]).any() else np.ones(len(nd),bool)
            in_color = (s['gr']>=c0)&(s['gr']<c1) if np.isfinite([c0,c1]).any() else np.ones(len(nd),bool)
            phot = in_r&in_color
            for region,k in regions:
                subset = phot&k
                both = subset&ns&ss
                if not subset.any():
                    continue
                meta = dict(seam_side=region,south_r_min=r0,south_r_max=r1,south_gr_min=c0,south_gr_max=c1)
                row = dict(**meta,n=int(subset.sum()),both=int(both.sum()),
                           north_only=int((subset&ns&~ss).sum()),south_only=int((subset&ss&~ns).sum()),
                           neither=int((subset&~ns&~ss).sum()))
                if both.any():
                    center = np.median(delta[both])
                    row.update(distance_median_relative=float(center),
                               distance_nmad_relative=float(1.4826*np.median(abs(delta[both]-center))),
                               width_median_difference_mag=float(np.median(wn[both]-ws[both])),
                               north_multiple_modes=int((n['native_pdf_modes'][both]>1).sum()),
                               south_multiple_modes=int((s['native_pdf_modes'][both]>1).sum()),
                               mean_shell_total_variation=float(.5*abs(n['distance_shell_mass'][both]
                                                                     -s['distance_shell_mass'][both]).sum(axis=1).mean()))
                    for i in range(4):
                        for j in range(4):
                            migration.append(dict(**meta,south_shell=i,north_shell=j,
                                                  data_count=int((both&(south_shell==i)&(north_shell==j)).sum())))
                rows.append(row)
    if not rows or rows[0]['n']!=PAIR_COUNT or rows[0]['both']!=int((ns&ss).sum()):
        raise ValueError('Seam summary lost pairs')
    csv_rows(output/'production_comparison.csv',rows)
    csv_rows(output/'shell_migration.csv',migration)


def main():
    require_compute()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('north','south','compare'))
    stage = p.parse_args().stage
    DEST.mkdir(parents=True,exist_ok=True)
    marker = RELEASE/'complete.json'
    receipt = json.loads(marker.read_text())
    seam_marker = verified(RELEASE/'seam/complete.json',
                           receipt['sources'][str((RELEASE/'seam/complete.json').relative_to(ROOT))])
    seam = json.loads(seam_marker.read_text())
    pairs = read_columns(verified(RELEASE/'seam/paired_sources.npz',seam['outputs']['paired_sources.npz']),
                         ('TYPE','south_ls_id','north_ls_id','south_dec','north_dec',
                          'south_prediction_state','north_prediction_state'))
    if not np.all(pairs['TYPE']=='PSF'):
        raise ValueError('Expected verified native PSF seam pairs')
    inputs = {str(marker.relative_to(ROOT)):sha(marker),str(seam_marker.relative_to(ROOT)):sha(seam_marker)}
    if stage=='compare':
        tables = {}
        for side in ('south','north'):
            ready = DEST/f'{side}.json'
            record = json.loads(ready.read_text())
            if record['inputs']!=inputs:
                raise ValueError('Seam projections use different production versions')
            path = DEST/f'{side}.npz'
            t = read_columns(verified(path,record['outputs'][path.name]),
                             ('TYPE','ls_id','r','gr','state','qrf_quantiles_kpc','distance_shell_mass','native_pdf_modes'))
            np.testing.assert_array_equal(t['ls_id'],pairs[f'{side}_ls_id'])
            np.testing.assert_array_equal(t['TYPE'],pairs['TYPE'])
            tables[side] = t
        compare(tables['north'],tables['south'],pairs,DEST)
        outputs = ('production_comparison.csv','shell_migration.csv')
    else:
        index = verified(RELEASE/'unified_parts.csv',receipt['outputs']['unified_parts.csv'])
        with index.open() as f:
            parts = list(csv.DictReader(f))
        outputs = (f'{stage}.npz',)
        read_side(stage,pairs[f'{stage}_ls_id'],pairs[f'{stage}_prediction_state'],parts,DEST/outputs[0])
    save_json(DEST/f'{stage}.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,pairs=PAIR_COUNT,
              modes='Native-bin density, one-bin Gaussian smoothing, peaks at least 10% of maximum',
              outputs={name:sha(DEST/name) for name in outputs}))


if __name__=='__main__':
    main()
