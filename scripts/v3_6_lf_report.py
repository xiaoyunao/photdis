#!/usr/bin/env python3
"""Estimate population sensitivity and sky-block uncertainty on the LF draw."""
import json
import os
from pathlib import Path

import healpy as hp
import numpy as np

from photdis.stage1 import require_compute
from v2_1_near import ROOT, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_6_lf_sensitivity import NAMES, OUT as FORECAST


OUT = FORECAST/'report'
SHELLS = ('0–2','2–5','5–10','10–20','20–50','>50')
SEED = 360918
DRAWS = 1024


def contributions(joint,qrf):
    shells = joint.sum(axis=2)
    true_giant = joint[:,:,1]
    # Both model probabilities are functions of the same measured row. A QRF
    # assignment is a fixed weight, while the population scenario describes
    # true class/distance conditional on that row's photometry.
    far_giant = true_giant[:,:,3:].sum(axis=2)
    far_any = shells[:,:,3:].sum(axis=2)
    weighted_far_giant = far_giant[:,:,None]*qrf[:,None,:3]
    weighted_far_any = far_any[:,:,None]*qrf[:,None,:3]
    labels = ('near_population','near_giant_population','far_giant_assigned_near','far_any_assigned_near')
    return {k:x for k,x in zip(labels,(shells[:,:,:3],true_giant[:,:,:3],weighted_far_giant,weighted_far_any))}


def sky_totals(value,weight,block,draw):
    flat = value.reshape(len(weight),-1)
    nblocks = draw.shape[1]
    denominator = np.bincount(block,weights=weight,minlength=nblocks)
    sums = np.zeros((nblocks,flat.shape[1]))
    np.add.at(sums,block,flat*weight[:,None])
    unit = draw@denominator
    live = unit>0
    shifted = (draw[live]@sums)/unit[live,None]*weight.sum()
    shape = value.shape[1:]
    return sums.sum(axis=0).reshape(shape),shifted.reshape(-1,*shape),int((denominator>0).sum())


def design_se(values,weight,cell):
    """Sampling variance under SRS without replacement within the frozen cells."""
    flat = values.reshape(len(values),-1)
    var = np.zeros(flat.shape[1])
    # In every stratum all drawn rows are in or out of a reported cut/sector.
    order = np.argsort(cell,kind='stable')
    bound = np.r_[0,np.flatnonzero(np.diff(cell[order]))+1,len(cell)]
    for a,b in zip(bound[:-1],bound[1:]):
        ix = order[a:b]
        n = len(ix)
        w = weight[ix[0]]
        if not np.allclose(weight[ix],w,rtol=0,atol=1e-12):
            raise ValueError('A design stratum has nonconstant sampling weights')
        if w<=1:
            continue
        if n<2:
            raise ValueError('A subsampled design stratum has only one row')
        var += n*w*(w-1)*flat[ix].var(axis=0,ddof=1)
    return np.sqrt(var).reshape(values.shape[1:])


def summary(p):
    joint,qrf = p['class_shell'],p['qrf_bins']
    if joint.shape[1:]!=(len(NAMES),3,len(SHELLS)) or qrf.shape[1]!=len(SHELLS):
        raise ValueError('Unexpected class/distance grid')
    weight,cell = p['weight'],p['sampling_cell']
    lon,lat = p['galactic_l'],p['galactic_b']
    _,block = np.unique(hp.ang2pix(4,p['RA'],p['DEC'],lonlat=True,nest=True),return_inverse=True)
    nblocks = block.max()+1
    draw = np.random.default_rng(SEED).multinomial(nblocks,np.full(nblocks,1/nblocks),size=DRAWS)
    metrics = contributions(joint,qrf)
    results, spatial, migration = [],[],[]
    groups = [('all_sky',np.ones(len(weight),bool))]
    groups += [(f'{side}_{lo}_{lo+90}',(lon>=lo)&(lon<lo+90)&hemisphere)
               for side,hemisphere in (('N',lat>=0),('S',lat<0)) for lo in (0,90,180,270)]
    for cut in ('primary','strict'):
        selected = p[cut]
        for scope,sky in groups:
            keep = selected&sky
            if not keep.any():
                continue
            w = weight[keep]
            qrf_counts,qrf_rep,_ = sky_totals(qrf[keep,:3],w,block[keep],draw)
            for estimand,value in metrics.items():
                a = value[keep]
                total,rep,nb = sky_totals(a,w,block[keep],draw)
                uncertainty = design_se(a,w,cell[keep])
                for j,name in enumerate(NAMES):
                    for shell,label in enumerate(SHELLS[:3]):
                        n = total[j,shell]
                        assigned = 'assigned' in estimand
                        denominator = qrf_counts[shell] if assigned else w.sum()
                        rep_denom = qrf_rep[:,shell] if assigned else np.full(len(rep),w.sum())
                        # Paired sky draws share exactly the same replicated blocks.
                        conf = np.quantile(rep[:,j,shell],[.025,.975])
                        delta = rep[:,j,shell]-rep[:,0,shell]
                        interval = np.quantile(delta,[.025,.975])
                        valid = rep_denom>0
                        fraction_ci = np.quantile(rep[valid,j,shell]/rep_denom[valid],[.025,.975]) if valid.any() else (None,None)
                        row = dict(selection=cut,sky=scope,scenario=name,shell_kpc=label,estimand=estimand,
                            draw_rows=len(w),sky_blocks=nb,weighted_selected=float(w.sum()),
                            weighted_qrf_shell=float(qrf_counts[shell]),expected_count=float(n),
                            fraction=float(n/denominator) if denominator>0 else None,
                            denominator='QRF_assigned_shell' if assigned else 'selected_rows',
                            fraction_sky_95_lo=fraction_ci[0],fraction_sky_95_hi=fraction_ci[1],
                            design_se=float(uncertainty[j,shell]),sky_95_lo=float(conf[0]),sky_95_hi=float(conf[1]),
                            delta_nominal=float(n-total[0,shell]),delta_sky_95_lo=float(interval[0]),
                            delta_sky_95_hi=float(interval[1]))
                        (results if scope=='all_sky' else spatial).append(row)
        pairs = np.einsum('n,nscd,np->scdp',weight[selected],joint[selected],qrf[selected])
        np.testing.assert_allclose(pairs.sum(axis=(1,2,3)),weight[selected].sum(),rtol=0,atol=1e-6)
        for j,name in enumerate(NAMES):
            for c,label in enumerate(('dwarf','giant','other')):
                for t,true_shell in enumerate(SHELLS):
                    for d,assigned in enumerate(SHELLS):
                        migration.append(dict(selection=cut,scenario=name,latent_class=label,
                            latent_shell_kpc=true_shell,qrf_assigned_shell_kpc=assigned,
                            expected_count=float(pairs[j,c,t,d])))
    return results,spatial,migration


def main():
    require_compute()
    marker = FORECAST/'complete.json'
    receipt = json.loads(marker.read_text())
    source = verified(FORECAST/'predictions.npz',receipt['outputs']['predictions.npz'])
    keys = ('class_shell','qrf_bins','weight','sampling_cell','galactic_l','galactic_b',
            'RA','DEC','primary','strict','TYPE','scenario')
    with np.load(source,allow_pickle=False) as z:
        p = {k:z[k] for k in keys}
    if not (p['TYPE']=='PSF').all() or list(p['scenario'])!=list(NAMES):
        raise ValueError('LF count report requires the verified native PSF scenario draw')
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'complete.json').exists():
        raise FileExistsError(OUT/'complete.json')
    names = ('shell_counts.csv','spatial_counts.csv','migration.csv')
    for name,rows in zip(names,summary(p)):
        csv_rows(OUT/name,rows)
    save_json(OUT/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),draw_rows=len(p['weight']),
        inputs={str(x.relative_to(ROOT)):sha(x) for x in (marker,Path(__file__))},
        sky_bootstrap=dict(nside=4,draws=DRAWS,seed=SEED,holding_selected_total_fixed=True),
        interpretation='Conditional counts under the named population hypotheses. Paired sky-block intervals describe spatial variation and design SE describes the stratified draw. Neither includes all astrophysical model uncertainty.',
        outputs={x:sha(OUT/x) for x in names}))


if __name__=='__main__':
    main()
