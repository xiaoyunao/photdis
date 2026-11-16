#!/usr/bin/env python3
"""Check overlap CMD, populations, and the north–south selection seam."""
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from photdis.stage1 import require_compute
from photdis_figure_style import STYLE
from v2_1_near import ROOT, csv_rows, save_json, sha
from v3_6_distance_evidence import OUT as EVIDENCE
from v3_6_reference_photometry import OUT as PHOTO
from v3_6_vac_distances import read_columns
from v3_8_candidate import read_cache
from v3_8_direct_figures import density, save
from v3_8_north import OUT as NORTH

OUT = ROOT/'data/v3_8/north_report'


def associate(query,ids):
    order = np.argsort(ids)
    j = order[np.searchsorted(ids[order],query)]
    np.testing.assert_array_equal(ids[j],query)
    return j


def main():
    require_compute()
    OUT.mkdir(parents=True,exist_ok=True)
    marker = OUT/'complete.json'
    if marker.exists():
        raise FileExistsError(marker)
    pairs = read_columns(read_cache(NORTH,'pairs.npz'),('ls_id','TYPE','south_ls_id','north_q',
        'south_q','reference_kpc','north_mag','south_gr','support','dwarf_score','sed_near','north_dec'))
    south = read_columns(read_cache(EVIDENCE,'Gaia.npz'),('ls_id','TYPE','primary','consensus_class'))
    photo = read_columns(read_cache(PHOTO,'photometry.npz'),('ls_id','TYPE','magnitude'))
    e,p = (associate(pairs['south_ls_id'],a['ls_id']) for a in (south,photo))
    if not all((x=='PSF').all() for x in (pairs['TYPE'],south['TYPE'][e],photo['TYPE'][p])):
        raise ValueError('Paired comparisons require both original photometric systems to pass native PSF')
    sd = south['primary'][e]
    class_label = south['consensus_class'][e]
    north_sel = (pairs['dwarf_score']>=.80)&(pairs['sed_near']>=.90)
    score = pairs['dwarf_score']
    sed = np.isfinite(pairs['sed_near'])
    rows = []
    for cls,name in ((0,'dwarf'),(1,'giant'),(2,'other'),(-1,'unknown')):
        compared = pairs['support']&(class_label==cls)
        ss = compared&np.isfinite(score)
        test = compared&sed
        rows.append(dict(class_name=name,qualified_native_pairs=int(compared.sum()),north_scores=int(ss.sum()),
            mean_dwarf_score=float(score[ss].mean()) if ss.any() else None,
            dwarf_score_pass=int(np.sum(ss&(score>=.80))),sed_trials=int(test.sum()),
            north_pass=int(np.sum(test&north_sel)),south_pass=int(np.sum(test&sd))))
    csv_rows(OUT/'classification_comparison.csv',rows)
    seam = []
    for lo,hi in ((30,32.375),(32.375,35),(35,40),(40,45)):
        g = (pairs['north_dec']>=lo)&(pairs['north_dec']<hi)&pairs['support']
        t = g&sed
        delta = pairs['north_q'][g,2]/pairs['south_q'][g,2]-1
        seam.append(dict(dec_lo=lo,dec_hi=hi,n=int(g.sum()),sed_trials=int(t.sum()),
            north_selected=int(np.sum(t&north_sel)),south_selected=int(np.sum(t&sd)),
            median_relative_distance=float(np.median(delta)) if len(delta) else None))
    csv_rows(OUT/'declination_seam.csv',seam)
    n = pairs['north_mag']
    s = photo['magnitude'][p]
    z = pairs['reference_kpc']
    keep = pairs['support']&np.isfinite(n[:,0])&np.isfinite(n[:,1])&np.isfinite(s[:,0])&np.isfinite(s[:,1])
    gr = (n[keep,0]-n[keep,1],s[keep,0]-s[keep,1])
    xb,yb = np.linspace(-.4,2.6,101),np.linspace(-4.,16.,111)
    hist = {}
    with plt.rc_context(STYLE):
        f,axes = plt.subplots(2,2,figsize=(13,10),layout='constrained')
        for i,(label,base,d) in enumerate((('North',n[keep,1],pairs['north_q'][keep,2]),
                                           ('South',s[keep,1],pairs['south_q'][keep,2]))):
            for j,(dist,name) in enumerate(((z[keep],'Gaia reference'),(d,'Photometric distance'))):
                ax = axes[i,j]
                hist[f'panel_{i}_{j}'] = density(f,ax,gr[i],base-10-5*np.log10(dist),xb,yb)
                ax.set(ylabel='Absolute r magnitude',xlabel='Dereddened g − r (mag)',
                    ylim=(yb[-1],yb[0]),title=f'{label} · {name} · {keep.sum():,} matched stars')
        f.suptitle('Native north and south passbands on the same held-out overlap stars')
        save(f,OUT,'A02_north_south_cmd',seam,dict(color_edges=xb,absolute_edges=yb,**hist))
    files = (NORTH/'complete.json',EVIDENCE/'complete.json',PHOTO/'complete.json',Path(__file__).resolve())
    save_json(marker,dict(job=os.getenv('SLURM_JOB_ID'),inputs={str(p.relative_to(ROOT)):sha(p) for p in files},
        outputs={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p!=marker}))


if __name__=='__main__':
    main()
