#!/usr/bin/env python3
"""Freeze the faint Gaia cohort and measure its low-luminosity evidence."""
import argparse
import json
import os

import numpy as np

from photdis.gaia import join_gaia
from photdis.stage1 import require_compute
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_4_near_validate import GAIA
from v3_6_vac_distances import read_columns
from v3_8_direct_figures import MABS, STYLE, XCOL, density, plt, save
from v3_10_evidence import DEST as EVIDENCE, RVS_PARAMETERS
from v3_10_figures import check_templates

DEST = ROOT/'data/v3_10/branch_cases'


def low_luminosity(t, gaia):
    mu = 10+5*np.log10(t['reference_kpc'])
    props = dict(
        M_g=t['g0']-mu, M_r=t['r0']-mu, M_G=gaia['phot_g_mean_mag']-mu,
        bp_rp=gaia['bp_rp'], g_r=t['g0']-t['r0'], ruwe=gaia['ruwe'],
        gaia_ipd=gaia['ipd_frac_multi_peak'], gaia_dup=gaia['duplicated_source'],
        gaia_pm=np.hypot(gaia['pmra'],gaia['pmdec']))
    dra = ((t['RA']-gaia['ra']+180)%360-180)*np.cos(np.deg2rad(t['DEC']))
    sep = 3600*np.hypot(dra,t['DEC']-gaia['dec'])
    # The ordinate is observed Gaia M_G, as in the historical candidate audit.
    # Keep the reference-epoch offset as a measurement rather than rejecting
    # high-proper-motion stars by an unmatched-epoch separation threshold.
    evidence = gaia['gaia_match'] & (props['bp_rp']<1.5) & (props['M_G']>10) & (props['M_g']>10)
    good = (props['ruwe']<1.4)&~props['gaia_dup']&(props['gaia_ipd']<=10)
    return dict(**props,raw_position_offset_arcsec=sep,low_luminosity_cmd=evidence,
                clean_low_luminosity=evidence&good)


def hrd(t, out):
    f,axes = plt.subplots(3,2,figsize=(13,15),layout='constrained')
    mu = 10+5*np.log10(t['qrf_quantiles_kpc'][:,2])
    xy = ((t['g_r'],t['M_r'],t['r0']-mu,'Dereddened g − r (mag)','Absolute r magnitude',
           np.ones(len(mu),bool),XCOL,'Legacy Surveys'),
          (t['bp_rp'],t['M_G'],t['gaia_g']-mu,'Gaia BP − RP (mag)','Observed absolute G magnitude',
           np.ones(len(mu),bool),XCOL,'Gaia DR3'),
          (t['bp_rp'],t['M_G'],t['gaia_g']-mu,'Gaia BP − RP (mag)','Observed absolute G magnitude',
           t['clean_low_luminosity'],XCOL,'Blue subset'))
    tables,counts = {},[]
    for row,(x,y_ref,y_photo,xlabel,ylabel,k,xbin,name) in enumerate(xy):
        for j,(y,kind) in enumerate(((y_ref,'Gaia reference'),(y_photo,'Photometric distance'))):
            ax = axes[row,j]
            hh = density(f,ax,x[k],y[k],xbin,MABS)
            ax.set(ylim=(MABS[-1],MABS[0]),xlabel=xlabel,ylabel=ylabel,
                   title=f'{name} · {kind} · N = {k.sum():,}')
            tables[f'hist_{row}_{j}'] = hh
            counts.append(dict(sample=name,distance=kind,rows=int(k.sum()),displayed=int(hh.sum())))
    f.suptitle('Same-star color–magnitude diagrams · faint southern Gaia reference sample')
    save(f,out,'faint_blue_cmd',counts,tables)


def spectral_rows(t):
    rows = []
    q = t['qrf_quantiles_kpc'][:,2]
    g_r = t['g0']-t['r0']
    good = (t['row_state']==7)&t['evaluation_row']&np.isfinite(q)&(q>0)
    good &= (t['class_label']==0)&np.isfinite(t['rvs_feh'])&(t['rvs_feh_err']>0)
    for source,d,v in (('SpecDis',t['specdis_kpc'],t['specdis_quality']),
                       ('SPDist',t['spdist_kpc'],t['spdist_quality'])):
        common = good&v&(d>=5)&(d<10)
        for lo,hi in ((-np.inf,20),(20,21)):
            for c0,c1 in ((-np.inf,np.inf),(-np.inf,.5),(.5,.8),(.8,1.4),(1.4,np.inf)):
                for a,b in ((-np.inf,np.inf),(-np.inf,-1.5),(-1.5,-1),(-1,-.5),(-.5,np.inf)):
                    k = common&(t['r']>=lo)&(t['r']<hi)&(g_r>=c0)&(g_r<c1)
                    k &= (t['rvs_feh']>=a)&(t['rvs_feh']<b)
                    row = dict(source=source,r_min=lo,r_max=hi,color_min=c0,color_max=c1,feh_min=a,feh_max=b,n=int(k.sum()))
                    if k.any():
                        row.update(median_relative=float(np.median(q[k]/d[k]-1)),
                                   **{key:float(np.median(t[key][k])) for key in ('rvs_feh','rvs_logg','rvs_teff')})
                    rows.append(row)
    return rows


def main():
    require_compute()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render-only',action='store_true')
    args = parser.parse_args()
    lock = check_templates()
    if args.render_only:
        marker = DEST/'complete.json'
        record = json.loads(marker.read_text())
        keys = ('qrf_quantiles_kpc','g_r','M_r','r0','bp_rp','M_G','gaia_g','clean_low_luminosity')
        data = read_columns(verified(DEST/'faint_gaia.npz',record['outputs']['faint_gaia.npz']),keys)
        root = DEST/'verified_plate'
        root.mkdir(exist_ok=True)
        with plt.rc_context(STYLE):
            hrd(data,root)
        save_json(root/'complete.json',dict(style_lock_sha256=lock,source_sha256=sha(marker),
            job=os.getenv('SLURM_JOB_ID'),outputs={p.name:sha(p) for p in root.iterdir() if p.name!='complete.json'}))
        return
    inputs,data,stats = {},{},[]
    keys = ('TYPE','ls_id','source_id','RA','DEC','r','g0','r0','row_state','gaia_state','qrf_quantiles_kpc',
            'class_label','sky_role','evaluation_row','reference_kpc','specdis_kpc','spdist_kpc',
            'specdis_quality','spdist_quality',*RVS_PARAMETERS)
    for region in ('south_faint','south_low','north'):
        root = EVIDENCE/region
        path = root/'complete.json'
        record = json.loads(path.read_text())
        t = read_columns(verified(root/'references.npz',record['outputs']['references.npz']),keys)
        if not np.all(t['TYPE']=='PSF'):
            raise ValueError('Reference cohorts require native DR11 PSF')
        inputs[str(path.relative_to(ROOT))] = sha(path)
        stats.extend(dict(region=region,**row) for row in spectral_rows(t))
        if region=='south_faint':
            use = (t['row_state']==7)&t['evaluation_row']&(t['gaia_state']==6)
            use &= (t['r']>=20)&(t['r']<21)&np.isfinite(t['qrf_quantiles_kpc'][:,2])
            data = {name:values[use] for name,values in t.items()}
    if len(data['ls_id'])!=2003 or len(np.unique(data['ls_id']))!=2003:
        raise ValueError('Faint Gaia sample differs from the frozen v3.9 evaluation table')
    g = join_gaia(data['source_id'],GAIA)
    if not g['gaia_match'].all():
        raise ValueError('Faint reference cohort lost its Gaia identities')
    data.update(low_luminosity(data,g),gaia_g=g['phot_g_mean_mag'])
    DEST.mkdir(parents=True,exist_ok=True)
    atomic_npz(DEST/'faint_gaia.npz',**data)
    csv_rows(DEST/'dwarf_reference_strata.csv',stats)
    with plt.rc_context(STYLE):
        hrd(data,DEST)
    delta = 5*np.log10(data['qrf_quantiles_kpc'][:,2]/data['reference_kpc'])
    counts = dict(total=len(delta),severe=int((abs(delta)>1).sum()),
                  clean_low_luminosity=int(data['clean_low_luminosity'].sum()),
                  archived_spectral_label=int((data['class_label']>=0).sum()))
    save_json(DEST/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,**counts,
        style_lock_sha256=lock,outputs={p.name:sha(p) for p in DEST.iterdir() if p.name!='complete.json'}))
    print(counts,flush=True)


if __name__=='__main__':
    main()
