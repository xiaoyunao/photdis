#!/usr/bin/env python3
"""Audit native DR11 cluster detection and fitted-source selection on frozen rows."""
import argparse
import json
import os
from pathlib import Path

import fitsio
import numpy as np
from scipy.spatial import cKDTree

from photdis.photometry import CLEAN_REJECT_MASKBITS
from photdis.reference_v1_4 import unique_join
from photdis.stage1 import galactic, legacy_ids, require_compute, tile_bounds
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_5_dr11_area import BANDS, random_memmap
from v3_8_direct_figures import BLUE, ORANGE, STYLE, plt, save
from v3_10_figures import check_templates
from v3_11_layers import read_receipt
from v3_13_clusters import DEST as OLD, MAG, SYSTEMS, forecasts, native_config, phase, photos, zones
from v3_13_cmd import APP, C, DEST as CMD, M5_CENTER, M5_D
from v3_12_reference import separation
from v3_9_references import subset

DEST = ROOT/'data/v3_14/audit'
MASK = 1<<13
FORCED = 1
GAIA = 1<<7
EXTRA = ('MASKBITS','FITBITS','REF_CAT','REF_ID','FRACFLUX_G','FRACFLUX_R','FRACFLUX_Z')


def read_flags(name,t,inputs):
    _,cfg,_ = native_config(name)
    center,_ = SYSTEMS[name]
    rows = {}
    found = np.zeros(len(t['ls_id']),bool)
    for path in sorted(Path(cfg['sweep_root']).glob('sweep-*.fits')):
        r0,b0,r1,b1 = tile_bounds(path)
        if not (r0<center[0]+2 and r1>center[0]-2 and b0<center[1]+1.25 and b1>center[1]-1.25):
            continue
        with fitsio.FITS(str(path)) as hdus:
            source = hdus[1].read(columns=['RELEASE','BRICKID','OBJID','TYPE',*EXTRA])
        idx = unique_join(t['ls_id'],legacy_ids(source))
        j = idx>=0
        if not j.any():
            continue
        if (found&j).any() or not np.all(np.char.strip(source['TYPE'][idx[j]].astype('U'))=='PSF'):
            raise ValueError(f'Invalid exact native join {path}')
        for key in EXTRA:
            if key not in rows:
                rows[key] = np.empty(len(found),dtype=source[key].dtype)
            rows[key][j] = source[key][idx[j]]
        found |= j
        s = path.stat()
        inputs[str(path)] = dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns)
    if not found.all():
        raise ValueError('The raw DR11 flags did not cover the frozen roster')
    return rows


def area(name,inputs):
    side,cfg,dust = native_config(name)
    center,_ = SYSTEMS[name]
    path = Path(cfg['sweep_root']).parent.parent/'randoms'/f'randoms-{side}-1-0.fits'
    raw,density,keys = random_memmap(path)
    reject = sum(1<<k for k in CLEAN_REJECT_MASKBITS)
    counts = np.zeros((2,2,5),np.int64)
    depth = (2.5/np.log(10)/.1/10**((22.5-np.arange(20,24))/2.5))**2
    for i in range(0,len(raw),300000):
        a = raw[i:i+300000]
        ra,dec = a[keys['RA']],a[keys['DEC']]
        use = (abs(ra-center[0])<2)&(abs(dec-center[1])<1.25)
        if not use.any():
            continue
        a,ra,dec = a[use],ra[use],dec[use]
        data = a[keys['MASKBITS']].astype('i8')
        valid = (abs(galactic(ra,dec)[1])>30)&(a[keys['RELEASE']]==cfg['release'])
        valid &= np.char.strip(a[keys['PHOTSYS']])==(b'N' if side=='north' else b'S')
        valid &= (a[keys['EBV']]>=0)&(a[keys['EBV']]<dust)&((data&reject)==0)
        valid &= np.logical_and.reduce([a[keys['NOBS_'+b]]>0 for b in BANDS])
        psf = np.minimum.reduce([a[keys['PSFDEPTH_'+b]] for b in BANDS])
        masked = (data&MASK)!=0
        for j,region in enumerate(zones(name,ra,dec)):
            for k in (0,1):
                p = valid&region&(masked==bool(k))
                counts[j,k,0] += p.sum()
                counts[j,k,1:] += (p[:,None]&(psf[:,None]>=depth)).sum(axis=0)
    stat = path.stat()
    inputs[str(path)] = dict(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)
    return [dict(area=name,zone=zone,masked=bool(k),limiting_r=lim,area_deg2=counts[j,k,i]/density)
            for j,zone in enumerate(('target','background')) for k in (0,1)
            for i,lim in enumerate((0,20,21,22,23))]


def get_rows(name,inputs):
    if name!='M5':
        return read_receipt(OLD/name,'complete.json','sources.npz',
                ('TYPE','ls_id','source_id','RA','DEC','target','member','g0','r0','r','z0','error_grz',
                 'qrf_kpc','linear_mu','phase_box','training_or_calibration'),inputs)
    SYSTEMS[name] = M5_CENTER,M5_D
    original = read_receipt(CMD,'complete.json','m5_rows.npz',
                   ('TYPE','ls_id','target','common','confirmed_member'),inputs)
    p,_ = photos(name,inputs)
    gr = p['g0']-p['r0']
    p = subset(p,(gr>=C[0])&(gr<C[-1])&(p['r0']>=APP[0])&(p['r0']<APP[-1]))
    j = unique_join(original['ls_id'],p['ls_id'])
    if np.any(j<0) or not np.all(original['TYPE']=='PSF') or not np.all(p['TYPE'][j]=='PSF'):
        raise ValueError('Missing native rows in M5 replay')
    p = subset(p,j)
    np.testing.assert_array_equal(p['target'],original['target'])
    forecasts('south',p,inputs)
    np.testing.assert_array_equal(np.isfinite(p['qrf_kpc'][:,2]),original['common'])
    p['member'] = original['confirmed_member']
    p['phase_box'] = phase(p['g0']-p['r0'],p['r0']-10-5*np.log10(M5_D))
    return p


def annotate(t,name):
    t['cluster_mask'] = (t['MASKBITS'].astype('i8')&MASK)!=0
    t['forced_psf'] = (t['FITBITS'].astype('i8')&FORCED)!=0
    t['fit_gaia'] = (t['FITBITS'].astype('i8')&GAIA)!=0
    t['gaia_key'] = np.char.startswith(np.char.strip(t['REF_CAT'].astype('U')),'G')&(t['REF_ID']>0)
    t['max_fracflux'] = np.maximum.reduce([t[f'FRACFLUX_{k}'] for k in 'GRZ'])
    t['max_error'] = np.max(t['error_grz'],axis=1)
    t['radius_deg'] = separation(t['RA'],t['DEC'],SYSTEMS[name][0])
    ra,dec = np.deg2rad([t['RA'],t['DEC']])
    vec = np.column_stack((np.cos(ra)*np.cos(dec),np.sin(ra)*np.cos(dec),np.sin(dec)))
    x = cKDTree(vec).query(vec,k=2)[0][:,1]
    t['nearest_admitted_arcsec'] = np.rad2deg(2*np.arcsin(np.clip(x/2,0,1)))*3600
    t['low_crowding'] = (t['max_fracflux']<.10)&(t['nearest_admitted_arcsec']>=2.)


def statistics(name,t):
    delta = 5*np.log10(SYSTEMS[name][1]/t['qrf_kpc'][:,2])
    mu_ref = 10+5*np.log10(SYSTEMS[name][1])
    gr = t['g0']-t['r0']
    common = t['target']&t['member']&np.isfinite(delta)
    mask = t['cluster_mask']
    samples,paired = [],[]
    for lo,hi in MAG:
        mag = (t['r']>=lo)&(t['r']<hi)
        for field,zone in [('target',t['target']),('background',~t['target'])]:
            for phase_label in ('all',*np.unique(t['phase_box'])):
                region = np.ones(len(gr),bool) if phase_label=='all' else t['phase_box']==phase_label
                for masked in (False,True):
                    for crowd in ('all','low_crowding','higher_crowding'):
                        cr = np.ones(len(gr),bool) if crowd=='all' else t['low_crowding']==(crowd=='low_crowding')
                        a = mag&zone&region&(mask==masked)&cr
                        if not a.any():
                            continue
                        k = a&t['member']&np.isfinite(delta)
                        y = delta[k]
                        z = (mu_ref-t['linear_mu'])[a&t['member']&np.isfinite(t['linear_mu'])]
                        samples.append(dict(r_lo=lo,r_hi=hi,area=field,phase=phase_label,masked=masked,
                          crowding=crowd,psf=int(a.sum()),forced_psf=int((a&t['forced_psf']).sum()),
                          gaia_fit=int((a&t['fit_gaia']).sum()),gaia_key=int((a&t['gaia_key']).sum()),
                          trained_qrf_excluded=int((a&t['training_or_calibration']).sum()),
                          members=int((a&t['member']).sum()),qrf_members=int(k.sum()),
                          qrf_delta_mr=np.median(y) if len(y) else None,
                          relation_members=len(z),relation_delta_mr=np.median(z) if len(z) else None,
                          median_error=np.median(t['max_error'][a]),median_fracflux=np.median(t['max_fracflux'][a]),
                          median_radius_deg=np.median(t['radius_deg'][a]),
                          median_nearest_arcsec=np.median(t['nearest_admitted_arcsec'][a])))
        for phase_label in np.unique(t['phase_box']):
            k = mag&common&(t['phase_box']==phase_label)&t['low_crowding']
            cells = np.column_stack((np.floor(gr/.10),np.floor(t['r']/.5),np.floor(t['max_error']/.025)))
            for cell in np.unique(cells[k],axis=0):
                q = k&np.all(cells==cell,axis=1)
                a,b = delta[q&mask],delta[q&~mask]
                if min(len(a),len(b))<5:
                    continue
                paired.append(dict(r_lo=lo,r_hi=hi,phase=phase_label,gr_lo=cell[0]*.10,r_cell_lo=cell[1]*.5,
                    error_lo=cell[2]*.025,inside=len(a),outside=len(b),
                    median_inside=np.median(a),median_outside=np.median(b),
                    inside_minus_outside=np.median(a)-np.median(b)))
    return samples,paired


def run(name):
    inputs = {'style':check_templates(),'bits':'https://www.legacysurvey.org/dr11/bitmasks/'}
    t = get_rows(name,inputs)
    t.update(read_flags(name,t,inputs))
    annotate(t,name)
    samples,paired = statistics(name,t)
    out = DEST/name
    out.mkdir(parents=True,exist_ok=True)
    csv_rows(out/'matched_cells.csv',paired)
    csv_rows(out/'area.csv',area(name,inputs))
    with plt.rc_context(STYLE):
        f,axes = plt.subplots(1,4,figsize=(19,5.5),layout='constrained',sharey=True)
        delta = 5*np.log10(SYSTEMS[name][1]/t['qrf_kpc'][:,2])
        for ax,(lo,hi) in zip(axes,MAG):
            k = t['target']&t['member']&t['low_crowding']&(t['r']>=lo)&(t['r']<hi)
            for value,color,label in ((False,BLUE,'Outside CLUSTER'),(True,ORANGE,'Inside CLUSTER')):
                j = k&(t['cluster_mask']==value)&np.isfinite(delta)
                ax.scatter(t['g0'][j]-t['r0'][j],delta[j],color=color,s=4,alpha=.4,label=f'{label} ({j.sum()})',
                           rasterized=True)
            ax.axhline(0,color='0.5',lw=1,ls='--')
            ax.set(title=f'r < {hi:g}' if not np.isfinite(lo) else f'{lo:g} ≤ r < {hi:g}',
                   xlabel='Dereddened g − r (mag)',ylim=(-2.5,3.5))
            ax.legend()
        axes[0].set_ylabel('Predicted minus reference absolute r (mag)')
        f.suptitle(f'{name} · native DR11 mask and crowding audit')
        save(f,out,'mask_residuals',samples)
    atomic_npz(out/'native_join.npz',**t)
    save_json(out/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,rows=len(t['TYPE']),
        mask_rows=int(t['cluster_mask'].sum()),matched_cells=len(paired),
        outputs={f.name:sha(f) for f in out.iterdir() if f.name!='complete.json'}))


if __name__=='__main__':
    require_compute()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name',choices=(*SYSTEMS,'M5'))
    run(parser.parse_args().name)
