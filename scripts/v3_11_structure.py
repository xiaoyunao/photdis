#!/usr/bin/env python3
"""Test full distance distributions in two fixed northern cluster fields."""

import csv
import json
import os
from pathlib import Path

import joblib
import numpy as np

from photdis.photometry import CLEAN_REJECT_MASKBITS
from photdis.stage1 import galactic, require_compute, tile_bounds
from photdis.v3.controls import apply_temperature
from v2_1_near import ROOT, atomic_npz, save_json, sha
from v3_3_sed import verified
from v3_5_dr11_area import BANDS, random_memmap, tile_map
from v3_6_vac_distances import read_columns
from v3_8_direct_figures import BLUE, GRAY, ORANGE, STYLE, plt, save
from v3_9_catalog import OUT as EXPANSION, settings
from v3_9_delivery import DEST as RELEASE
from v3_9_references import join, predictor_features, subset
from v3_9_seam import BOUNDARY
from v3_10_coverage import ANNULI, DEST as COVERAGE, M92, m92_zones
from v3_10_figures import check_templates
from v3_10_science import DISTANCE
from v3_10_selection import DEST as OLD
from v3_10_working_sample import DEST as WORK, RADIAL_MASS, shell_mass
from v3_11_selection import DEST as NEW

DEST = ROOT/'data/v3_11/structure'
CLUSTERS = {'M92':(M92,8.50), 'M13':((250.42181,36.45986),7.42)}
REF = 'https://academic.oup.com/view-large/353698133'


def zones(name, ra, dec):
    if name=='M92':
        return m92_zones(ra,dec)
    center,_ = CLUSTERS[name]
    x,y = np.deg2rad([ra,dec])
    a,b = np.deg2rad(center)
    sep = np.rad2deg(np.arccos(np.clip(np.sin(y)*np.sin(b)+np.cos(y)*np.cos(b)*np.cos(x-a),-1.,1.)))
    return [(sep>=lo)&(sep<hi) for lo,hi in ANNULI]


def areas(name, inputs):
    if name=='M92':
        marker = COVERAGE/'north/complete.json'
        file = marker.parent/'area.npz'
        record = json.loads(marker.read_text())
        inputs[str(marker.relative_to(ROOT))] = sha(marker)
        return read_columns(verified(file,record['outputs'][file.name]),('m92_area_deg2',))['m92_area_deg2'][3]
    side = 'north'
    native = settings()[side]
    path = Path(native['sweep_root']).parent.parent/'randoms/randoms-north-1-0.fits'
    raw,density,names = random_memmap(path)
    marker = EXPANSION/side/'inventory.json'
    inventory = json.loads(marker.read_text())
    tiles = tile_map([s for group in inventory['groups'] for s in group])
    good_bits = sum(1<<i for i in CLEAN_REJECT_MASKBITS)
    threshold = (2.5/np.log(10)/.1/(10**((22.5-20.)/2.5)))**2
    nums = np.zeros(2,dtype=np.int64)
    for start in range(0,len(raw),300000):
        z = raw[start:start+300000]
        ra,dec = z[names['RA']],z[names['DEC']]
        center = CLUSTERS[name][0]
        keep = (abs(ra-center[0])<2) & (abs(dec-center[1])<1.25)
        if not keep.any():
            continue
        z = z[keep]
        ra,dec = ra[keep],dec[keep]
        b = galactic(ra,dec)[1]
        ij = tiles[(ra%360).astype(int),np.minimum((dec+90).astype(int),179)]
        admitted = (b>30)&(dec>=BOUNDARY)&(ij>=0)&(z[names['EBV']]>=0)&(z[names['EBV']]<.03)
        admitted &= (z[names['RELEASE']]==native['release']) & (np.char.strip(z[names['PHOTSYS']])==b'N')
        admitted &= (z[names['MASKBITS']].astype('i8')&good_bits)==0
        for band in BANDS:
            admitted &= (z[names['NOBS_'+band]]>0)&(z[names['PSFDEPTH_'+band]]>=threshold)
        nums += [(zz&admitted).sum() for zz in zones(name,ra,dec)]
    inputs[str(marker.relative_to(ROOT))] = sha(marker)
    inputs['m13_random_catalog'] = dict(file=str(path),size=path.stat().st_size,
                                       mtime_ns=path.stat().st_mtime_ns,density=density)
    return nums/density


def overlaps(name, sweep):
    center,_ = CLUSTERS[name]
    r0,b0,r1,b1 = tile_bounds(sweep+'.fits')
    return r0<center[0]+2. and r1>center[0]-2. and b0<center[1]+1.25 and b1>center[1]-1.25


def scan(name, model, excluded, inputs):
    root = RELEASE
    marker = root/'complete.json'
    receipt = json.loads(marker.read_text())
    inputs[str(marker.relative_to(ROOT))] = sha(marker)
    with verified(root/'unified_parts.csv',receipt['outputs']['unified_parts.csv']).open() as f:
        parts = {part['pdf_file']:part for part in csv.DictReader(f) if part['region']=='north' and
                 part['pdf_file'] and overlaps(name,part['sweep'])}
    sources, count_excluded = [],0
    for shard in range(48):
        base = WORK/'north'
        path = base/f'{shard:02d}.json'
        meta = json.loads(path.read_text())
        if not meta['rows']:
            continue
        n = f'{shard:02d}.csv'
        with verified(base/n,meta['outputs'][n]).open() as f:
            available = {int(k['part_index']):k for k in csv.DictReader(f) if k['file'] in parts}
        if not available:
            continue
        inputs[str(path.relative_to(ROOT))] = sha(path)
        n = f'{shard:02d}.npz'
        p = read_columns(verified(base/n,meta['outputs'][n]),
            ('ls_id','TYPE','RA','DEC','g0','r0','state','part_index','pdf_rows','radial_mass','score'))
        on,off = zones(name,p['RA'],p['DEC'])
        in_window = (on|off)&(p['state']>=2)&np.isin(p['part_index'],list(available))
        in_training = np.isin(p['ls_id'],excluded)
        count_excluded += int((in_training&in_window).sum())
        p = subset(dict(p,target=on),in_window&~in_training)
        if not len(p['ls_id']):
            continue
        cuts = 10+5*np.log10(DISTANCE)
        p['pdf_mass'] = np.empty((len(p['ls_id']),len(cuts)-1))
        p['new_score'] = np.empty(len(p['ls_id']))
        for i in np.unique(p['part_index']):
            k = np.flatnonzero(p['part_index']==i)
            point = available[i]
            z = read_columns(verified(ROOT/point['file'],point['pdf_sha256']),
                             ('TYPE','ls_id','photo_row','qrf_mu_edges','qrf_mu_mass'))
            a = p['pdf_rows'][k]
            part = parts[point['file']]
            x = read_columns(verified(ROOT/part['photo_file'],part['photo_sha256']),
                             ('TYPE','ls_id','mag_grz'))
            t = z['photo_row'][a]
            np.testing.assert_array_equal(z['ls_id'][a],p['ls_id'][k])
            np.testing.assert_array_equal(x['ls_id'][t],p['ls_id'][k])
            if not np.all((p['TYPE'][k]=='PSF')&(z['TYPE'][a]=='PSF')&(x['TYPE'][t]=='PSF')):
                raise ValueError('Expected native PSF in all cluster joins')
            f = predictor_features(x['mag_grz'][t])
            p['new_score'][k] = apply_temperature(model['model'].predict_proba(f),model['temperature'])[:,0]
            p['pdf_mass'][k] = shell_mass(z['qrf_mu_edges'][a],z['qrf_mu_mass'][a],cuts)
        sources.append(p)
    return join(sources),count_excluded


def report(name,t,area,threshold):
    r = t['target']
    color = t['g0']-t['r0']
    radial = t['radial_mass'].sum(axis=1)>=RADIAL_MASS
    stages = [('Supported',t['state']>=2),('v3.10',t['state']==4),
              ('Revised class',t['new_score']>=threshold),
              ('Revised combined',(t['new_score']>=threshold)&radial)]
    ref = CLUSTERS[name][1]
    rows,values = [],{}
    centers = .5*(DISTANCE[1:]+DISTANCE[:-1])
    window = abs(centers-ref)<=1.
    with plt.rc_context(STYLE):
        fig,axes = plt.subplots(2,4,figsize=(20,9),layout='constrained',sharex=True)
        for row,(a,b) in enumerate(((-np.inf,np.inf),(-np.inf,.5))):
            k = (color>=a)&(color<b)
            for col,(view,g) in enumerate(stages):
                target,control = (k&g&r,k&g&~r)
                on,off = [t['pdf_mass'][j].sum(axis=0)/n for j,n in zip((target,control),area)]
                excess = on-off
                peak = int(np.argmax(excess))
                rows.append(dict(cluster=name,view=view,gr_min=a,gr_max=b,
                    target_n=int(target.sum()),control_n=int(control.sum()),reference_kpc=ref,
                    target_area_deg2=area[0],control_area_deg2=area[1],
                    reference_window_excess_per_deg2=float(excess[window].sum()),
                    full_range_peak_kpc=float(centers[peak]) if excess[peak]>0 else None,
                    full_range_peak_excess_per_deg2=float(excess[peak])))
                values[f'{row}_{col}'] = np.stack([on,off,excess])
                ax = axes[row,col]
                ax.stairs(on,DISTANCE,color=BLUE,label='Cluster annulus')
                ax.stairs(off,DISTANCE,color=ORANGE,label='Control annulus')
                ax.stairs(excess,DISTANCE,color=GRAY,label='Excess')
                ax.axvline(ref,color=GRAY,ls='--',lw=1.25)
                label = 'All colors' if row==0 else 'Dereddened g − r < 0.50'
                ax.set(xlabel='Distance (kpc)',ylabel='Stars per cell per sq. deg.',
                       title=f'{view} · {label}',xlim=(1,20))
        axes[0,0].legend()
        fig.suptitle(f'{name} · cluster and background distance distributions')
        save(fig,DEST,name.lower(),rows,dict(distance_bins=DISTANCE,**values))


def run():
    inputs = {'figure_templates':check_templates(),'distance_source':REF}
    marker = NEW/'north/complete.json'
    m = json.loads(marker.read_text())
    inputs[str(marker.relative_to(ROOT))] = sha(marker)
    model = joblib.load(verified(marker.parent/'model.joblib',m['outputs']['model.joblib']))
    held = read_columns(verified(marker.parent/'scores.npz',m['outputs']['scores.npz']),
                        ('ls_id','TYPE','new_train','new_cal'))
    marker = OLD/'north/complete.json'
    old = read_columns(verified(marker.parent/'scores.npz',json.loads(marker.read_text())[
        'outputs']['scores.npz']),('ls_id','TYPE','training','calibration'))
    inputs[str(marker.relative_to(ROOT))] = sha(marker)
    np.testing.assert_array_equal(old['ls_id'],held['ls_id'])
    np.testing.assert_array_equal(old['TYPE'],held['TYPE'])
    excluded = held['ls_id'][held['new_train']|held['new_cal']|old['training']|old['calibration']]
    DEST.mkdir(parents=True,exist_ok=True)
    reports = {}
    for name in CLUSTERS:
        t,n = scan(name,model,excluded,inputs)
        area = areas(name,inputs)
        if not t or (area<=0).any():
            reports[name] = dict(status='no supported native footprint')
            continue
        if len(t['ls_id'])!=len(np.unique(t['ls_id'])):
            raise ValueError('Cluster contains duplicate physical catalog keys')
        report(name,t,area,model['threshold'])
        atomic_npz(DEST/f'{name.lower()}_sources.npz',**t)
        reports[name] = dict(status='measured',sources=len(t['ls_id']),direct_training_exclusions=n,
                             target_area_deg2=area[0],control_area_deg2=area[1])
    save_json(DEST/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,clusters=reports,
        outputs={f.name:sha(f) for f in DEST.iterdir() if f.name!='complete.json'}))


if __name__=='__main__':
    require_compute()
    run()
