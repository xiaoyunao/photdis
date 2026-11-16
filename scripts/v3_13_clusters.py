#!/usr/bin/env python3
"""Measure member luminosity residuals and depth-dependent structure in three fields."""
import argparse
import csv
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.special import ndtr

from photdis.baseline_v1 import support
from photdis.photometry import CLEAN_REJECT_MASKBITS
from photdis.reference_v1_4 import unique_join
from photdis.stage1 import galactic, require_compute, tile_bounds
from photdis.v3_3 import frozen_qrf
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_4_near_forecast import prepare
from v3_5_dr11_area import BANDS, random_memmap
from v3_8_direct_figures import BLUE, GRAY, ORANGE, STYLE, density, plt, save
from v3_9_catalog import LEDGER, from_north, read_native, scan_chunk, settings
from v3_9_references import join, subset
from v3_10_coverage import ANNULI, m92_zones
from v3_10_figures import check_templates
from v3_10_science import DISTANCE
from v3_10_working_sample import shell_mass
from v3_11_layers import read_receipt
from v3_11_structure import CLUSTERS
from v3_12_reference import DEST as RELATION, M30, predict, separation
from v3_12_structure import D30, INPUT, MEMBERS_SHA, load_model

DEST = ROOT/'data/v3_13/clusters'
SYSTEMS = {**CLUSTERS,'M30':(M30,D30)}
MAG = ((-np.inf,20.),(20.,21.),(21.,22.),(22.,23.))
GR = np.linspace(-.5,2.5,121)
MR = np.linspace(-3,10,131)
_STATE = None


def zones(name,ra,dec):
    if name=='M92':
        return m92_zones(ra,dec)
    distance = separation(ra,dec,SYSTEMS[name][0])
    return [(distance>=lo)&(distance<hi) for lo,hi in ANNULI]


def native_config(name):
    side = 'south' if name=='M30' else 'north'
    all_cfg = settings()
    cfg = all_cfg[side]
    ebv = .08 if side=='south' else .03
    cfg['maximum_grz_error_mag'] = all_cfg['maximum_grz_error_mag']
    cfg['domains'] = {'cluster':{'r_max':23.,'abs_b_min':30.,'ebv_max':ebv}}
    return side,cfg,ebv


def photos(name,inputs):
    side,cfg,_ = native_config(name)
    center,_ = SYSTEMS[name]
    path = Path(cfg['sweep_root'])
    records,roster = [],[]
    for f in sorted(path.glob('sweep-*.fits')):
        r0,b0,r1,b1 = tile_bounds(f)
        if not (r0<center[0]+2. and r1>center[0]-2. and b0<center[1]+1.25 and b1>center[1]-1.25):
            continue
        raw = read_native(f)
        on,off = zones(name,raw['RA'],raw['DEC'])
        if not (on|off).any():
            continue
        stat = f.stat()
        inputs[str(f)] = dict(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)
        selected = raw[on|off]
        with np.errstate(divide='ignore',invalid='ignore'):
            r = (22.5-2.5*np.log10(selected['FLUX_R'].astype(float))).astype('f4')
        for photo,code in scan_chunk(selected,(r0,b0,r1,b1),cfg,0):
            if photo:
                p = subset(photo,photo['row_state']==7)
                keys = ('TYPE','ls_id','source_id','identity','RA','DEC','r','g0','r0','z0',
                        'mag_grz','error_grz','EBV')
                records.append(dict(**{key:p[key] for key in keys},target=zones(name,p['RA'],p['DEC'])[0]))
        groups = [((r>=lo)&(r<hi),lo,hi) for lo,hi in MAG]
        groups += [(~(np.isfinite(r)&(r<23)),23.,np.inf)]
        for mask,lo,hi in groups:
            roster.append(dict(sweep=f.name,r_min=lo,r_max=hi,
                      **dict(zip(LEDGER,np.bincount(code[mask],minlength=len(LEDGER))))))
    t = join(records)
    if not t or not np.all(t['TYPE']=='PSF') or len(np.unique(t['ls_id']))!=len(t['ls_id']):
        raise ValueError('Missing or repeated native PSF sources')
    if sum(row['precise_parent'] for row in roster)!=len(t['ls_id']):
        raise ValueError('Native magnitude-tier admissions do not reconcile')
    return t,roster


def exposure(name,inputs):
    side,cfg,ebv = native_config(name)
    center,_ = SYSTEMS[name]
    root = Path(cfg['sweep_root']).parent.parent/'randoms'/f'randoms-{side}-1-0.fits'
    raw,density_random,names = random_memmap(root)
    limits = np.square((2.5/np.log(10))/.1/10**((22.5-np.array([20,21,22,23]))/2.5))
    masks = sum(1<<i for i in CLEAN_REJECT_MASKBITS)
    counts = np.zeros((5,2),dtype=np.int64)
    for i in range(0,len(raw),300000):
        part = raw[i:i+300000]
        r,b = part[names['RA']],part[names['DEC']]
        keep = (abs(r-center[0])<2)&(abs(b-center[1])<1.25)
        if not keep.any():
            continue
        a,r,b = part[keep],r[keep],b[keep]
        lat = galactic(r,b)[1]
        admitted = (abs(lat)>30)&(a[names['RELEASE']]==cfg['release'])
        admitted &= (np.char.strip(a[names['PHOTSYS']])==(b'S' if side=='south' else b'N'))
        admitted &= (a[names['EBV']]>=0)&(a[names['EBV']]<ebv)&((a[names['MASKBITS']].astype('i8')&masks)==0)
        admitted &= np.logical_and.reduce([a[names['NOBS_'+band]]>0 for band in BANDS])
        minimum_depth = np.minimum.reduce([a[names['PSFDEPTH_'+band]] for band in BANDS])
        for j,z in enumerate(zones(name,r,b)):
            counts[0,j] += (admitted&z).sum()
            counts[1:,j] += ((admitted&z)[:,None]&(minimum_depth[:,None]>=limits)).sum(axis=0)
    s = root.stat()
    inputs[str(root)] = dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,density=density_random)
    result = counts/density_random
    if np.any(result[0]<=0) or np.any(np.diff(result,axis=0)>1e-12):
        raise ValueError('Invalid angular/depth coverage')
    return result


def model_and_history(side,inputs):
    if side=='north':
        model,digest = from_north()
        from v3_9_catalog import OUT
        root = OUT/'north/native_model'
        rows = read_receipt(root,'complete.json','roles.npz',('TYPE','source_id','role'),inputs)
        selected = np.isin(rows['role'],('gaia_fit','specdis_fit','calibration'))
        if not np.all(rows['TYPE'][selected]=='PSF'):
            raise ValueError('Training ledger includes non-PSF entries')
        ids = rows['source_id'][selected]
        inputs['native_north_model'] = digest
        return model['qrf'],model['mu_offset'],ids
    model,_,bound = prepare()
    inputs.update({str(p.relative_to(ROOT)):sha(p) for p in bound})
    return model['qrf'],0.,model['qrf'].training_identity


def predict_batch(j):
    model,t,offset = _STATE
    q = frozen_qrf(model,t['mag_grz'][j],t['error_grz'][j],t['identity'][j])
    k = q['supported']
    edges = q['mu_edges'][k]+offset
    w = q['mass'][k]
    cuts = np.r_[-np.inf,10+5*np.log10(DISTANCE),np.inf]
    bins = shell_mass(edges,w,cuts)
    np.testing.assert_allclose(bins.sum(axis=1),1.,atol=1e-10,rtol=0)
    return dict(row=j[k],distance=q['distance'][k]*10**(offset/5),hist=bins.astype('f4'))


def forecasts(side,t,inputs):
    global _STATE
    model,offset,history = model_and_history(side,inputs)
    t['training_or_calibration'] = np.isin(t['identity'],history)
    ix = np.flatnonzero(~t['training_or_calibration']&
                        support(model,np.pad(t['mag_grz'],((0,0),(0,2)))))
    if not len(ix):
        raise ValueError('No frozen forecast support')
    _STATE = model,t,offset
    size = int(os.environ.get('SLURM_CPUS_PER_TASK','1'))
    groups = [ix[i:i+384] for i in range(0,len(ix),384)]
    with ProcessPoolExecutor(max_workers=min(size,16),mp_context=mp.get_context('fork')) as pool:
        q = join(list(pool.map(predict_batch,groups)))
    _STATE = None
    t['qrf_kpc'] = np.full((len(t['r']),5),np.nan)
    t['qrf_kpc'][q['row']] = q['distance']
    t['qrf_mass'] = np.zeros((len(t['r']),len(DISTANCE)+1),dtype='f4')
    t['qrf_mass'][q['row']] = q['hist']
    emp = load_model(RELATION/side,'models.joblib',inputs)['relation']
    t['linear_mu'],t['linear_width'] = predict(emp,t['g0']-t['r0'],t['r0'])
    t['linear_mu'][t['training_or_calibration']] = np.nan


def get_members(name,t):
    public = {}
    with verified(INPUT,MEMBERS_SHA).open() as f:
        for row in csv.DictReader(f):
            if row['cluster']==name and float(row['memberprob'])>.9:
                public[int(row['source_id'])] = (float(row['ra']),float(row['dec']))
    t['member'] = np.isin(t['source_id'],list(public))&(unique_join(t['source_id'],t['source_id'])>=0)
    d = np.array(list(public.values()))
    return int(zones(name,d[:,0],d[:,1])[0].sum())


def phase(cr,mr):
    """Broad empirical CMD boxes for old metal-poor populations."""
    return np.select([
        (mr>=5.4)&(cr>=.28)&(cr<1.6),
        (mr>=3.6)&(mr<5.4)&(cr>=.10)&(cr<.65),
        (mr>=2.5)&(mr<3.6)&(cr>=.35)&(cr<.8),
    ],['lower_main_sequence','turnoff_region','subgiant_region'],default='outside_boxes')


def distribution(t,dist,area):
    m = t['r0']-10-5*np.log10(dist)
    gr,rz = t['g0']-t['r0'],t['r0']-t['z0']
    t['Mr_reference'] = m
    t['phase_box'] = phase(gr,m)
    with np.errstate(invalid='ignore',divide='ignore'):
        qmu = 10+5*np.log10(t['qrf_kpc'][:,2])
    true_mu = 10+5*np.log10(dist)
    styles = {'QRF':qmu,'Color relation':t['linear_mu']}
    linear_cdf = ndtr((10+5*np.log10(DISTANCE)[None,:]-t['linear_mu'][:,None])/t['linear_width'][:,None])
    hist = {'QRF':t['qrf_mass'][:,1:-1],'Color relation':np.nan_to_num(np.diff(linear_cdf,axis=1))}
    member = t['member']&t['target']&~t['training_or_calibration']
    windows = [('all','all',np.ones(len(gr),bool))]
    for key,x,edges in [('gr',gr,[-np.inf,.2,.35,.5,.8,1.4,np.inf]),
                         ('rz',rz,[-np.inf,.0,.15,.3,.5,.8,np.inf])]:
        windows += [(key,f'{a:g}:{b:g}',(x>=a)&(x<b)) for a,b in zip(edges[:-1],edges[1:])]
    windows += [('phase',k,t['phase_box']==k) for k in np.unique(t['phase_box'])]
    rows,struct,arrays = [],[],{}
    mids = (DISTANCE[1:]+DISTANCE[:-1])/2
    w = np.clip((np.minimum(DISTANCE[1:],dist+1)-np.maximum(DISTANCE[:-1],dist-1))/.25,0,1)
    for i,(lo,hi) in enumerate(MAG):
        tier = (t['r']>=lo)&(t['r']<hi)
        on,off = tier&t['target'],tier&~t['target']
        for axis,interval,group in windows:
            k = member&tier&group
            for method,y in styles.items():
                mask = k&np.isfinite(y)
                delta = true_mu-y[mask]
                row = dict(r_min=lo,r_max=hi,axis=axis,interval=interval,method=method,
                           member_denominator=int(k.sum()),n=int(mask.sum()))
                if len(delta):
                    med = np.median(delta)
                    row.update(delta_Mr_median=float(med),delta_Mr_NMAD=float(1.4826*np.median(abs(delta-med))),
                        distance_median_relative=float(np.median(10**(-delta/5)-1)),
                        severe=float((abs(delta)>1.).mean()))
                rows.append(row)
        for region in ('all',*np.unique(t['phase_box'])):
            group = np.ones(len(gr),bool) if region=='all' else t['phase_box']==region
            # Phase-magnitude boxes require the external distance; only all-rows
            # profiles are blind peak diagnostics.
            for method,y in styles.items():
                good = np.isfinite(y)&group
                k,j = on&good,off&good
                target,control = hist[method][k],hist[method][j]
                net = target.sum(axis=0)/area[0]-control.sum(axis=0)/area[1]
                a,b = target@w,control@w
                se = np.sqrt(np.sum((a/area[0])**2)+np.sum((b/area[1])**2))
                peak = mids[np.argmax(net)] if len(target) and net.max()>0 else np.nan
                row = dict(r_min=lo,r_max=hi,region=region,method=method,target_psf=int(on.sum()),
                    control_psf=int(off.sum()),target_support=int(k.sum()),control_support=int(j.sum()),
                    target_members=int((member&k).sum()),control_members=int((t['member']&j).sum()),
                    peak_kpc=peak,peak_bias=peak/dist-1,window_excess=float(net@w),poisson_error=se,
                    within_10_percent=bool(np.isfinite(peak) and abs(peak/dist-1)<=.1))
                struct.append(row)
                if region=='all':
                    arrays[f'distance_{i}_{method}'] = np.stack(
                        [target.sum(axis=0)/area[0],control.sum(axis=0)/area[1],net])
        for a,k in enumerate([on,off]):
            arrays[f'cmd_{i}_{a}'] = np.histogram2d(gr[k],m[k],bins=(GR,MR))[0]/area[a]
        arrays[f'cmd_excess_{i}'] = arrays[f'cmd_{i}_0']-arrays[f'cmd_{i}_1']
    return rows,struct,arrays


def plot(name,t,rows,struct,arrays,path):
    with plt.rc_context(STYLE):
        f,axes = plt.subplots(2,4,figsize=(20,9),layout='constrained')
        for i,(lo,hi) in enumerate(MAG):
            ax,bx = axes[:,i]
            for method,color in (('QRF',BLUE),('Color relation',ORANGE)):
                ax.stairs(arrays[f'distance_{i}_{method}'][2],DISTANCE,color=color,label=method)
            ax.axvline(SYSTEMS[name][1],ls='--',color=GRAY)
            ax.set(title=f'Observed r < {hi:g}' if i==0 else f'{lo:g} ≤ observed r < {hi:g}',
                   xlabel='Distance (kpc)',ylabel='Excess stars per cell per sq. deg.')
            ax.set_xlim(1,20)
            net = arrays[f'cmd_excess_{i}']
            extent = np.max(np.abs(net))
            image = bx.pcolormesh(GR,MR,net.T,cmap='RdBu_r',vmin=-extent,vmax=extent,rasterized=True)
            f.colorbar(image,ax=bx,pad=.02,label='Excess stars per cell per sq. deg.')
            bx.set(xlabel='Dereddened g − r (mag)',ylabel='Absolute r at cluster distance',
                   xlim=(-.2,1.7),ylim=(9,-2))
        axes[0,0].legend()
        f.suptitle(f'{name} · target minus background across magnitude tiers')
        save(f,path,'depth',struct,dict(distance_bins=DISTANCE,gr=GR,Mr=MR,**arrays))
        f,axes = plt.subplots(2,3,figsize=(18,9),layout='constrained')
        keep = t['target']&t['member']
        true_mu = 10+5*np.log10(SYSTEMS[name][1])
        with np.errstate(invalid='ignore',divide='ignore'):
            q = 10+5*np.log10(t['qrf_kpc'][:,2])
        fields = (('Dereddened g − r (mag)',t['g0']-t['r0'],GR),
                  ('Dereddened r − z (mag)',t['r0']-t['z0'],GR),
                  ('Observed r (mag)',t['r'],np.linspace(14,23,91)))
        for a,(method,y) in enumerate((('QRF',q),('Color relation',t['linear_mu']))):
            d = true_mu - y
            for b,(label,x,bins) in enumerate(fields):
                ax = axes[a,b]
                density(f,ax,x[keep],d[keep],bins,np.linspace(-2,2,121))
                ax.set(xlabel=label,ylabel='Predicted minus reference absolute r (mag)',title=method)
                ax.axhline(0,ls='--',color=ORANGE,lw=1.25)
        f.suptitle(f'{name} · luminosity residuals for Gaia matched members')
        save(f,path,'luminosity',rows)


def run(name):
    inputs = {'style':check_templates(),'members':MEMBERS_SHA}
    path = DEST/name
    path.mkdir(parents=True,exist_ok=True)
    t,roster = photos(name,inputs)
    side,_,ebv = native_config(name)
    area = exposure(name,inputs)
    available = get_members(name,t)
    forecasts(side,t,inputs)
    rows,struct,arrays = distribution(t,SYSTEMS[name][1],area[0])
    csv_rows(path/'admission.csv',roster)
    csv_rows(path/'area.csv',[dict(limiting_r=r,target_deg2=a[0],control_deg2=a[1])
                              for r,a in zip((0,20,21,22,23),area)])
    plot(name,t,rows,struct,arrays,path)
    atomic_npz(path/'sources.npz',**t)
    save_json(path/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),name=name,inputs=inputs,
        admitted=len(t['r']),frozen_training_excluded=int(t['training_or_calibration'].sum()),
        public_target_members=available,matched_target_members=int((t['target']&t['member']).sum()),
        ebv_max=ebv,outputs={p.name:sha(p) for p in path.iterdir() if p.name!='complete.json'}))


if __name__=='__main__':
    require_compute()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name',choices=tuple(SYSTEMS))
    run(parser.parse_args().name)
