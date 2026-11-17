#!/usr/bin/env python3
"""Recover the v3.16 native parent and sample corrected GD-1 ribbons."""
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from astropy.io import fits

from photdis.reference_v1_4 import unique_join
from photdis.stage1 import galactic, require_compute, tile_bounds
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_5_dr11_area import random_memmap
from v3_6_vac_distances import read_columns
from v3_9_catalog import CONFIG as CATALOG, settings
from v3_9_references import join, subset
from v3_11_layers import read_receipt
from v3_13_clusters import model_and_history
from v3_16_gd1 import (BLOCKS, CONFIG as OLD_CONFIG, DEST as OLD, EDGES, LIMITS,
                       MASK, NSEG, NSTEP, OFFSETS, P, ROTATION, WIDTH, angles,
                       classify, coordinates, required_depth, unit)
from v3_16_sample import APPARENT, BANDS, COLOR, NORTH, PHOTO

DEST = ROOT/'data/v3_17'
CONFIG = ROOT/'configs/v3_17_gd1.json'
C = json.loads(CONFIG.read_text())
INPUT = DEST/'input'
TRACK_SHA = '9b995704f6de33b9ffb0c6894f09410be7a89b8c2840af3deaa472e914cf6e55'
NEW = np.r_[0, C['far_offsets_deg'], C['blank_offsets_deg']]


def read_track():
    raw = fits.getdata(verified(INPUT/'v3_Table3.fits',TRACK_SHA),1)
    return np.sort(raw,order='phi1')


def center(lon,nodes):
    return np.interp(lon,nodes['phi1'],nodes['phi2'])


def bounds(nodes):
    lon = np.linspace(EDGES[0],EDGES[-1],401)
    ra,dec = angles(ROTATION.T@unit(np.tile(lon,2),np.r_[
        center(lon,nodes)+NEW.min()-WIDTH,center(lon,nodes)+NEW.max()+WIDTH]))
    ra %= 360
    return ra.min()-1,dec.min()-1,ra.max()+1,dec.max()+1


def newer(ra,dec,nodes):
    l,b = coordinates(ra,dec)
    delta = b-center(l,nodes)
    rb = np.full(len(l),-1,dtype='i2')
    for k,offset in enumerate(NEW):
        rb[np.abs(delta-offset)<WIDTH] = k
    block = np.floor((l-EDGES[0])/P['sky_block_deg']).astype('i2')
    return l,delta,block,rb,(block>=0)&(block<BLOCKS)&(rb>=0)


def random_areas(nodes,path):
    rows,density,names = random_memmap(path)
    a,b,c,d = bounds(nodes)
    result = np.zeros((len(LIMITS),BLOCKS,len(NEW)),int)
    release = settings()['north']['release']
    depth = required_depth()
    for first in range(0,len(rows),350000):
        x = rows[first:first+350000]
        ra,dec = x[names['RA']],x[names['DEC']]
        take = (ra>=a)&(ra<c)&(dec>=max(b,P['north_declination_min']))&(dec<d)
        x,ra,dec = x[take],ra[take],dec[take]
        _,_,block,ribbon,ok = newer(ra,dec,nodes)
        ok &= (galactic(ra,dec)[1]>P['minimum_galactic_latitude'])&(x[names['RELEASE']]==release)
        ok &= (x[names['EBV']]>=0)&(x[names['EBV']]<P['maximum_ebv'])
        ok &= ((x[names['MASKBITS']].astype('i8')&MASK)==0)&(np.char.strip(x[names['PHOTSYS']])==b'N')
        ok &= np.column_stack([x[names[f'NOBS_{band}']]>0 for band in BANDS]).all(axis=1)
        dp = np.column_stack([x[names[f'PSFDEPTH_{band}']] for band in BANDS])
        for i,threshold in enumerate(depth):
            k = ok&(dp>=threshold).all(axis=1)
            np.add.at(result[i],(block[k],ribbon[k]),1)
    return result/density


def extract(tile,nodes,old_ok,new_ok,history):
    photo = NORTH/'photo'/tile
    preds = NORTH/'predict'/tile
    rec = photo/'complete.json'
    if not rec.is_file():
        return {},{},[],{tile:'no archived native sweep'}
    pmark = json.loads(rec.read_text())
    pm = json.loads((preds/'complete.json').read_text())
    ph = sha(rec)
    if pm['binding']['photo']!=ph:
        raise ValueError(f'Unbound photometry for {tile}')
    binding = {str(rec.relative_to(ROOT)):ph,
               str((preds/'complete.json').relative_to(ROOT)):sha(preds/'complete.json')}
    rows = [[],[]]
    topology = []
    for name,digest in pmark['outputs'].items():
        if name=='parent_state.npz':
            continue
        p = read_columns(verified(photo/name,digest),PHOTO)
        old = classify(p['RA'],p['DEC'])
        new = newer(p['RA'],p['DEC'],nodes)
        sky = old[-1]|new[-1]
        if not sky.any():
            continue
        p = subset(p,sky)
        if not (p['TYPE']=='PSF').all():
            raise ValueError('Unverified native type')
        tier = np.searchsorted(LIMITS,p['r'],side='right')
        k = np.minimum(tier,len(LIMITS)-1)
        depth = np.column_stack([p[f'PSFDEPTH_{band}'] for band in BANDS])
        frac = np.column_stack([p[f'FRACFLUX_{band}'] for band in BANDS])
        color = p['g0']-p['r0']
        qual = (p['row_state']==7)&p['north_primary_region']&((p['MASKBITS'].astype('i8')&MASK)==0)
        qual &= (frac<P['maximum_grz_fracflux']).all(axis=1)
        use = qual&(tier<len(LIMITS))&(depth>=required_depth()[k]).all(axis=1)
        use &= (color>=COLOR[0])&(color<COLOR[-1])&(p['r0']>=APPARENT[0])&(p['r0']<APPARENT[-1])
        use &= ~np.isin(p['identity'],history)
        qrf = np.zeros(len(p['ls_id']),bool)
        if use.any():
            fname = 'state_'+name
            state = read_columns(verified(preds/fname,pm['outputs'][fname]),('TYPE','ls_id','state'))
            m = unique_join(p['ls_id'],state['ls_id'])
            if (m<0).any() or not (state['TYPE'][m]=='PSF').all():
                raise ValueError('QRF state roster changed')
            qrf = state['state'][m]==3
        for w,(g,e) in enumerate(((old,old_ok),(new,new_ok))):
            lon,delta,block,ribbon,keep = (a[sky] for a in g)
            ix = np.flatnonzero(keep&use)
            ix = ix[e[k[ix],block[ix]//NSTEP,ribbon[ix]]]
            if len(ix):
                rows[w].append(dict(TYPE=p['TYPE'][ix],ls_id=p['ls_id'][ix],identity=p['identity'][ix],
                    phi1=lon[ix],phi2_offset=delta[ix],tier=tier[ix],block=block[ix],ribbon=ribbon[ix],
                    r=p['r'][ix],r0=p['r0'][ix],color=color[ix],qrf=qrf[ix]))
                topology.append(dict(geometry=w,tile=tile,part=name,n=len(ix),qrf=int(qrf[ix].sum())))
    return join(rows[0]),join(rows[1]),topology,binding


def sample():
    binding = {'config':sha(CONFIG),'baseline_config':sha(OLD_CONFIG),'native_config':sha(CATALOG),
               'track':TRACK_SHA}
    nodes = read_track()
    past = read_receipt(OLD/'footprint','complete.json','area.npz',('coverage_ok','area'),binding)
    old = read_receipt(OLD/'sample','complete.json','sources.npz',('TYPE','ls_id'),binding)
    random = Path(settings()['north']['sweep_root']).parent.parent/'randoms/randoms-north-1-0.fits'
    size = random.stat()
    binding['random'] = dict(path=str(random),size=size.st_size,mtime=size.st_mtime_ns)
    a = random_areas(nodes,random)
    exposure = a.reshape(len(LIMITS),NSEG,NSTEP,len(NEW)).sum(axis=2)
    good = exposure >= (np.diff(EDGES)*2*WIDTH*P['minimum_usable_area_fraction'])[None,:,None]
    ra0,dec0,ra1,dec1 = bounds(nodes)
    cfg = settings()['north']
    files = []
    for f in sorted(Path(cfg['sweep_root']).glob('sweep-*.fits')):
        x,y,z,w = tile_bounds(f)
        if z>ra0 and x<ra1 and w>dec0 and y<dec1:
            files.append(f.stem)
    _,_,hist = model_and_history('north',binding)
    def one(stem):
        return extract(stem,nodes,past['coverage_ok'],good,hist)
    with ThreadPoolExecutor(max_workers=min(8,int(os.getenv('SLURM_CPUS_PER_TASK',1)))) as pool:
        pieces = list(pool.map(one,files))
    folder = DEST/'sample'
    folder.mkdir(parents=True,exist_ok=True)
    identities = {}
    for k,name in enumerate(('legacy','desi')):
        t = join([x[k] for x in pieces if x[k]])
        assert all(t['TYPE']=='PSF') and len(np.unique(t['ls_id']))==len(t['ls_id'])
        if k==0:
            assert len(t['ls_id'])==175328
            np.testing.assert_array_equal(np.sort(t['ls_id'][t['qrf']]),np.sort(old['ls_id']))
        atomic_npz(folder/(name+'.npz'),**t)
        identities[name] = [len(t['ls_id']),int(t['qrf'].sum())]
    atomic_npz(folder/'area.npz',legacy=past['area'][1:],desi=a,legacy_good=past['coverage_ok'],desi_good=good)
    csv_rows(folder/'selection.csv',[r for x in pieces for r in x[2]])
    for x in pieces:
        binding.update(x[3])
    outputs = {p.name:sha(p) for p in folder.iterdir() if p.name!='complete.json'}
    save_json(folder/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=binding,
                                         populations=identities,outputs=outputs))


if __name__=='__main__':
    require_compute()
    sample()
