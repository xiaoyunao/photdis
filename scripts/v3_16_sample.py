#!/usr/bin/env python3
"""Collect fixed native north GD-1 photometry and rebin archived QRF PDFs."""
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from photdis.stage1 import require_compute, tile_bounds
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_6_vac_distances import read_columns
from v3_9_catalog import OUT, settings
from v3_9_references import join, subset
from v3_10_working_sample import shell_mass
from v3_11_layers import read_receipt
from v3_13_clusters import model_and_history
from v3_16_gd1 import CONFIG, DEST, LIMITS, MASK, NSTEP, P, bounding_box, classify, required_depth

NORTH = OUT/'north'
BANDS = 'GRZ'
D_EDGES = np.arange(5.,14.01,.25)
D = (D_EDGES[1:]+D_EDGES[:-1])/2
COLOR = np.linspace(-.12,1.32,37)
APPARENT = np.linspace(14.,22.4,85)
PHOTO = ('TYPE','ls_id','identity','RA','DEC','MASKBITS','r','g0','r0','row_state',
         'north_primary_region',*(f'{field}_{b}' for field in ('PSFDEPTH','FRACFLUX') for b in BANDS))


def detect_tiles():
    ra0,dec0,ra1,dec1 = bounding_box()
    for source in sorted(Path(settings()['north']['sweep_root']).glob('sweep-*.fits')):
        a,b,c,d = tile_bounds(source)
        if c>ra0 and a<ra1 and d>dec0 and b<dec1:
            yield source.stem


def take_sources(tile, footprint, history):
    home = NORTH/'photo'/tile
    arch = NORTH/'predict'/tile
    photo_receipt = home/'complete.json'
    if not photo_receipt.is_file():
        return {}, [], {tile:'no archived native sweep'}
    metadata = json.loads(photo_receipt.read_text())
    prediction = arch/'complete.json'
    inputs = {str(photo_receipt.relative_to(ROOT)):sha(photo_receipt)}
    if not prediction.is_file():
        raise ValueError(f'Missing native prediction receipt for {tile}')
    pm = json.loads(prediction.read_text())
    if pm['binding']['photo']!=inputs[str(photo_receipt.relative_to(ROOT))]:
        raise ValueError(f'Prediction input changed for {tile}')
    inputs[str(prediction.relative_to(ROOT))] = sha(prediction)
    rows,ledger = [],[]
    for name,digest in metadata['outputs'].items():
        if name=='parent_state.npz':
            continue
        p = read_columns(verified(home/name,digest),PHOTO)
        l,offset,block,ribbon,sky = classify(p['RA'],p['DEC'])
        p = subset(p,sky)
        block, ribbon = block[sky],ribbon[sky]
        tier = np.searchsorted(LIMITS,p['r'],side='right')
        if not np.all(p['TYPE']=='PSF'):
            raise ValueError('Archived non-PSF reached native admission')
        good = tier<len(LIMITS)
        choose = np.minimum(tier,len(LIMITS)-1)
        depth = np.column_stack([p[f'PSFDEPTH_{b}'] for b in BANDS])
        frac = np.column_stack([p[f'FRACFLUX_{b}'] for b in BANDS])
        color = p['g0']-p['r0']
        clean = (p['row_state']==7)&p['north_primary_region']&((p['MASKBITS'].astype('i8')&MASK)==0)
        clean &= (frac<P['maximum_grz_fracflux']).all(axis=1)
        use = good&clean&(depth>=required_depth()[choose]).all(axis=1)
        use &= (color>=COLOR[0])&(color<COLOR[-1])&(p['r0']>=APPARENT[0])&(p['r0']<APPARENT[-1])
        use &= footprint[choose,block//NSTEP,ribbon]
        train = np.isin(p['identity'],history)
        use &= ~train
        accepted = np.zeros(len(use),bool)
        if use.any() and name in pm['outputs']:
            part = read_columns(verified(arch/name,pm['outputs'][name]),
                                ('TYPE','ls_id','qrf_mu_edges','qrf_mu_mass'))
            src = np.flatnonzero(use)
            ind = np.argsort(part['ls_id'])
            x = part['ls_id'][ind]
            if not len(x) or np.any(np.diff(x)<=0):
                raise ValueError('Empty or repeated archived QRF keys')
            pos = np.minimum(np.searchsorted(x,p['ls_id'][src]),len(x)-1)
            matched = x[pos]==p['ls_id'][src]
            src,qidx = src[matched],ind[pos[matched]]
            if not np.all(part['TYPE'][qidx]=='PSF'):
                raise ValueError('Non-PSF archived PDF')
            accepted[src] = True
            bins = shell_mass(part['qrf_mu_edges'][qidx],part['qrf_mu_mass'][qidx],
                              np.r_[-np.inf,10+5*np.log10(D_EDGES),np.inf])
            np.testing.assert_allclose(bins.sum(axis=1),1.,rtol=0,atol=2e-7)
            rows.append(dict(TYPE=p['TYPE'][src],ls_id=p['ls_id'][src],phi1=l[sky][src],
                             phi2_offset=offset[sky][src],r=p['r'][src],r0=p['r0'][src],
                             color=color[src],block=block[src],ribbon=ribbon[src],tier=tier[src],
                             qrf=bins.astype('f4')))
        ledger.append(dict(tile=tile,chunk=name,sky=int(sky.sum()),native_quality=int((clean&good).sum()),
                           training_excluded=int((good&clean&train).sum()),
                           ready=int(use.sum()),common=int(accepted.sum())))
    inputs[tile+'_native_ledger'] = metadata.get('counts',{})
    return join(rows),ledger,inputs


def run():
    binding = {'geometry':sha(CONFIG)}
    area = read_receipt(DEST/'footprint','complete.json','area.npz',('coverage_ok','passed'),binding)
    if area['passed'][0].sum()<P['minimum_viable_segments']:
        raise RuntimeError('GD-1 fails the frozen bright coverage requirement')
    _,_,history = model_and_history('north',binding)
    def work(tile):
        return take_sources(tile,area['coverage_ok'],history)
    workers = min(6,int(os.getenv('SLURM_CPUS_PER_TASK',1)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        result = list(pool.map(work,detect_tiles()))
    t = join([x for x,_,_ in result if x])
    if not t or len(np.unique(t['ls_id']))!=len(t['ls_id']) or not np.all(t['TYPE']=='PSF'):
        raise ValueError('Empty or nonunique common native PSF roster')
    for _,_,extra in result:
        binding.update(extra)
    folder = DEST/'sample'
    folder.mkdir(parents=True,exist_ok=True)
    atomic_npz(folder/'sources.npz',**t)
    csv_rows(folder/'admission.csv',[k for _,ledger,_ in result for k in ledger])
    save_json(folder/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=binding,
        rows=len(t['TYPE']),outputs={k:sha(folder/k) for k in ('sources.npz','admission.csv')}))


if __name__=='__main__':
    require_compute()
    run()
