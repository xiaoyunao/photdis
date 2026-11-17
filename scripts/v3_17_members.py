#!/usr/bin/env python3
"""Audit externally identified GD-1 sources against native DR11 rows."""
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fitsio
import numpy as np
from astropy.io import fits
from scipy.spatial import cKDTree

from photdis.stage1 import legacy_ids, require_compute, tile_bounds
from v2_1_near import ROOT, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_5_dr11_pilot import in_tile
from v3_6_vac_distances import read_columns
from v3_9_catalog import scan_chunk, settings
from v3_11_layers import read_receipt
from v3_16_gd1 import EDGES, MASK, NSTEP, OFFSETS, classify, coordinates, unit
from v3_17_sample import C, CONFIG, DEST, INPUT, NEW, NORTH, center, newer, read_track

TABLES = {
    'v4_Table1.fits':'e70a661e26feac5574bbd33f964eea0bf2d8adbd1f6ab8e31e76baa72b04c29d',
    'v4_Table7_all.fits':'80fdb61ab706b8be960a69f920e3af64dd124de3bb6a3b00c9ca297c1ab34378',
    'v4_Table8_all.fits':'c54dfbc1f88c2af0d7ad44ca90fa528a17f9aafca4e63a2998df779993a725e1',
    'v3_Table7.fits':'20ad8f91d56616bacf6ff972f133a342696cb32a9089716055f036688a6dd242',
    'v3_Table8.fits':'498ee03069b3602a2a3553525eead4e45c7155ac4de4a0cbcfced31e5ab9c216',
    'v4_Figure3_empirical_isochrone.fits':'d59d99bf3143bc3dfc91d9076468a01d8e87d8ccf9318f137ffd8d017f3ec0bc',
    'v4_Figure3_dotter_isochrone.fits':'2af9aed0494d78f6b4a84aa302765501728ab753d34ebf5adf7800c7e7136c25',
}
RAD = np.deg2rad(C['position_match_arcsec']/3600)
MODEL_NAMES = ('thin','cocoon','ambiguous','foreground')
CFG = settings()
NATIVE = dict(CFG['north'],maximum_grz_error_mag=CFG['maximum_grz_error_mag'])
RAW_COLUMNS = ('RA','DEC','RELEASE','BRICKID','OBJID','TYPE','REF_CAT','REF_ID')
MEMBERS = DEST/'members_precise'


def table(file):
    return fits.getdata(verified(INPUT/file,TABLES[file]),1)


def external():
    for number,kind in [(7,'DESI'),(8,'BHB')]:
        t = np.sort(table(f'v3_Table{number}.fits'),order='SOURCE_ID')
        update = np.sort(table(f'v4_Table{number}_all.fits'),order='SOURCE_ID')
        np.testing.assert_array_equal(t['SOURCE_ID'],update['SOURCE_ID'])
        # Version four rounds positions to 0.001 deg. Use the original positions
        # while retaining the updated mixture probabilities on the identical IDs.
        if len(np.unique(t['SOURCE_ID']))!=len(t):
            raise ValueError('Duplicate published DESI identity')
        l,b = coordinates(t['RA'],t['Dec'])
        if np.max(abs(l-t['phi1']))>.002 or np.max(abs(b-t['phi2']))>.002:
            raise ValueError('Published coordinates do not reproduce the frozen frame')
        q = (l>=EDGES[0])&(l<EDGES[-1])
        for row,payload,x,y in zip(t[q],update[q],l[q],b[q]):
            p,c = (float(payload[k]) for k in ('P_THIN','P_COCOON')) if kind=='DESI' else (np.nan,np.nan)
            if kind=='DESI' and not (p>=0 and c>=0 and p+c<=1.000001):
                raise ValueError('Invalid DESI mixture probabilities')
            name = 'BHB' if kind=='BHB' else MODEL_NAMES[0 if p>.5 else 1 if c>.5 else 2 if p+c>.5 else 3]
            yield dict(catalog=kind,group=name,source_id=int(row['SOURCE_ID']),ra=float(row['RA']),
                dec=float(row['Dec']),phi1=x,phi2=y,p_thin=p,p_cocoon=c,
                p_membership=p+c,external_r0=float(row['RMAG0']),external_color=float(row['GMAG0']-row['RMAG0']))
    path = INPUT/'tavangar_candidates.csv'
    if path.is_file():
        spec = json.loads((INPUT/'tavangar_source.json').read_text())
        verified(path,spec['candidate_sha256'])
        with path.open() as f:
            for a in csv.DictReader(f):
                p = {k:float(a[k]) for k in ('ra','dec','phi1','phi2','p_membership','external_r0','external_color')}
                lon,lat = coordinates([p['ra']],[p['dec']])
                if EDGES[0]<=lon[0]<EDGES[-1]:
                    p.update(phi1_catalog=p['phi1'],phi1=lon[0],phi2=lat[0])
                    yield dict(catalog='Gaia+PS1',group='candidate',source_id=int(a['source_id']),
                               **p,p_thin=np.nan,p_cocoon=np.nan)


def native_matches(path,e):
    bounds = tile_bounds(path)
    p = np.array([[a['ra'],a['dec']] for a in e])
    a,b,c,d = bounds
    near = np.flatnonzero((p[:,0]>=a-.005)&(p[:,0]<c+.005)&(p[:,1]>=b-.005)&(p[:,1]<d+.005))
    with fits.open(path,memmap=True) as hdus:
        table = hdus[1].data
        ra,dec = table['RA'],table['DEC']
        lo,hi = p[near].min(axis=0)-.005,p[near].max(axis=0)+.005
        i = np.flatnonzero((ra>=lo[0])&(ra<=hi[0])&(dec>=lo[1])&(dec<=hi[1]))
        i = i[in_tile(ra[i],dec[i],bounds)]
        if not len(i):
            return []
        tree = cKDTree(unit(ra[i],dec[i]).T)
    centers = unit(p[near,0],p[near,1]).T
    choose = tree.query_ball_point(centers,2*np.sin(RAD/2))
    links = []
    indices_all = np.unique([i[k] for indices in choose for k in indices])
    if not len(indices_all):
        return links
    rows = fitsio.read(path,rows=indices_all,columns=list(RAW_COLUMNS),ext=1)
    for jj,indices in enumerate(choose):
        j = near[jj]
        sel = i[indices]
        if not len(sel):
            continue
        r = rows[np.searchsorted(indices_all,sel)]
        xyz = unit(r['RA'],r['DEC']).T
        arcsec = np.rad2deg(2*np.arcsin(np.linalg.norm(xyz-centers[jj],axis=1)/2))*3600
        ids = legacy_ids(r)
        for k,raw_index in enumerate(sel):
            if r['RELEASE'][k]!=NATIVE['release']:
                continue
            flag = str(r['REF_CAT'][k]).strip()
            if isinstance(r['REF_CAT'][k],bytes):
                flag = r['REF_CAT'][k].decode().strip()
            known = flag=='G3' and r['REF_ID'][k]>0
            morphology = np.char.strip(r['TYPE'][k:k+1].astype('U'))[0]
            links.append((j,int(raw_index),int(ids[k]),float(arcsec[k]),morphology,
                          bool(known and r['REF_ID'][k]==e[j]['source_id']),bool(known)))
    return links


def resolve(options):
    direct = [x for x in options if x[0][5]]
    if len(direct)==1:
        return direct[0],'gaia_id'
    if len(direct)>1 or len(options)>1:
        return None,'ambiguous'
    if not len(options):
        return None,'unmatched'
    if options[0][0][6]:
        return None,'gaia_conflict'
    return options[0],'unique_position'


def qualities(path,links):
    ix = np.unique([a[0][1] for a in links if a[0][4]=='PSF'])
    if not len(ix):
        return {}
    raw = fitsio.read(path,rows=ix,ext=1)
    host = path.stem
    ans = {}
    for p,_ in scan_chunk(raw,tile_bounds(path),NATIVE,0):
        if not p:
            continue
        frac = np.column_stack([p['FRACFLUX_'+b] for b in 'GRZ'])
        qual = (p['row_state']==7)&p['north_primary_region']&((p['MASKBITS'].astype('i8')&MASK)==0)
        qual &= (frac<.10).all(axis=1)
        for j,key in enumerate(p['ls_id']):
            ans[int(key)] = dict(clean=bool(qual[j]),r=float(p['r'][j]),
                r0=float(p['r0'][j]),color=float(p['g0'][j]-p['r0'][j]),qrf_archived=False)
    receipt = NORTH/'predict'/host/'complete.json'
    if not ans or not receipt.is_file():
        return ans
    o = json.loads(receipt.read_text())
    keys = np.array(list(ans))
    for n,digest in o['outputs'].items():
        if n.startswith('state_'):
            data = read_columns(verified(receipt.parent/n,digest),('TYPE','ls_id','state'))
            keep = (data['state']==3)&np.isin(data['ls_id'],keys)
            assert np.all(data['TYPE'][keep]=='PSF')
            for key in data['ls_id'][keep]:
                ans[key]['qrf_archived'] = True
    return ans


def run():
    e = list(external())
    inputs = {'config':sha(CONFIG),'external':TABLES}
    optional = INPUT/'tavangar_source.json'
    if optional.is_file(): inputs['gaia_audit']=sha(optional)
    nodes = read_track()
    ra,dec = [np.array([a[k] for a in e]) for k in ('ra','dec')]
    _,d0,block,r0,old = classify(ra,dec)
    _,d1,_,r1,new = newer(ra,dec,nodes)
    for j,z in enumerate(e):
        z.update(segment=block[j]//NSTEP,legacy_delta=d0[j],desi_delta=d1[j],
                 legacy_ribbon=r0[j] if old[j] else -1,desi_ribbon=r1[j] if new[j] else -1)
    roster = []
    for name in ('legacy','desi'):
        t = read_receipt(DEST/'sample','complete.json',name+'.npz',('TYPE','ls_id','qrf'),inputs)
        assert (t['TYPE']=='PSF').all()
        roster.append((set(t['ls_id']),set(t['ls_id'][t['qrf']])))
    margin = .005
    files = []
    for p in sorted(Path(CFG['north']['sweep_root']).glob('sweep-*.fits')):
        a,b,c,d = tile_bounds(p)
        if ((ra>=a-margin)&(ra<c+margin)&(dec>=b-margin)&(dec<d+margin)).any():
            files.append(p)
    def work(p):
        return native_matches(p,e)
    with ThreadPoolExecutor(max_workers=min(8,int(os.getenv('SLURM_CPUS_PER_TASK',1)))) as pool:
        result = list(pool.map(work,files))
    grouped = [[] for _ in e]
    for path,options in zip(files,result):
        info = path.stat()
        inputs[str(path)] = [info.st_size,info.st_mtime_ns]
        for x in options:
            grouped[x[0]].append((x,path))
    by_tile = {}
    for x,options in zip(e,grouped):
        chosen,mode = resolve(options)
        x.update(match=mode,n_candidates=len(options),native_type='',psf=False,clean=False,
                 qrf_archived=False,legacy_native=False,legacy_qrf=False,desi_native=False,desi_qrf=False)
        if chosen is None:
            continue
        hit,path = chosen
        _,_,key,sep,morph,_,_ = hit
        x.update(native_type=morph,psf=morph=='PSF',separation_arcsec=sep)
        if morph=='PSF':
            x['ls_id'] = key
            for (a,b),name in zip(roster,('legacy','desi')):
                x[name+'_native'],x[name+'_qrf'] = key in a,key in b
            by_tile.setdefault(path,[]).append(chosen)
    phot = {}
    for path,links in by_tile.items():
        phot.update(qualities(path,links))
    for x in e:
        x.update(phot.get(x.get('ls_id'),{}))
    out = MEMBERS
    out.mkdir(parents=True,exist_ok=True)
    csv_rows(out/'members.csv',e)
    by = []
    for source,group in sorted({(a['catalog'],a['group']) for a in e}):
        for j in range(len(EDGES)-1):
            for geometry,offsets in [('legacy',OFFSETS),('desi',NEW)]:
                for k in (-2,-1,*range(len(offsets))):
                    match = [a for a in e if (a['catalog'],a['group'],a['segment'])==(source,group,j)
                             and (k==-2 or a[geometry+'_ribbon']==k)]
                    if not match:
                        continue
                    by.append(dict(catalog=source,group=group,segment=j,geometry=geometry,ribbon=k,
                        offset=offsets[k] if k>=0 else None,count=len(match),
                        raw=sum(x['match'] in ('gaia_id','unique_position') for x in match),
                        ambiguous=sum(x['match'] in ('ambiguous','gaia_conflict') for x in match),
                        psf=sum(x['psf'] for x in match),clean=sum(x['clean'] for x in match),
                        archived_qrf=sum(x['qrf_archived'] for x in match),
                        in_experiment=sum(x[geometry+'_native'] for x in match),
                        supported=sum(x[geometry+'_qrf'] for x in match)))
    csv_rows(out/'retention.csv',by)
    save_json(out/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,
        rows=len(e),outputs={p.name:sha(p) for p in out.iterdir() if p.name!='complete.json'}))


if __name__=='__main__':
    require_compute()
    run()
