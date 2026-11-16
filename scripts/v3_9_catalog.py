#!/usr/bin/env python3
"""Build resumable photometry and PDF shards from the native DR11 parent sweeps."""
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import json
import os
from pathlib import Path

import joblib
import numpy as np

from photdis.baseline_v1 import support
from photdis.stage1 import galactic, require_compute
from photdis.v3.controls import apply_temperature, optical_magnitudes, probe_features
from photdis.v3.data import stable_rank
from photdis.v3_3 import frozen_qrf
from v2_1_near import ROOT, atomic_npz, load, save_json, sha
from v3_0_native import native_payload, read_native, science_domain, subset
from v3_3_sed import verified
from v3_4_near_forecast import prepare, qrf_bins
from v3_5_dr11_inventory import describe
from v3_5_dr11_pilot import in_tile

CONFIG = ROOT/'configs/v3_9_catalog.json'
OUT = ROOT/'data/v3_9/expansion'
LEDGER = ('outside_tile', 'other_release', 'missing_type', 'non_psf',
          'outside_science_domain', 'optical_rejected', 'noise_control_parent', 'precise_parent')
GRZ = (0, 1, 3)
PHOTO = ('ls_id', 'TYPE', 'source_id', 'identity', 'RA', 'DEC', 'EBV', 'MASKBITS',
         'band_mask', 'band_state', 'flux', 'flux_error', 'transmission', 'photometric_system',
         'r', 'g0', 'r0', 'z0', 'psf')
DENSITY_FIELDS = tuple(f'{field}_{b}' for field in ('FRACFLUX', 'PSFDEPTH', 'PSFSIZE') for b in 'GRZ')
REQUIRED = ('RELEASE', 'BRICKID', 'OBJID', 'TYPE', 'RA', 'DEC', 'EBV', 'REF_ID', 'REF_CAT', 'MASKBITS')
_MODEL = None


def settings():
    return json.loads(CONFIG.read_text())


def confirm(marker, binding):
    if not marker.exists():
        return None
    result = json.loads(marker.read_text())
    if result['binding'] != binding:
        raise ValueError(f'Inputs changed: {marker}')
    for name, digest in result['outputs'].items():
        verified(marker.parent/name, digest)
    return result


def inventory(side, groups):
    cfg = settings()[side]
    root = Path(cfg['sweep_root'])
    with ThreadPoolExecutor(max_workers=int(os.getenv('SLURM_CPUS_PER_TASK','4'))) as pool:
        files = list(pool.map(partial(describe, sweep=True), sorted(root.glob('sweep-*.fits'))))
    if not files:
        raise ValueError(f'Empty native sweep inventory: {root}')
    roster = [dict(s, missing_fields=sorted(set(REQUIRED)-set(s.get('columns',()))))
              for s in files]
    lanes, sizes = [[] for _ in range(groups)], np.zeros(groups, dtype='i8')
    for source in sorted(roster, key=lambda s:-s['size']):
        i = sizes.argmin()
        lanes[i].append(source)
        sizes[i] += source['size']
    binding = dict(config=sha(CONFIG), script=sha(Path(__file__)))
    path = OUT/side/'inventory.json'
    if path.exists():
        saved = json.loads(path.read_text())
        if saved['groups'] != lanes or saved['binding'] != binding:
            raise ValueError('Existing census inventory has different inputs')
        return
    save_json(path, dict(binding=binding, groups=lanes, group_bytes=sizes,
                        job=os.getenv('SLURM_JOB_ID'), sweeps=len(files)))
    print(side, len(files), 'native sweep headers', flush=True)


def thin(photo, take, cap, salt):
    """Deterministic sampling within a sweep by observed magnitude and color."""
    rows = np.flatnonzero(take)
    r = np.digitize(photo['r'][rows], [14, 16, 18, 19, 20, 21, 22])
    c = np.digitize(photo['g0'][rows]-photo['r0'][rows], [.2, .5, .8, 1.1, 1.4, 2.])
    cells = r*7+c
    rank = stable_rank(photo['ls_id'][rows], salt)
    chosen, weights = [], []
    for cell in np.unique(cells):
        j = np.flatnonzero(cells==cell)
        k = j[np.argsort(rank[j], kind='stable')[:cap]]
        chosen.append(rows[k])
        weights.append(np.repeat(len(j)/len(k), len(k)))
    if not chosen:
        return np.empty(0, int), np.empty(0)
    return np.concatenate(chosen), np.concatenate(weights)


def scan_chunk(raw, bounds, cfg, first):
    status = np.full(len(raw), len(LEDGER)-1, dtype='i1')
    kind = np.char.strip(raw['TYPE'].astype('U4'))
    tests = (in_tile(raw['RA'], raw['DEC'], bounds), raw['RELEASE']==cfg['release'],
             kind!='', kind=='PSF')
    for k, valid in enumerate(tests):
        status[(status==len(LEDGER)-1)&~valid] = k
    i = np.flatnonzero(status==len(LEDGER)-1)
    _, b = galactic(raw['RA'][i], raw['DEC'][i])
    with np.errstate(divide='ignore', invalid='ignore'):
        r = (22.5-2.5*np.log10(raw['FLUX_R'][i].astype('f8'))).astype('f4')
    status[i] = 4
    for region, domain in cfg['domains'].items():
        inside = science_domain(r, b, raw['EBV'][i], **domain)
        j = i[inside]
        if not len(j):
            continue
        t = native_payload(raw[j], cfg['system'], domain=domain)
        clean = t['photometric_entry'] & t['band_mask'][:,GRZ].all(axis=1)
        mag, err = np.full((len(j),3),np.nan), np.full((len(j),3),np.nan)
        mag[clean], err[clean] = optical_magnitudes(subset(t,clean))
        clean &= np.isfinite(mag).all(axis=1)&np.isfinite(err).all(axis=1)
        precise = clean&(err < cfg['maximum_grz_error_mag']).all(axis=1)
        if np.any(status[j]!=4) or not np.all(t['TYPE']=='PSF'):
            raise ValueError('Nonexclusive science domain or non-PSF admission')
        status[j] = np.where(precise, 7, np.where(clean, 6, 5))
        p = {k:t[k] for k in PHOTO}
        north_owner = (t['DEC']>=32.375)&(b[inside]>0)
        p.update(parent_row=first+j, mag_grz=mag, error_grz=err,
                 observed_r=t['r'], dereddened_g_r=t['g0']-t['r0'],
                 north_primary_region=north_owner, galactic_latitude=b[inside],
                 row_state=status[j], region=np.full(len(j),region,dtype='U11'))
        for key in DENSITY_FIELDS:
            p[key] = raw[key][j] if key in raw.dtype.names else np.full(len(j), np.nan, 'f4')
        yield p, status
    yield {}, status


def scan_one(source, *, side, cfg, binding):
    path = Path(cfg['sweep_root'])/source['filename']
    root = OUT/side/'photo'/path.stem
    marker = root/'complete.json'
    bound = dict(**binding, size=source['size'], mtime=source['mtime_ns'])
    if confirm(marker, bound):
        return
    st = path.stat()
    if (st.st_size, st.st_mtime_ns)!=(source['size'], source['mtime_ns']):
        raise ValueError(f'Raw sweep changed: {path}')
    if source.get('error') or source['missing_fields']:
        save_json(marker, dict(binding=bound, outputs={}, rows=source.get('n_rows'),
                              error=source.get('error'), missing_fields=source['missing_fields']))
        return
    raw = read_native(path)
    root.mkdir(parents=True, exist_ok=True)
    ledger = np.empty(len(raw), dtype='i1')
    outputs = {}
    for i in range(0, len(raw), 150_000):
        for photo, status in scan_chunk(raw[i:i+150_000], source['bounds_deg'], cfg, i):
            if not photo:
                continue
            name = f"{i:09d}_{photo['region'][0]}.npz"
            atomic_npz(root/name, **photo)
            outputs[name] = sha(root/name)
        ledger[i:i+len(status)] = status
    if len(raw)!=source['n_rows']:
        raise ValueError(f'Incorrect row census in {path}')
    atomic_npz(root/'parent_state.npz', state=ledger)
    outputs['parent_state.npz'] = sha(root/'parent_state.npz')
    counts = dict(zip(LEDGER, np.bincount(ledger, minlength=len(LEDGER))))
    save_json(marker, dict(binding=bound, outputs=outputs, rows=len(raw), counts=counts,
                           job=os.getenv('SLURM_JOB_ID')))
    print(side, path.name, {k:counts[k] for k in LEDGER[5:]}, flush=True)


def from_north():
    path = OUT/'north/native_model/model.joblib'
    record = json.loads((path.parent/'complete.json').read_text())
    return joblib.load(verified(path, record['outputs'][path.name])), sha(path)


def predict_one(source, *, side, cfg, binding):
    models, probe, model_binding = _MODEL
    root = OUT/side/'photo'/Path(source['filename']).stem
    parent = root/'complete.json'
    dest = OUT/side/'predict'/root.name
    receipt = dest/'complete.json'
    bound = dict(**binding, photo=sha(parent), models=model_binding)
    if confirm(receipt, bound):
        return
    dest.mkdir(parents=True, exist_ok=True)
    photos = json.loads(parent.read_text())['outputs']
    outputs, totals = {}, np.zeros(5, dtype='i8')
    for name, digest in photos.items():
        if name=='parent_state.npz':
            continue
        p = load(verified(root/name,digest))
        if not (p['TYPE']=='PSF').all() or len(np.unique(p['ls_id']))!=len(p['ls_id']):
            raise ValueError(f'Identity check failed in {root/name}')
        precise = p['row_state']==7
        low_rows, weight = thin(p, p['row_state']==6, cfg['noise_control_per_stratum_per_sweep'],3901)
        wanted = precise.copy()
        wanted[low_rows] = True
        good = np.flatnonzero(wanted)
        state = np.full(len(precise), 0, 'u1')
        state[p['row_state']==6] = 1
        state[wanted] = 2
        archive = []
        model = models['qrf']
        offset = models.get('mu_offset',0.)
        inside = good
        if len(good):
            inside = good[support(model,np.pad(p['mag_grz'][good],((0,0),(0,2))))]
        for start in range(0, len(inside), cfg['batch']):
            j = inside[start:start+cfg['batch']]
            q = frozen_qrf(model, p['mag_grz'][j], p['error_grz'][j], p['identity'][j])
            j = j[q['supported']]
            if not len(j):
                continue
            state[j] = 3
            k = q['supported']
            edges = q['mu_edges'][k]+offset
            mass = q['mass'][k]
            np.testing.assert_allclose(mass.sum(axis=1),1,atol=1e-12,rtol=0)
            bins = qrf_bins(edges, mass)
            np.testing.assert_allclose(bins.sum(axis=1),1,atol=1e-12,rtol=0)
            score = np.full(len(j),np.nan)
            if probe is not None:
                score = apply_temperature(probe['model'].predict_proba(probe_features(subset(p,j),'grz')),
                                          probe['temperature'])[:,0]
            elif models.get('classifier') is not None:
                classifier = models['classifier']
                mag = p['mag_grz'][j]
                features = np.column_stack((mag[:,0]-mag[:,1], mag[:,1]-mag[:,2], mag[:,1]))
                if 0 in classifier.classes_:
                    score = classifier.predict_proba(features)[:,np.flatnonzero(classifier.classes_==0)[0]]
            archive.append(dict(photo_row=j, ls_id=p['ls_id'][j], TYPE=p['TYPE'][j],
                qrf_quantiles_kpc=q['distance'][k]*10**(offset/5), qrf_mu_edges=edges,
                qrf_mu_mass=mass, qrf_bins=bins, dwarf_score=score))
        fname = 'state_'+name
        atomic_npz(dest/fname, ls_id=p['ls_id'], TYPE=p['TYPE'], state=state,
                   noise_control_photo_row=low_rows, noise_control_weight=weight)
        outputs[fname] = sha(dest/fname)
        if archive:
            atomic_npz(dest/name, **{key:np.concatenate([a[key] for a in archive]) for key in archive[0]})
            outputs[name] = sha(dest/name)
        totals += (len(state), precise.sum(), len(low_rows), np.sum((state==3)&precise),np.sum((state==3)&~precise))
    save_json(receipt, dict(binding=bound, outputs=outputs, job=os.getenv('SLURM_JOB_ID'),
                           counts=dict(zip(('parent','precise','noise_control','precise_pdf','control_pdf'),totals))))
    print(side, root.name, totals.tolist(), flush=True)


def run(stage, side, shard, smoke):
    global _MODEL
    cfg = settings()
    description = OUT/side/'inventory.json'
    inventory = json.loads(description.read_text())
    sources = inventory['groups'][shard]
    binding = dict(inventory=sha(description), script=sha(Path(__file__)),
                   payload=sha(ROOT/'scripts/v3_0_native.py'))
    if stage=='predict':
        if side=='north':
            model,digest = from_north()
            _MODEL = model,None,{'north':digest}
        else:
            model,probe,paths = prepare()
            _MODEL = model,probe,{p.name:sha(p) for p in paths}
    worker = partial(scan_one if stage=='scan' else predict_one, side=side,
                     cfg={**cfg,**cfg[side]}, binding=binding)
    if smoke:
        worker(sources[0])
        return
    workers = int(os.getenv('SLURM_CPUS_PER_TASK','4'))
    if stage=='scan':
        workers = min(workers,8)
    if not sources:
        return
    with ProcessPoolExecutor(max_workers=min(workers,len(sources))) as pool:
        list(pool.map(worker,sources))


def main():
    require_compute()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('inventory','scan','predict'))
    parser.add_argument('side',choices=('south','north'))
    parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--groups',type=int,default=48)
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    if args.stage=='inventory':
        inventory(args.side,args.groups)
    else:
        run(args.stage,args.side,args.shard,args.smoke)


if __name__=='__main__':
    main()
