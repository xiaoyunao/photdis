#!/usr/bin/env python3
"""Existing-field census and identity-safe labels; run only in an allocation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'.deps')]
import fitsio
import numpy as np
from photdis.audit import rvs_metallicity
from photdis.class_evidence import desi_class_evidence_v3
from photdis.giant_information import BANDS, ERRORS, abundance_error, linked_groups
from photdis.stage1 import galactic, require_compute, write_json

CONFIG = Path('configs/giant_information_v1.json')
COLUMNS = list(BANDS + ERRORS) + [
    'ls_id', 'source_id', 'ra', 'dec', 'ebv', 'r', 'grz_base', 'grz_clean',
    'w1_clean', 'w2_clean', 'snr_w1', 'snr_w2', 'desi_match',
    'desi_match_ambiguous', 'desi_object_id', 'desi_snr', 'desi_spectype',
    'desi_quality_reason', 'desi_teff', 'desi_teff_err', 'desi_logg',
    'desi_logg_err', 'desi_feh', 'desi_feh_err']


def availability(table, select, cfg):
    sn1, sn2 = table['snr_w1'], table['snr_w2']
    masks = {'optical': select,
             'w1_snr5': select & table['w1_clean'] & (sn1>=cfg['wise_snr_min']),
             'w1_snr20': select & table['w1_clean'] & (sn1>=cfg['wise_high_snr'])}
    masks['w1w2_snr5'] = masks['w1_snr5'] & table['w2_clean'] & (sn2>=cfg['wise_snr_min'])
    masks['w1w2_snr20'] = masks['w1_snr20'] & table['w2_clean'] & (sn2>=cfg['wise_high_snr'])
    values = {'r': table['r'], 'gr': table['g0']-table['r0'],
              'optical_error': np.maximum.reduce([table[k] for k in ERRORS[:3]])}
    bins = {key: cfg[key+'_bins'] for key in values}
    return {'total': {k: int(m.sum()) for k, m in masks.items()},
            'histograms': {key: {k: np.histogram(value[m], bins=bins[key])[0].tolist()
                                 for k, m in masks.items()} for key, value in values.items()},
            'outside_histograms': {key: {k: {
                'below': int((m & (value<bins[key][0])).sum()),
                'above': int((m & (value>bins[key][-1])).sum()),
                'nonfinite': int((m & ~np.isfinite(value)).sum())}
                for k, m in masks.items()} for key, value in values.items()}}


def add_counts(destination, addition):
    for key, value in addition.items():
        if isinstance(value, dict):
            add_counts(destination.setdefault(key, {}), value)
        elif isinstance(value, list):
            destination[key] = (np.asarray(destination.get(key, np.zeros(len(value))), dtype=int)
                                + value).tolist()
        else:
            destination[key] = destination.get(key, 0) + value


def main():
    require_compute()
    cfg = json.loads(CONFIG.read_text())
    out = Path(cfg['output_dir']); out.mkdir(parents=True, exist_ok=True)
    if (out/'labels.npz').exists():
        raise FileExistsError(out/'labels.npz')
    fields = [f for p in cfg['registries'] for f in json.loads(Path(p).read_text())['fields']]
    assert len(fields) == 14 and len({f['config'] for f in fields}) == 14
    fragments, identities, records = [], [], []
    for number, entry in enumerate(fields):
        task = json.loads(Path(entry['config']).read_text())
        path = Path(task['output_dir'])/'field_diagnostics.fits'
        stat = path.stat()
        info = {'field': task['name'], 'field_index': number, 'original_role': entry['role'],
                'config': entry['config'], 'input': str(path), 'input_size': stat.st_size,
                'input_mtime_ns': stat.st_mtime_ns, 'systems': {}, 'cutflow': {}}
        id_hash = hashlib.sha256()
        with fitsio.FITS(str(path)) as hdus:
            hdu = hdus[1]
            missing = set(COLUMNS) - set(hdu.get_colnames())
            if missing:
                raise ValueError(f'{path}: required columns absent {sorted(missing)}')
            for begin in range(0, hdu.get_nrows(), 100000):
                rows = hdu.read(rows=np.arange(begin, min(begin+100000, hdu.get_nrows())), columns=COLUMNS)
                t = {k: rows[k] for k in rows.dtype.names}
                id_hash.update(np.asarray(t['ls_id'], '<i8').tobytes())
                _, b = galactic(t['ra'], t['dec'])
                sky = (np.abs(b)>cfg['abs_b_min']) & (t['ebv']>=0) & (t['ebv']<cfg['ebv_max'])
                base = sky & t['grz_base']; clean = base & t['grz_clean']
                add_counts(info['cutflow'], {'stored_parent': len(rows), 'sky': int(sky.sum()),
                                            'base': int(base.sum()), 'clean': int(clean.sum())})
                release = t['ls_id'].astype(np.int64) >> 40
                unknown = set(map(str, np.unique(release))) - set(cfg['release_system'])
                if unknown:
                    raise ValueError(f'Unregistered native release: {unknown}')
                evidence = desi_class_evidence_v3(t)
                label_text = evidence['class_evidence'].astype('U')
                focus = (t['desi_teff']>=cfg['teff_bounds'][0]) & (t['desi_teff']<cfg['teff_bounds'][1])
                add_counts(info['cutflow'], {
                    'clean_unambiguous_desi': int((clean & t['desi_match'] & ~t['desi_match_ambiguous']).sum()),
                    'clean_K_giant_records': int((clean & focus & (label_text=='conservative_giant')).sum()),
                    'clean_K_dwarf_records': int((clean & focus & (label_text=='conservative_dwarf')).sum())})
                for rel in np.unique(release):
                    system = cfg['release_system'][str(rel)]
                    selected = clean & (release==rel)
                    stats = availability(t, selected, cfg)
                    stats['class_evidence'] = {k: int((selected & (label_text==k)).sum())
                                              for k in np.unique(label_text)}
                    add_counts(info['systems'].setdefault(system, {}), stats)
                identities.append(t['ls_id'][clean])
                # Keep uncertain/nonstellar spectral evidence too, for audit.
                keep = clean & t['desi_match'] & ~t['desi_match_ambiguous']
                selected = {k: v[keep] for k, v in t.items()}
                selected['field_index'] = np.full(keep.sum(), number, dtype=np.int16)
                selected['release'] = release[keep]
                selected.update({k: v[keep] for k, v in evidence.items()})
                fragments.append(selected)
        info['ordered_ls_ids_sha256'] = id_hash.hexdigest()
        records.append(info)
        print(json.dumps({'field': info['field'], 'cutflow': info['cutflow']}), flush=True)
    t = {k: np.concatenate([x[k] for x in fragments]) for k in fragments[0]}
    groups = linked_groups(t['ls_id'], t['source_id'], t['desi_object_id'])
    t['identity_group'] = groups
    text = t['class_evidence'].astype('U')
    t['label'] = np.where(text=='conservative_giant', 1, np.where(text=='conservative_dwarf', 0, -1))
    conflict, representatives, linked = np.zeros(len(groups), bool), [], []
    order = np.argsort(groups, kind='stable')
    blocks = np.split(order, np.flatnonzero(np.diff(groups[order]))+1)
    for positions in blocks:
        if not len(positions):
            continue
        group = groups[positions[0]]
        owner = int(t['field_index'][positions].min())
        known = np.unique(t['label'][positions][t['label'][positions]>=0])
        conflict[positions] = len(known)>1
        candidates = positions[t['field_index'][positions]==owner]
        rank = np.lexsort((t['ls_id'][candidates], -np.nan_to_num(t['desi_snr'][candidates], nan=-1.)))
        representatives.append(candidates[rank[0]])
        if len(positions)>1:
            linked.append({'group': int(group), 'n_records': len(positions),
                           'fields': np.unique(t['field_index'][positions]).tolist(),
                           'class_conflict': bool(conflict[positions[0]])})
    t['identity_conflict'] = conflict
    t['representative'] = np.zeros(len(t['label']), bool)
    representatives = np.asarray(representatives, dtype=np.int64)
    t['representative'][representatives] = True
    t['feh_corrected'], _ = rvs_metallicity(t['desi_feh'], t['desi_teff'], t['desi_logg'])
    t['feh_sigma'], t['feh_sigma_random'] = abundance_error(
        t['desi_feh_err'], t['desi_teff'], t['desi_teff_err'], cfg['feh_systematic_floor'])
    np.savez_compressed(out/'labels.npz', **t)
    write_json(out/'identity_links.json', linked)
    allids = np.concatenate(identities)
    result = {'config_sha256': hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
              'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'fields': records, 'n_spectral_records': len(groups),
              'n_spectral_identity_groups': len(representatives),
              'n_conflicting_groups': int(conflict[representatives].sum()),
              'post_identity_K_labels': {
                  'giants': int((t['representative'] & ~conflict & (t['label']==1)
                                 & (t['desi_teff']>=4200) & (t['desi_teff']<5000)).sum()),
                  'dwarfs': int((t['representative'] & ~conflict & (t['label']==0)
                                 & (t['desi_teff']>=4200) & (t['desi_teff']<5000)).sum())},
              'n_clean_photometric_records': len(allids),
              'n_unique_clean_ls_measurements': len(np.unique(allids)),
              'n_crossfield_groups': sum(len(x['fields'])>1 for x in linked),
              'identity_definition': 'Connected LS/Gaia/DESI identities; clean LS measurement count is not an independent-star count.'}
    write_json(out/'inventory.json', result)
    print(json.dumps({k:v for k,v in result.items() if k!='fields'}), flush=True)


if __name__ == '__main__':
    main()
