#!/usr/bin/env python3
"""Recover the two published v3.5 blue cohorts by exact frozen identity."""
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from pathlib import Path

import fitsio
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from photdis.gaia import SOURCE_ID_HEALPIX_FACTOR
from photdis.stage1 import legacy_ids, require_compute
from v2_1_near import ROOT, atomic_npz, csv_rows, load, save_json, sha
from v3_3_sed import verified
from v3_4_near_recover_labels import archived_labels
from v3_4_near_validate import geometry, tile_inventory
from v3_5_dr11_inventory import CONFIG
from v3_5_dr11_production import CUTS, FREEZE, PRODUCTION
from v3_5_dr11_spatial import DELIVERY
from v3_5_dr11_validate import OUT as VALIDATION, join_parts, qualification, take_rows


OUT = ROOT/'data/v3_6/blue_replay'
BANDS = ('G','R','Z','I','W1','W2')
COLS = ('ls_id','identity','source_id','TYPE','RA','DEC','EBV','parent_row','observed_r',
        'dereddened_g_r','primary','strict','band_state','band_mask','dwarf_selection_score',
        'nominal_qnear','qrf_qnear','qrf_quantiles_kpc','qrf_mu_edges','qrf_mu_mass')
NATIVE = ('RELEASE','BRICKID','OBJID','REF_CAT','REF_ID','REF_EPOCH','PMRA','PMDEC','TYPE',
          'MASKBITS','FITBITS', *[f'{key}_{b}' for b in BANDS for key in
                                    ('FLUX','FLUX_IVAR','MW_TRANSMISSION','NOBS')],
          *[f'{key}_{b}' for b in ('G','R','Z','I') for key in
            ('FRACMASKED','FRACIN','FRACFLUX','RCHISQ')],
          'WISEMASK_W1','WISEMASK_W2','FRACFLUX_W1','FRACFLUX_W2')


def blue_draw(name, qualified_ids, digest):
    dest = PRODUCTION/Path(name).stem
    rec = json.loads(verified(dest/'complete.json',digest).read_text())
    src = verified(dest/'identities.npz',rec['outputs']['identities.npz'])
    with np.load(src) as z:
        wanted = z['ls_id'][np.isin(z['ls_id'],qualified_ids)]
    if not len(wanted):
        return {}
    found = []
    for f,h in sorted(rec['outputs'].items()):
        if not f.startswith('part_'):
            continue
        with np.load(dest/f) as z:
            mask = np.isin(z['ls_id'],wanted)
            if not mask.any():
                continue
            color = z['dereddened_g_r']
            mask &= (color>=.5)&(color<.8)
            if mask.any():
                verified(dest/f,h)
                d = {k:z[k][mask] for k in COLS}
                d['sweep'] = np.full(mask.sum(),name)
                d['part'] = np.full(mask.sum(),f)
                d['part_row'] = np.flatnonzero(mask)
                found.append(d)
    return join_parts(found)


def gaia_replay(photo, frozen, pool):
    ids = photo['source_id']
    original = frozen['gaia_tiles']
    current = tile_inventory(ids)
    if any(original.get(p)!=v for p,v in current.items()):
        raise ValueError('Gaia tiles differ from the frozen validation inputs')
    pixel = ids//SOURCE_ID_HEALPIX_FACTOR
    order = np.argsort(pixel,kind='stable')
    bounds = np.r_[0,1+np.flatnonzero(np.diff(pixel[order])),len(order)]
    # Balanced contiguous tile groups share no input files across workers.
    groups = [order[bounds[g[0]]:bounds[g[-1]+1]] for g in
              np.array_split(np.arange(len(bounds)-1),48) if len(g)]
    result = list(pool.map(lambda j:qualification(take_rows(photo,j),ids[j]),groups))
    back = np.argsort(np.concatenate(groups))
    gaia = take_rows(join_parts([r[0] for r in result]),back)
    flags = take_rows(join_parts([r[1] for r in result]),back)
    if not np.concatenate([r[2] for r in result]).all():
        raise ValueError('A frozen qualified source failed repeated qualification')
    return gaia,flags


def native_payload(selected, pool):
    cfg = json.loads(CONFIG.read_text())
    raw = Path(cfg['sweep_root'])
    names = selected['sweep']
    groups = [np.flatnonzero(names==s) for s in np.unique(names)]
    def read(j):
        rows = selected['parent_row'][j]
        order = np.argsort(rows)
        with fitsio.FITS(str(raw/names[j[0]])) as f:
            available = set(f[1].get_colnames())
            keys = [k for k in NATIVE if k in available]
            t = f[1].read(columns=keys,rows=rows[order])[np.argsort(order)]
        np.testing.assert_array_equal(legacy_ids(t),selected['ls_id'][j])
        np.testing.assert_array_equal(t['REF_ID'],selected['source_id'][j])
        result = {k:t[k] if k in keys else np.full(len(j),np.nan) for k in NATIVE}
        for k in ('TYPE','REF_CAT'):
            result[k] = np.char.strip(result[k].astype('U'))
        if not (result['TYPE']=='PSF').all() or not (result['REF_CAT']=='G3').all():
            raise ValueError(f'Unverified morphology or Gaia association in {names[j[0]]}')
        return result
    return take_rows(join_parts(list(pool.map(read,groups))),np.argsort(np.concatenate(groups)))


def epoch_offset(photo,native,gaia):
    epoch = native['REF_EPOCH']
    ok = (epoch>1900)&(epoch<2100)&np.isfinite(gaia['pmra'])&np.isfinite(gaia['pmdec'])
    out = np.full(len(epoch),np.nan)
    src = SkyCoord(ra=gaia['ra'][ok]*u.deg,dec=gaia['dec'][ok]*u.deg,
                   pm_ra_cosdec=gaia['pmra'][ok]*u.mas/u.yr,pm_dec=gaia['pmdec'][ok]*u.mas/u.yr,
                   obstime=Time(gaia['ref_epoch'][ok],format='jyear'))
    moved = src.apply_space_motion(new_obstime=Time(epoch[ok],format='jyear'))
    out[ok] = moved.separation(SkyCoord(photo['RA'][ok]*u.deg,photo['DEC'][ok]*u.deg)).arcsec
    return out


def photometric_diagnostics(t):
    mu = 10+5*np.log10(t['reference_kpc'])
    mu_qrf = 10+5*np.log10(t['qrf_quantiles_kpc'][:,2])
    for b in BANDS:
        flux, transmission = t[f'FLUX_{b}'],t[f'MW_TRANSMISSION_{b}']
        with np.errstate(divide='ignore',invalid='ignore'):
            mag = np.where((flux>0)&(transmission>0),22.5-2.5*np.log10(flux/transmission),np.nan)
        t[f'{b}_dereddened'] = mag
        t[f'M_{b}_reference'] = mag-mu
        t[f'M_{b}_qrf'] = mag-mu_qrf
    t['M_Gaia_G_observed'] = t['gaia_phot_g_mean_mag']-mu
    pm = np.hypot(t['gaia_pmra'],t['gaia_pmdec'])
    with np.errstate(divide='ignore',invalid='ignore'):
        t['gaia_reduced_proper_motion_G'] = t['gaia_phot_g_mean_mag']+5*np.log10(pm/1000)+5
    t['low_luminosity_locus'] = ((t['gaia_bp_rp']<1.5)&(t['M_Gaia_G_observed']>10)&
                                (t['M_G_reference']>10))


def main():
    require_compute()
    paths = [Path(__file__),CUTS,CONFIG,FREEZE,ROOT/'manifests/v2_0_revision/gaia_anomalies.csv']
    hashes = {str(p.relative_to(ROOT)):sha(p) for p in paths}
    val_path = VALIDATION/'complete.json'
    val = json.loads(val_path.read_text())
    frozen = json.loads(verified(VALIDATION/'freeze.json',val['freeze_sha256']).read_text())
    roster = load(verified(VALIDATION/'gaia_qualification_roster.npz',
                           val['outputs']['gaia_qualification_roster.npz']))
    unique = roster['qualified']&(roster['native_source_id']>0)
    qual = np.sort(roster['ls_id'][unique])
    delivery = json.loads(verified(DELIVERY/'complete.json',
                          frozen['hashes']['data/v3_5/delivery/complete.json']).read_text())
    with verified(DELIVERY/'sweeps.csv',delivery['outputs']['sweeps.csv']).open() as f:
        digests = {r['sweep']:r['receipt_sha256'] for r in csv.DictReader(f)}
    plan = json.loads(verified(FREEZE,delivery['production_freeze_sha256']).read_text())
    with ThreadPoolExecutor(max_workers=min(12,int(os.getenv('SLURM_CPUS_PER_TASK','1')))) as pool:
        parts = pool.map(lambda n:blue_draw(n,qual,digests[n]),chain.from_iterable(plan['groups']))
        photo = join_parts([p for p in parts if p])
        pos = np.searchsorted(qual,photo['ls_id'])
        np.testing.assert_array_equal(qual[pos],photo['ls_id'])
        roster_idx = np.argsort(roster['ls_id'][unique])[pos]
        np.testing.assert_array_equal(roster['native_source_id'][unique][roster_idx],photo['source_id'])
        assert (photo['TYPE']=='PSF').all() and photo['primary'].all()
        assert len(np.unique(photo['ls_id']))==len(pos)==87257
        print(f'Frozen qualified blue objects: {len(pos)}',flush=True)
        gaia,flags = gaia_replay(photo,frozen,pool)
        d,lo,hi = geometry(flags['p_corr'],gaia['parallax_error'],np.ones(len(pos),bool),
                            json.loads(CUTS.read_text()))
        r = photo['observed_r']
        nearby = np.isfinite(d)&(d<2)&(r>=18)
        distant = (d>=2)&(d<5)
        assert (nearby.sum(),(nearby&photo['strict']).sum(),distant.sum())==(116,17,189)
        assert ((distant&(r>=14)&(r<16)).sum(),(distant&(r>=16)&(r<18)).sum())==(148,41)
        use = nearby|distant
        t,gaia,flags = [take_rows(a,use) for a in (photo,gaia,flags)]
        t.update(sample=np.where(nearby[use],'faint_blue_near','blue_2_to_5'),
                 reference_kpc=d[use],reference_lo_kpc=lo[use],reference_hi_kpc=hi[use])
        raw = native_payload(t,pool)
    with paths[-1].open() as f:
        old = list(csv.DictReader(f))
    t['old_anomaly_legacy_match'] = np.isin(t['ls_id'],[int(r['ls_id']) for r in old])
    t['old_anomaly_gaia_match'] = np.isin(t['source_id'],[int(r['dr3_source_id']) for r in old])
    t['gaia_unique'] = t['source_id']>0
    label,_,conflict,held = archived_labels(t,hashes)
    t.update(archived_class=label,archived_class_key_conflict=conflict,archived_heldout=held,
             gaia_epoch_offset_arcsec=epoch_offset(t,raw,gaia))
    t.update({f'gaia_{k}':v for k,v in {**gaia,**flags}.items()})
    t.update(raw)
    photometric_diagnostics(t)
    t['distance_ratio'] = t['qrf_quantiles_kpc'][:,2]/t['reference_kpc']
    OUT.mkdir(parents=True,exist_ok=True)
    p = OUT/'blue_objects.npz'
    atomic_npz(p,**t)
    cols = {k:v for k,v in t.items() if v.ndim==1}
    cols.update(qrf_median_kpc=t['qrf_quantiles_kpc'][:,2],
                w1_state=t['band_state'][:,4],w2_state=t['band_state'][:,5])
    table = OUT/'blue_objects.csv'
    csv_rows(table,({k:v[i].item() for k,v in cols.items()} for i in range(len(t['ls_id']))))
    hashes[str(val_path.relative_to(ROOT))] = sha(val_path)
    save_json(OUT/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=hashes,
              counts={s:int((t['sample']==s).sum()) for s in np.unique(t['sample'])},
              old_gaia_overlap=int(t['old_anomaly_gaia_match'].sum()),
              outputs={f.name:sha(f) for f in (table,p)}))


if __name__ == '__main__':
    main()
