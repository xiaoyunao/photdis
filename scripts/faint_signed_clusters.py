#!/usr/bin/env python3
"""Bounded independent-scale inventory before opening any candidate distances."""
import argparse
import csv
import json
from pathlib import Path
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
C = json.loads((ROOT/'configs/faint_signed_20261120.json').read_text())
M = ROOT/'manifests'
NAMESPACE = 'faint_signed_20261120'


def public_rows():
    path = M/'sky_sequence_20261119/coverage/sky_coverage.npz'
    z = np.load(path)
    l = -np.rad2deg((z['l'][:-1]+z['l'][1:])/2)
    b = np.rad2deg((z['b'][:-1]+z['b'][1:])/2)
    mask = z['mask']
    depth = z['values'][2]
    ebv = z['values'][3]
    old_path = M/'v3_1/external_inventory_finite_sky/cluster_inventory.csv'
    with old_path.open() as f:
        old = {r['name']:r for r in csv.DictReader(f)}
    g = json.loads((M/'giant_parameter_inventory_v1.json').read_text())['systems']
    catalog = {}
    for name,meta in C['cluster_candidates'].items():
        lon,lat = meta['galactic']
        ra_sep = ((l-lon+180)%360-180)*np.cos(np.deg2rad(lat))
        nearby = np.hypot(ra_sep[None,:],b[:,None]-lat)<0.5
        seen = nearby&mask
        k = name.replace('_','')[:7]
        old_m = old.get(k,{})
        catalog[name] = dict(system=name,longitude=lon,latitude=lat,
              comparison_kpc=meta['scale_kpc'],rrl_w1_modulus=meta['rrl_mu'],
              rrl_w1_stat_sigma=meta['rrl_sigma_mag'],
              rrl_scale_source=meta['independent_scale'],
              combined_area_cells=int(seen.sum()),aperture_cells=int(nearby.sum()),
              median_shallower_gr_depth=float(np.nanmedian(depth[seen])) if seen.any() else '',
              median_sfd=float(np.nanmedian(ebv[seen])) if seen.any() else '',
              old_member_inventory_rows=old_m.get('rows',''),
              old_admitted_science=old_m.get('admitted_science',''),
              prior_native_k_giant_members=g.get(name,{}).get('unique_K_members',''))
    return catalog


def public(dest):
    catalog = public_rows()
    dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=next(iter(catalog.values())),lineterminator="\n")
        writer.writeheader()
        writer.writerows(catalog.values())


def native():
    import environment_pilot as pilot
    import environment_results as scores
    import environment_dust as dust
    from photdis.stage1 import require_compute
    from v2_1_near import atomic_npz, csv_rows, save_json, sha
    from v3_4_near_forecast import prepare
    from photdis.v3_3 import frozen_qrf

    require_compute()
    target = ROOT/'data'/NAMESPACE/'results'
    target.mkdir(parents=True,exist_ok=True)
    path = ROOT/'data/reference/vasiliev_baumgardt_2021_members.zip'
    models,_,_ = prepare()
    model = models['qrf']
    output = []
    eligible = {}
    with zipfile.ZipFile(path) as archive:
        for name,meta in C['cluster_candidates'].items():
            pilot.C = dict(pilot.C,field_radius_deg=.5)
            parts,led,_ = pilot.field_rows([name,*meta['galactic']])
            prefix = '_'.join(name.split('_')[:2])
            files = [f for f in archive.namelist() if f.startswith('catalogues/') and
                     '_'.join(Path(f).stem.split('_')[:2]) == prefix]
            if len(files)!=1:
                raise ValueError(f'Expected one membership table for {name}: {files}')
            lines = archive.read(files[0]).decode().splitlines()
            records = [line.split() for line in lines if line.strip() and not line.startswith('#')]
            ids = np.array([int(r[0]) for r in records if float(r[-1])>.9],dtype='i8')
            row = dict(system=name,members_p90=len(ids),intersecting_sweeps=len(led),
                       raw_rows=sum(r['rows'] for r in led),missing_type=sum(r['missing_type'] for r in led),
                       rejected_non_psf=sum(r['non_psf'] for r in led))
            if not parts:
                row.update(photometric_member_psf=0,eligible=0,faint_lower_sequence=0)
            else:
                t = pilot.join(parts)
                assert (t['TYPE']=='PSF').all()
                unique = pilot.unique_join(t['source_id'],t['source_id'])>=0
                matched = t['source_id']>0
                matched &= unique&np.isin(t['source_id'],ids)
                t = pilot.subset(t,matched)
                hist, audit = pilot.historical(t)
                fit = np.isin(t['identity'],model.training_identity) | np.isin(t['ls_id'],-model.training_identity)
                mass_support = pilot.support(model,np.pad(t['mag_grz'],((0,0),(0,2))))
                clean = mass_support&~(hist|fit)
                mr = t['mag_grz'][:,1]-meta['rrl_mu']
                gr = t['mag_grz'][:,0]-t['mag_grz'][:,1]
                lower = (mr>=5)&(gr>=.40)
                faint = (t['r']>=20)&(t['r']<22)
                row.update(photometric_member_psf=len(hist),history=int(hist.sum()),
                       fit=int(fit.sum()),qrf_support=int(mass_support.sum()),eligible=int(clean.sum()),
                       faint_lower_sequence=int((clean&lower&faint).sum()),
                       bright=int((clean&(t['r']<20)).sum()),faint=int((clean&faint).sum()),
                       median_ebv=float(np.median(t['EBV'])) if len(hist) else '')
                if row['eligible']>=C['cluster_minimum_psf'] and row['faint_lower_sequence']>=C['cluster_minimum_faint_lower_sequence']:
                    eligible[name] = pilot.subset(dict(t,lower_sequence_proxy=lower),clean),meta
            output.append(row)
    csv_rows(target/'cluster_native_inventory.csv',output)
    seal = target/'cluster_freeze.json'
    save_json(seal,dict(config=sha(ROOT/'configs/faint_signed_20261120.json'),
                    member_archive=sha(path), inventory=sha(target/'cluster_native_inventory.csv'),
                    admitted={name:t['ls_id'].tolist() for name,(t,_) in eligible.items()}))
    forecast = []
    for name,(t,meta) in eligible.items():
        f = frozen_qrf(model,t['mag_grz'],t['error_grz'],t['identity'])
        assert f['supported'].all()
        reference = np.full(len(t['ls_id']),meta['rrl_mu'])
        sigma = np.full_like(reference,np.hypot(meta['rrl_sigma_mag'],meta['zeropoint_stress_mag']))
        p = scores.measure(f['mu_edges'],f['mass'],reference,sigma,np.ones(len(reference)))
        atomic_npz(target.parent/(name+'_fixed_pdfs.npz'),ls_id=t['ls_id'],TYPE=t['TYPE'],
                   edges=f['mu_edges'],mass=f['mass'])
        forecast.extend(dict(system=name,ls_id=int(i),TYPE=ty,r=r,lower_sequence_proxy=lower,delta_mu=b,crps=c)
                        for i,ty,r,lower,b,c in zip(t['ls_id'],t['TYPE'],t['r'],
                                                  t['lower_sequence_proxy'],p['bias'],p['crps']))
    csv_rows(target/'cluster_fixed_scores.csv',forecast)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--public',type=Path)
    arg = parser.parse_args()
    if arg.public:
        public(arg.public)
    else:
        native()
