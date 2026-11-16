#!/usr/bin/env python3
"""Test luminosity branches in grz and optical-error matched neighborhoods."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from photdis.stage1 import require_compute
from v2_1_near import ROOT, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_6_blue_replay import OUT as BLUE
from v3_6_distance_evidence import OUT as EVIDENCE
from v3_6_reference_photometry import OUT as PHOTO
from v3_6_vac_distances import read_columns

OUT = ROOT/'data/v3_7/blue_neighborhood'
# Observed r, both grz colors and logarithms of both propagated color errors.
TOL = np.array([.25,.05,.05,np.log(1.5),np.log(1.5)])
BANDS = {'i':(1,2),'W1':(3,4),'W2':(4,5)}


def coordinates(mag,error,r):
    gr = mag[:,0]-mag[:,1]
    rz = mag[:,1]-mag[:,3]
    sgr = np.hypot(error[:,0],error[:,1])
    srz = np.hypot(error[:,1],error[:,3])
    with np.errstate(divide='ignore',invalid='ignore'):
        return np.column_stack((r,gr,rz,np.log(sgr),np.log(srz)))/TOL


def pdf_window(edges,mass,center,width=.5):
    overlap = np.clip(np.minimum(edges[1:],center+width)-np.maximum(edges[:-1],center-width),0.,None)
    return float(np.sum(mass*overlap/np.diff(edges)))


def linked(table,photo):
    order = np.argsort(photo['ls_id'])
    identity = photo['ls_id'][order]
    ix = np.searchsorted(identity,table['ls_id'])
    if (ix>=len(identity)).any():
        raise ValueError('Photometry association is incomplete')
    j = order[ix]
    np.testing.assert_array_equal(photo['ls_id'][j],table['ls_id'])
    if not (photo['TYPE'][j]=='PSF').all():
        raise ValueError('Native PSF required after identity join')
    return photo['magnitude'][j],photo['magnitude_error'][j]


def compare(case,neighbors,photo):
    cm,ce = linked(case,photo)
    nm,ne = linked(neighbors,photo)
    cx,nx = coordinates(cm,ce,case['observed_r']),coordinates(nm,ne,neighbors['observed_r'])
    finite = np.isfinite(nx).all(axis=1)
    tree = cKDTree(nx[finite])
    parent = np.flatnonzero(finite)
    cmu = 10+5*np.log10(case['reference_kpc'])
    nmu = 10+5*np.log10(neighbors['reference_kpc'])
    absolute = nm[:,0]-nmu
    for i,coord in enumerate(cx):
        js = parent[tree.query_ball_point(coord,1.,p=np.inf)] if np.isfinite(coord).all() else parent[:0]
        edges,mass = case['qrf_mu_edges'][i],case['qrf_mu_mass'][i]
        truth = pdf_window(edges,mass,cmu[i])
        row = dict(cohort=case['sample'][i],ls_id=case['ls_id'][i],TYPE=case['TYPE'][i],
                   reference_g_absolute=cm[i,0]-cmu[i],qrf_reference_window_probability=truth,
                   controls=len(js))
        for name,(a,b) in BANDS.items():
            c = cm[i,a]-cm[i,b]
            color = nm[js,a]-nm[js,b]
            available = np.isfinite(color)&np.isfinite(c)
            offer = js[available]
            restricted = js[available&(abs(color-c)<=.2)]
            for design,use in (('available_optical_neighbors',offer),('additional_color_within_0.2',restricted)):
                v = absolute[use]
                yield dict(**row,band=name,design=design,n=len(use),
                    within_1mag_fraction=(abs(v-row['reference_g_absolute'])<=1.).mean() if len(v) else None,
                    separated_by_2mag_fraction=(abs(v-row['reference_g_absolute'])>=2.).mean() if len(v) else None,
                    M_g_p10=np.quantile(v,.1) if len(v) else None,M_g_p50=np.median(v) if len(v) else None,
                    M_g_p90=np.quantile(v,.9) if len(v) else None)
        v = absolute[js]
        # Convert each control's luminosity to a branch for this case.
        alternative = cm[i,0]-v
        yield dict(**row,band='grz',design='all_optical_neighbors',n=len(js),
                   within_1mag_fraction=(abs(v-row['reference_g_absolute'])<=1.).mean() if len(v) else None,
                   separated_by_2mag_fraction=(abs(v-row['reference_g_absolute'])>=2.).mean() if len(v) else None,
                   M_g_p10=np.quantile(v,.1) if len(v) else None,M_g_p50=np.median(v) if len(v) else None,
                   M_g_p90=np.quantile(v,.9) if len(v) else None,
                   mean_qrf_other_branch_probability=np.mean([
                       pdf_window(edges,mass,m) for m in alternative[abs(alternative-cmu[i])>=2.]])
                       if (abs(alternative-cmu[i])>=2.).any() else None)


def main():
    require_compute()
    marker = OUT/'complete.json'
    if marker.exists():
        raise FileExistsError(marker)
    inputs = [x/'complete.json' for x in (BLUE,EVIDENCE,PHOTO)]
    b,e,p = (json.loads(x.read_text()) for x in inputs)
    case = read_columns(verified(BLUE/'blue_objects.npz',b['outputs']['blue_objects.npz']),
        ('ls_id','TYPE','observed_r','reference_kpc','qrf_mu_edges','qrf_mu_mass','sample'))
    ref = read_columns(verified(EVIDENCE/'Gaia.npz',e['outputs']['Gaia.npz']),
        ('ls_id','TYPE','observed_r','reference_kpc','high_quality','qrf_training'))
    photo = read_columns(verified(PHOTO/'photometry.npz',p['outputs']['photometry.npz']),
                        ('ls_id','TYPE','magnitude','magnitude_error'))
    excluded = {label:{'non_psf':int(((a['TYPE']!='')&(a['TYPE']!='PSF')).sum()),
                        'unverified_type':int((a['TYPE']=='').sum())} for label,a in (('case',case),('control',ref))}
    case = {k:v[case['TYPE']=='PSF'] for k,v in case.items()}
    use = (ref['TYPE']=='PSF')&ref['high_quality']&~ref['qrf_training']&~np.isin(ref['ls_id'],case['ls_id'])
    ref = {k:v[use] for k,v in ref.items()}
    OUT.mkdir(parents=True,exist_ok=True)
    target = OUT/'neighborhoods.csv'
    csv_rows(target,compare(case,ref,photo))
    inputs.append(Path(__file__).resolve())
    save_json(marker,dict(inputs={str(x.relative_to(ROOT)):sha(x) for x in inputs},
        cases=len(case['ls_id']),native_parent=len(ref['ls_id']),excluded_morphology=excluded,
        outputs={target.name:sha(target)}))


if __name__=='__main__':
    main()
