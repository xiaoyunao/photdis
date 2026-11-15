#!/usr/bin/env python3
"""Compare the frozen blue anomalies with brightness- and color-matched Gaia stars."""
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from photdis.stage1 import require_compute
from v2_1_near import ROOT, atomic_npz, csv_rows, load, save_json, sha
from v3_3_sed import verified
from v3_5_dr11_spatial import DELIVERY
from v3_5_dr11_validate import join_parts
from v3_6_blue_replay import OUT as BLUE, native_payload
from v3_6_gaia_census import OUT as GAIA
from v3_6_reference_index import OUT as INDEX, workers
from v3_6_vac_distances import read_columns
from v3_6_vac_flow import gather_forecasts


OUT = ROOT/'data/v3_6/blue_matched_controls'
CALIPER = np.array([.25,.05])
FIELDS = ('ls_id','source_id','catalog_row','TYPE','observed_r','dereddened_g_r','qrf_quantiles_kpc',
          'reference_kpc','reference_lo_kpc','reference_hi_kpc','gaia_ruwe','gaia_bp_rp',
          'gaia_phot_g_mean_mag','gaia_pmra','gaia_pmdec','gaia_state','strict')
NATIVE_DIAGNOSTICS = ('RCHISQ_G','RCHISQ_R','RCHISQ_Z','FRACFLUX_G','FRACFLUX_R','FRACFLUX_Z',
                      'FRACMASKED_R','FRACIN_R','FLUX_IVAR_R','MASKBITS','FITBITS')
PAIR_FIELDS = ('ls_id','source_id','observed_r','dereddened_g_r','reference_kpc','distance_ratio','strict',
               'gaia_ruwe','gaia_bp_rp','absolute_gaia_g','proper_motion_mas_per_year',*NATIVE_DIAGNOSTICS)


def gaia_pool(case, record, pool):
    low,high = case['observed_r'].min()-.25,case['observed_r'].max()+.25
    clo,chi = case['dereddened_g_r'].min()-.05,case['dereddened_g_r'].max()+.05

    def read(item):
        name,digest = item
        marker = verified(GAIA/name,digest)
        child = json.loads(marker.read_text())
        path = marker.with_suffix('.npz')
        with np.load(verified(path,child['outputs'][path.name]),allow_pickle=False) as f:
            r,c = f['observed_r'],f['dereddened_g_r']
            mask = ((f['gaia_state']==6)&f['primary']&(r>=low)&(r<=high)&(c>=clo)&(c<=chi)
                    &~np.isin(f['source_id'],case['source_id'])&~np.isin(f['ls_id'],case['ls_id']))
            return {key:f[key][mask] for key in FIELDS} if mask.any() else {}

    a = join_parts([p for p in pool.map(read,record['child_receipts'].items()) if p])
    if not a:
        raise ValueError('No eligible Gaia controls in the case photometry range')
    if not (a['TYPE']=='PSF').all():
        raise ValueError('Matched controls require verified native PSF')
    sid = a['source_id']
    _,back,counts = np.unique(sid,return_inverse=True,return_counts=True)
    use = (sid>0)&(counts[back]==1)
    return {key:value[use] for key,value in a.items()}


def coordinates(a):
    return np.column_stack((a['observed_r'],a['dereddened_g_r']))/CALIPER


def matches(case,pool):
    points = coordinates(pool)
    query = coordinates(case)
    candidates = cKDTree(points).query_ball_point(query,1,p=np.inf)
    with np.errstate(divide='ignore',invalid='ignore'):
        delta = 5*np.log10(pool['qrf_quantiles_kpc'][:,2]/pool['reference_kpc'])
    distance = pool['reference_kpc']
    for j,(point,indices) in enumerate(zip(query,candidates)):
        indices = np.asarray(indices,dtype=int)
        separation = np.sum((points[indices]-point)**2,axis=1)
        low,high = (0.,2.) if case['sample'][j]=='faint_blue_near' else (2.,5.)
        in_shell = (distance[indices]>=low)&(distance[indices]<high)
        for mode,keep in (('accurate_distance',abs(delta[indices])<=.5),
                          ('same_reference_shell',in_shell),
                          ('same_shell_accurate',in_shell&(abs(delta[indices])<=.5))):
            options = indices[keep]
            if not len(options):
                yield j,mode,-1,0
                continue
            order = np.lexsort((pool['ls_id'][options],separation[keep]))
            yield j,mode,int(options[order[0]]),len(options)


def diagnostics(a):
    return dict(distance_ratio=a['qrf_quantiles_kpc'][:,2]/a['reference_kpc'],
                absolute_gaia_g=a['gaia_phot_g_mean_mag']-10-5*np.log10(a['reference_kpc']),
                proper_motion_mas_per_year=np.hypot(a['gaia_pmra'],a['gaia_pmdec']))


def make_pairs(case, controls, matches, positions):
    result = []
    for i,mode,j,n in matches:
        row = dict(cohort=case['sample'][i],control_design=mode,candidates=n,
                   case_r_group='14–16' if case['observed_r'][i]<16 else
                   '16–18' if case['observed_r'][i]<18 else '18–20')
        row.update({f'case_{key}':case[key][i] for key in PAIR_FIELDS})
        if j>=0:
            k = np.searchsorted(positions,j)
            row.update({f'control_{key}':controls[key][k] for key in PAIR_FIELDS})
        result.append(row)
    return result


def summarize(pairs):
    groups = sorted({(r['cohort'],r['control_design']) for r in pairs})
    keys = ('RCHISQ_R','FRACFLUX_R','gaia_ruwe','absolute_gaia_g','proper_motion_mas_per_year','distance_ratio')
    for cohort,design in groups:
        base = [r for r in pairs if r['cohort']==cohort and r['control_design']==design]
        for magnitude in ('all','14–16','16–18','18–20'):
            part = [r for r in base if magnitude=='all' or r['case_r_group']==magnitude]
            if not part:
                continue
            available = [r for r in part if r['candidates']]
            row = dict(cohort=cohort,control_design=design,case_r_group=magnitude,
                       cases=len(part),matched=len(available),
                       unique_controls=len({r['control_ls_id'] for r in available}))
            for key in keys:
                a,b = (np.asarray([r[f'{side}_{key}'] for r in available],dtype=float)
                       for side in ('case','control'))
                valid = np.isfinite(a)&np.isfinite(b)
                row[f'{key}_pair_n'] = int(valid.sum())
                for k,v in (('case',a),('control',b),('case_minus_control',a-b)):
                    row[f'{key}_{k}_median'] = float(np.median(v[valid])) if valid.any() else None
            yield row


def main():
    require_compute()
    bm,gm,im,dm = (path/'complete.json' for path in (BLUE,GAIA,INDEX,DELIVERY))
    blue,gaia,index,delivery = [json.loads(p.read_text()) for p in (bm,gm,im,dm)]
    case = load(verified(BLUE/'blue_objects.npz',blue['outputs']['blue_objects.npz']))
    if not (case['TYPE']=='PSF').all():
        raise ValueError('Case sample requires verified native PSF')
    with ThreadPoolExecutor(max_workers=workers()) as threads:
        roster = gaia_pool(case,gaia,threads)
        matched = list(matches(case,roster))
        positions = np.unique(np.array([p[2] for p in matched if p[2]>=0],dtype=int))
        catalog = read_columns(verified(INDEX/'catalog_rows.npz',index['outputs']['catalog_rows.npz']),
                               ('ls_id','sweep','sweep_offset'))
        with verified(DELIVERY/'sweeps.csv',delivery['outputs']['sweeps.csv']).open() as f:
            receipts = {r['sweep']:r['receipt_sha256'] for r in csv.DictReader(f)}
        controls = {key:val[positions] for key,val in roster.items()}
        if len(positions):
            forecast = gather_forecasts(controls['catalog_row'],catalog,receipts)
            np.testing.assert_array_equal(forecast['ls_id'],controls['ls_id'])
            np.testing.assert_array_equal(forecast['source_id'],controls['source_id'])
            raw = native_payload(forecast,threads)
            controls.update({key:raw[key] for key in NATIVE_DIAGNOSTICS})
        controls.update(diagnostics(controls))
        case.update(diagnostics(case))
    rows = make_pairs(case,controls,matched,positions)
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'complete.json').exists():
        raise FileExistsError(OUT/'complete.json')
    csv_rows(OUT/'pairs.csv',rows)
    csv_rows(OUT/'summary.csv',summarize(rows))
    atomic_npz(OUT/'control_roster.npz',**controls)
    save_json(OUT/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),candidates=len(roster['ls_id']),
        cases=len(case['ls_id']),unique_controls=len(positions),caliper=CALIPER.tolist(),
        inputs={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),bm,gm,im,dm)},
        interpretation=('Fixed post hoc comparison of the original 305 blue cases. Both candidates '
                        'and cases satisfy the native PSF and Gaia geometry rules. Controls match '
                        'observed r and dereddened g-r before considering three distance conditions. '
                        'Control matches may repeat. These are descriptive comparisons in validation '
                        'data and do not identify the cause of a residual or change any cut.'),
        outputs={p.name:sha(p) for p in OUT.iterdir() if p.name!='complete.json'}))
    print(len(case['ls_id']),len(roster['ls_id']),len(positions),flush=True)


if __name__=='__main__':
    main()
