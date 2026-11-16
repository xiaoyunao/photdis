#!/usr/bin/env python3
"""Bind physical class evidence and paired reference scales to native v3.9 rows."""
import argparse
import json
import os

import numpy as np

from photdis.reference_v1_4 import unique_join
from photdis.stage1 import require_compute
from photdis.v3.metrics import distance_summary
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_4_near_reference_support import separation
from v3_6_vac_distances import read_columns
from v3_8_redrock import ASSESSED, OUT as REDROCK
from v3_9_catalog import OUT
from v3_9_references import OLDER, matched, sorted_key

DEST = ROOT/'data/v3_10/evidence'
REGIONS = ('south_faint','south_low','north')
RVS_PARAMETERS = ('rvs_feh','rvs_feh_err','rvs_logg','rvs_logg_err','rvs_teff','rvs_teff_err')
RR_QUALITY = ('redrock_SPECTYPE','redrock_ZWARN','redrock_DELTACHI2','redrock_TARGETID',
              'redrock_quality_assessed','redrock_separation_arcsec')


def rvs_evidence(t, inputs):
    root = OLDER.parent
    roster = root/'sources.json'
    receipt = root/'source_evidence.json'
    audit = json.loads(receipt.read_text())
    expected = json.loads(roster.read_text())['source_sha256']
    if expected!=audit['input_sha256']['sources.npz']:
        raise ValueError('Raw RVS positions and classifications come from different archives')
    s = read_columns(verified(root/'sources.npz',expected),
        ('source_id','ra','dec','rvs_targetid',*RVS_PARAMETERS))
    a = read_columns(verified(root/'source_evidence.npz',audit['output_sha256']),
        ('source_id','original_row','class_label','class_identity_conflict','target_identity_conflict'))
    np.testing.assert_array_equal(a['original_row'],np.arange(len(s['source_id'])))
    np.testing.assert_array_equal(a['source_id'],s['source_id'])
    rows = np.flatnonzero((s['source_id']>0)&(a['class_label']>=0)&
                          ~a['class_identity_conflict']&~a['target_identity_conflict'])
    label_rows = matched(t,dict(reference_ra=s['ra'],reference_dec=s['dec'],
                                _gaia=sorted_key(s['source_id'],rows)))
    found = label_rows>=0
    j = label_rows[found]
    result = dict(rvs_archive_row=label_rows,rvs_label=np.full(len(found),-1,'i1'),
                  rvs_targetid=np.full(len(found),-1,'i8'))
    result['rvs_label'][found] = a['class_label'][j]
    result['rvs_targetid'][found] = s['rvs_targetid'][j]
    for name in RVS_PARAMETERS:
        y = s[name][j].astype(float)
        value = np.full(len(found),np.nan,'f4')
        value[found] = np.where(np.isfinite(y)&(y>-9000),y,np.nan)
        result[name] = value
    for path in (receipt,roster):
        inputs[str(path.relative_to(ROOT))] = sha(path)
    return result


def redrock_evidence(t, inputs):
    marker = REDROCK/'complete.json'
    data = json.loads(marker.read_text())
    path = verified(ASSESSED,data['inputs'][str(ASSESSED.relative_to(ROOT))])
    a = read_columns(path,('identity','TYPE','RA','DEC',*RR_QUALITY))
    if not np.all((a['TYPE']=='PSF')&a['redrock_quality_assessed']&(a['redrock_ZWARN']==0)&
                  (a['redrock_DELTACHI2']>100)):
        raise ValueError('Unqualified historical Redrock objects')
    identity = np.where(t['source_id']>0,t['source_id'],-t['ls_id'])
    row = unique_join(identity,a['identity'])
    j = np.flatnonzero(row>=0)
    valid = separation(t['RA'][j],t['DEC'][j],a['RA'][row[j]],a['DEC'][row[j]])
    row[j[~valid]] = -1
    j = np.flatnonzero(row>=0)
    attached = dict(redrock_archive_row=row)
    for name in RR_QUALITY:
        value = a[name]
        column = np.zeros(len(row),dtype=value.dtype)
        column[j] = value[row[j]]
        attached[name] = column
    inputs[str(marker.relative_to(ROOT))] = sha(marker)
    return attached


def scale_comparison(t):
    q = t['qrf_quantiles_kpc']
    a,b = t['specdis_kpc'],t['spdist_kpc']
    joint = t['evaluation_row']&(t['row_state']==7)&np.isfinite(q).all(axis=1)
    joint &= t['specdis_quality']&t['spdist_quality']&(a>0)&(b>0)
    edges = (0,2,5,10,np.inf)
    rows = []
    for lo,hi in ((-np.inf,20),(20,21),(21,22),(22,23)):
        magnitude = (t['r']>=lo)&(t['r']<hi)
        for c in (-2,-1,0,1,2):
            base = joint&magnitude
            if c!=-2:
                base &= t['class_label']==c
            for label,d,other in (('SpecDis',a,b),('SPDist',b,a)):
                for left,right in zip(edges[:-1],edges[1:]):
                    j = base&(d>=left)&(d<right)
                    ranks = 10+5*np.log10(q[j])
                    actual = 10+5*np.log10(d[j])
                    md = float(np.median(other[j]/d[j]-1)) if j.any() else None
                    rows.append(dict(r_min=lo,r_max=hi,reference_bin=label,class_label=c,
                        d_min=left,d_max=right,other_reference_relative_median=md,
                        both_in_bin=int((j&(other>=left)&(other<right)).sum()),**distance_summary(ranks,actual)))
    return rows


def run(region):
    source = OUT/'figures_labeled'/region/'complete.json'
    receipt = json.loads(source.read_text())
    path = 'figure_reference_rows.npz'
    t = read_columns(verified(source.parent/path,receipt['outputs'][path]),
        ('TYPE','ls_id','source_id','RA','DEC','r','g0','r0','error_grz','gaia_state','row_state',
         'class_label','class_label_raw_rvs','evaluation_row','qrf_quantiles_kpc','dwarf_score',
         'spectype','reference_kpc','specdis_kpc','spdist_kpc','specdis_quality','spdist_quality',
         'nominal_qnear','nominal_numerical_flag'))
    if not np.all(t['TYPE']=='PSF') or len(np.unique(t['ls_id']))!=len(t['ls_id']):
        raise ValueError('Expected one verified native PSF per expansion reference')
    inputs = {str(source.relative_to(ROOT)):sha(source)}
    rvs = rvs_evidence(t,inputs)
    np.testing.assert_array_equal(rvs['rvs_label'],t['class_label_raw_rvs'])
    rr = redrock_evidence(t,inputs)
    for p in RVS_PARAMETERS:
        t[p] = rvs[p]
    t.update(rvs_archive_row=rvs['rvs_archive_row'],rvs_targetid=rvs['rvs_targetid'],**rr)
    roles = (np.floor(t['RA']/10).astype(int)+3*np.floor((t['DEC']+90)/10).astype(int))%5
    t['sky_role'] = roles.astype('u1')
    if region=='north':
        np.testing.assert_array_equal(t['evaluation_row'],roles==4)
    dest = DEST/region
    dest.mkdir(parents=True,exist_ok=True)
    csv_rows(dest/'photdis_specdis_spdist.csv',scale_comparison(t))
    qa = dict(rows=len(t['ls_id']),label_counts={str(c):int((t['class_label']==c).sum()) for c in (-1,0,1,2)},
              raw_rvs=int((t['rvs_archive_row']>=0).sum()),qualified_redrock=int((rr['redrock_archive_row']>=0).sum()),
              raw_nonstellar=int(np.isin(t['spectype'],('QSO','GALAXY')).sum()))
    atomic_npz(dest/'references.npz',**t)
    save_json(dest/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,**qa,
              outputs={k:sha(dest/k) for k in ('references.npz','photdis_specdis_spdist.csv')}))
    print(region,qa,flush=True)


if __name__=='__main__':
    require_compute()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('region',choices=REGIONS)
    run(parser.parse_args().region)
