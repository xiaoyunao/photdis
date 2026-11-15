#!/usr/bin/env python3
"""Bind source-specific distance evidence to common DR11 prediction and class fields."""
import json
import os
from pathlib import Path

import numpy as np
from scipy.special import ndtr

from photdis.reference_v1_4 import unique_join
from photdis.stage1 import require_compute
from photdis.v3.metrics import distance_summary
from v2_1_near import ROOT, atomic_npz, csv_rows, load, save_json, sha
from v3_3_sed import BASE, MANIFESTS, verified
from v3_5_dr11_validate import OUT as VALIDATION, join_parts, validation_history
from v3_6_external_reference import OUT as EXTERNAL
from v3_6_gaia_census import OUT as GAIA
from v3_6_reference_index import OUT as INDEX
from v3_6_vac_distances import ARCHIVE, DISTANCE_EDGES, read_columns
from v3_6_vac_flow import OUT as VAC


OUT = ROOT/'data/v3_6/distance_evidence'
COMMON = ('ls_id','identity','source_id','catalog_row','TYPE','RA','DEC','observed_r','dereddened_g_r',
          'band_state','nominal_qnear','qrf_qnear','dwarf_selection_score','qrf_quantiles_kpc',
          'qrf_bins','primary','strict')
CLASSES = ('dwarf','giant','other')
SHELLS = ('0–2','2–5','5–10','10–20','20–50','>50')


def class_index(paths,size,inputs):
    marker = INDEX/'complete.json'
    ref = json.loads(marker.read_text())
    inputs[str(marker.relative_to(ROOT))] = sha(marker)
    first = np.full(size,-1,'i1')
    conflict = np.zeros(size,bool)
    archive_roles = {}
    for name,path in paths.items():
        x = load(verified(INDEX/f'{name}.npz',ref['outputs'][f'{name}.npz']))
        cols = ('class_label','class_eligible','split') if name=='baseline_v1_4' else ('class_label','class_eligible','role')
        t = read_columns(verified(path,ref['inputs'][str(path.relative_to(ROOT))]),cols)
        ok = (x['catalog_row']>=0)&t['class_eligible']&np.isin(t['class_label'],np.arange(3))
        row = x['catalog_row'][ok]
        label = t['class_label'][ok]
        # Compare all repeated classifications before constructing consensus.
        for k in range(3):
            j = np.unique(row[label==k])
            changed = (first[j]>=0)&(first[j]!=k)
            conflict[j[changed]] = True
            first[j[first[j]<0]] = k
        mapped = x['catalog_row']>=0
        label_key = 'split' if name=='baseline_v1_4' else 'role'
        archive_roles[name] = (x['catalog_row'][mapped],t[label_key][mapped])
    first[conflict] = -1
    return first,conflict,archive_roles


def source(path,inputs):
    receipt = json.loads(path.read_text())
    inputs[str(path.relative_to(ROOT))] = sha(path)
    return receipt


def geometry_rows(inputs):
    r = source(GAIA/'complete.json',inputs)
    chunks = []
    for name,digest in r['child_receipts'].items():
        marker = verified(GAIA/name,digest)
        child = json.loads(marker.read_text())
        path = marker.with_suffix('.npz')
        with np.load(verified(path,child['outputs'][path.name]),allow_pickle=False) as z:
            mask = z['gaia_state']==6
            if mask.any():
                row = {key:z[key][mask] for key in (*COMMON,'reference_kpc','reference_lo_kpc',
                                                   'reference_hi_kpc','p_corr','gaia_parallax_error')}
                chunks.append(row)
    a = join_parts(chunks)
    a['high_quality'] = np.ones(len(a['ls_id']),bool)
    return a


def vac_rows(kind,inputs):
    p = VAC/kind
    r = source(p/'complete.json',inputs)
    keys = ('class_label','distance_kpc','sigma_mu','historical_quality','usable_uncertainty','vac_row',
            'rvs_logg','rvs_teff','rvs_feh','rvs_parameter_quality')
    a = read_columns(verified(p/'matched.npz',r['outputs']['matched.npz']),(*COMMON,*keys))
    a['reference_kpc'] = a['distance_kpc']
    width = np.where(a['usable_uncertainty'],a['sigma_mu'],np.nan)
    a['reference_lo_kpc'] = a['reference_kpc']*10**(-width/5)
    a['reference_hi_kpc'] = a['reference_kpc']*10**(width/5)
    a['high_quality'] = a['historical_quality']
    return a


def external_rows(inputs):
    r = source(EXTERNAL/'complete.json',inputs)
    for f,h in r['outputs'].items():
        if not f.endswith('_matched.npz'):
            continue
        keys = ('class_label','distance_kpc','distance_error_kpc','distance_available','distance_precise',
                'historical_evidence','inventory_row','feh','feh_valid')
        a = read_columns(verified(EXTERNAL/f,h),(*COMMON,*keys))
        d,s = a['distance_kpc'],a['distance_error_kpc']
        a['reference_kpc'] = d
        a['reference_lo_kpc'] = np.maximum(d-s,0.)
        a['reference_hi_kpc'] = d+s
        a['high_quality'] = a['distance_precise']
        yield f.removesuffix('_matched.npz'),a


def attach(a,first,conflict,roles,training,history):
    row = a['catalog_row']
    if not (a['TYPE']=='PSF').all() or (row<0).any():
        raise ValueError('Distance reconciliation requires native PSFs and checked ownership')
    own = a.get('class_label',np.full(len(row),-1,'i1'))
    label = first[row]
    valid = np.isin(own,np.arange(3))
    disagree = conflict[row]|(valid&(label>=0)&(own!=label))
    a['source_class'] = own
    a['consensus_class'] = np.where(disagree,-1,np.where(valid,own,label)).astype('i1')
    a['class_disagreement'] = disagree
    for k,identities in training.items():
        a[k] = np.isin(a['identity'],identities)|np.isin(a['source_id'],identities)
    identities,legacy = history
    a['recorded_history'] = (np.isin(a['identity'],identities)|np.isin(a['source_id'],identities)
                             |np.isin(a['ls_id'],legacy)|a.get('historical_evidence',False))
    for name,(pointer,key) in roles.items():
        ix = unique_join(row,pointer)
        have = ix>=0
        role = np.full(len(row),-1,dtype='i4') if key.dtype.kind in 'iu' else np.full(len(row),'',dtype=key.dtype)
        role[have] = key[ix[have]]
        a[f'{name}_role'] = role


def reference_agreement(a):
    """Reference-error probability inside the stored interval, with each source's error law."""
    q = a['qrf_quantiles_kpc']
    d = a['reference_kpc']
    result = []
    for lo,hi in ((1,3),(0,4)):
        if 'p_corr' in a:
            p,err = a['p_corr'],a['gaia_parallax_error']
            of_reference = (ndtr((1/q[:,lo]-p)/err)-ndtr((1/q[:,hi]-p)/err))/ndtr(p/err)
        elif 'sigma_mu' in a:
            e = a['sigma_mu']
            with np.errstate(divide='ignore',invalid='ignore'):
                of_reference = ndtr(5*np.log10(q[:,hi]/d)/e)-ndtr(5*np.log10(q[:,lo]/d)/e)
            of_reference[e<=0] = np.nan
        else:
            e = a['distance_error_kpc']
            with np.errstate(divide='ignore',invalid='ignore'):
                of_reference = (ndtr((q[:,hi]-d)/e)-ndtr((q[:,lo]-d)/e))/ndtr(d/e)
            of_reference[e<=0] = np.nan
        result.append(of_reference)
    return np.stack(result,axis=1)


def summaries(a,name):
    distance = a['reference_kpc']
    pred = a['qrf_quantiles_kpc']
    with np.errstate(divide='ignore',invalid='ignore'):
        ref = 10+5*np.log10(distance)
        mu = 10+5*np.log10(pred)
    valid = np.isfinite(ref)
    uncertainty = a['reference_error_agreement']
    r,c = a['observed_r'],a['dereddened_g_r']
    all_rows = np.ones(len(distance),bool)
    labels = a['consensus_class']
    conditions = [('all',all_rows),('blue',(c>=.5)&(c<.8)),('faint_blue',(r>=18)&(c>=.5)&(c<.8))]
    summary,migration = [],[]
    for cut,gate in (('supported',all_rows),('primary',a['primary']),('strict',a['strict'])):
        for group,cl in [('all',all_rows),*[(key,labels==i) for i,key in enumerate(CLASSES)],('unknown',labels<0)]:
            selected = gate&cl
            for case,qual in (('available',all_rows),('high_quality',a['high_quality'])):
                for label,history in (('all',all_rows),('outside_qrf_train',~a['qrf_training']),
                                      ('outside_recorded_history',~a['recorded_history']&~a['qrf_training'])):
                    scope = selected&qual&history
                    for observed,valid_obs in conditions:
                        for j,(lo,hi) in enumerate(zip(DISTANCE_EDGES[:-1],DISTANCE_EDGES[1:])):
                            keep = scope&valid_obs&valid&(distance>=lo)&(distance<hi)
                            agreement = uncertainty[keep]
                            means = [float(np.mean(x[np.isfinite(x)])) if np.isfinite(x).any() else None
                                     for x in agreement.T]
                            summary.append(dict(source=name,selection=cut,reference_class=group,quality=case,
                                history=label,observed=observed,reference_lo_kpc=lo,reference_hi_kpc=hi,
                                row_denominator=int((scope&valid_obs).sum()),
                                qrf_training_n=int((keep&a['qrf_training']).sum()),
                                qrf_d50_gt10_n=int((keep&(pred[:,2]>10.)).sum()),
                                qrf_qnear_lt_half_n=int((keep&(a['qrf_qnear']<.5)).sum()),
                                ref_error_agreement68=means[0],ref_error_agreement95=means[1],
                                error_defined_n=int(np.isfinite(agreement).all(axis=1).sum()),
                                **distance_summary(mu[keep],ref[keep])))
                            if case=='high_quality' and label=='all' and observed=='all':
                                for k,(plo,phi) in enumerate(zip(DISTANCE_EDGES[:-1],DISTANCE_EDGES[1:])):
                                    migration.append(dict(source=name,selection=cut,reference_class=group,
                                        reference_shell=SHELLS[j],predicted_shell=SHELLS[k],
                                        hard_count=int((keep&(pred[:,2]>=plo)&(pred[:,2]<phi)).sum()),
                                        qrf_weighted_count=float(a['qrf_bins'][keep,k].sum())))
    return summary,migration


def main():
    require_compute()
    inputs = {str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))}
    size = source(INDEX/'complete.json',inputs)['supported_rows']
    first,conflict,roles = class_index({'baseline_v1_4':ARCHIVE/'references.npz','v3_0':BASE/'references.npz'},size,inputs)
    controls = source(MANIFESTS/'controls/complete.json',inputs)
    q = read_columns(verified(BASE/'controls/training.npz',controls['outputs']['training.npz']),('identity','class_label'))
    cloud = source(MANIFESTS/'sed_cloud.json',inputs)
    sed = read_columns(verified(BASE/'sed_cloud.npz',cloud['sha256']),('identity',))['identity']
    np.testing.assert_array_equal(sed,q['identity'])
    training = dict(qrf_training=q['identity'],nominal_sed_training=sed,
                    selector_training=q['identity'][q['class_label']>=0])
    previous,legacy = validation_history(inputs,None)
    original = source(VALIDATION/'complete.json',inputs)
    for kind in ('gaia_qualification_roster.npz','spectra_reference_roster.npz'):
        path = verified(VALIDATION/kind,original['outputs'][kind])
        with np.load(path,allow_pickle=False) as z:
            legacy = np.union1d(legacy,z['ls_id'])
            if 'native_source_id' in z:
                previous = np.union1d(previous,z['native_source_id'])
    history = previous,legacy
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'complete.json').exists():
        raise FileExistsError(OUT/'complete.json')
    cases = [('Gaia',geometry_rows(inputs)),('SpecDis',vac_rows('specdis',inputs)),
             ('SPDist',vac_rows('spdist',inputs)),*external_rows(inputs)]
    all_rows,all_migration = [],[]
    for name,a in cases:
        attach(a,first,conflict,roles,training,history)
        a['reference_error_agreement'] = reference_agreement(a)
        atomic_npz(OUT/f'{name}.npz',**a)
        summary,migration = summaries(a,name)
        all_rows.extend(summary)
        all_migration.extend(migration)
        print(name,len(a['ls_id']),int(a['qrf_training'].sum()),int(a['class_disagreement'].sum()),flush=True)
    csv_rows(OUT/'distance_cells.csv',all_rows)
    csv_rows(OUT/'migration.csv',all_migration)
    save_json(OUT/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,
        rows={name:len(a['ls_id']) for name,a in cases},
        interpretation='Per-source native PSF projections. Reference sets overlap. QRF training matches are fit consistency. Error-integrated agreement uses Gaussian parallax, modulus, or distance reference errors as available.',
        outputs={p.name:sha(p) for p in OUT.iterdir() if p.name!='complete.json'}))


if __name__=='__main__':
    main()
