#!/usr/bin/env python3
"""Measure the composition and error of actual predicted QRF distance layers."""

import argparse
import csv
import json
import os

import numpy as np

from photdis.reference_v1_4 import unique_join
from photdis.stage1 import require_compute
from photdis.v3.metrics import histogram_reference_score
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_6_vac_distances import read_columns
from v3_6_weak_gaia import histogram, score
from v3_9_catalog import OUT as EXPANSION
from v3_9_references import choose, join, subset
from v3_10_evidence import DEST as EVIDENCE
from v3_10_redrock import DEST as REDROCK
from v3_10_working_sample import DEST as WORK, DOMAINS
from v3_11_selection import LABELS, SHELLS, gravity_labels

DEST = ROOT/'data/v3_11/layers'
COLORS = ((-np.inf,np.inf),(-np.inf,.5),(.5,.8),(.8,1.4),(1.4,np.inf))
MAGNITUDES = ((-np.inf,20.),(-np.inf,18.),(18.,19.),(19.,20.))
CUTS = np.array([2.,5.,10.])
EXTRA = ('reference_lo_kpc','reference_hi_kpc','specdis_sigma_mu','spdist_sigma_mu',
         'signed_parallax_mas','parallax_error_mas')


def read_receipt(root, receipt, name, fields, inputs):
    path = root/receipt
    m = json.loads(path.read_text())
    inputs[str(path.relative_to(ROOT))] = sha(path)
    return read_columns(verified(root/name,m['outputs'][name]),fields)


def evidence(side, inputs):
    fields = ('TYPE','ls_id','class_label','evaluation_row','sky_role','gaia_state','reference_kpc',
              'specdis_kpc','specdis_quality','spdist_kpc','spdist_quality',
              'rvs_archive_row','rvs_logg','rvs_logg_err','rvs_teff',
              'redrock_quality_assessed','redrock_SPECTYPE')
    t = read_receipt(EVIDENCE/DOMAINS[side],'complete.json','references.npz',fields,inputs)
    if not np.all(t['TYPE']=='PSF') or len(np.unique(t['ls_id']))!=len(t['ls_id']):
        raise ValueError('Invalid science-reference identity')
    t['label'] = gravity_labels(t['rvs_logg'],t['rvs_logg_err'],t['rvs_teff'],
                               (t['rvs_archive_row']>=0)&(t['class_label']>=0))
    z = read_receipt(REDROCK,'complete.json','qualified.npz',('TYPE','ls_id','qualified'),inputs)
    j = unique_join(t['ls_id'],z['ls_id'])
    known = j>=0
    np.testing.assert_array_equal(t['TYPE'][known],z['TYPE'][j[known]])
    rr = np.where(t['redrock_quality_assessed'],t['redrock_SPECTYPE'],'').astype('U8')
    known[known] &= z['qualified'][j[known]]!=''
    a = z['qualified'][j[known]]
    rr[known] = np.where((rr[known]!='')&(rr[known]!=a),'',a)
    t['nonstellar'] = rr
    wd = ROOT/'manifests/v3_10/branch_cases/wd/wd_evidence.csv'
    with wd.open() as f:
        values = list(csv.DictReader(f))
    ids = [int(row['ls_id']) for row in values if row['TYPE']=='PSF' and
           (row['spectroscopic_wd']=='True' or (row['published_pwd'] and float(row['published_pwd'])>.75))]
    inputs[str(wd.relative_to(ROOT))] = sha(wd)
    t['wd_crossmatch'] = np.isin(t['ls_id'],ids)
    for key in EXTRA:
        t[key] = np.full(len(t['ls_id']),np.nan)
    folder = EXPANSION/'validation'/side
    markers = sorted(folder.glob('[0-9][0-9].json'))
    if not markers:
        raise ValueError(f'Missing reference errors in {folder}')
    filled = np.zeros(len(t['ls_id']),bool)
    for marker in markers:
        obj = json.loads(marker.read_text())
        name = marker.stem+'_reference.npz'
        if name not in obj['outputs']:
            continue
        ref = read_receipt(folder,marker.name,name,('TYPE','ls_id',*EXTRA),inputs)
        j = unique_join(ref['ls_id'],t['ls_id'])
        k = j>=0
        if np.any(filled[j[k]]) or not np.all(ref['TYPE'][k]=='PSF'):
            raise ValueError('Repeated or unverified reference uncertainty')
        for key in EXTRA:
            t[key][j[k]] = ref[key][k]
        filled[j[k]] = True
    t['has_reference_uncertainties'] = filled
    return t


def forecast(side, shard, ref, inputs):
    root = WORK/side
    receipt = root/f'{shard:02d}.json'
    meta = json.loads(receipt.read_text())
    inputs[str(receipt.relative_to(ROOT))] = sha(receipt)
    if not meta['rows']:
        return {}
    fields = ('TYPE','ls_id','RA','DEC','r','g0','r0','state','pdf_rows','part_index',
              'radial_mass','qrf_median_kpc','error_grz')
    p = read_receipt(root,receipt.name,f'{shard:02d}.npz',fields,inputs)
    p = subset(p,p['state']>=2)
    if not len(p['ls_id']):
        return {}
    if not np.all(p['TYPE']=='PSF'):
        raise ValueError('Unverified morphology in PDF index')
    with verified(root/f'{shard:02d}.csv',meta['outputs'][f'{shard:02d}.csv']).open() as f:
        pointers = list(csv.DictReader(f))
    p['role'] = (np.floor(p['RA']/10).astype(int)+3*np.floor((p['DEC']+90)/10).astype(int))%5
    j = unique_join(p['ls_id'],ref['ls_id'])
    p['reference_row'] = j
    p['pdf_shell'] = np.empty((len(j),4),dtype='f4')
    p['spectro_log_score'] = np.full((len(j),2),np.nan)
    p['gaia_weak'] = np.full((len(j),5),np.nan)
    comparables = np.flatnonzero(j>=0)
    holdout = np.zeros(len(j),bool)
    holdout[comparables] = ref['evaluation_row'][j[comparables]] & (ref['sky_role'][j[comparables]]==4)
    y,e = [ref[key] for key in ('signed_parallax_mas','parallax_error_mas')]
    weak = holdout.copy()
    d = j[holdout]
    weak[holdout] &= (ref['gaia_state'][d]==4)&np.isfinite(y[d])&np.isfinite(e[d])&(e[d]>0)&(y[d]<5*e[d])
    # Uniform deterministic subsample of each archived shard before applying selection.
    weak_indices = choose(p['ls_id'],weak,256,31123)
    broad = np.linspace(0.,25.,251)
    broad_nodes = histogram(broad,np.diff(broad)/25.)
    for part_index in np.unique(p['part_index']):
        k = np.flatnonzero(p['part_index']==part_index)
        point = pointers[part_index]
        a = p['pdf_rows'][k]
        z = read_columns(verified(ROOT/point['file'],point['pdf_sha256']),
                         ('TYPE','ls_id','qrf_bins','qrf_mu_edges','qrf_mu_mass'))
        np.testing.assert_array_equal(z['ls_id'][a],p['ls_id'][k])
        np.testing.assert_array_equal(z['TYPE'][a],p['TYPE'][k])
        bins = z['qrf_bins'][a]
        p['pdf_shell'][k] = np.column_stack((bins[:,:3],bins[:,3:].sum(axis=1)))
        np.testing.assert_allclose(bins[:,1:3],p['radial_mass'][k],atol=1e-7,rtol=0)
        for x in k[holdout[k]]:
            h = j[x]
            row = p['pdf_rows'][x]
            edges,mass = z['qrf_mu_edges'][row],z['qrf_mu_mass'][row]
            for column,name in enumerate(('specdis','spdist')):
                distance = ref[name+'_kpc'][h]
                error = ref[name+'_sigma_mu'][h]
                if ref[name+'_quality'][h] and distance>0 and np.isfinite(error) and error>0:
                    with np.errstate(divide='ignore'):
                        p['spectro_log_score'][x,column] = histogram_reference_score(
                            edges,mass,10+5*np.log10(distance),error)
        for x in np.intersect1d(k,weak_indices,assume_unique=True):
            h = j[x]
            row = p['pdf_rows'][x]
            edges,mass = z['qrf_mu_edges'][row],z['qrf_mu_mass'][row]
            result = score(*histogram(edges,mass),y[h],e[h])
            p['gaia_weak'][x] = [result[0]-score(*broad_nodes,y[h],e[h])[0],result[1],
                                 y[h],result[3],result[2]]
    np.testing.assert_allclose(p['pdf_shell'].sum(axis=1),1.,rtol=0,atol=2e-7)
    return p


def counts(p, r):
    summary, migration, calibration = [], [], []
    j = p['reference_row']
    found = j>=0
    evaluated = np.zeros(len(j),bool)
    evaluated[found] = r['evaluation_row'][j[found]] & (r['sky_role'][j[found]]==4)
    color = p['g0']-p['r0']
    median = p['qrf_median_kpc']
    hard = np.digitize(median,CUTS)
    shell_mass = p['pdf_shell']
    dist,good,groups = {},{},[('all',evaluated)]
    for klass in range(len(LABELS)):
        groups.append((LABELS[klass],evaluated&(r['label'][j]==klass)))
    for name in ('QSO','GALAXY'):
        groups.append((name,evaluated&(r['nonstellar'][j]==name)))
    groups.append(('published_wd_or_spectroscopic_match',evaluated&r['wd_crossmatch'][j]))
    for source,name,k in (('Gaia','reference_kpc',r['gaia_state']==6),
                          ('SpecDis','specdis_kpc',r['specdis_quality']),
                          ('SPDist','spdist_kpc',r['spdist_quality'])):
        dist[source] = r[name][j]
        good[source] = evaluated & k[j] & np.isfinite(dist[source]) & (dist[source]>0)
    for view,selection in (('supported',p['state']>=2),('working',p['state']==4)):
        for m0,m1 in MAGNITUDES:
            for c0,c1 in COLORS:
                use = selection & (p['r']>=m0) & (p['r']<m1) & (color>=c0) & (color<c1)
                if not use.any():
                    continue
                meta = dict(view=view,r_min=m0,r_max=m1,gr_min=c0,gr_max=c1)
                for pred,(lo,hi) in enumerate(SHELLS):
                    h = use & (hard==pred)
                    w = shell_mass[:,pred]
                    base = dict(**meta,pred_lo=lo,pred_hi=hi)
                    ident = (p['role']==4)&use
                    summary.append(dict(**base,component='role4_catalog',hard_n=int((h&ident).sum()),
                                       pdf_n=float(w[ident].sum(dtype=float))))
                    for group,lab in groups:
                        k = ident & lab
                        summary.append(dict(**base,component=group,hard_n=int((h&k).sum()),
                                           pdf_n=float(w[k].sum(dtype=float))))
                        if m0!=-np.inf or m1!=20.:
                            continue
                        for source in dist:
                            g = k & good[source]
                            if not g.any():
                                continue
                            center = dist[source]
                            severe = np.zeros(len(g),bool)
                            severe[g] = np.abs(5*np.log10(median[g]/center[g]))>1.
                            for t0,t1 in SHELLS:
                                t = g & (center>=t0) & (center<t1)
                                if not t.any():
                                    continue
                                migration.append(dict(**base,component=group,reference=source,
                                    ref_lo=t0,ref_hi=t1,ref_denominator=int(t.sum()),
                                    hard_n=int((h&t).sum()),pdf_n=float(w[t].sum(dtype=float)),
                                    severe_hard_n=int((h&t&severe).sum()),
                                    severe_pdf_n=float(w[t&severe].sum(dtype=float))))
                    weak = h & np.isfinite(p['gaia_weak'][:,0])
                    if weak.any():
                        a = p['gaia_weak'][weak]
                        calibration.append(dict(**base,reference='Gaia weak',n=len(a),
                            average_gain=float(a[:,0].mean()),observed_mas=float(a[:,2].mean()),
                            predicted_mas=float(a[:,3].mean()),negative_observed=float((a[:,2]<0).mean()),
                            negative_predicted=float(a[:,4].mean()),median_pit=float(np.median(a[:,1]))))
                    for name,column in (('SpecDis',0),('SPDist',1)):
                        a = p['spectro_log_score'][:,column]
                        k = h & np.isfinite(a)
                        if k.any():
                            calibration.append(dict(**base,reference=name,n=int(k.sum()),
                                                     mean_convolved_log_score=float(a[k].mean())))
    return summary,migration,calibration


def run(side):
    inputs = {}
    r = evidence(side,inputs)
    p = join([part for g in range(48) if (part:=forecast(side,g,r,inputs))])
    if len(np.unique(p['ls_id']))!=len(p['ls_id']):
        raise ValueError('Repeated identity in the working domains')
    summary,migration,calibration = counts(p,r)
    dest = DEST/side
    dest.mkdir(parents=True,exist_ok=True)
    csv_rows(dest/'composition.csv',summary)
    csv_rows(dest/'migration.csv',migration)
    csv_rows(dest/'calibration.csv',calibration)
    save_json(dest/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,
        predicted=len(p['ls_id']),working=int((p['state']==4).sum()),
        role4=int((p['role']==4).sum()),reference_rows=len(r['ls_id']),
        reference_uncertainties=int((r['has_reference_uncertainties']).sum()),
        outputs={f:sha(dest/f) for f in ('composition.csv','migration.csv','calibration.csv')}))


if __name__=='__main__':
    require_compute()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('side',choices=tuple(DOMAINS))
    run(parser.parse_args().side)
