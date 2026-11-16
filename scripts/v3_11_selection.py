#!/usr/bin/env python3
"""Compare RVS label semantics with frozen v3.10 predictors and sky roles."""

import argparse
import json
import os

import joblib
import numpy as np

from photdis.reference_v1_4 import unique_join
from photdis.stage1 import require_compute
from photdis.v3.metrics import distance_summary
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_6_vac_distances import read_columns
from v3_10_evidence import DEST as EVIDENCE
from v3_10_science import working_index
from v3_10_selection import DEST as BASE, ROUTES, features, fit, references
from v3_10_working_sample import DOMAINS, RADIAL_MASS, SCORE

DEST = ROOT/'data/v3_11/selection'
LABELS = ['dwarf', 'giant', 'intermediate', 'boundary', 'hot', 'unknown']
SHELLS = ((0.,2.), (2.,5.), (5.,10.), (10.,np.inf))


def gravity_labels(logg, error, teff, admitted):
    good = admitted & np.isfinite(logg) & np.isfinite(error) & (error>0)
    good &= np.isfinite(teff) & (teff>0)
    hot = good & (teff>=7000)
    cool = good & ~hot
    lower = logg-error
    upper = logg+error
    labels = np.full(len(logg), 5, dtype=np.int8)
    labels[hot] = 4
    labels[cool] = 3
    labels[cool & (lower>4.2)] = 0
    labels[cool & (upper<3.2)] = 1
    labels[cool & (lower>3.2) & (upper<4.2)] = 2
    return labels


def matched_threshold(score, baseline, labels, eligible):
    """Use the loosest threshold meeting both calibration leakage budgets."""
    lower_bounds, budgets = [], {}
    for klass in (1,2):
        use = eligible & (labels==klass)
        if not use.any():
            raise ValueError(f'Missing calibration control for {LABELS[klass]}')
        allowed = int(np.count_nonzero(use & (baseline>=SCORE)))
        ranked = np.sort(score[use])[::-1]
        lower_bounds.append(np.nextafter(ranked[allowed],np.inf) if allowed<len(ranked) else 0.)
        budgets[LABELS[klass]] = dict(available=int(use.sum()),baseline_pass=allowed)
    threshold = float(max(lower_bounds))
    for klass in (1,2):
        budgets[LABELS[klass]]['candidate_pass'] = int(np.count_nonzero(
            eligible & (labels==klass) & (score>=threshold)))
    return threshold,budgets


def metrics(t, labels, eligible, radial, score, candidate, threshold):
    distance_rows, class_rows = [], []
    color = t['g0']-t['r0']
    finite = np.isfinite(score) & np.isfinite(candidate)
    gates = (
        ('supported',finite), ('radial_only',finite & radial),
        ('original_class',score>=SCORE), ('original_combined',(score>=SCORE)&radial),
        ('new_fixed_class',candidate>=SCORE), ('new_fixed_combined',(candidate>=SCORE)&radial),
        ('new_matched_class',candidate>=threshold),
        ('new_matched_combined',(candidate>=threshold)&radial),
    )
    groups = [(name, labels==i) for i,name in enumerate(LABELS)]
    groups += [('original_dwarf',t['class_label']==0)]
    for kind in ('QSO','GALAXY'):
        groups.append((kind,t['qualified']==kind))
    for lo,hi in ((-np.inf,np.inf),(-np.inf,.5),(.5,.8),(.8,1.4),(1.4,np.inf)):
        color_cut = eligible & (color>=lo) & (color<hi)
        for label,klass in groups:
            base = color_cut & klass
            if not base.any():
                continue
            for stage,gate in gates:
                keep = base & gate
                class_rows.append(dict(color_min=lo,color_max=hi,label=label,stage=stage,
                                       eligible=int(base.sum()),selected=int(keep.sum())))
                if label in ('QSO','GALAXY'):
                    continue
                for source,ref,quality in (
                    ('SpecDis',t['specdis_kpc'],t['specdis_quality']),
                    ('SPDist',t['spdist_kpc'],t['spdist_quality']),
                    ('Gaia',t['reference_kpc'],t['gaia_state']==6),
                ):
                    for d0,d1 in SHELLS:
                        cohort = base & quality & (ref>0) & (ref>=d0) & (ref<d1)
                        if not cohort.any():
                            continue
                        k = cohort & gate
                        result = distance_summary(
                            10+5*np.log10(t['qrf_quantiles_kpc'][k]),10+5*np.log10(ref[k]))
                        distance_rows.append(dict(color_min=lo,color_max=hi,label=label,stage=stage,
                                                  reference=source,d_min=d0,d_max=d1,
                                                  eligible=int(cohort.sum()),**result))
    columns = ('n','median_relative','nmad_relative','median_mu_residual','severe_fraction',
               'coverage68','coverage95','median_width68','median_width95')
    return class_rows,[{**dict.fromkeys(columns),**row} for row in distance_rows]


def run(side):
    t,inputs = references(side)
    extra = ('TYPE','ls_id','g0','r0','rvs_logg','rvs_logg_err','rvs_teff','rvs_archive_row')
    for key in extra[2:]:
        t[key] = np.full(len(t['ls_id']),np.nan)
    covered = np.zeros(len(t['ls_id']),bool)
    for region in ROUTES[side]:
        marker = EVIDENCE/region/'complete.json'
        digest = json.loads(marker.read_text())['outputs']['references.npz']
        p = read_columns(verified(marker.parent/'references.npz',digest),extra)
        j = unique_join(p['ls_id'],t['ls_id'])
        source = np.flatnonzero(j>=0)
        dest = j[source]
        if np.any(covered[dest]) or not np.all(p['TYPE'][source]=='PSF'):
            raise ValueError('Overlapping or non-PSF augmented reference')
        for key in extra[2:]:
            t[key][dest] = p[key][source]
        covered[dest] = True
    if not covered.all():
        raise ValueError('Missing RVS evidence rows for the fixed comparisons')
    labels = gravity_labels(t['rvs_logg'],t['rvs_logg_err'],t['rvs_teff'],
                            (t['rvs_archive_row']>=0)&(t['class_label']>=0))
    marker = BASE/side/'complete.json'
    inputs[str(marker.relative_to(ROOT))] = sha(marker)
    old = read_columns(verified(marker.parent/'scores.npz',
        json.loads(marker.read_text())['outputs']['scores.npz']),
        ('ls_id','TYPE','score','training','calibration','held_out'))
    np.testing.assert_array_equal(t['ls_id'],old['ls_id'])
    np.testing.assert_array_equal(t['TYPE'],old['TYPE'])
    x = features(side,t,inputs)
    model,new_score,new_train,new_cal,new_test = fit(
        side,x,dict(t,class_label=np.where(labels<3,labels,-1)),inputs)
    np.testing.assert_array_equal(new_test,old['held_out'])
    if np.any(new_train&~old['training']) or np.any(new_cal&~old['calibration']):
        raise ValueError('New labels modified the frozen role eligibility')
    _, sample = working_index(side,inputs)
    j = unique_join(t['ls_id'],sample['ls_id'])
    support = j>=0
    support[support] &= sample['state'][j[support]]>=2
    radial = np.zeros(len(j),bool)
    radial[support] = sample['radial_mass'][j[support]].sum(axis=1)>=RADIAL_MASS
    bright = support & (t['region']==DOMAINS[side])
    threshold,budgets = matched_threshold(new_score,old['score'],labels,new_cal & bright)
    from v3_10_redrock import DEST as REDROCK
    marker = REDROCK/'complete.json'
    inputs[str(marker.relative_to(ROOT))] = sha(marker)
    z = read_columns(verified(REDROCK/'qualified.npz',
        json.loads(marker.read_text())['outputs']['qualified.npz']),('ls_id','TYPE','qualified'))
    jj = unique_join(t['ls_id'],z['ls_id'])
    found = jj>=0
    np.testing.assert_array_equal(t['TYPE'][found],z['TYPE'][jj[found]])
    t['qualified'] = np.where(t['redrock_quality_assessed'],t['redrock_SPECTYPE'],'').astype('U8')
    q = found.copy()
    q[found] &= z['qualified'][jj[found]]!=''
    vals = z['qualified'][jj[q]]
    conflict = (t['qualified'][q]!='') & (t['qualified'][q]!=vals)
    t['qualified'][q] = np.where(conflict,'',vals)
    eligible = bright & old['held_out']
    retention,distances = metrics(t,labels,eligible,radial,old['score'],new_score,threshold)
    folder = DEST/side
    folder.mkdir(parents=True,exist_ok=True)
    joblib.dump(dict(model,threshold=threshold),folder/'model.joblib')
    csv_rows(folder/'retention.csv',retention)
    csv_rows(folder/'distances.csv',distances)
    atomic_npz(folder/'scores.npz',ls_id=t['ls_id'],TYPE=t['TYPE'],labels=labels,
               new_train=new_train,new_cal=new_cal,held_out=eligible,radial_pass=radial,
               score_original=old['score'],score_candidate=new_score)
    save_json(folder/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,
        calibration_leakage=budgets,threshold=threshold,fit=int(new_train.sum()),
        fit_original=int(old['training'].sum()),evaluation=int(eligible.sum()),
        outputs={name:sha(folder/name) for name in ('model.joblib','retention.csv','distances.csv','scores.npz')}))


if __name__=='__main__':
    require_compute()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('side',choices=tuple(ROUTES))
    run(parser.parse_args().side)
