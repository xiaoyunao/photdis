#!/usr/bin/env python3
"""Validate actual v2.4 artifacts without modifying frozen science products."""
import csv
import json
import os
import numpy as np
from v2_1_near import ROOT,load,sha,save_json,csv_rows
from v2_4_audit import OLD,DEST,TABLES
from photdis.stage1 import require_compute


def read(name):
    with open(TABLES/(name+'.csv')) as f:return list(csv.DictReader(f))


def diagnostic_summaries():
    a=read('anomaly_objects')
    # Concrete diagnosis summaries preserve full711/147 and explicit missing references.
    summary=[]
    for group in ('all','low','remaining'):
        q=[r for r in a if group=='all' or (r['low_pit']=='True')==(group=='low')]
        matched=[r for r in q if r['SPDist_qualified']=='True']
        for model in ('control','candidate'):
            for category in ('main_near_reference','weak_reference_peak','no_reference_peak','broad_displaced','no_SPDist'):
                n=0
                for row in q:
                    if row['SPDist_qualified']!='True':c='no_SPDist'
                    else:
                        delta=abs(float(row[model+'_mode_mu'])-float(row['SPDist_mu']));mass=float(row[model+'_mass_within1']);peak=float(row[model+'_peak_near_reference'])>0
                        c='main_near_reference' if delta<=1 else ('weak_reference_peak' if peak and mass<.1 else ('no_reference_peak' if not peak and mass<.1 else 'broad_displaced'))
                    n+=c==category
                summary.append(dict(group=group,model=model,category=category,n=n,denominator=len(q),matched=len(matched)))
    csv_rows(TABLES/'anomaly_diagnosis.csv',summary)
    consistency=[]
    for low in (False,True):
        q=[r for r in a if (r['low_pit']=='True')==low and r['SPDist_qualified']=='True'];v=np.array([float(r['Gaia_minus_SPDist_sigma']) for r in q])
        consistency.append(dict(low_pit=low,n=len(q),median_Gaia_SPDist_standardized=float(np.median(v)),below_minus3=int((v< -3).sum()),
            median_RUWE=float(np.median([float(r['RUWE']) for r in q])),same_direction_XD=int(sum(float(r['candidate_conditional_pit'])<.05 for r in q))))
    csv_rows(TABLES/'anomaly_reference_consistency.csv',consistency)


def main():
    require_compute()
    from v2_4_information_followup import main as information_support
    information_support()
    checks={}
    def check(k,v):checks[k]=bool(v);assert v,k
    freeze=json.loads((OLD/'models_frozen.json').read_text())
    for n,h in freeze['models'].items():check('unchanged_model:'+n,sha(OLD/(n+'.joblib'))==h)
    for n,h in freeze['inputs'].items():check('old_freeze:'+n,sha(ROOT/n)==h)
    for stage in ('audit','population','recovery'):
        for path,h in json.loads((DEST/(stage+'_freeze.json')).read_text()).items():check('new_freeze:'+path,sha(ROOT/path)==h)
    for path,h in json.loads((DEST/'reservation_io_freeze.json').read_text()).items():check('reservation_io_freeze:'+path,sha(ROOT/path)==h)
    info=json.loads((DEST/'information_complete.json').read_text())
    check('information_support_source',sha(ROOT/'scripts/v2_4_information_followup.py')==info['source_sha256'])
    check('information_support_amendment',sha(ROOT/'docs/v2_4_information_amendment.md')==info['amendment_sha256'])
    a=read('anomaly_objects');check('anomaly_denominator',len(a)==711);check('anomaly_numerator',sum(x['low_pit']=='True' for x in a)==147)
    check('anomaly_unique_identity',len({x['identity'] for x in a})==711)
    check('anomaly_PSF',all(x['TYPE']=='PSF' for x in a))
    for i,row in enumerate(a):
        for name in ('control','candidate'):
            pit=float(row[name+'_conditional_pit']);check(f'finite_same_identity_PIT:{i}:{name}',np.isfinite(pit) and 0<=pit<=1)
    r=load(DEST/'prospective_identity_reservation.npz');check('reservation_PSF',np.all(r['TYPE']=='PSF'));check('reservation_unique',len(np.unique(r['identity']))==len(r['identity']))
    for path in ['data/baseline_v0/references.npz','data/baseline_v1_3/references.npz','data/baseline_v1_4/references.npz','data/v2_2/application_inputs.npz']:
        with np.load(ROOT/path) as z:check('reservation_disjoint:'+path,not np.intersect1d(r['identity'],z['identity']).size)
    meta=json.loads((DEST/'reservation.json').read_text());check('reservation_unopened',not meta['reference_outcomes_opened'] and not meta['predictions_generated'])
    for folder in ('application_joint_chunks','recovery_chunks'):
        for p in sorted((DEST/folder).glob('*.npz')):
            m=json.loads(p.with_suffix('.json').read_text());check('chunk_hash:'+str(p.relative_to(DEST)),sha(p)==m['sha256'])
            z=load(p);check('chunk_PSF:'+str(p.relative_to(DEST)),(z['TYPE']=='PSF').all())
            check('chunk_mass:'+str(p.relative_to(DEST)),np.allclose(z['joint_mass'][:,:4].sum(axis=(1,2)),1,atol=2e-6))
    d=load(DEST/'recovery_design.npz');train=load(OLD/'training.npz');check('template_not_fit',not np.intersect1d(d['identity'],train['identity']).size)
    # Audit nearest available noise matches rather than implying new faint truth.
    noise=[]
    for group in np.unique(d['group']):
        for distance in np.unique(d['distance_bin']):
            u=(d['group']==group)&(d['distance_bin']==distance)&(d['replica']>=0)
            match=d['target_color_distance'][u]
            noise.append(dict(template_group=int(group),distance_kpc=[1,3.5,7.5,15,30][distance],n=int(u.sum()),
                median_match_distance=float(np.median(match)),p95_match_distance=float(np.quantile(match,.95)),
                fraction_match_gt1=float(np.mean(match>1)),median_r=float(np.median(d['r'][u])),
                median_max_grz_error=float(np.median(np.max(d['errors'][u,:3],axis=1)))))
    csv_rows(TABLES/'population_noise_support.csv',noise)
    rows=read('population_recovery')
    for pop in sorted({r['population'] for r in rows}):
        for condition in ('clean','target_noise'):
            for selection in ('all','observable_parent','near_scope'):
                q=[r for r in rows if (r['population'],r['condition'],r['selection'])==(pop,condition,selection)]
                allrows=[r for r in q if r['channel']=='all'];cl=[r for r in q if r['channel']!='all']
                check('population_conservation:'+':'.join([pop,condition,selection]),np.isclose(sum(float(r['pdf_count']) for r in allrows),sum(float(r['pdf_count']) for r in cl),atol=1e-5))
                if selection=='all':check('known_design_count:'+pop+condition,np.isclose(sum(float(r['expected_truth']) for r in allrows),1000))
    counts=read('application_joint_counts')
    for selection in ('observable_parent','near_scope'):
        q=[r for r in counts if r['selection']==selection and r['group']=='all' and r['high_kpc']!='10']
        # Full disjoint bins include5–10; remove only extra0–10 aggregate.
        q=[r for r in counts if r['selection']==selection and r['group']=='all' and not (float(r['low_kpc'])==0 and float(r['high_kpc'])==10)]
        cl=sum(float(r['expected_count']) for r in q if r['channel'].endswith('_reference'))
        src=sum(float(r['expected_count']) for r in q if r['channel'].endswith('_label'))
        check('joint_source_class_total:'+selection,np.isclose(cl,src,atol=.01))
    diagnostic_summaries()
    # Verify all source-side pair counts never exceed original2055; source errors
    # and reusable ordinary controls remain separate axes of evidence.
    for i,row in enumerate(read('metal_source_pairs')):check('pair_unique_counts:'+str(i),0<=int(row['unique_controls'])<=int(row['n'])<=2055)
    result=dict(actual_checks=len(checks),all_passed=all(checks.values()),checks=checks,job=os.getenv('SLURM_JOB_ID'))
    save_json(DEST/'validation.json',result);print('V2.4 VALIDATION',len(checks),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--diagnostics-only',action='store_true');args=parser.parse_args()
    if args.diagnostics_only:
        require_compute()
        from v2_4_information_followup import main as information_support
        information_support();diagnostic_summaries()
        print('DIAGNOSTIC TABLES COMPLETE; full validation still pending',flush=True)
    else:main()
