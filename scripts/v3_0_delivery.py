#!/usr/bin/env python3
"""Final evidence-linked delivery checks; scientific failures remain explicit."""
import argparse
import json
import os
import numpy as np
from v2_1_near import ROOT,load,sha,save_json
from photdis.stage1 import require_compute
from v3_0_validation_inputs import check_freeze
D=ROOT/'data/v3_0'


def main():
    require_compute();p=argparse.ArgumentParser();p.add_argument('--recovery',default='all32_native_final');a=p.parse_args()
    method=check_freeze();checks=[];inputs={};limitations=[]
    def check(name,condition,value=None):
        checks.append(dict(name=name,passed=bool(condition),value=value))
        if not condition:raise AssertionError(name)
    def read(path):
        inputs[str(path.relative_to(ROOT))]=sha(path)
        return json.loads(path.read_text())
    def outputs(base,record):
        for name,digest in record.get('outputs',{}).items():
            check('hash:'+str((base/name).relative_to(ROOT)),sha(base/name)==digest)
    check('immutable physical method',len(method)==64,method)
    milestone=read(D/'milestone_delivery_audit.json')
    check('actual A-D milestone evidence',milestone['passed'] and milestone['method_freeze']==method)
    check('milestone auditor version',milestone['source_sha256']==sha(ROOT/'scripts/v3_0_milestone_audit.py'))
    for name,digest in milestone['inputs'].items():check('milestone evidence:'+name,sha(ROOT/name)==digest)
    unit=read(ROOT/'manifests/v3_0/final_unit_validation.json');check('mathematical tests',unit['failed']==0 and unit['passed']>=68,unit['passed'])
    for name,digest in unit.get('test_hashes',{}).items():check('test source:'+name,sha(ROOT/name)==digest)
    ensemble=D/'ensemble_products/all32_native_refined';ens=read(ensemble/'complete.json')
    check('sample joint numerical gate',ens['passed'],ens['errors']);check('sample size',ens['n']==26000)
    check('all32 response prior',len(ens['response_draw_ids'])==32)
    cdf=read(ensemble/'cdf_revision_audit.json')
    check('CDF repair preserves densities and counts',cdf['passed'] and cdf['objects']==26000)
    check('CDF repair linked to delivered ensemble',cdf['new_complete_sha256']==sha(ensemble/'complete.json'))
    check('sample count hash',sha(ensemble/'counts.npz')==ens['counts_sha256'])
    count=load(ensemble/'counts.npz');check('sample counts int64',count['count_draws'].dtype==np.int64)
    check('sample fixed total',(count['count_draws'].sum(axis=1)==26000).all())
    check('sample mean total',abs(count['mean'].sum()-26000)<1e-6)
    if ens['response_effective_sample_size']<10:limitations.append('Finite reference-response prior concentrates on too few realizations; continuous calibration/model uncertainty not converged scientifically.')
    design=read(ensemble/'design_diagnostic/complete.json');outputs(ensemble/'design_diagnostic',design)
    check('design sample identities',design['n']==26000);check('design parent',design['represented_parent']==8899334)
    recbase=D/'shared_recovery_summary'/a.recovery;rec=read(recbase/'complete.json');outputs(recbase,rec)
    check('shared recovery coverage',rec['catalogs']==112 and len(rec['response_ids'])==32)
    check('shared recovery numerical gate',rec['numerically_passed']==112,rec['numerically_passed'])
    for name,n in [('reservation',3250),('new_sky',2048),('independent_scale',8192)]:
        base=D/'validation'/name
        phot=read(base/'photometry.json');evidence=read(base/'evidence.json')
        check('prospective freeze:'+name,phot['method_freeze']==method and evidence['method_freeze']==method)
        check('prospective source hash:'+name,sha(base/'photometry.npz')==phot['sha256'])
        check('prospective evidence hash:'+name,sha(base/'evidence.npz')==evidence['sha256'])
        reference=read(D/'evaluation_results'/f'final_{name}_multiband_fit_0/complete.json')
        check('prospective output count:'+name,reference['rows']==n)
        transferbase=D/'transfer_evaluation'/name;transfer=read(transferbase/'complete.json');outputs(transferbase,transfer)
        check('paired transfer count:'+name,transfer['n']==n and transfer['method_freeze']==method)
    numericbase=D/'transfer_evaluation/numerical_audit';numeric=read(numericbase/'complete.json');outputs(numericbase,numeric)
    check('prospective numerical flags',numeric['structurally_passed'] and numeric['rows']==13490)
    for cohort in numeric['cohorts']:
        if cohort['combined_failures']:
            limitations.append(f"{cohort['cohort']}: {cohort['combined_failures']} prospective objects carry numerical flags and remain in scientific scores; these rows do not pass the frozen numerical criterion.")
    for name,digest in numeric['inputs'].items():check('prospective numerical provenance:'+name,sha(ROOT/name)==digest)
    plan=read(D/'catalog/transfer_plan.json');check('plan linked to ensemble',plan['ensemble_sha256']==sha(ensemble/'complete.json'))
    check('global response pruning bound',plan['discarded_response_mass']<=1e-10)
    summary=read(D/'catalog/summary/complete.json');outputs(D/'catalog/summary',summary)
    check('full original parent',summary['parent']==9390518)
    check('full qualified parent',summary['qualified']==1183990)
    if summary['target_mc_failures']:limitations.append(f"{summary['target_mc_failures']} target candidates fail the frozen final joint Monte Carlo tolerance and remain flagged; no accuracy claim for these rows.")
    total=0;original_numeric_flags=0;export_reference_failures=0;export_mc_failures=0;export_failures=0
    response=plan['responses'][0];target_name=f'refit_{response["draw"]:02}_s{response["draws"]}'
    for field,n in enumerate(plan['field_counts']):
        base=D/'catalog'/f'field_{field:02}';prep=read(base/'prepared.json')
        check('admission count:'+str(field),prep['candidate']==n)
        if not n:continue
        audit=read(base/'reference_audit.json');check('reference structural audit:'+str(field),audit['structural_passed'] and audit['n']==n)
        original_numeric_flags+=audit['numerical_failures']
        export=read(base/'candidate_v3_0.json');check('candidate export count:'+str(field),export['rows']==n and export['method_freeze']==method)
        check('candidate compact hash:'+str(field),sha(base/'candidate_v3_0.npz')==export['table_sha256'])
        check('candidate joint index hash:'+str(field),sha(base/'candidate_v3_0_chunks.csv')==export['chunk_index_sha256'])
        table=load(base/'candidate_v3_0.npz')
        check('candidate exact identities:'+str(field),table['identity'].dtype==np.int64 and len(np.unique(table['identity']))==n)
        check('candidate PSF:'+str(field),(table['TYPE']=='PSF').all())
        check('candidate joint masses:'+str(field),table['probability'].shape==(n,3,6) and np.isfinite(table['probability']).all()
              and np.allclose(table['probability'].sum(axis=(1,2)),1.,atol=1e-7,rtol=0))
        check('candidate near mass:'+str(field),np.allclose(table['qnear'],table['probability'][:,:,:3].sum(axis=(1,2)),atol=1e-7,rtol=0))
        ref_failed=~table['reference_numerical_converged'];mc_failed=table['final_half_delta']>=.005;failed=ref_failed|mc_failed
        check('candidate numerical flag union:'+str(field),np.array_equal(~table['numerically_converged'],failed))
        check('candidate numerical flag count:'+str(field),int(failed.sum())==export['numerical_failed'])
        export_reference_failures+=int(ref_failed.sum());export_mc_failures+=int(mc_failed.sum());export_failures+=int(failed.sum())
        if ref_failed.any():
            auditbase=base/'population_transfer'/target_name/'reference_numeric_audit'
            numeric_ref=read(auditbase/'complete.json');outputs(auditbase,numeric_ref)
            check('conditional reference refinement:'+str(field),numeric_ref['method_freeze']==method and numeric_ref['physical_changes'] is False
                  and numeric_ref['flagged_objects']==int(ref_failed.sum()))
            for name,digest in numeric_ref['inputs'].items():check('conditional refinement provenance:'+name,sha(ROOT/name)==digest)
        total+=n
    check('candidate total',total==1183990)
    check('catalog reference flags agree with exports',export_reference_failures==summary['target_reference_numerical_failures'])
    check('catalog MC flags agree with exports',export_mc_failures==summary['target_mc_failures'])
    check('catalog combined flags agree with exports',export_failures==summary['target_numerical_failures'])
    check('catalog numerical acceptance includes both tests',summary['passed']==(export_failures==0))
    if export_reference_failures:limitations.append(f'{export_reference_failures} conditional reference-quadrature flags remain in the exported catalog; isolated same-model refinements are sidecar diagnostics, not silent replacement of original rows.')
    if original_numeric_flags:limitations.append(f'{original_numeric_flags} original reference-prior numerical flag(s) retained; isolated quadrature refinements reported separately.')
    for name in ['reference',target_name]:
        base=D/'scale_example'/name;record=read(base/'complete.json');outputs(base,record)
        check('independent scale members:'+name,record['members']==1442 and record['method_freeze']==method)
    qa=read(ROOT/'manifests/v3_0/figure_delivery_validation.json')
    check('figure format/hash/English checks',qa['passed'] and qa['figures']>=42,qa['figures'])
    check('figure visual review',qa['visually_reviewed']==qa['figures'])
    for name,digest in qa['files'].items():check('figure file:'+name,sha(ROOT/name)==digest)
    for name in ['v3_0_result.md','v3_0_methods.md','v3_0_atlas.md','v3_0_requirements_result.md','v3_0_catalog.md','v3_0_transfer_scope.md']:
        path=ROOT/'docs'/name;check('document:'+name,path.exists() and path.stat().st_size>500);inputs[str(path.relative_to(ROOT))]=sha(path)
    limitations += ['Reference and application selection are conditional; no all-DR11 denominator/effective area.',
        'Original3250reservation has no qualifying real distance evidence; no successful accuracy claim.',
        'Real interval undercoverage, poor-metal scale and narrow-structure recovery failures remain.',
        'Candidate population transfer is modular from26000targets, not full-catalog generative refitting.',
        'Model-expected near/far contamination is distinct from independently measured application contamination.',
        'Northern system, faint populations and much of the far population remain unvalidated; deployment unchanged.']
    save_json(D/'delivery.json',dict(method_freeze=method,checks=checks,passed=all(x['passed'] for x in checks),
        check_count=len(checks),inputs=inputs,scientific_limitations=limitations,
        scope='Complete research-system/artifact delivery; not universal scientific acceptance or replacement of deployed v1.',job=os.getenv('SLURM_JOB_ID')))
    print('DELIVERY CHECKS',len(checks),'PASSED; scientific limitations preserved',flush=True)


if __name__=='__main__':main()
