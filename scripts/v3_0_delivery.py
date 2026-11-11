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
    unit=read(ROOT/'manifests/v3_0/final_unit_validation.json');check('mathematical tests',unit['failed']==0 and unit['passed']>=68,unit['passed'])
    for name,digest in unit.get('test_hashes',{}).items():check('test source:'+name,sha(ROOT/name)==digest)
    ensemble=D/'ensemble_products/all32_native_refined';ens=read(ensemble/'complete.json')
    check('sample joint numerical gate',ens['passed'],ens['errors']);check('sample size',ens['n']==26000)
    check('all32 response prior',len(ens['response_draw_ids'])==32)
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
    plan=read(D/'catalog/transfer_plan.json');check('plan linked to ensemble',plan['ensemble_sha256']==sha(ensemble/'complete.json'))
    check('global response pruning bound',plan['discarded_response_mass']<=1e-10)
    summary=read(D/'catalog/summary/complete.json');outputs(D/'catalog/summary',summary)
    check('full original parent',summary['parent']==9390518)
    check('full qualified parent',summary['qualified']==1183990)
    if summary['target_mc_failures']:limitations.append(f"{summary['target_mc_failures']} target candidates fail the frozen final joint Monte Carlo tolerance and remain flagged; no accuracy claim for these rows.")
    total=0;original_numeric_flags=0
    for field,n in enumerate(plan['field_counts']):
        base=D/'catalog'/f'field_{field:02}';prep=read(base/'prepared.json')
        check('admission count:'+str(field),prep['candidate']==n)
        if not n:continue
        audit=read(base/'reference_audit.json');check('reference structural audit:'+str(field),audit['structural_passed'] and audit['n']==n)
        original_numeric_flags+=audit['numerical_failures']
        export=read(base/'candidate_v3_0.json');check('candidate export count:'+str(field),export['rows']==n and export['method_freeze']==method)
        check('candidate compact hash:'+str(field),sha(base/'candidate_v3_0.npz')==export['table_sha256'])
        check('candidate joint index hash:'+str(field),sha(base/'candidate_v3_0_chunks.csv')==export['chunk_index_sha256'])
        total+=n
    check('candidate total',total==1183990)
    if original_numeric_flags:limitations.append(f'{original_numeric_flags} original reference-prior numerical flag(s) retained; isolated quadrature refinements reported separately.')
    response=plan['responses'][0];target_name=f'refit_{response["draw"]:02}_s{response["draws"]}'
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
