#!/usr/bin/env python3
"""Actual full qualified-catalog counts, preserving common parameter draw IDs."""
import json
import os
import numpy as np
from v2_1_near import ROOT,load,sha,save_json,atomic_npz,csv_rows
from photdis.stage1 import require_compute
from photdis.v3.ensemble_counts import merge_chunk_counts,combine_counts
from v3_0_validation_inputs import check_freeze
D=ROOT/'data/v3_0'


def quantile(x,w):
    i=np.argsort(x);return x[i][np.searchsorted(np.cumsum(w[i])/w.sum(),[.025,.5,.975])]


def count_rows(record,scope):
    mean=record['mean'].reshape(3,6);draws=record['count_draws'].reshape(-1,3,6);w=record['draw_weights'];rows=[]
    for c in [-1,0,1,2]:
        for name,bins in [('0_2',[0]),('2_5',[1]),('5_10',[2]),('near_10',[0,1,2]),('10_20',[3]),('20_50',[4]),('10_50',[3,4]),('above_50',[5]),('all',list(range(6)))]:
            value=draws[:,c,bins].sum(axis=1) if c>=0 else draws[:,:,bins].sum(axis=(1,2))
            avg=mean[c,bins].sum() if c>=0 else mean[:,bins].sum()
            lo,median,hi=quantile(value,w)
            rows.append(dict(scope=scope,class_index=c,distance_group=name,mean=float(avg),lower95=int(lo),median=int(median),upper95=int(hi)))
    return rows


def main():
    require_compute();method=check_freeze();plan_path=D/'catalog/transfer_plan.json';plan=json.loads(plan_path.read_text())
    assert plan['method_freeze']==method
    out=D/'catalog/summary';out.mkdir(exist_ok=True)
    inputs={str(plan_path.relative_to(ROOT)):sha(plan_path)};field_rows=[];counts_rows=[];global_records=[]
    response_weight=np.array([r['retained_weight'] for r in plan['responses']]);reference_mean=np.zeros((3,6));matched_reference_mean=np.zeros((3,6));selected_n=0;selected_near=0.;expected_near=0.;selected_far=0.
    for response in plan['responses']:
        field_counts=[]
        for field,n in enumerate(plan['field_counts']):
            if n==0:continue
            directory=D/'catalog'/f'field_{field:02}'/'population_transfer'/f'refit_{response["draw"]:02}_s{response["draws"]}'
            marker=directory/'complete.json';record=json.loads(marker.read_text())
            assert record['rows']==n and record['integration_passed']
            assert record['inputs']['method_freeze']==method
            assert sha(directory/'counts.npz')==record['counts_sha256']
            inputs[str(marker.relative_to(ROOT))]=sha(marker)
            field_counts.append(load(directory/'counts.npz'))
        global_records.append(merge_chunk_counts(field_counts))
    with np.errstate(divide='ignore'):logweights=np.log(response_weight)
    combined=combine_counts(global_records,logweights);combined['response_draw_ids']=np.array([r['draw'] for r in plan['responses']],'i8')
    assert int(combined['n'])==plan['qualified_parent'];atomic_npz(out/'counts.npz',**combined);counts_rows+=count_rows(combined,'all_qualified_fields')
    half_delta=0.;rows_total=0;identities=[];mc_failures=0;mask_summary={}
    for field,n in enumerate(plan['field_counts']):
        base=D/'catalog'/f'field_{field:02}';prep=json.loads((base/'prepared.json').read_text())
        status=load(base/'parent_status.npz');assert len(status['status'])==prep['original_parent']
        assert sha(base/'inputs.npz')==prep['outputs']['inputs.npz']
        original=load(base/'inputs.npz');assert (original['TYPE']=='PSF').all();identities.append(original['identity'])
        assert int((status['status']==0).sum())==n
        inputs[str((base/'prepared.json').relative_to(ROOT))]=sha(base/'prepared.json')
        if not n:
            field_rows.append(dict(field=field,parent=prep['original_parent'],candidate=0,reference_converged=0,system='unmodeled_north'));continue
        audit=json.loads((base/'reference_audit.json').read_text());assert audit['structural_passed'] and audit['n']==n
        inputs[str((base/'reference_audit.json').relative_to(ROOT))]=sha(base/'reference_audit.json')
        directories=[base/'population_transfer'/f'refit_{r["draw"]:02}_s{r["draws"]}' for r in plan['responses']]
        count=combine_counts([load(p/'counts.npz') for p in directories],logweights);counts_rows+=count_rows(count,f'field_{field:02}')
        atomic_npz(out/f'field_{field:02}_counts.npz',**count)
        field_selected=0;field_leakage=0.;field_near=0.;field_mc_failures=0;flags=dict(multimodal=0,prior_sensitive=0)
        files=sorted(p.name for p in (directories[0]/'chunks').glob('chunk_*.npz') if not p.stem.endswith('_counts'))
        for name in files:
            pieces=[]
            for directory in directories:
                p=directory/'chunks'/name;m=json.loads(p.with_suffix('.json').read_text());assert sha(p)==m['outputs'][name];pieces.append(load(p))
            for p in pieces:np.testing.assert_array_equal(p['identity'],pieces[0]['identity']);assert (p['TYPE']=='PSF').all()
            prob=sum(w*p['probability'] for w,p in zip(response_weight,pieces));ha=sum(w*p['half_a'] for w,p in zip(response_weight,pieces));hb=sum(w*p['half_b'] for w,p in zip(response_weight,pieces))
            half_delta=max(half_delta,float(abs(ha-hb).max()));q=prob[:,:,:3].sum(axis=(1,2));selected=q>.95
            field_mc_failures+=int((abs(ha-hb).max(axis=(1,2))+2*plan['discarded_response_mass']>=.005).sum())
            mask_code=np.sum(pieces[0]['band_mask'].astype('i8')*(1<<np.arange(6))[None,:],axis=1)
            for code in np.unique(mask_code):
                use=mask_code==code
                aggregate=mask_summary.setdefault(int(code),dict(n=0,near95=0,expected_far=0.,joint=np.zeros((3,6))))
                aggregate['n']+=int(use.sum());aggregate['near95']+=int((use&selected).sum())
                aggregate['expected_far']+=float((1-q[use&selected]).sum());aggregate['joint']+=prob[use].sum(axis=0)
            field_selected+=int(selected.sum());field_leakage+=float((1-q[selected]).sum());field_near+=float(q.sum())
            selected_near+=float(q[selected].sum());rows_total+=len(q)
            # Descriptive union flags remain conservative if multiple responses survive.
            flags['multimodal']+=int(np.logical_or.reduce([p['multimodal'] for p in pieces]).sum())
            flags['prior_sensitive']+=int(np.logical_or.reduce([p['prior_sensitive'] for p in pieces]).sum())
        for p in sorted((base/'predictions').glob('chunk_*.npz')):
            if not p.stem.endswith('_joint'):reference_mean+=load(p)['class_distance'].sum(axis=0)
        for weight,directory in zip(response_weight,directories):
            for p in sorted((directory/'reference_prediction').glob('chunk_*.npz')):
                if not p.stem.endswith('_joint'):matched_reference_mean+=weight*load(p)['class_distance'].sum(axis=0)
        selected_n+=field_selected;selected_far+=field_leakage;expected_near+=field_near;mc_failures+=field_mc_failures
        field_rows.append(dict(field=field,parent=prep['original_parent'],candidate=n,reference_converged=audit['numerically_converged'],system='DECam_DR11',
            target_mc_failures=field_mc_failures,near95_selected=field_selected,model_expected_far_leakage=field_leakage,model_expected_near=field_near,**flags))
    assert rows_total==plan['qualified_parent'];assert abs(combined['mean'].sum()-rows_total)<1e-6
    assert len(np.unique(np.concatenate(identities)))==rows_total
    assert abs(combined['covariance'].sum())<1e-4*max(1,float(np.diag(combined['covariance']).sum()))
    csv_rows(out/'fields.csv',field_rows);csv_rows(out/'joint_counts.csv',counts_rows)
    band_rows=[]
    for code,record in sorted(mask_summary.items()):
        label='+'.join(name for i,name in enumerate(['g','r','i','z','W1','W2']) if code&(1<<i))
        for c in range(3):
            for b in range(6):
                band_rows.append(dict(mask_code=code,bands=label,objects=record['n'],near95_selected=record['near95'],
                    model_expected_far_selected=record['expected_far'],class_index=c,distance_bin=b,expected_count=float(record['joint'][c,b])))
    assert sum(x['n'] for x in mask_summary.values())==rows_total
    csv_rows(out/'band_coverage_counts.csv',band_rows)
    csv_rows(out/'prior_comparison.csv',[dict(class_index=c,distance_bin=b,nominal_reference_mean=float(reference_mean[c,b]),matched_response_reference_mean=float(matched_reference_mean[c,b]),target_mean=float(combined['mean'].reshape(3,6)[c,b])) for c in range(3) for b in range(6)])
    save_json(out/'complete.json',dict(method_freeze=method,inputs=inputs,qualified=rows_total,parent=sum(x['parent'] for x in field_rows),
        final_marginal_half_delta=half_delta,target_mc_failures=mc_failures,passed=half_delta+2*plan['discarded_response_mass']<.005,
        selected_near95=selected_n,model_expected_far_leakage=selected_far,model_expected_far_fraction=selected_far/selected_n,
        model_expected_near_retention=selected_near/expected_near,independently_measured_application_contamination=None,
        missing_contamination_reason='No representative independent distance truth across qualified parent; new-sky and member-conditioned results are separate evidence.',
        response_compression_bounds=plan,scope='Full qualified observed catalog: unit weights, conditional transferred joint posterior. No sampling expansion; no intrinsic Galactic density or deployment claim. Finite response support and external calibration remain limitations.',
        outputs={p.name:sha(p) for p in out.iterdir() if p.suffix in ('.npz','.csv')},job=os.getenv('SLURM_JOB_ID')))
    print('FULL CATALOG COUNTS',rows_total,'finalhalf',half_delta,'near95',selected_n,'expected leakage',selected_far,flush=True)


if __name__=='__main__':main()
