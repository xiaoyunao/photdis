#!/usr/bin/env python3
"""Summarize actual matched mechanism controls; no correction is fitted here."""
import argparse
import json
import os
import numpy as np
from v2_1_near import ROOT,sha,save_json,csv_rows,load
from photdis.stage1 import require_compute
from photdis.v3_1.comparison import match_controls,paired_mean


def main():
    require_compute(); p=argparse.ArgumentParser(); p.add_argument('--stage',choices=['pilot','diagnostic'],default='pilot'); a=p.parse_args()
    base=ROOT/'data/v3_1/diagnostics'/a.stage; completion=json.loads((base/'complete.json').read_text())
    assert sha(base/'cohort.json') == completion['cohort_sha256']
    records=[]
    for name,digest in completion['objects'].items():
        assert sha(base/name) == digest
        marker=json.loads((base/name).read_text()); assert marker['inputs'] == completion['inputs']
        assert sha((base/name).with_suffix('.npz')) == marker['pdf_sha256']
        records.extend(marker['results'])
    assert len(records) == completion['views']
    out=base/'paired_report';out.mkdir(exist_ok=True)
    rows=[]; modes=[]; scores=[]; flags=[]
    controls=sorted({r['control'] for r in records})
    for control in controls:
        pairs=match_controls(records,control)
        for label in [None,-1,0,1,2]:
            selected=[(r,b) for r,b in pairs if label is None or r['class_label']==label]
            if not selected:continue
            name='all' if label is None else str(label)
            def summarize(metric,values,rr=selected,reference='none'):
                return dict(control=control,baseline='response12',reference_class=name,metric=metric,reference=reference,
                    **paired_mean(values,[r['design_weight'] for r,_ in rr],[r['cell'] for r,_ in rr]))
            rows.append(summarize('mu50_shift_mag',[r['mu'][2]-b['mu'][2] for r,b in selected]))
            rows.append(summarize('qnear_shift',[r['qnear']-b['qnear'] for r,b in selected]))
            for source in ('primary','specdis','spdist','gaia_precise'):
                rr=[(r,b) for r,b in selected if source in r['references'] and source in b['references']]
                if not rr:continue
                for r,b in rr:
                    assert r['references'][source]['mu'] == b['references'][source]['mu']
                    assert r['references'][source]['error'] == b['references'][source]['error']
                rows.append(summarize('reference_logscore_gain',[r['references'][source]['logscore']-b['references'][source]['logscore'] for r,b in rr],rr,source))
                rows.append(summarize('absolute_mu50_error_change',[abs(r['mu'][2]-r['references'][source]['mu'])-abs(b['mu'][2]-b['references'][source]['mu']) for r,b in rr],rr,source))
                for level,lo,hi in [(68,1,3),(95,0,4)]:
                    delta=[int(r['mu'][lo]<=r['references'][source]['mu']<=r['mu'][hi])-int(b['mu'][lo]<=b['references'][source]['mu']<=b['mu'][hi]) for r,b in rr]
                    rows.append(summarize(f'reference_point_agreement{level}_change',delta,rr,source))
            rr=[(r,b) for r,b in selected if 'gaia' in r and 'gaia' in b]
            if rr: rows.append(summarize('signed_gaia_logscore_gain',[r['gaia']['log_score']-b['gaia']['log_score'] for r,b in rr],rr,'signed_gaia'))
    for r in records:
        assert r['TYPE']=='PSF'
        significant=[m for m in r['modes'] if m['mass']>.05]
        modes.append(dict(identity=r['identity'],control=r['control'],reference_class=r['class_label'],
            total_modes=len(r['modes']),significant_modes=len(significant),
            mode_mass_sum=sum(m['mass'] for m in r['modes']),
            high_single_modes=sum(m['binary_probability']<.1 for m in significant),
            high_binary_modes=sum(m['binary_probability']>.9 for m in significant),
            dominant_classes=len({int(np.argmax(m['class_probability'])) for m in significant}),
            dominant_parent_components=len({m['dominant_parent'] for m in significant}),
            negative_kernel_support_proxy_max=max(r['kernel_negative_support_proxy_by_band'])))
        if not r['convergence']['converged']:flags.append(dict(identity=r['identity'],control=r['control'],**r['convergence']))
        for source,ref in r['references'].items():
            scores.append(dict(identity=r['identity'],TYPE=r['TYPE'],control=r['control'],reference_class=r['class_label'],
                source=source,reference_mu=ref['mu'],predicted_mu50=r['mu'][2],qnear=r['qnear'],
                model_expected_far=1-r['qnear'],reference_far=ref['mu']>=15,
                high_qnear=r['qnear']>.95,reference_logscore=ref['logscore'],
                inclusion_probability=r['inclusion_probability'],cell=r['cell']))
    csv_rows(out/'paired.csv',rows);csv_rows(out/'mode_attribution.csv',modes);csv_rows(out/'reference_scores.csv',scores)
    save_json(out/'numerical_flags.json',flags)
    frozen_path=ROOT/'data/v3_0/evaluation_results/development_multiband_fit_0/per_star.npz'
    frozen_marker=json.loads(frozen_path.with_name('complete.json').read_text())
    assert sha(frozen_path)==frozen_marker['outputs'][frozen_path.name]
    frozen=load(frozen_path); index={int(v):i for i,v in enumerate(frozen['identity'])}
    fidelity=[]
    for r in records:
        if r['control']!='nominal':continue
        j=index[r['identity']]
        fidelity.append(dict(identity=r['identity'],
            max_mu_quantile_change=float(np.max(abs(np.asarray(r['mu'])-frozen['mu'][j]))),
            qnear_change=float(abs(r['qnear']-frozen['qnear'][j])),
            max_joint_change=float(np.max(abs(np.asarray(r['class_distance'])-frozen['class_distance'][j])))))
    csv_rows(out/'frozen_nominal_fidelity.csv',fidelity)
    checks=dict(exact_views=len(records)==completion['views'],native_psf=all(r['TYPE']=='PSF' for r in records),
        joint_normalization=all(abs(np.sum(r['class_distance'])-1)<1e-10 for r in records),
        mode_conservation=all(abs(m['mode_mass_sum']-1)<1e-10 for m in modes),
        flags_retained=len(flags)==completion['nonconverged'],
        frozen_nominal_fidelity=all(r['max_mu_quantile_change']<=.03 and r['qnear_change']<=.003 and r['max_joint_change']<=.003 for r in fidelity))
    save_json(out/'complete.json',dict(inputs={str(x.relative_to(ROOT)):sha(x) for x in [base/'complete.json',frozen_path,ROOT/'scripts/v3_1_diagnostic_report.py',ROOT/'src/photdis/v3_1/comparison.py']},
        outputs={x.name:sha(x) for x in out.iterdir() if x.name!='complete.json'},checks=checks,passed=all(checks.values()),
        job=os.getenv('SLURM_JOB_ID'),stage=a.stage,
        interpretation='Seen, stratified mechanism contrasts. Sky-block bootstrap is conditional on source selection; small pilots cannot establish calibrated domains. Truth-assisted improvements are not production performance. Reference-point agreement is not latent true-distance coverage; model leakage is not measured population contamination.'))
    assert all(checks.values())
    print('PAIRED DIAGNOSTICS',a.stage,len(records),'views',len(rows),'contrasts',len(flags),'numerical flags',flush=True)


if __name__=='__main__':main()
