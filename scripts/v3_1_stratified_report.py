#!/usr/bin/env python3
"""Seen-development stratified paired evidence and mode support; no model fit."""
import argparse
import json
import os
import joblib
import numpy as np
from v2_1_near import ROOT,load,sha,save_json,csv_rows
from photdis.stage1 import require_compute
from photdis.v3.controls import optical_magnitudes
from photdis.v3_1.comparison import match_controls,paired_mean


def main():
    require_compute(); p=argparse.ArgumentParser(); p.add_argument('--stage',default='diagnostic',choices=['pilot','diagnostic']); a=p.parse_args()
    base=ROOT/'data/v3_1/diagnostics'/a.stage; out=base/'stratified_report';out.mkdir(exist_ok=True)
    completion=json.loads((base/'complete.json').read_text()); records=[]
    for name,digest in completion['objects'].items():
        path=base/name; assert sha(path)==digest; record=json.loads(path.read_text())
        assert sha(path.with_suffix('.npz'))==record['pdf_sha256'];records.extend(record['results'])
    paths=[ROOT/'data/v3_0/cohorts/development.npz',ROOT/'data/v3_0/uncertainty_refit/draw_12/model.joblib']
    for path in paths: assert sha(path)==completion['inputs'][str(path.relative_to(ROOT))]
    table=load(paths[0]); model=joblib.load(paths[1]); mag,error=optical_magnitudes(table)
    r=mag[:,1]-2.5*np.log10(table['transmission'][:,1]); color=mag[:,0]-mag[:,1]
    baseline={x['identity']:x for x in records if x['control']=='response12'}
    axes={}; counts=[]
    for identity,b in baseline.items():
        j=b['row']; assert table['identity'][j]==identity and table['TYPE'][j]=='PSF'
        feh=b['reference_feh']; optional=table['band_mask'][j,[2,4,5]]
        axes[identity]=dict(all='all',feh=('missing' if feh is None else ['poor','middle','rich'][np.digitize(feh,[-1.5,-.5])]),
            apparent_r=['lt18','18_20','20_22','ge22'][np.digitize(r[j],[18,20,22])],
            intrinsic_gr=['lt0p3','0p3_0p8','0p8_1p3','ge1p3'][np.digitize(color[j],[.3,.8,1.3])],
            optional_bands=''.join(str(int(x)) for x in optional),sky=str(b['cell']))
    controls=sorted({r['control'] for r in records}); all_rows={axis:[] for axis in next(iter(axes.values()))}
    for control in controls:
        pairs=match_controls(records,control)
        for axis,output in all_rows.items():
            for group in sorted({x[axis] for x in axes.values()}):
                for label in [None,-1,0,1,2]:
                    rr=[(v,b) for v,b in pairs if axes[v['identity']][axis]==group and (label is None or v['class_label']==label)]
                    common=dict(control=control,baseline='response12',axis=axis,group=group,reference_class='all' if label is None else str(label))
                    counts.append(dict(common,n=len(rr),design_weight=sum(v['design_weight'] for v,b in rr),
                        cells=len({v['cell'] for v,b in rr})))
                    if not rr:continue
                    def append(metric,values,selected=rr,source='none'):
                        output.append(dict(common,source=source,metric=metric,
                            **paired_mean(values,[v['design_weight'] for v,b in selected],[v['cell'] for v,b in selected])))
                    append('mu50_shift_mag',[v['mu'][2]-b['mu'][2] for v,b in rr])
                    for source in ['primary','specdis','spdist','gaia_precise']:
                        selected=[(v,b) for v,b in rr if source in v['references'] and source in b['references']]
                        if not selected:continue
                        append('absolute_mu_error_change',[abs(v['mu'][2]-v['references'][source]['mu'])-abs(b['mu'][2]-b['references'][source]['mu']) for v,b in selected],selected,source)
                        append('reference_logscore_gain',[v['references'][source]['logscore']-b['references'][source]['logscore'] for v,b in selected],selected,source)
                        # Same baseline-selected high-q sample, not two different
                        # samples silently described as a paired contamination test.
                        high=[(v,b) for v,b in selected if b['qnear']>.95]
                        if high:
                            append('baseline_highq_reference_far_frequency',[float(v['references'][source]['mu']>=15) for v,b in high],high,source)
                            append('baseline_highq_model_far_change',[b['qnear']-v['qnear'] for v,b in high],high,source)
                    selected=[(v,b) for v,b in rr if 'gaia' in v and 'gaia' in b]
                    if selected:
                        append('signed_gaia_logscore_gain',[v['gaia']['log_score']-b['gaia']['log_score'] for v,b in selected],selected,'signed_gaia')
                        append('signed_gaia_PIT_tail_change',[float(v['gaia']['pit']<.025 or v['gaia']['pit']>.975)-float(b['gaia']['pit']<.025 or b['gaia']['pit']>.975) for v,b in selected],selected,'signed_gaia')
            csv_rows(out/(axis+'.csv'),output)
            print('STRATIFIED',control,axis,len(output),flush=True)
    csv_rows(out/'denominators.csv',counts)
    support=[]
    for value in records:
        if value['control']=='nominal':continue
        for i,mode in enumerate(value['modes']):
            parent=mode['dominant_parent']
            support.append(dict(identity=value['identity'],TYPE=value['TYPE'],control=value['control'],reference_class=value['class_label'],
                mode=i,mass=mode['mass'],mu=mode['mu'],weak_basin=mode['weak_basin'],
                binary_probability=mode['binary_probability'],dominant_parent=parent,
                dominant_parent_probability=mode['dominant_parent_probability'],effective_parent_components=mode['effective_parent_components'],
                dominant_parent_effective_reference_sources=float(model.effective_reference_sources[parent]),
                dominant_parent_missing_class_fraction=float(model.missing_class_reference_fraction[parent]),
                class_dwarf_probability=mode['class_probability'][0],class_evolved_probability=mode['class_probability'][1],
                class_other_probability=mode['class_probability'][2]))
    csv_rows(out/'mode_reference_support.csv',support)
    paths += [base/'complete.json',ROOT/'scripts/v3_1_stratified_report.py',ROOT/'src/photdis/v3_1/comparison.py']
    save_json(out/'complete.json',dict(inputs={str(p.relative_to(ROOT)):sha(p) for p in paths},
        outputs={p.name:sha(p) for p in out.glob('*.csv')},stars=len(baseline),views=len(records),
        job=os.getenv('SLURM_JOB_ID'),stage=a.stage,
        scope='Seen diagnostic design-weighted paired contrasts, fixed bins, sky-block bootstrap. Single-cell intervals absent. Unknown class separate. Baseline-high-q frequency uses reference-point labels and is not independently measured application contamination.',
        support_scope='Effective training sources of each dominant parent kernel only; not an exact posterior effective source count. Truth-assisted controls remain diagnostic.'))


if __name__=='__main__':main()
