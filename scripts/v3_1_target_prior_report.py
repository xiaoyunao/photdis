#!/usr/bin/env python3
"""Same-identity target-prior contrasts with design and sky uncertainty."""
import argparse
import json
import os
import numpy as np
from v2_1_near import ROOT,load,sha,save_json,csv_rows
from photdis.stage1 import require_compute
from photdis.v3.controls import optical_magnitudes
from photdis.v3_1.comparison import match_controls,paired_mean


def main():
    require_compute();p=argparse.ArgumentParser();p.add_argument('--stage',choices=['pilot','diagnostic'],default='diagnostic');a=p.parse_args()
    old=ROOT/'data/v3_1/diagnostics'/a.stage;new=ROOT/'data/v3_1/target_prior'/a.stage
    completed=[json.loads((x/'complete.json').read_text()) for x in (old,new)];records=[]
    for directory,done in zip((old,new),completed):
        for name,digest in done['objects'].items():
            path=directory/name;assert sha(path)==digest;record=json.loads(path.read_text());assert record['inputs']==done['inputs']
            assert sha(path.with_suffix('.npz'))==record['pdf_sha256']
            records.extend(r for r in record['results'] if r['control'] in ['nominal','response12','response12_target'])
    target=[r for r in records if r['control']=='response12_target'];assert len(target)==completed[1]['stars']
    tablepath=ROOT/'data/v3_0/cohorts/development.npz';assert sha(tablepath)==completed[1]['inputs'][str(tablepath.relative_to(ROOT))]
    table=load(tablepath);mag,_=optical_magnitudes(table);apparent=mag[:,1]-2.5*np.log10(table['transmission'][:,1]);color=mag[:,0]-mag[:,1]
    axes={}
    for r in target:
        j=r['row'];assert table['identity'][j]==r['identity'] and table['TYPE'][j]=='PSF';feh=r['reference_feh']
        axes[r['identity']]=dict(all='all',feh='missing' if feh is None else ['poor','middle','rich'][np.digitize(feh,[-1.5,-.5])],
            apparent_r=['lt18','18_20','20_22','ge22'][np.digitize(apparent[j],[18,20,22])],
            intrinsic_gr=['lt0p3','0p3_0p8','0p8_1p3','ge1p3'][np.digitize(color[j],[.3,.8,1.3])],
            optional_bands=''.join(str(int(x)) for x in table['band_mask'][j,[2,4,5]]),sky=str(r['cell']),
            target_field_scope='fitted_field' if r['existing_target_field'] else 'new_cell_hypermean_transfer')
    out=new/'paired_report';out.mkdir(exist_ok=True);outputs={axis:[] for axis in next(iter(axes.values()))};denominators=[]
    for baseline in ['response12','nominal']:
        pairs=match_controls(records,'response12_target',baseline)
        for axis,output in outputs.items():
            for group in sorted({x[axis] for x in axes.values()}):
                for label in [None,-1,0,1,2]:
                    rr=[(r,b) for r,b in pairs if axes[r['identity']][axis]==group and (label is None or r['class_label']==label)]
                    common=dict(control='response12_target',baseline=baseline,axis=axis,group=group,reference_class='all' if label is None else str(label))
                    denominators.append(dict(common,n=len(rr),cells=len({r['cell'] for r,b in rr}),design_weight=sum(r['design_weight'] for r,b in rr)))
                    if not rr:continue
                    def add(metric,values,subset=rr,source='none'):
                        output.append(dict(common,source=source,metric=metric,**paired_mean(values,[r['design_weight'] for r,b in subset],[r['cell'] for r,b in subset])))
                    add('mu50_shift_mag',[r['mu'][2]-b['mu'][2] for r,b in rr])
                    add('qnear_shift',[r['qnear']-b['qnear'] for r,b in rr])
                    add('highq_fraction_change',[float(r['qnear']>.95)-float(b['qnear']>.95) for r,b in rr])
                    for source in ['primary','specdis','spdist','gaia_precise']:
                        selected=[(r,b) for r,b in rr if source in r['references'] and source in b['references']]
                        if not selected:continue
                        for r,b in selected:
                            assert r['references'][source]['mu']==b['references'][source]['mu'] and r['references'][source]['error']==b['references'][source]['error']
                        add('reference_logscore_gain',[r['references'][source]['logscore']-b['references'][source]['logscore'] for r,b in selected],selected,source)
                        add('absolute_mu_error_change',[abs(r['mu'][2]-r['references'][source]['mu'])-abs(b['mu'][2]-b['references'][source]['mu']) for r,b in selected],selected,source)
                        for level,lo,hi in [(68,1,3),(95,0,4)]:
                            add(f'reference_point_agreement{level}_change',[float(r['mu'][lo]<=r['references'][source]['mu']<=r['mu'][hi])-float(b['mu'][lo]<=b['references'][source]['mu']<=b['mu'][hi]) for r,b in selected],selected,source)
                        high=[(r,b) for r,b in selected if b['qnear']>.95]
                        if high:
                            add('baseline_highq_reference_far_frequency',[float(r['references'][source]['mu']>=15) for r,b in high],high,source)
                            add('baseline_highq_model_far_change',[b['qnear']-r['qnear'] for r,b in high],high,source)
                    signed=[(r,b) for r,b in rr if 'gaia' in r and 'gaia' in b]
                    if signed:
                        add('signed_gaia_logscore_gain',[r['gaia']['log_score']-b['gaia']['log_score'] for r,b in signed],signed,'signed_gaia')
                        add('signed_gaia_PIT_tail_change',[float(r['gaia']['pit']<.025 or r['gaia']['pit']>.975)-float(b['gaia']['pit']<.025 or b['gaia']['pit']>.975) for r,b in signed],signed,'signed_gaia')
    for axis,rows in outputs.items():csv_rows(out/(axis+'.csv'),rows)
    csv_rows(out/'denominators.csv',denominators)
    flags=[dict(identity=r['identity'],**r['convergence']) for r in target if not r['convergence']['converged']];save_json(out/'numerical_flags.json',flags)
    checks=dict(exact_pairs=all(len(match_controls(records,'response12_target',b))==len(target) for b in ['nominal','response12']),
        all_native_PSF=all(r['TYPE']=='PSF' for r in records),joint_normalization=all(abs(np.sum(r['class_distance'])-1)<1e-10 for r in records),
        mode_conservation=all(abs(sum(x['mass'] for x in r['modes'])-1)<1e-10 for r in target),flags_retained=len(flags)==completed[1]['nonconverged'])
    paths=[old/'complete.json',new/'complete.json',tablepath,ROOT/'scripts/v3_1_target_prior_report.py',ROOT/'src/photdis/v3_1/comparison.py']
    save_json(out/'complete.json',dict(inputs={str(p.relative_to(ROOT)):sha(p) for p in paths},outputs={p.name:sha(p) for p in out.iterdir() if p.name!='complete.json'},
        job=os.getenv('SLURM_JOB_ID'),stage=a.stage,checks=checks,passed=all(checks.values()),stars=len(target),
        paired_contrasts=sum(len(x) for x in outputs.values()),numerical_flags=len(flags),
        scope='Seen diagnostic sample, original design weights and sky bootstrap. Source-reference-point agreement is not true coverage; model leakage is not independently measured contamination. Target prior is conditional on26000target fit and response12; new cells use sampled hypermean effects.'))
    assert all(checks.values())
    print('TARGET PRIOR PAIRED COMPLETE',len(target),len(flags),flush=True)


if __name__=='__main__':main()
