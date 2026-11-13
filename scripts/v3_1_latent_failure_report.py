#!/usr/bin/env python3
"""Join latent-sign support with same-object development errors and modes."""
import csv
import json
import os
import numpy as np
from v2_1_near import ROOT,sha,save_json,csv_rows
from photdis.stage1 import require_compute
from photdis.v3_1.comparison import paired_mean


def main():
    require_compute();base=ROOT/'data/v3_1/diagnostics/diagnostic';latent=base/'latent_full';out=base/'latent_failure_report';out.mkdir(exist_ok=True)
    paths=[base/'complete.json',latent/'complete.json',latent/'posterior_negative_flux.csv',ROOT/'scripts/v3_1_latent_failure_report.py',ROOT/'src/photdis/v3_1/comparison.py']
    inputs={str(p.relative_to(ROOT)):sha(p) for p in paths};done=json.loads(paths[0].read_text());ld=json.loads(paths[1].read_text())
    assert sha(paths[2])==ld['outputs'][paths[2].name]
    metadata={}
    for name,digest in done['objects'].items():
        path=base/name;assert sha(path)==digest
        for r in json.loads(path.read_text())['results']:
            if r['control'] in ('response12','class_feh'):
                key=r['identity'],r['control'];assert key not in metadata;metadata[key]=r
    with paths[2].open() as stream:latent_rows=list(csv.DictReader(stream))
    rows=[];score_rows=[];indexed={}
    for l in latent_rows:
        identity=int(l['identity']);key=identity,l['control'];r=metadata[key]
        assert r['TYPE']==l['TYPE']=='PSF' and r['cell']==int(l['cell'])
        assert abs(r['inclusion_probability']-float(l['inclusion_probability']))<1e-14
        if l['supported']!='True':continue
        lower=float(l['any_negative_lower']);upper=float(l['any_negative_upper'])
        group='below0.001' if upper<.001 else '0.001to0.01' if upper<.01 else '0.01to0.1' if upper<.1 else 'atleast0.1'
        modes=[m for m in r['modes'] if m['mass']>.05]
        item=dict(identity=identity,TYPE='PSF',control=r['control'],reference_class=r['class_label'],cell=r['cell'],
            design_weight=r['design_weight'],negative_group=group,negative_lower=lower,negative_upper=upper,
            W1_negative=float(l['posterior_negative_W1']),W2_negative=float(l['posterior_negative_W2']),
            qnear=r['qnear'],mu50=r['mu'][2],significant_modes=len(modes),
            high_binary_mode_mass=sum(m['mass'] for m in modes if m['binary_probability']>.9),
            multiple_dominant_classes=float(len({int(np.argmax(m['class_probability'])) for m in modes})>1),
            maximum_mode_mass=max(m['mass'] for m in r['modes']),reference_channels=len(r['references']))
        rows.append(item);indexed[key]=item
        for source,ref in r['references'].items():
            score_rows.append(dict(**item,source=source,reference_mu=ref['mu'],reference_error=ref['error'],
                absolute_mu_error=abs(r['mu'][2]-ref['mu']),severe_abs_mu_gt1=float(abs(r['mu'][2]-ref['mu'])>1),
                reference_logscore=ref['logscore'],point_agreement68=float(r['mu'][1]<=ref['mu']<=r['mu'][3]),
                high_q_reference_far=float(r['qnear']>.95 and ref['mu']>=15),reference_far=float(ref['mu']>=15)))
        if 'gaia' in r:
            pit=r['gaia']['pit'];score_rows.append(dict(**item,source='signed_gaia',gaia_logscore=r['gaia']['log_score'],gaia_PIT_tail=float(pit<.025 or pit>.975)))
    summaries=[]
    for control in ('response12','class_feh'):
        for label in ('all',-1,0,1,2):
            for group in ('all','below0.001','0.001to0.01','0.01to0.1','atleast0.1'):
                selected=[r for r in rows if r['control']==control and (label=='all' or r['reference_class']==label) and (group=='all' or r['negative_group']==group)]
                for metric in ('negative_lower','negative_upper','W1_negative','W2_negative','significant_modes','high_binary_mode_mass','multiple_dominant_classes','qnear'):
                    summaries.append(dict(control=control,reference_class=label,negative_group=group,source='none',metric=metric,
                        **paired_mean([r[metric] for r in selected],[r['design_weight'] for r in selected],[r['cell'] for r in selected])))
                for source in ('primary','specdis','spdist','gaia_precise','signed_gaia'):
                    ss=[r for r in score_rows if r['source']==source and r['control']==control and (label=='all' or r['reference_class']==label) and (group=='all' or r['negative_group']==group)]
                    metrics=('gaia_logscore','gaia_PIT_tail') if source=='signed_gaia' else ('absolute_mu_error','severe_abs_mu_gt1','reference_logscore','point_agreement68','high_q_reference_far')
                    for metric in metrics:
                        summaries.append(dict(control=control,reference_class=label,negative_group=group,source=source,metric=metric,
                            **paired_mean([r[metric] for r in ss],[r['design_weight'] for r in ss],[r['cell'] for r in ss])))
    paired=[]
    for identity,control in indexed:
        if control!='class_feh':continue
        a=indexed[identity,'response12'];b=indexed[identity,control]
        item=dict(identity=identity,TYPE='PSF',reference_class=b['reference_class'],cell=b['cell'],design_weight=b['design_weight'],
            negative_upper_change=b['negative_upper']-a['negative_upper'],negative_lower_change=b['negative_lower']-a['negative_lower'],
            mu50_change=b['mu50']-a['mu50'],qnear_change=b['qnear']-a['qnear'])
        aa=metadata[identity,'response12'];bb=metadata[identity,control]
        for source in ('primary','specdis','spdist','gaia_precise'):
            if source not in aa['references'] or source not in bb['references']:continue
            assert aa['references'][source]['mu']==bb['references'][source]['mu'];ref=aa['references'][source]['mu']
            paired.append(dict(**item,source=source,absolute_mu_error_change=abs(bb['mu'][2]-ref)-abs(aa['mu'][2]-ref),
                reference_logscore_change=bb['references'][source]['logscore']-aa['references'][source]['logscore']))
    for name,values in [('objects.csv',rows),('summary.csv',summaries),('paired_class_feh.csv',paired)]:csv_rows(out/name,values)
    keys=sorted(set().union(*(r.keys() for r in score_rows)));csv_rows(out/'scores.csv',[{k:r.get(k) for k in keys} for r in score_rows])
    save_json(out/'complete.json',dict(inputs=inputs,outputs={p.name:sha(p) for p in out.glob('*.csv')},job=os.getenv('SLURM_JOB_ID'),
        views=len(rows),unsupported=len(latent_rows)-len(rows),all_PSF=True,paired_reference_rows=len(paired),
        scope='Exploratory same-identity seen-development associations, not causal attribution or new validation. Native latent negative-flux bounds differ from legal negative measured flux. Class+FeH is a truth-assisted diagnostic; source-correlated references are kept separate. High-q/reference-far joint fraction is not application contamination.'))


if __name__=='__main__':main()
