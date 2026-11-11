#!/usr/bin/env python3
"""Select numerical refinements from actual final joint marginal failures."""
import argparse
import csv
import json
import os
import numpy as np
from v2_1_near import ROOT,load,sha,save_json
from photdis.stage1 import require_compute
D=ROOT/'data/v3_0'


def main():
    require_compute();p=argparse.ArgumentParser();p.add_argument('--report',default='all32_native_refined')
    p.add_argument('--name',default='recovery_precision_wave1');a=p.parse_args()
    base=D/'shared_recovery_summary'/a.report;complete=json.loads((base/'complete.json').read_text())
    numerical=list(csv.DictReader((base/'numerical.csv').open()));weights=list(csv.DictReader((base/'response_weights.csv').open()))
    inputs={str((base/'complete.json').relative_to(ROOT)):sha(base/'complete.json')};cases={8192:[],32768:[],131072:[]};details=[]
    for row in numerical:
        if row['numerical_passed']=='True':continue
        catalog=int(row['catalog']);candidates=[]
        for w in weights:
            if int(w['catalog'])!=catalog:continue
            draw=int(w['response']);weight=float(w['weight'])
            original=D/'shared_recovery'/f'refit_{draw:02}'/f'catalog_{catalog:03}_s2048_native'
            paths=sorted(original.parent.glob(f'catalog_{catalog:03}_s*_native/complete.json'),key=lambda x:int(x.parent.name.split('_s')[1].split('_')[0]),reverse=True)
            path=next(x for x in paths if str(x.relative_to(ROOT)) in complete['inputs'])
            assert sha(path)==complete['inputs'][str(path.relative_to(ROOT))]
            v=json.loads(path.read_text());stars=load(path.parent/'per_star.npz')
            contribution=weight*float(np.max(abs(stars['half_a']-stars['half_b'])))
            candidates.append(dict(draw=draw,catalog=catalog,weight=weight,contribution=contribution,passed=v['passed'],
                current_draws=int(path.parent.name.split('_s')[1].split('_')[0]),source_sha256=sha(path)))
        candidates.sort(key=lambda x:x['contribution'],reverse=True)
        remaining=sum(x['contribution'] for x in candidates);chosen=[]
        for case in candidates:
            # Keep the unrefined response contribution below a fifth of tolerance.
            # A failed sampler is refined regardless of small posterior weight.
            if remaining>.001 or not case['passed']:
                chosen.append(case);remaining-=case['contribution']
        for case in chosen:
            n=next((x for x in cases if x>case['current_draws']),None)
            if n is None:raise RuntimeError('Explicit review required above131072 draws')
            cases[n].append(dict(draw=case['draw'],catalog=catalog,
                reason=f"Final joint half-difference {row['final_marginal_half_delta']}; weighted response contribution {case['contribution']}; numerical-only refinement"))
        details.extend(dict(x,selected=x in chosen,remaining_conditional_bound=remaining) for x in candidates)
    for n,values in cases.items():
        path=D/'execution'/f'{a.name}_s{n}.json'
        if not values:continue
        assert not path.exists()
        save_json(path,dict(inputs=inputs,cases=values,draws=n,physical_changes=False,job=os.getenv('SLURM_JOB_ID')))
        print('PRECISION MANIFEST',path,n,len(values),flush=True)
    save_json(D/'execution'/(a.name+'_audit.json'),dict(inputs=inputs,details=details,
        selection='Refine actual failed-final-marginal cases, retaining unrefined weighted conditional maximum sum <=0.001; preserve all failed original products.'))


if __name__=='__main__':main()
