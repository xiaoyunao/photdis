#!/usr/bin/env python3
"""Inspect and refine flagged reference quadrature without overwriting the catalog."""
import argparse
import json
import os
import joblib
import numpy as np
from v2_1_near import ROOT,load,sha,save_json,csv_rows
from photdis.stage1 import require_compute
from photdis.v3.model import inference_model
from photdis.v3.inference import infer_reference
from v3_0_response import observation
from v3_0_validation_inputs import check_freeze
D=ROOT/'data/v3_0'


def main():
    require_compute();p=argparse.ArgumentParser();p.add_argument('--field',type=int,required=True);a=p.parse_args()
    method=check_freeze();base=D/'catalog'/f'field_{a.field:02}'
    source=load(base/'inputs.npz');model=inference_model(joblib.load(D/'sed_fit.joblib'));rows=[];inputs={}
    for marker in sorted((base/'predictions').glob('chunk_*.json')):
        meta=json.loads(marker.read_text())
        if meta['converged']==meta['n']:continue
        path=marker.with_suffix('.npz');assert sha(path)==meta['summary_sha256'];star=load(path)
        inputs[str(marker.relative_to(ROOT))]=sha(marker)
        for i in np.flatnonzero(~star['converged']):
            source_row=int(star['rows'][i]);identity=int(star['identity'][i]);assert source['identity'][source_row]==identity
            for step in [.05,.025]:
                result=infer_reference(model,observation(source,source_row),initial_step=step)
                rows.append(dict(identity=identity,source_row=source_row,initial_step=step,
                    original_probability_delta=float(star['numerical_probability_delta'][i]),
                    original_qnear=float(star['qnear'][i]),refined_qnear=result['qnear'],
                    maximum_joint_change=float(abs(result['class_distance']-star['class_distance'][i]).max()),
                    median_mu_change=float(result['mu_quantiles'][2]-star['mu'][i,2]),
                    **result['posterior'].convergence))
    out=base/'reference_numeric_audit';out.mkdir(exist_ok=True)
    csv_rows(out/'refinements.csv',rows)
    save_json(out/'complete.json',dict(inputs=inputs,method_freeze=method,field=a.field,
        flagged_objects=len({r['identity'] for r in rows}),checks=rows,
        outputs={'refinements.csv':sha(out/'refinements.csv')},
        scope='Numerical-only same-model quadrature check; original flagged catalog and audit preserved. No automatic replacement of physical model or production rows.',job=os.getenv('SLURM_JOB_ID')))
    print('REFERENCE NUMERIC AUDIT',a.field,rows,flush=True)


if __name__=='__main__':main()
