#!/usr/bin/env python3
"""Apply posterior latent-sign diagnostics to the existing pilot without refits."""
import json
import os
import joblib
import numpy as np
from v2_1_near import ROOT,load,sha,save_json,csv_rows
from photdis.stage1 import require_compute
from photdis.v3.model import inference_model
from photdis.v3.systematics import calibration_vectors
from photdis.v3.likelihood import normalize_joint
from photdis.v3_1.diagnostics import control_log_density,condition_feh
from photdis.v3_1.latent_flux import negative_flux_support
from v3_0_response import observation


def main():
    require_compute(); d=ROOT/'data/v3_0'; base=ROOT/'data/v3_1/diagnostics/pilot'
    done=json.loads((base/'complete.json').read_text())
    source=d/'cohorts/development.npz'; evidence=d/'evaluation/development.npz'
    model_path=d/'uncertainty_refit/draw_12/model.joblib'; shared_path=d/'uncertainty/shared_draws.npz'
    for path in [source,evidence,model_path,shared_path]:assert sha(path)==done['inputs'][str(path.relative_to(ROOT))]
    table=load(source); ev=load(evidence); model=inference_model(joblib.load(model_path))
    calibration,_=calibration_vectors(table,load(shared_path),12)
    rows=[]; max_fidelity=0.
    for name,digest in done['objects'].items():
        marker=base/name;assert sha(marker)==digest
        record=json.loads(marker.read_text()); assert sha(marker.with_suffix('.npz'))==record['pdf_sha256']
        pdf=load(marker.with_suffix('.npz'))
        for r in record['results']:
            if r['control'] not in ('response12','class_feh'):continue
            j=r['row'];assert int(table['identity'][j])==r['identity'] and table['TYPE'][j]=='PSF'
            o=observation(table,j); grid=pdf[r['control']+'_mu']
            control=dict(calibration_mag=calibration[j]); conditioned=model
            if r['control']=='class_feh':
                control.update(reference_class=r['class_label'],reference_feh=r['reference_feh'],reference_feh_error=ev['rvs_feh_err'][j])
                conditioned,_=condition_feh(model,r['reference_feh'],ev['rvs_feh_err'][j])
            p=normalize_joint(grid,control_log_density(model,o,grid,**control))
            fidelity=float(np.max(abs(p.quantiles()-r['mu'])));max_fidelity=max(fidelity,max_fidelity)
            assert fidelity<1e-9
            value=negative_flux_support(conditioned,o,p,calibration_mag=calibration[j])
            assert 0<=value['any_negative_lower']<=value['any_negative_upper']<=1+1e-12
            row=dict(identity=r['identity'],TYPE='PSF',control=r['control'],reference_class=r['class_label'],
                inclusion_probability=r['inclusion_probability'],any_negative_lower=value['any_negative_lower'],
                any_negative_upper=value['any_negative_upper'],kernel_proxy_max=max(r['kernel_negative_support_proxy_by_band']))
            row.update({f'posterior_negative_{band}':float(p) for band,p in zip(['G','R','I','Z','W1','W2'],value['probability_by_band'])})
            rows.append(row)
        print('LATENT SIGN AUDIT',len(rows),flush=True)
    out=base/'latent_audit';out.mkdir(exist_ok=True)
    csv_rows(out/'posterior_negative_flux.csv',rows)
    paths=[base/'complete.json',source,evidence,model_path,shared_path,ROOT/'scripts/v3_1_latent_audit.py',
        ROOT/'src/photdis/v3_1/latent_flux.py',ROOT/'src/photdis/v3_1/diagnostics.py']
    save_json(out/'complete.json',dict(inputs={str(p.relative_to(ROOT)):sha(p) for p in paths},
        output_sha256=sha(out/'posterior_negative_flux.csv'),views=len(rows),max_quantile_reproduction_delta=max_fidelity,
        scope='Pilot diagnostic only; Gaussian conditional under frozen mean-linearized dust, any-band event bounded without independence assumption. Negative observed flux remains legal. No model refit or positivity correction.',job=os.getenv('SLURM_JOB_ID')))


if __name__=='__main__':main()
