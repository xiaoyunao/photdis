#!/usr/bin/env python3
"""Publish compact source-bound results, leaving full-member PDFs on NERSC."""
from seq7_common import *
from photdis.stage1 import require_compute
import shutil


def main():
    require_compute();dest=ROOT/'manifests/sequence_mechanisms_20261119';fig=ROOT/'figures/sequence_mechanisms_20261119';dest.mkdir(parents=True,exist_ok=True);fig.mkdir(parents=True,exist_ok=True)
    files={'prepare':['admission.csv','identities.csv','model_measures.csv','complete.json'],
           'predict':['legacy_A_replay.csv','QRF_replay.csv','convergence.csv','quadrature_metrics.csv','precision.json','complete.json'],
           'curves':['support_status.csv','complete.json'],
           'evaluate':['summary.csv','paired.csv','phase_gaps.csv','complete.json'],
           'examples':['selected.csv','prediction_color_residuals.csv','QRF_M13_lowerMS_per_star.csv','QRF_centering_explanation.json','complete.json'],
           'validation':['legacy_scores_replay.csv','grid_precision.csv','style_regression.json','complete.json']}
    outputs={}
    for stage,names in files.items():
        marker=json.loads((OUT/stage/'complete.json').read_text());target=dest/stage;target.mkdir(exist_ok=True)
        for name in names:
            p=OUT/stage/name
            if name!='complete.json':verified(p,marker['outputs'][name])
            shutil.copy2(p,target/name);outputs[str((target/name).relative_to(ROOT))]=sha(p)
    for path in (OUT/'public_data').iterdir():
        target=dest/'data'/path.name;target.parent.mkdir(exist_ok=True);shutil.copy2(path,target);outputs[str(target.relative_to(ROOT))]=sha(path)
    for path in (OUT/'figures').iterdir():
        shutil.copy2(path,fig/path.name);outputs[str((fig/path.name).relative_to(ROOT))]=sha(path)
    sky=ROOT/'figures/sky_sequence_20261119/selected_channels/rgb_selected_fixed_gain.png';expected='411b0086e9d4c6f9f208805f02f064e2c24bb117d61293083b9bb51dc7faaed5';assert sha(sky)==expected
    save_json(dest/'release.json',dict(config=CH,job=os.getenv('SLURM_JOB_ID'),private_root=str(OUT),production=C['production'],style=check_templates(),sky_unchanged_sha256=expected,outputs=outputs,source={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'scripts').glob('seq7_*.py')},scope='five controls only; no production/model/manuscript edits; source PDF caches and shared per-member draws remain private on NERSC'))
    print('Published',len(outputs),'bound outputs',flush=True)


if __name__=='__main__':main()
