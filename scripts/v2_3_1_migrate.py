#!/usr/bin/env python3
"""Journaled inference-only correction; never reuse old candidate caches."""
import ast
import json
import subprocess
from v2_3_prepare import ROOT,DEST
from v2_1_near import sha,save_json
from v2_3_run import freeze_inputs,candidate_inference_stamp
from photdis.stage1 import require_compute


def functions(source):
    return {n.name:ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef)}


def main():
    require_compute();marker=DEST/'inference_plus_g_migration.json';current=freeze_inputs()
    if marker.exists():
        m=json.loads(marker.read_text());assert m['replacement_inputs']==current
        for n,h in m['models'].items():assert sha(DEST/(n+'.joblib'))==h
        frozen=m['previous_freeze'];frozen['inputs']=current;frozen['inference_correction']=sha(marker)
        save_json(DEST/'models_frozen.json',frozen)
        return
    pending=DEST/'inference_plus_g_pending.json';archive=DEST/'inference_minus_g_archive'
    if not pending.exists():
        freeze=json.loads((DEST/'models_frozen.json').read_text())
        for path,h in freeze['inputs'].items():
            if path in ('src/photdis/density_v2_3.py','scripts/v2_3_run.py'):
                old=subprocess.check_output(['git','show','569d6ff:'+path],cwd=ROOT)
                import hashlib
                assert hashlib.sha256(old).hexdigest()==h
                before=functions(old.decode());after=functions((ROOT/path).read_text())
                keep=('fit','expectation','observation_covariance','conditional_latent','predict') if 'density' in path else ('fit','support')
                for name in keep:assert before[name]==after[name],name
            else:assert current[path]==h,path
        for n,h in freeze['models'].items():assert sha(DEST/(n+'.joblib'))==h
        paths=list((DEST/'chunks').glob('*_candidate_*.npz'))+list((DEST/'chunks').glob('*_candidate_*.json'))
        paths+=list((DEST/'tables').glob('*.csv'))
        paths=[p for p in paths if p.name not in ('noise_baselines.csv','reference_inventory.csv')]
        for role in ('development','reserved','application'):
            paths.extend(p for p in [DEST/f'{role}_candidate.npz',DEST/f'{role}_candidate_complete.json',DEST/f'{role}_candidate_teacher.npz'] if p.exists())
        paths.extend(p for p in [DEST/'decision.json',DEST/'diagnostics_complete.json',DEST/'far_complete.json',DEST/'validation.json',DEST/'models_frozen.json',DEST/'reserved_opened.json',DEST/'metal_poor_pairs.npz',DEST/'metal_poor_matching.json'] if p.exists())
        controls=list((DEST/'chunks').glob('*_control_*'))+[DEST/f'{r}_control.npz' for r in ('development','reserved','application')]
        record=dict(previous_freeze=freeze,replacement_inputs=current,models=freeze['models'],
            controls={str(p.relative_to(DEST)):sha(p) for p in controls},
            archived=[dict(path=str(p.relative_to(DEST)),sha256=sha(p)) for p in paths],
            inference=candidate_inference_stamp(),already_inspected_reserve=True,no_refit=True,
            reason='Correct forward shared-g covariance sign; original training and QRF unchanged')
        save_json(pending,record)
    record=json.loads(pending.read_text());assert record['replacement_inputs']==current
    for item in record['archived']:
        path=DEST/item['path'];target=archive/item['path'];target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert sha(target)==item['sha256'] and not path.exists()
        else:assert sha(path)==item['sha256'];path.rename(target)
    save_json(marker,record)
    freeze=record['previous_freeze'];freeze['inputs']=current;freeze['inference_correction']=sha(marker)
    save_json(DEST/'models_frozen.json',freeze)
    print('INFERENCE CORRECTION',len(record['archived']),'archived; models/QRF unchanged',flush=True)


if __name__=='__main__':main()
