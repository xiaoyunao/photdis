#!/usr/bin/env python3
"""Refine only numerical PDF grid after measured compression failure."""
import json
import subprocess
from v2_3_prepare import ROOT,DEST
from v2_1_near import sha,save_json
from photdis.stage1 import require_compute
from v2_3_run import freeze_inputs


def main():
    require_compute();marker=DEST/'grid_migration.json';current=freeze_inputs()
    if marker.exists():
        assert json.loads(marker.read_text())['replacement_inputs']==current
        return
    freeze=json.loads((DEST/'models_frozen.json').read_text())
    old_hash='e5977cd03a6643dcfb1fe0939b82a26f367d6d1520b4c51e64174eee69c05d66'
    for path,h in freeze['inputs'].items():assert h==(old_hash if path=='scripts/v2_3_run.py' else current[path])
    old=subprocess.check_output(['git','show','05a3421:scripts/v2_3_run.py'],cwd=ROOT,text=True)
    assert old.replace('np.arange(-30,40.0001,.1)','np.arange(-30,40.0001,.01)')==(ROOT/'scripts/v2_3_run.py').read_text()
    for name,h in freeze['models'].items():assert sha(DEST/(name+'.joblib'))==h
    audit=json.loads((DEST/'precision_audit.json').read_text())
    assert all(r['step_max_q_error']['0.01']<1e-3 for r in audit['results'].values())
    archive=DEST/'coarse_grid_results';archive.mkdir(exist_ok=False)
    paths=list((DEST/'chunks').glob('*_candidate_*.npz'))+list((DEST/'chunks').glob('*_candidate_*.json'))
    paths+=list((DEST/'tables').glob('*.csv'))
    paths=[p for p in paths if p.name not in ('noise_baselines.csv','reference_inventory.csv')]
    for role in ('development','reserved','application'):
        paths.extend(p for p in [DEST/f'{role}_candidate.npz',DEST/f'{role}_candidate_complete.json',DEST/f'{role}_candidate_teacher.npz'] if p.exists())
    paths.extend(p for p in [DEST/'candidate_grid.npz',DEST/'decision.json',DEST/'diagnostics_complete.json',DEST/'far_complete.json'] if p.exists())
    entries=[]
    for path in paths:
        relative=path.relative_to(DEST);target=archive/relative;target.parent.mkdir(parents=True,exist_ok=True)
        entries.append(dict(path=str(relative),sha256=sha(path)));path.rename(target)
    record=dict(previous_freeze=freeze,replacement_inputs=current,archived=entries,
        numerical_grid_change='.1mag to .01mag; target max analytic-q versus histogram mass error<1e-3',
        precision_audit=sha(DEST/'precision_audit.json'),model_hashes_unchanged=freeze['models'],
        scientific_selection_and_parameters_unchanged=True,scientific_scores_not_used_to_choose_grid=True)
    save_json(marker,record);freeze['inputs']=current;freeze['grid_migration']=sha(marker)
    save_json(DEST/'models_frozen.json',freeze)
    print('GRID MIGRATION',len(entries),'files archived; no refit',flush=True)


if __name__=='__main__':main()
