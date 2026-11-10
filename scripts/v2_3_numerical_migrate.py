#!/usr/bin/env python3
"""One explicit numerical-only migration; fitted models are byte-identical."""
import ast
import json
import subprocess
from pathlib import Path
from v2_3_prepare import ROOT,DEST
from v2_1_near import sha,save_json
from photdis.stage1 import require_compute
from v2_3_run import freeze_inputs

OLD={'src/photdis/density_v2_3.py':'70fe5b3fe6d6f059b64de41aff769a1170c7f24daf16d28cd9346973b80658dc',
     'scripts/v2_3_run.py':'9e12d5c595a8995ee0d8c304f5e205dc8ca1e891c6eb2b7e40508578ac1d6ff4'}


def main():
    require_compute();marker=DEST/'numerical_migration.json';current=freeze_inputs()
    if marker.exists():
        assert json.loads(marker.read_text())['replacement_inputs']==current
        print('NUMERICAL MIGRATION already verified',flush=True);return
    frozen=json.loads((DEST/'models_frozen.json').read_text())
    for path,h in frozen['inputs'].items():
        assert h==(OLD[path] if path in OLD else current[path])
    for name,h in frozen['models'].items():assert sha(DEST/(name+'.joblib'))==h
    # No training or conditional-mixture formula change: AST-equivalent methods.
    old=ast.parse(subprocess.check_output(['git','show','6555c0d:src/photdis/density_v2_3.py'],cwd=ROOT,text=True))
    new=ast.parse((ROOT/'src/photdis/density_v2_3.py').read_text())
    def functions(tree):return {n.name:ast.dump(n,include_attributes=False) for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)}
    a,b=functions(old),functions(new)
    assert set(a)==set(b)
    for name in a:
        if name!='predict':assert a[name]==b[name],name
    assert not (DEST/'diagnostics_complete.json').exists()
    archive=DEST/'legacy_candidate_numerics';archive.mkdir(exist_ok=False)
    paths=list((DEST/'chunks').glob('*_candidate_*.npz'))+list((DEST/'chunks').glob('*_candidate_*.json'))
    for role in ('development','reserved','application'):
        paths.extend(p for p in [DEST/f'{role}_candidate.npz',DEST/f'{role}_candidate_complete.json'] if p.exists())
    entries=[]
    for path in paths:
        relative=path.relative_to(DEST);target=archive/relative;target.parent.mkdir(parents=True,exist_ok=True)
        entries.append(dict(path=str(relative),sha256=sha(path)));path.rename(target)
    if (DEST/'reserved_opened.json').exists():
        (archive/'reserved_opened_before_numerical_repair.json').write_bytes((DEST/'reserved_opened.json').read_bytes())
    record=dict(previous_freeze=frozen,replacement_inputs=current,model_hashes_unchanged=frozen['models'],archived=entries,
        reason='Stable survival-CDF differences preserve tiny Gaussian right tails; candidate PDFs saved float64. No fitted parameter, sample, convergence, hyperparameter or conditional-mixture change.',
        reserved_generation_had_started=(DEST/'reserved_opened.json').exists(),
        performance_inspected_before_repair=False,proof='No aggregate diagnostics exist; repair follows analytic Gaussian-tail regression test, not observed predictive scores.')
    save_json(marker,record)
    frozen['inputs']=current;frozen['numerical_migration']=sha(marker)
    save_json(DEST/'models_frozen.json',frozen)
    print('NUMERICAL MIGRATION',len(entries),'archived files; fitted models unchanged',flush=True)


if __name__=='__main__':main()
