#!/usr/bin/env python3
"""Bounded 64-worker pilot using the original immutable context refinement."""
import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
import resource
import time
import joblib
import numpy as np
import v3_1_structure_entry_refine as original
from v2_1_near import ROOT,load,sha,save_json,csv_rows
from photdis.stage1 import require_compute
from photdis.v3.model import inference_model
from photdis.v3.systematics import calibration_vectors

EXPECTED_SOURCE='123013843b1296c0df8c43542817ac37306e66eb5da435554580a9c59c35290d'


def main():
    require_compute();p=argparse.ArgumentParser();p.add_argument('--draw',type=int,required=True)
    p.add_argument('--workers',type=int,choices=[64],default=64);a=p.parse_args();assert 0<=a.draw<32
    start=time.monotonic();root=ROOT/'data/v3_1/structure';old=ROOT/'data/v3_0'
    scientific_source=ROOT/'scripts/v3_1_structure_entry_refine.py';assert sha(scientific_source)==EXPECTED_SOURCE
    # The measured48-worker pilot used <4.1GiB maximum child RSS and159.1GiB
    # batch RSS. Reserve5GiB per worker and use <=70% of the actual allocation.
    basis=root/'resource_execution/draw_10/job_61240361.json'
    measured=json.loads(basis.read_text())
    assert measured['execution_complete'] and measured['workers']==48
    assert measured['scientific_source_sha256']==EXPECTED_SOURCE
    assert measured['maximum_child_RSS_KiB']<4.1*1024**2
    memory=os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')
    declared_memory=int(os.getenv('SLURM_MEM_PER_NODE','0'))
    if declared_memory>0:memory=min(memory,declared_memory*1024**2)
    assert a.workers*5*1024**3<=.7*memory
    assert a.workers<=len(os.sched_getaffinity(0))
    original._SOURCE=root/f'entry/draw_{a.draw:02}'
    paths=[original._SOURCE/'complete.json',root/'photometry.npz',old/f'uncertainty_refit/draw_{a.draw:02}/model.joblib',old/'uncertainty/shared_draws.npz',
        scientific_source,ROOT/'docs/v3_1_structure_entry_refinement_protocol.md']
    paths += [ROOT/'src/photdis/v3_1'/n for n in ['structure_entry.py','selection_dust.py','candidate_selection.py']]
    original._INPUTS={str(path.relative_to(ROOT)):sha(path) for path in paths};original._DONE=json.loads(paths[0].read_text())
    for path in paths[1:4]:assert sha(path)==original._DONE['inputs'][str(path.relative_to(ROOT))]
    original._TABLE=load(paths[1]);assert len(original._TABLE['identity'])==512 and (original._TABLE['TYPE']=='PSF').all()
    original._MODEL=inference_model(joblib.load(paths[2]));original._CAL,_=calibration_vectors(original._TABLE,load(paths[3]),a.draw)
    original._ROWS=np.arange(512);original._OUT=root/f'entry_refined/draw_{a.draw:02}';original._OUT.mkdir(parents=True,exist_ok=True)
    freeze=original._OUT/'inputs.json'
    if freeze.exists():assert json.loads(freeze.read_text())==original._INPUTS
    else:save_json(freeze,original._INPUTS)
    complete=original._OUT/'complete.json'
    if complete.exists():raise RuntimeError('Complete structure refinement already exists; do not replace its provenance')
    audit_dir=root/'resource_execution'/f'draw_{a.draw:02}';audit_dir.mkdir(parents=True,exist_ok=True)
    audit=audit_dir/f'job_{os.environ["SLURM_JOB_ID"]}.json'
    record=dict(scientific_inputs=original._INPUTS,scientific_source_sha256=EXPECTED_SOURCE,
        execution_source_sha256=sha(ROOT/'scripts/v3_1_structure_entry_resume64.py'),workers=a.workers,
        memory_limit_bytes=memory,planning_bound_per_worker_GiB=5,resource_basis={str(basis.relative_to(ROOT)):sha(basis)},job=os.getenv('SLURM_JOB_ID'),
        scope='Original per-context function, same input dictionary/outputs/integrand/seeds. Only execution concurrency and resource guards change; original checkpoints and failures retained.')
    save_json(audit,record)
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('fork')) as pool:
        records=list(pool.map(original.one,range(512)))
    csv_rows(original._OUT/'summary.csv',[dict(context=r['context'],row=r['row'],identity=r['identity'],TYPE='PSF',branch=b['branch'],
        original_passed=b['original_passed'],passed=b['passed'],**{k:v for k,v in (b['checks'] or {}).items() if k!='passed'}) for r in records for b in r['results']])
    record.update(elapsed_seconds=time.monotonic()-start,maximum_child_RSS_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,execution_complete=True)
    save_json(audit,record)
    save_json(complete,dict(inputs=original._INPUTS,contexts={f'context_{i}/complete.json':sha(original._OUT/f'context_{i}/complete.json') for i in range(512)},
        outputs={'summary.csv':sha(original._OUT/'summary.csv')},response_draw=a.draw,rows=original._ROWS.tolist(),n=512,all_PSF=True,
        passed=all(r['passed'] for r in records),original_failures=sum(not r['original_passed'] for r in records),unresolved=sum(not r['passed'] for r in records),
        resource_execution=str(audit.relative_to(ROOT)),resource_execution_sha256=sha(audit),job=os.getenv('SLURM_JOB_ID'),
        scope='Same original refinement function and bounded2^23 integrals, resource-only concurrency change; all original files/flags preserved.'))
    print('EXACT STRUCTURE REFINEMENT RESUME COMPLETE',a.draw,record['elapsed_seconds'],flush=True)


if __name__=='__main__':main()
