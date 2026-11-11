#!/usr/bin/env python3
"""Use exact parallel likelihood evaluations in the frozen evidence driver."""
import os
from v2_1_near import ROOT,sha
from v3_0_parallel_bridge import parallel_bridge
import v3_0_evidence_refine as original


def main():
    workers=int(os.getenv('SLURM_CPUS_PER_TASK','32'))
    original.structured_bridge=lambda *args,**kwargs:parallel_bridge(*args,workers=workers,**kwargs)
    original_save=original.save_json
    def save(path,record):
        if path.name=='complete.json':
            record['parallel_evaluation']=dict(workers=workers,exact_frozen_algorithm=True,
                code_sha256={name:sha(ROOT/name) for name in ['scripts/v3_0_evidence_parallel.py','scripts/v3_0_parallel_bridge.py']})
        return original_save(path,record)
    original.save_json=save
    original.main()


if __name__=='__main__':main()
