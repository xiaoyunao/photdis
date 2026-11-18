#!/usr/bin/env python3
"""Finish source audits, calibration-disjoint measurements and figure validation."""
from photdis.stage1 import require_compute
from sv18_parallel import run

if __name__=='__main__':
    require_compute()
    for stage,args in (('provenance',()),('cohorts',()),('populations',('--exclude-calibration',)),
                        ('oracle',()),('figures',()),('verify',())):
        if run(stage,args):
            raise SystemExit(1)
