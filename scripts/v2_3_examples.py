#!/usr/bin/env python3
from v2_3_diagnostics import posterior_examples
from v2_3_prepare import ROOT,DEST
from v2_1_near import load,save_json,sha
from photdis.stage1 import require_compute
if __name__=='__main__':
    require_compute();posterior_examples(load(ROOT/'data/v2_1/development.npz'))
    save_json(DEST/'examples_complete.json',dict(source_sha256=sha(ROOT/'scripts/v2_3_diagnostics.py'),csv_sha256=sha(DEST/'tables/posterior_examples.csv')))
