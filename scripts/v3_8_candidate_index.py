#!/usr/bin/env python3
"""Publish the evaluated candidate membership and unknown-gate ownership."""
import json
from pathlib import Path

import numpy as np

from photdis.stage1 import require_compute
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_6_catalog_projection import OUT as PROJECTION
from v3_7_selection_replay import OUT as SELECTION, probabilities
from v3_8_candidate import gates
from v3_8_direct_figures import catalog

OUT = ROOT/'data/v3_8/candidate_index'
DISTANCE = (0.,1.,5.,10.,np.inf)
NAMES = ('evaluated_reject','published_primary','confirmed_addition','missing_nominal_gate')


def main():
    require_compute()
    OUT.mkdir(parents=True,exist_ok=True)
    marker = OUT/'complete.json'
    if marker.exists():
        raise FileExistsError(marker)
    p = catalog()
    q = probabilities(p,gates())
    dwarf = p['dwarf_selection_score']
    primary = (dwarf>=.8)&(q>=.90)
    np.testing.assert_array_equal(p['primary'],primary)
    verified = (dwarf>=.65)&(q>=.90)
    missing = (dwarf>=.65)&~np.isfinite(q)
    status = np.zeros(len(q),'u1')
    status[primary] = 1
    status[verified&~primary] = 2
    status[missing] = 3
    d = p['qrf_quantiles_kpc'][:,2]
    keep = status!=0
    atomic_npz(OUT/'membership.npz',catalog_row=p['catalog_row'][keep],ls_id=p['ls_id'][keep],
               TYPE=np.full(keep.sum(),'PSF'),status=status[keep],qrf_median_kpc=d[keep])
    rows = [dict(status=name,rows=int((status==i).sum()),lo_kpc=lo,hi_kpc=hi,
                 shell_rows=int(np.sum((status==i)&(d>=lo)&(d<hi))))
            for i,name in enumerate(NAMES) for lo,hi in zip(DISTANCE[:-1],DISTANCE[1:])]
    for name in NAMES:
        members = [r for r in rows if r['status']==name]
        assert members[0]['rows']==sum(r['shell_rows'] for r in members),name
    csv_rows(OUT/'membership.csv',rows)
    paths = (SELECTION/'complete.json',PROJECTION/'complete.json',Path(__file__).resolve())
    save_json(marker,dict(inputs={str(x.relative_to(ROOT)):sha(x) for x in paths},
        status_codes=dict(enumerate(NAMES)),excluded_morphology=dict(non_psf=0,unverified_type=0),
        note='Sparse index omits known rejects. Unknown-gate rows are included with code 3 and carry no decision.',
        outputs={f.name:sha(f) for f in OUT.iterdir() if f.is_file() and f!=marker}))


if __name__=='__main__':
    main()
