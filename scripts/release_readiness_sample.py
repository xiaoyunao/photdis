#!/usr/bin/env python3
"""Export the frozen Gaia comparison identity set and a blinded reference frame."""
import json
import os

import numpy as np

from photdis.stage1 import require_compute
from v2_1_near import ROOT, atomic_npz, save_json, sha
from v3_3_sed import verified
from v3_6_vac_distances import read_columns
from final_science_catalog import DEST as FINAL

DEST = ROOT/'data/release_readiness'
FIELDS = ('TYPE','ls_id','source_id','RA','DEC','r','color','region','science','gaia_state','heldout',
          'q','bins','specdis_kpc','spdist_kpc','rvs_label','rvs_feh','rvs_feh_err')


def main():
    require_compute()
    marker = FINAL/'evidence/complete.json'
    r = json.loads(marker.read_text())
    t = read_columns(verified(marker.with_name('selected.npz'),r['outputs']['selected.npz']),FIELDS)
    select = (t['science']==2)&np.isin(t['gaia_state'],(3,4,5))
    j = np.flatnonzero(select)
    j = j[np.argsort(t['source_id'][j])]
    t = {k:v[j] for k,v in t.items()}
    ids = t['source_id']
    if len(ids)!=1150705 or (np.diff(ids)<=0).any() or (t['TYPE']!='PSF').any():
        raise ValueError('Frozen working sample does not have the expected unique PSF identities')
    DEST.mkdir(exist_ok=True,parents=True)
    path = DEST/'gaia_ids.npy'
    np.save(path,ids)
    atomic_npz(DEST/'cohort.npz',**t)
    print('Unique Gaia IDs within selected working sample:',len(ids),flush=True)
    save_json(DEST/'cohort_receipt.json',dict(job=os.getenv('SLURM_JOB_ID'),rows=len(ids),
        final_science_sha256=sha(marker),outputs={p.name:sha(p) for p in (path,DEST/'cohort.npz')}))


if __name__=='__main__':
    main()
