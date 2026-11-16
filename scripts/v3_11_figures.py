#!/usr/bin/env python3
"""Render the trial and baseline cohorts with the locked v3.8 templates."""

import json
import os

import numpy as np

from photdis.reference_v1_4 import unique_join
from photdis.stage1 import require_compute
from v2_1_near import ROOT, save_json, sha
from v3_8_direct_figures import STYLE, plt, render_reference_figures
from v3_10_evidence import DEST as EVIDENCE
from v3_10_figures import check_templates
from v3_10_working_sample import DOMAINS, SCORE
from v3_11_layers import read_receipt
from v3_11_selection import DEST as TRIAL

DEST = ROOT/'data/v3_11/figures'


def run():
    inputs = {'figure_templates':check_templates()}
    for side,region in DOMAINS.items():
        p = read_receipt(EVIDENCE/region,'complete.json','references.npz',
            ('ls_id','TYPE','g0','r0','qrf_quantiles_kpc','gaia_state','reference_kpc',
             'specdis_kpc','specdis_quality','spdist_kpc','spdist_quality'),inputs)
        f = read_receipt(TRIAL/side,'complete.json','scores.npz',
            ('ls_id','TYPE','held_out','radial_pass','score_original','score_candidate'),inputs)
        if not np.all(f['TYPE']=='PSF'):
            raise ValueError('Non-native candidate row')
        j = unique_join(f['ls_id'],p['ls_id'])
        region_match = f['held_out']&(j>=0)
        i,k = np.flatnonzero(region_match),j[region_match]
        if int(region_match.sum())!=int(f['held_out'].sum()) or not np.all(p['TYPE'][k]=='PSF'):
            raise ValueError('Plot data lose the fixed role-4 PSF cohort')
        meta = json.loads((TRIAL/side/'complete.json').read_text())
        phot = np.column_stack([p['g0'][k],p['r0'][k]])
        for version,value,cut in (('v3.10','score_original',SCORE),
                                  ('v3.11 trial','score_candidate',meta['threshold'])):
            chosen = f['radial_pass'][i]&(f[value][i]>=cut)
            cohorts = {}
            for source,distance,flag in (
                ('Gaia','reference_kpc',p['gaia_state']==6),
                ('SpecDis','specdis_kpc',p['specdis_quality']),
                ('SPDist','spdist_kpc',p['spdist_quality']),
            ):
                d = p[distance][k]
                q = p['qrf_quantiles_kpc'][k]
                g = chosen & flag[k] & np.isfinite(d) & (d>0)
                g &= np.isfinite(q).all(axis=1) & (q>0).all(axis=1)
                cohorts[source] = (d[g],q[g],phot[g])
            short = 'trial' if version=='v3.11 trial' else 'baseline'
            out = DEST/side/short
            out.mkdir(parents=True,exist_ok=True)
            domain = 'Northern sample' if side=='north' else 'Southern low-extinction sample'
            with plt.rc_context(STYLE):
                render_reference_figures(out,cohorts,
                    distance_title=f'{domain} · {version} selection · qualified references')
    save_json(DEST/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,
        outputs={str(f.relative_to(DEST)):sha(f) for f in DEST.glob('*/*/*') if f.is_file()}))


if __name__=='__main__':
    require_compute()
    run()
