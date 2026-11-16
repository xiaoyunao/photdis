#!/usr/bin/env python3
"""Compare grz error cuts within matched reference populations at the faint end."""

import json
import os

import numpy as np

from photdis.stage1 import require_compute
from photdis.v3.metrics import distance_summary
from v2_1_near import ROOT, csv_rows, save_json, sha
from v3_10_evidence import DEST as EVIDENCE
from v3_11_layers import COLORS, read_receipt
from v3_11_selection import LABELS, SHELLS, gravity_labels

DEST = ROOT/'data/v3_11/depth'
MAGS = ((-np.inf,20.),(20.,21.),(21.,22.),(22.,23.))
ERROR_CUTS = (.20,.15,.10,.07,.05,.03)


def run():
    inputs = {}
    rows = []
    for region in ('north','south_low','south_faint'):
        t = read_receipt(EVIDENCE/region,'complete.json','references.npz',
             ('TYPE','ls_id','r','g0','r0','error_grz','qrf_quantiles_kpc','evaluation_row',
              'sky_role','row_state','gaia_state','reference_kpc','specdis_kpc','specdis_quality',
              'spdist_kpc','spdist_quality','rvs_archive_row','class_label','rvs_teff',
              'rvs_logg','rvs_logg_err'),inputs)
        if not np.all(t['TYPE']=='PSF') or len(np.unique(t['ls_id']))!=len(t['ls_id']):
            raise ValueError('New depth comparisons require unique native PSF records')
        color = t['g0']-t['r0']
        error = t['error_grz'].max(axis=1)
        q = t['qrf_quantiles_kpc']
        label = gravity_labels(t['rvs_logg'],t['rvs_logg_err'],t['rvs_teff'],
                               (t['class_label']>=0)&(t['rvs_archive_row']>=0))
        base = t['evaluation_row']&(t['sky_role']==4)&np.isfinite(error)
        base &= np.isfinite(color)&np.isin(t['row_state'],(6,7))
        for source,ref,good in (('Gaia',t['reference_kpc'],t['gaia_state']==6),
                                ('SpecDis',t['specdis_kpc'],t['specdis_quality']),
                                ('SPDist',t['spdist_kpc'],t['spdist_quality'])):
            for r0,r1 in MAGS:
                mag = (t['r']>=r0)&(t['r']<r1)
                for c0,c1 in COLORS:
                    col = (color>=c0)&(color<c1)
                    for i,name in enumerate(['all',*LABELS],start=-1):
                        cls = label==i if i>=0 else np.ones(len(q),bool)
                        u = base&good&mag&col&cls&np.isfinite(ref)&(ref>0)
                        if not u.any():
                            continue
                        for d0,d1 in SHELLS:
                            eligible = u&(ref>=d0)&(ref<d1)
                            if not eligible.any():
                                continue
                            supported = eligible&np.isfinite(q).all(axis=1)&(q>0).all(axis=1)
                            for cut in ERROR_CUTS:
                                adm = eligible&(error<cut)
                                s = supported&adm
                                rows.append(dict(region=region,reference=source,r_min=r0,r_max=r1,
                                    gr_min=c0,gr_max=c1,category=name,reference_d_min=d0,
                                    reference_d_max=d1,max_grz_error=cut,eligible=int(eligible.sum()),
                                    supported=int(supported.sum()),pass_error_cut=int(adm.sum()),
                                    **distance_summary(10+5*np.log10(q[s]),10+5*np.log10(ref[s]))))
    DEST.mkdir(parents=True,exist_ok=True)
    csv_rows(DEST/'quality_distance.csv',rows)
    save_json(DEST/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,
               comparisons=len(rows),outputs={'quality_distance.csv':sha(DEST/'quality_distance.csv')}))


if __name__=='__main__':
    require_compute()
    run()
