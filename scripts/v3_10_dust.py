#!/usr/bin/env python3
"""Replay the frozen Bayestar19 pilot on a common PS1-to-SFD reddening scale."""
import json
import os

import joblib
import numpy as np

from photdis.stage1 import require_compute
from photdis.v3.model import inference_model
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_3_catalog import read_frozen
from v3_3_sed import verified
from v3_6_vac_distances import read_columns
from v3_9_dust import (
    GRZ, MU, MU_EDGES, OUT as OLD_DUST, SFD, marginal_log_density,
    matched_reference_summary, normalize, photometry,
)

DEST = ROOT/'data/v3_10/dust/replay'
BAYESTAR_TO_QUOTED_EBV = .883
# Match the same PS1 g-r extinction between the two published conventions.
# Green et al. 2019 §2.1 and Table 1; Schlafly & Finkbeiner 2011 Table 6, RV=3.1.
PS1_BAYESTAR = (3.518,2.617)
PS1_SFD = (3.172,2.271)
BAYESTAR_TO_SFD = (PS1_BAYESTAR[0]-PS1_BAYESTAR[1])/(PS1_SFD[0]-PS1_SFD[1])
DECam_PER_BAYESTAR = SFD*BAYESTAR_TO_SFD


def replay():
    original = OLD_DUST/'bayestar19'
    marker = original/'complete.json'
    previous = json.loads(marker.read_text())
    old = read_columns(verified(original/'distance_pdfs.npz',previous['outputs']['distance_pdfs.npz']),
        ('TYPE','ls_id','mu_edges','pdf','map_reliable','ebv_samples','reference_kpc',
         'gaia_reference_lo','gaia_reference_hi'))
    p = photometry()
    np.testing.assert_array_equal(p['ls_id'],old['ls_id'])
    np.testing.assert_array_equal(p['TYPE'],old['TYPE'])
    np.testing.assert_array_equal(old['mu_edges'],MU_EDGES)
    if len(p['ls_id'])!=311 or len(np.unique(p['ls_id']))!=311 or not np.all(p['TYPE']=='PSF'):
        raise ValueError('The replay requires exactly the 311 original native PSF sources')
    reports = np.genfromtxt(verified(original/'sightlines.csv',previous['outputs']['sightlines.csv']),
                            delimiter=',',names=True,usecols=('ls_id','gaia_state'),dtype='i8')
    np.testing.assert_array_equal(reports['ls_id'],p['ls_id'])
    entry = read_frozen()['models']['nominal']
    if entry['sha256'] != previous['nominal_sha256']:
        raise ValueError('The v3.9 nominal SED must be identical for the paired replay')
    model = inference_model(joblib.load(verified(ROOT/entry['path'],entry['sha256'])))
    pdf = old['pdf'].copy()
    valid = old['map_reliable'].sum(axis=1)>=3
    changed = []
    for i in np.flatnonzero(valid):
        use = old['map_reliable'][i]
        sample = np.where(use[None,:],old['ebv_samples'][i]/BAYESTAR_TO_QUOTED_EBV,0.)
        flux,err = p['flux'][i,GRZ],p['flux_error'][i,GRZ]
        sfd = np.broadcast_to(p['transmission'][i,GRZ],(1,len(MU),3))
        old_sfd = marginal_log_density(model,flux,err,sfd)
        np.testing.assert_allclose(normalize(old_sfd,use),pdf[i,1],rtol=0,atol=1e-12)
        trans = 10**(-.4*sample[:,:,None]*DECam_PER_BAYESTAR)
        pdf[i,2] = normalize(marginal_log_density(model,flux,err,trans),use)
        changed.append(dict(ls_id=int(p['ls_id'][i]),TYPE='PSF',field=int(p['field'][i]),
                            total_variation=float(.5*abs(old['pdf'][i,2]-pdf[i,2]).sum())))
    args = (p,reports['gaia_state'],old['reference_kpc'],old['gaia_reference_lo'],
            old['gaia_reference_hi'],old['map_reliable'])
    old_rows = matched_reference_summary(*args,old['pdf'])
    new_rows = matched_reference_summary(*args,pdf)
    root = DEST
    root.mkdir(parents=True,exist_ok=True)
    csv_rows(root/'matched_gaia.csv',[dict(scale=name,**r)
                                  for name,rows in (('v3.9',old_rows),('ps1_to_sfd',new_rows)) for r in rows])
    csv_rows(root/'pdf_change.csv',changed)
    atomic_npz(root/'distance_pdfs.npz',**{**old,'pdf':pdf})
    save_json(root/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs={str(marker.relative_to(ROOT)):sha(marker)},
        map_sha256=previous['map_sha256'],coefficients_decamin_grz=DECam_PER_BAYESTAR,
        bayestar_to_sfd=BAYESTAR_TO_SFD,rows=len(pdf),common_map=int(valid.sum()),
        outputs={x.name:sha(x) for x in root.iterdir() if x.name!='complete.json'}))


if __name__=='__main__':
    require_compute()
    replay()
