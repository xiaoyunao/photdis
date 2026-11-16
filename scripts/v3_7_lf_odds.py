#!/usr/bin/env python3
"""Reweight the v3.6 class-distance posterior without changing either selection."""
import json
from pathlib import Path

import healpy as hp
import numpy as np

from photdis.stage1 import require_compute
from v2_1_near import ROOT, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_6_lf_sensitivity import OUT as FORECAST
from v3_6_vac_distances import DISTANCE_EDGES, read_columns
from v3_6_distance_evidence import SHELLS
from v3_7_weak_gaia import PROTOCOL

OUT = ROOT/'data/v3_7/lf_odds'


def reweight(class_shell,factor):
    result = class_shell.copy()
    result[:,1,:] *= factor
    return result/result.sum(axis=(1,2),keepdims=True)


def block_resamples(sky,draws,seed):
    _,ix = np.unique(sky,return_inverse=True)
    blocks = len(np.unique(ix))
    rng = np.random.default_rng(seed)
    return ix,rng.integers(blocks,size=(draws,blocks))


def summarize(p,protocol):
    names = p['scenario']
    d50 = p['qrf_quantiles_kpc'][:,2]
    w = p['weight']
    sky = hp.ang2pix(protocol['sky_nside'],p['RA'],p['DEC'],nest=True,lonlat=True)
    all_rows = np.ones(len(w),bool)
    for cut,selected in (('supported',all_rows),('primary',p['primary']),('strict',p['strict'])):
        ix,draw = block_resamples(sky[selected],protocol['bootstrap_draws'],protocol['bootstrap_seed'])
        weight = w[selected]
        for j,(lo,hi) in enumerate(zip(DISTANCE_EDGES[:-1],DISTANCE_EDGES[1:])):
            hard = (d50[selected]>=lo)&(d50[selected]<hi)
            for estimand,mass in (('median_bin',hard),('pdf_weighted',p['qrf_bins'][selected,j])):
                v = weight*mass
                denominator = v.sum()
                den = np.bincount(ix,weights=v)
                counts = den[draw].sum(axis=1)
                for i,scenario in enumerate(names):
                    for factor in protocol['giant_odds_factors']:
                        joint = reweight(p['class_shell'][selected,i],factor)
                        wrong = 1.-joint[:,:,j].sum(axis=1)
                        values = {'giant':joint[:,1,:].sum(axis=1),
                                  'far_giant':joint[:,1,3:].sum(axis=1),
                                  'other':joint[:,2,:].sum(axis=1),
                                  'foreground':joint[:,:,:j].sum(axis=(1,2)),
                                  'background':joint[:,:,j+1:].sum(axis=(1,2)),
                                  'outside_assigned_shell':wrong}
                        for label,a in values.items():
                            num = np.bincount(ix,weights=v*a)
                            total = num.sum()
                            valid = counts>0
                            ci = np.quantile(num[draw].sum(axis=1)[valid]/counts[valid],[.025,.975]) if valid.any() else (None,None)
                            yield dict(selection=cut,assignment=estimand,shell=SHELLS[j],scenario=scenario,
                                       giant_odds_factor=factor,event=label,n=int((mass>0).sum()),
                                       effective_n=denominator**2/(v@v) if denominator else None,
                                       denominator=denominator,expected=total,fraction=total/denominator if denominator else None,
                                       lo=ci[0],hi=ci[1])


def main():
    require_compute()
    dest = OUT/'complete.json'
    if dest.exists():
        raise FileExistsError(dest)
    source = FORECAST/'complete.json'
    receipt = json.loads(source.read_text())
    fields = ('TYPE','qrf_quantiles_kpc','qrf_bins','weight','RA','DEC','primary','strict','scenario','class_shell')
    p = read_columns(verified(FORECAST/'predictions.npz',receipt['outputs']['predictions.npz']),fields)
    kind = p['TYPE']
    use = kind=='PSF'
    exclusions = dict(unverified_type=int((kind=='').sum()),non_psf=int(((kind!='')&~use).sum()))
    p = {k:v if k=='scenario' else v[use] for k,v in p.items()}
    if not use.any():
        raise ValueError('No admissible native PSF on the frozen LF draw')
    np.testing.assert_allclose(p['class_shell'].sum(axis=(2,3)),1.,rtol=0,atol=1e-12)
    np.testing.assert_allclose(p['qrf_bins'].sum(axis=1),1.,rtol=0,atol=1e-12)
    OUT.mkdir(parents=True,exist_ok=True)
    tab = OUT/'odds_leakage.csv'
    csv_rows(tab,summarize(p,json.loads(PROTOCOL.read_text())))
    save_json(dest,dict(inputs={str(x.relative_to(ROOT)):sha(x) for x in (source,PROTOCOL,Path(__file__).resolve())},
        rows=int(use.sum()),excluded_morphology=exclusions,outputs={tab.name:sha(tab)}))


if __name__=='__main__':
    main()
