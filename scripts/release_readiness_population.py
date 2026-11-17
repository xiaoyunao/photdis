#!/usr/bin/env python3
"""Replay class-conditional retention, distance migration, and coverage."""
import json
import os

import numpy as np

from photdis.reference_v1_4 import unique_join
from photdis.stage1 import require_compute
from photdis.v3.metrics import distance_summary
from v2_1_near import csv_rows, save_json, sha
from v3_3_sed import verified
from v3_6_vac_distances import read_columns
from v3_10_evidence import DEST as EVIDENCE
from v3_10_redrock import DEST as REDROCK
from v3_10_science import working_index
from v3_10_working_sample import DOMAINS
from final_science_evidence import SUMMARY_KEYS
from release_readiness_sample import DEST

EXTERIOR = (0.,2.,5.,10.,np.inf)
FIELDS = ('TYPE','ls_id','r','g0','r0','evaluation_row','sky_role','class_label','rvs_logg',
    'rvs_logg_err','rvs_feh','rvs_feh_err','qrf_quantiles_kpc','gaia_state','reference_kpc',
    'specdis_quality','specdis_kpc','spdist_quality','spdist_kpc',
    'redrock_quality_assessed','redrock_SPECTYPE')


def classes(t, redrock):
    label = t['class_label']
    f,df,g,dg = (t[k] for k in ('rvs_feh','rvs_feh_err','rvs_logg','rvs_logg_err'))
    reliable_m = np.isfinite(f)&(df>0)&(df<=.3)
    reliable_g = np.isfinite(g)&(dg>0)&(dg<=.3)
    result = np.full(len(label),'unclassified','U32')
    result[label==0] = 'dwarf_feh_unknown'
    result[(label==0)&reliable_m&(f>=-1.5)] = 'dwarf_feh_ge_minus1p5'
    result[(label==0)&reliable_m&(f<-1.5)] = 'dwarf_feh_lt_minus1p5'
    result[label==1] = 'giant'
    result[label==2] = 'other_stellar'
    result[(label==2)&reliable_g&(g>=3.2)&(g<4.2)] = 'intermediate_gravity'
    result[np.isin(redrock,('QSO','GALAXY'))&(label>=0)] = 'class_conflict'
    result[np.isin(redrock,('QSO','GALAXY'))&(label<0)] = 'qualified_nonstellar'
    return result


def build(side, inputs, rows, compositions):
    _, work = working_index(side,inputs)
    region = DOMAINS[side]
    marker = EVIDENCE/region/'complete.json'
    inputs[str(marker)] = sha(marker)
    meta = json.loads(marker.read_text())
    f = verified(marker.with_name('references.npz'),meta['outputs']['references.npz'])
    t = read_columns(f,FIELDS)
    j = unique_join(t['ls_id'],work['ls_id'])
    keep = (j>=0)&t['evaluation_row']&(t['sky_role']==4)
    if not np.all(t['TYPE'][keep]=='PSF'):
        raise ValueError('Reference row failed native PSF verification')
    t = {k:v[keep] for k,v in t.items()}
    j = j[keep]
    state,radial = work['state'][j],work['radial_mass'][j]
    quant = t['qrf_quantiles_kpc']
    support = state>=2
    median = quant[:,2]
    np.testing.assert_allclose(median[support],work['qrf_median_kpc'][j[support]],rtol=2e-6)
    rr = np.where(t['redrock_quality_assessed'],t['redrock_SPECTYPE'].astype('U8'),'')
    marker_rr = REDROCK/'complete.json'
    inputs[str(marker_rr)] = sha(marker_rr)
    a = read_columns(verified(REDROCK/'qualified.npz',
        json.loads(marker_rr.read_text())['outputs']['qualified.npz']),('ls_id','TYPE','qualified'))
    ix = unique_join(t['ls_id'],a['ls_id'])
    has = ix>=0
    np.testing.assert_array_equal(a['TYPE'][ix[has]],t['TYPE'][has])
    k = has.copy()
    k[has] = a['qualified'][ix[has]]!=''
    conflict = k.copy()
    conflict[k] = (rr[k]!='')&(rr[k]!=a['qualified'][ix[k]])
    rr[k] = a['qualified'][ix[k]]
    labels = classes(t,rr)
    labels[conflict] = 'class_conflict'
    color = t['g0']-t['r0']
    strata = (('all',np.ones(len(j),bool)),('faint',t['r']>=18),
              ('blue',color<.5),('faint_blue',(t['r']>=18)&(color<.5)),
              ('intermediate_color',(color>=.5)&(color<.8)),('redder',color>=.8))
    sel = state==4
    med_slices = dict(median_0_2=median<2,median_2_5=(median>=2)&(median<5),
                      median_5_10=(median>=5)&(median<10),median_10_plus=median>=10)
    for group in np.unique(labels):
        label = labels==group
        for name,obs in strata:
            k = label&obs
            for stage,w in (('matched',state>=0),('strict',state>=1),
                            ('supported',support),('score_pass',state>=3),('selected',sel),
                            ('selected_2_5',sel&med_slices['median_2_5']),
                            ('selected_5_10',sel&med_slices['median_5_10'])):
                compositions.append(dict(region=region,observable=name,population=group,stage=stage,
                                         n=int((w&k).sum())))
    for source,d,quality in (('Gaia',t['reference_kpc'],t['gaia_state']==6),
            ('SpecDis',t['specdis_kpc'],t['specdis_quality']),
            ('SPDist',t['spdist_kpc'],t['spdist_quality'])):
        base = quality&np.isfinite(d)&(d>0)
        for name,obs in strata:
            for group in ('all',*np.unique(labels)):
                label = True if group=='all' else labels==group
                for lo,hi in zip(EXTERIOR[:-1],EXTERIOR[1:]):
                    w = base&obs&label&(d>=lo)&(d<hi)
                    for stage,gate in (('supported',support),('selected',sel)):
                        k = w&gate
                        stats = distance_summary(10+5*np.log10(quant[k]),10+5*np.log10(d[k]))
                        rows.append(dict(region=region,source=source,population=group,observable=name,
                            ref_lo=lo,ref_hi=hi,stage=stage,qualified=int(w.sum()),
                            **{key:stats.get(key) for key in SUMMARY_KEYS},
                            **{key:int((k&m).sum()) for key,m in med_slices.items()},
                            pdf_2_5=float(radial[k,0].sum(dtype='f8')),
                            pdf_5_10=float(radial[k,1].sum(dtype='f8'))))


def main():
    require_compute()
    dest = DEST/'population'
    dest.mkdir(exist_ok=True,parents=True)
    inputs,rows,composition = {},[],[]
    for side in ('north','south'):
        build(side,inputs,rows,composition)
    csv_rows(dest/'retention_migration.csv',rows)
    csv_rows(dest/'reference_composition.csv',composition)
    save_json(dest/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,
        outputs={p.name:sha(p) for p in dest.glob('*.csv')}))


if __name__=='__main__':
    main()
