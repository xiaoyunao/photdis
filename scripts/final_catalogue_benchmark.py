#!/usr/bin/env python3
"""Test published distances before the Photdis class and estimated-distance cuts."""
import argparse
import json
import os

import numpy as np

from photdis.reference_v1_4 import unique_join
from photdis.stage1 import require_compute
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_6_vac_distances import read_columns
from v3_8_direct_figures import ORANGE, STYLE, density, plt, save
from v3_9_references import join, subset
from v3_10_evidence import DEST as EVIDENCE
from v3_10_figures import check_templates
from v3_10_science import working_index
from v3_10_working_sample import DOMAINS
from release_readiness_archive import PRODUCTS
from release_readiness_compare import NAMES, available, interval_ok, stats
from release_readiness_population import classes
from final_catalogue_raw import DEST

OUT = DEST/'benchmark'
FIELDS = ('TYPE','ls_id','source_id','RA','DEC','r','g0','r0','row_state','evaluation_row',
 'sky_role','gaia_state','class_label','rvs_feh','rvs_feh_err','rvs_logg','rvs_logg_err',
 'qrf_quantiles_kpc','specdis_quality','specdis_kpc','spdist_quality','spdist_kpc',
 'redrock_quality_assessed','redrock_SPECTYPE')
SHELLS = ((0.,np.inf),(0.,2.),(2.,5.),(5.,10.),(10.,np.inf))


def parent(side,inputs):
    region=DOMAINS[side]
    marker=EVIDENCE/region/'complete.json'
    t=read_columns(verified(marker.with_name('references.npz'),
                             json.loads(marker.read_text())['outputs']['references.npz']),FIELDS)
    _, work=working_index(side,inputs)
    j=unique_join(t['ls_id'],work['ls_id'])
    use=j>=0
    t=subset(t,use)
    assert (t['TYPE']=='PSF').all()
    k=j[use]
    t.update(state=work['state'][k],region=np.full(len(k),region,'U12'),
             radial_mass=work['radial_mass'][k])
    # Compare release PDF and index state without conditioning spectral admission.
    q=t['qrf_quantiles_kpc']
    j=t['state']>=2
    np.testing.assert_allclose(q[j,2],work['qrf_median_kpc'][k[j]],rtol=2e-6)
    inputs[str(marker.relative_to(ROOT))]=sha(marker)
    return t


def prepare():
    inputs={}
    t=join([parent(side,inputs) for side in ('north','south')])
    for k in ('specdis','spdist'):
        d=t[k+'_kpc']
        t[k+'_kpc']=np.where(t[k+'_quality']&np.isfinite(d)&(d>0),d,np.nan)
    ref=np.isfinite(t['specdis_kpc'])|np.isfinite(t['spdist_kpc'])
    j=(t['r']>=18)&(t['r']<20)&(t['row_state']==7)&t['evaluation_row']&(t['sky_role']==4)&ref
    t=subset(t,j)
    assert np.unique(t['ls_id']).size==len(t['ls_id']) and (t['TYPE']=='PSF').all()
    good=t['source_id']>0
    ids,counts=np.unique(t['source_id'][good],return_counts=True)
    t['unique_gaia']=good&~np.isin(t['source_id'],ids[counts>1])
    ids=ids[counts==1]
    OUT.mkdir(exist_ok=True,parents=True)
    atomic_npz(OUT/'parent.npz',**t)
    np.save(OUT/'ids.npy',ids)
    save_json(OUT/'parent.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,
         n=len(t['ls_id']),unique_gaia=len(ids),qrf_supported=int((t['state']>=2).sum()),
         outputs={x:sha(OUT/x) for x in ('parent.npz','ids.npy')}))


def load_parent():
    m=OUT/'parent.json'
    r=json.loads(m.read_text())
    return read_columns(verified(OUT/'parent.npz',r['outputs']['parent.npz']),
                        (*FIELDS,'state','region','radial_mass','unique_gaia')),r


def published(t, expected):
    q={'Photdis':t['qrf_quantiles_kpc'][:,1:4].copy()}
    q['Photdis'][t['state']<2]=np.nan
    found={'Photdis':t['state']>=2}
    ids=t['source_id']
    inputs={}
    j=np.flatnonzero(t['unique_gaia'])
    j=j[np.argsort(ids[j])]
    for name,(_,_,_,methods) in PRODUCTS.items():
        p=OUT/'archives'/name/'complete.json'
        a=json.loads(p.read_text())
        if a['inputs']['ids_sha256']!=expected or a['rows']!=len(j):
            raise ValueError(f'Archive cohort changed: {name}')
        data=read_columns(verified(p.with_name('matched.npz'),a['outputs']['matched.npz']),
                          ('source_id',*methods))
        k=j[np.searchsorted(ids[j],data['source_id'])]
        np.testing.assert_array_equal(ids[k],data['source_id'])
        assert len(np.unique(k))==len(k)==a['matched']
        inputs[name]=sha(p)
        linked=np.isin(t['gaia_state'][k],(3,4,5,6))
        for method in methods:
            q[method]=np.full((len(ids),3),np.nan)
            q[method][k[linked]]=data[method][linked]
            found[method]=np.zeros(len(ids),bool)
            found[method][k]=True
    return q,found,inputs


def masks(t):
    c=t['g0']-t['r0']
    rr=np.where(t['redrock_quality_assessed'],t['redrock_SPECTYPE'],'')
    group=classes(t,rr)
    weak=np.isin(t['gaia_state'],(3,4,5))
    sets=dict(all=np.ones(len(c),bool),north=t['region']=='north',south_low=t['region']=='south_low',
        blue=c<.5,intermediate_color=(c>=.5)&(c<.8),redder=c>=.8,
        dwarfs=t['class_label']==0,poor_dwarfs=group=='dwarf_feh_lt_minus1p5',
        ordinary_dwarfs=group=='dwarf_feh_ge_minus1p5',intermediate_gravity=group=='intermediate_gravity',
        giants=group=='giant',weak_gaia=weak,
        bright_half=t['r']<19,faint_half=t['r']>=19,
        reliable_geometry=t['gaia_state']==6,unqualified_link=t['gaia_state']<3,
        weak_gaia_dwarfs=weak&(t['class_label']==0),
        weak_gaia_ordinary=weak&(group=='dwarf_feh_ge_minus1p5'),
        weak_gaia_poor=weak&(group=='dwarf_feh_lt_minus1p5'),
        weak_gaia_intermediate=weak&(group=='intermediate_gravity'))
    return sets


def evaluate(out,t,q,found):
    moments,coverage=[],[]
    valid={name:available(v) for name,v in q.items()}
    common=np.logical_and.reduce(list(valid.values()))
    for group,w in masks(t).items():
        for label in ('SpecDis','SPDist'):
            d=t[label.lower()+'_kpc']
            good=w&np.isfinite(d)
            for lo,hi in SHELLS:
                base=good&(d>=lo)&(d<hi)
                for m,x in q.items():
                    coverage.append(dict(group=group,reference=label,lo=lo,hi=hi,method=m,n=int(base.sum()),
                      queried=int((base&t['unique_gaia']).sum()) if m!='Photdis' else int(base.sum()),
                      matched=int((base&found[m]).sum()),
                      qualified_link=int((base&found[m]&np.isin(t['gaia_state'],(3,4,5,6))).sum())
                         if m!='Photdis' else int((base&found[m]).sum()),
                      valid=int((base&valid[m]).sum()),
                      valid68=int((base&interval_ok(x)).sum())))
                for other in list(q)[1:]:
                    j=base&valid['Photdis']&valid[other]
                    for m in ('Photdis',other):
                        moments.append(dict(group=group,reference=label,lo=lo,hi=hi,
                            pair=other,method=m,**stats(q[m][j],d[j])))
                j=base&common
                for m,x in q.items():
                    moments.append(dict(group=group,reference=label,lo=lo,hi=hi,
                        pair='all_methods',method=m,**stats(x[j],d[j])))
    csv_rows(out/'coverage.csv',coverage)
    csv_rows(out/'same_star.csv',moments)
    return valid


def figure(out,t,q,valid):
    fig,axes=plt.subplots(2,2,figsize=(14,12),layout='constrained')
    rows,data=[],{}
    edges=np.geomspace(.3,25.,110)
    for i,label in enumerate(('SpecDis','SPDist')):
        d=t[label.lower()+'_kpc']
        j=(d>=2)&(d<5)&masks(t)['weak_gaia_dwarfs']&valid['Photdis']&valid['BJ_photogeometric']
        for k,m in enumerate(('Photdis','BJ_photogeometric')):
            ax=axes[i,k]
            p=q[m][j,1]
            h=density(fig,ax,d[j],p,edges,edges)
            ax.plot(edges[[0,-1]],edges[[0,-1]],'--',color=ORANGE,lw=1.5)
            ax.set(xscale='log',yscale='log',xlabel=f'{label} reference (kpc)',
                   ylabel='Estimated distance (kpc)',title=f'{NAMES[m]} · N = {len(p):,}')
            rows.append(dict(reference=label,method=m,n=len(p),displayed=int(h.sum())))
            data[f'panel_{i}_{k}']=h
    fig.suptitle('Faint spectroscopic dwarfs before class-score and distance selection')
    save(fig,out,'F15_independent_distances',rows,dict(edges=edges,**data))


def main():
    require_compute()
    a=argparse.ArgumentParser()
    a.add_argument('stage',choices=('prepare','compare'))
    if a.parse_args().stage=='prepare':
        prepare()
        return
    t,b=load_parent()
    q,found,inputs=published(t,b['outputs']['ids.npy'])
    out=OUT/'results'
    out.mkdir(exist_ok=True)
    valid=evaluate(out,t,q,found)
    with plt.rc_context(STYLE):
        figure(out,t,q,valid)
    save_json(out/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),parent=sha(OUT/'parent.json'),
        style=check_templates(),inputs=inputs,outputs={p.name:sha(p) for p in out.iterdir()
                                                         if p.name!='complete.json'}))


if __name__=='__main__':
    main()
