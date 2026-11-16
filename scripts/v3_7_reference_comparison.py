#!/usr/bin/env python3
"""Compare Gaia and the two VAC distances on exactly the same native objects."""
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from photdis.stage1 import require_compute
from photdis.v3.metrics import distance_summary
from v2_1_near import ROOT, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_6_distance_evidence import OUT as EVIDENCE, SHELLS
from v3_6_reference_photometry import OUT as PHOTO
from v3_6_vac_distances import DISTANCE_EDGES, read_columns


OUT = ROOT/'data/v3_7/reference_comparison'
SOURCES = ('Gaia','SpecDis','SPDist')
FIELDS = ('catalog_row','ls_id','TYPE','reference_kpc','reference_lo_kpc','reference_hi_kpc',
          'high_quality','qrf_training','recorded_history','qrf_quantiles_kpc','qrf_bins',
          'primary','strict','consensus_class','source_class','observed_r','dereddened_g_r')
LABELS = ('dwarf','giant','other','unknown')
PARAMETERS = ('rvs_feh','rvs_logg','rvs_parameter_quality')


def load_source(source,receipt):
    name = f'{source}.npz'
    path = verified(EVIDENCE/name,receipt['outputs'][name])
    with np.load(path,allow_pickle=False) as data:
        required = FIELDS if source=='Gaia' else (*FIELDS,*PARAMETERS)
        if 'TYPE' not in data:
            return {},dict(non_psf=0,unverified_type=len(data['catalog_row']))
        kind = data['TYPE']
        admissible = kind=='PSF'
        excluded = dict(non_psf=int(((kind!='')&~admissible).sum()),unverified_type=int((kind=='').sum()))
        center = data['reference_kpc']
        qualified = admissible&data['high_quality']&~data['qrf_training']&np.isfinite(center)&(center>0)
        result = {k:data[k][qualified] for k in required}
    ix = np.argsort(result['catalog_row'])
    if (np.diff(result['catalog_row'][ix])<=0).any():
        raise ValueError(f'{source}: reference catalog_row is not unique')
    return {k:v[ix] for k,v in result.items()},excluded


def matched(sources,tables):
    common = tables[sources[0]]['catalog_row']
    for source in sources[1:]:
        common = np.intersect1d(common,tables[source]['catalog_row'],assume_unique=True)
    roster = {}
    for source in sources:
        a = tables[source]
        pos = np.searchsorted(a['catalog_row'],common)
        roster[source] = {k:v[pos] for k,v in a.items()}
    pivot = roster[sources[0]]
    for a in roster.values():
        for k in ('ls_id','primary','strict','qrf_quantiles_kpc'):
            np.testing.assert_array_equal(a[k],pivot[k])
    return roster


def strata(a,feh,logg,rz):
    n = len(a['catalog_row'])
    yield 'all','all',np.ones(n,bool)
    for field,values,edges in (('r',a['observed_r'],[-np.inf,16.,18.,19.,20.,np.inf]),
                              ('g-r',a['dereddened_g_r'],[-np.inf,.2,.5,.8,1.1,np.inf]),
                              ('r-z',rz,[-np.inf,0.,.25,.5,.75,1.,np.inf]),
                              ('FeH',feh,[-np.inf,-1.5,-.7,0.,np.inf]),
                              ('logg',logg,[-np.inf,3.2,4.2,4.6,np.inf])):
        yield field,'unmeasured',~np.isfinite(values)
        for lo,hi in zip(edges[:-1],edges[1:]):
            yield field,f'[{lo:g}, {hi:g})',np.isfinite(values)&(values>=lo)&(values<hi)


def paired(roster,photo):
    names = tuple(roster)
    p = roster[names[0]]
    pos = np.searchsorted(photo['catalog_row'],p['catalog_row'])
    np.testing.assert_array_equal(photo['ls_id'][pos],p['ls_id'])
    if not (photo['TYPE'][pos]=='PSF').all():
        raise ValueError('Matching requires native PSF photometry')
    magnitude = photo['magnitude'][pos]
    rz = magnitude[:,1]-magnitude[:,3]
    labels = np.stack([a['consensus_class'] for a in roster.values()])
    consensus = np.where((labels==labels[0]).all(axis=0),labels[0],-1)
    # Spectroscopic values are DESI RVS fields stored with the VAC matches.
    vac = next(roster[name] for name in names if name!='Gaia')
    feh,logg = (np.where(vac['rvs_parameter_quality'],vac[k],np.nan)
                for k in ('rvs_feh','rvs_logg'))
    history = np.logical_or.reduce([a['recorded_history'] for a in roster.values()])
    all_rows = np.ones(len(history),bool)
    qrf = 10+5*np.log10(p['qrf_quantiles_kpc'])
    axis = list(strata(p,feh,logg,rz))
    for definition in names:
        d = roster[definition]['reference_kpc']
        shells = np.searchsorted(DISTANCE_EDGES,d,side='right')-1
        for selection,cut in (('supported',all_rows),('primary',p['primary']),('strict',p['strict'])):
            for exposure,exposure_mask in (('all',all_rows),('outside_recorded_history',~history)):
                for j,shell in enumerate(SHELLS):
                    for label,cl in [('all',all_rows),*((v,consensus==i) for i,v in enumerate(LABELS[:3])),
                                     ('unknown',consensus<0)]:
                        keep = cut&exposure_mask&(shells==j)&cl
                        for field,band,domain in axis:
                            use = keep&domain
                            row = dict(overlap='+'.join(names),anchor=definition,selection=selection,
                                       history=exposure,reference_shell=shell,reference_class=label,
                                       axis=field,bin=band,n=int(use.sum()))
                            for target,a in roster.items():
                                dv = a['reference_kpc'][use]
                                result = distance_summary(qrf[use],10+5*np.log10(dv))
                                ratio = np.median(dv/d[use]) if use.any() else None
                                yield dict(**row,reference=target,reference_to_anchor=ratio,
                                           **{k:v for k,v in result.items() if k!='n'})


def migration(source,a):
    d = a['reference_kpc']
    assignment = np.searchsorted(DISTANCE_EDGES,a['qrf_quantiles_kpc'][:,2],side='right')-1
    reference = np.searchsorted(DISTANCE_EDGES,d,side='right')-1
    p = a['qrf_bins']
    np.testing.assert_allclose(p.sum(axis=1),1.,rtol=0,atol=1e-12)
    every = np.ones(len(d),bool)
    for cut,selected in (('supported',every),('primary',a['primary']),('strict',a['strict'])):
        for role,history in (('all',every),('outside_recorded_history',~a['recorded_history'])):
            for label,group in [('all',every),*((x,a['consensus_class']==i) for i,x in enumerate(LABELS[:3])),
                                ('unknown',a['consensus_class']<0)]:
                for i,shell in enumerate(SHELLS):
                    denom = group&history&(reference==i)
                    use = selected&denom
                    counts = np.bincount(assignment[use],minlength=len(SHELLS))
                    soft = p[use].sum(axis=0)
                    for j,predicted in enumerate(SHELLS):
                        yield dict(source=source,selection=cut,history=role,reference_class=label,
                                   reference_shell=shell,predicted_shell=predicted,
                                   reference_denominator=int(denom.sum()),selected=int(use.sum()),
                                   hard_count=int(counts[j]),pdf_mass=soft[j])


def main():
    require_compute()
    output = OUT/'complete.json'
    if output.exists():
        raise FileExistsError(output)
    evidence,photometry = (p/'complete.json' for p in (EVIDENCE,PHOTO))
    inputs = [evidence,photometry,Path(__file__).resolve()]
    receipt,physical = (json.loads(p.read_text()) for p in inputs[:2])
    if physical['inputs'][str(evidence.relative_to(ROOT))]!=sha(evidence):
        raise ValueError('Photometry and distance evidence use different cohorts')
    photo = read_columns(verified(PHOTO/'photometry.npz',physical['outputs']['photometry.npz']),
                         ('catalog_row','TYPE','ls_id','magnitude'))
    tables,excluded = {},{}
    for source in SOURCES:
        tables[source],excluded[source] = load_source(source,receipt)
        if not tables[source]:
            raise ValueError(f'Missing admissible native morphology for {source}: {excluded[source]}')
    OUT.mkdir(parents=True,exist_ok=True)
    out = ['qualified_migration.csv','paired_distances.csv','overlap.csv']
    csv_rows(OUT/out[0],(row for source,a in tables.items() for row in migration(source,a)))
    counts = []
    def report():
        for names in (*combinations(SOURCES,2),SOURCES):
            group = matched(names,tables)
            p = group[names[0]]
            counts.append(dict(overlap='+'.join(names),n=len(p['ls_id']),
                               primary=int(p['primary'].sum()),strict=int(p['strict'].sum())))
            yield from paired(group,photo)
    csv_rows(OUT/out[1],report())
    csv_rows(OUT/out[2],counts)
    save_json(output,dict(inputs={str(x.relative_to(ROOT)):sha(x) for x in inputs},
        excluded_morphology=excluded,outputs={n:sha(OUT/n) for n in out}))


if __name__=='__main__':
    main()
