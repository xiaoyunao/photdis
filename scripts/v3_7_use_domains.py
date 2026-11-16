#!/usr/bin/env python3
"""Tabulate reference evidence in catalog-observable domains."""
import json
from pathlib import Path

import numpy as np

from photdis.stage1 import require_compute
from photdis.v3.metrics import distance_summary
from v2_1_near import ROOT, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_6_distance_evidence import SHELLS
from v3_6_vac_distances import DISTANCE_EDGES, read_columns
from v3_7_reference_comparison import EVIDENCE, PHOTO, SOURCES, load_source

OUT = ROOT/'data/v3_7/use_domains'
DOMAIN = ROOT/'configs/v3_4_near_release_domain.json'


def ranges(x,edges):
    for low,high in zip(edges[:-1],edges[1:]):
        yield low,high,(x>=low)&(x<high)


def summarize(source,a,w1):
    d = a['reference_kpc']
    q = a['qrf_quantiles_kpc']
    mod = 10+5*np.log10(q)
    ref = 10+5*np.log10(d)
    gr = a['dereddened_g_r']
    r = a['observed_r']
    sets = [('all',np.ones(len(r),bool)),('known_dwarf',a['consensus_class']==0),
            ('known_giant',a['consensus_class']==1)]
    for r0,r1,rr in ranges(r,(-np.inf,16.,18.,19.,20.)):
        for c0,c1,cc in ranges(gr,(-np.inf,.5,.8,1.1,1.4,np.inf)):
            cell = np.flatnonzero(rr&cc)
            for w in (-1,*np.unique(w1[cell])):
                b = cell if w==-1 else cell[w1[cell]==w]
                for j,(lo,hi) in enumerate(zip(DISTANCE_EDGES[:-1],DISTANCE_EDGES[1:])):
                    ix = b[(q[b,2]>=lo)&(q[b,2]<hi)]
                    for selection,accepted in (('primary',a['primary']),('strict',a['strict'])):
                        selected = ix[accepted[ix]]
                        for label,group in sets:
                            k = selected[group[selected]]
                            summary = distance_summary(mod[k],ref[k])
                            split = np.histogram(d[k],DISTANCE_EDGES)[0]
                            yield dict(source=source,selection=selection,reference_class=label,
                                       observed_r_lo=r0,observed_r_hi=r1,gr_lo=c0,gr_hi=c1,
                                       W1_state=int(w),predicted_shell=SHELLS[j],
                                       **summary,**{f'reference_shell_{l}':int(v) for l,v in enumerate(split)})


def main():
    require_compute()
    marker = OUT/'complete.json'
    if marker.exists():
        raise FileExistsError(marker)
    paths = (EVIDENCE/'complete.json',PHOTO/'complete.json',DOMAIN,Path(__file__).resolve())
    evidence,photo = (json.loads(p.read_text()) for p in paths[:2])
    p = read_columns(verified(PHOTO/'photometry.npz',photo['outputs']['photometry.npz']),
                     ('catalog_row','ls_id','TYPE','band_state'))
    if (np.diff(p['catalog_row'])<=0).any():
        raise ValueError('Reference photometry is not ordered by catalog row')
    excluded = {}

    def rows():
        for source in SOURCES:
            a,excluded[source] = load_source(source,evidence)
            i = np.searchsorted(p['catalog_row'],a['catalog_row'])
            np.testing.assert_array_equal(a['ls_id'],p['ls_id'][i])
            if not (p['TYPE'][i]=='PSF').all():
                raise ValueError('Only verified native PSFs may enter')
            yield from summarize(source,a,p['band_state'][i,4])

    OUT.mkdir(parents=True,exist_ok=True)
    table = OUT/'observable_domains.csv'
    csv_rows(table,rows())
    save_json(marker,dict(inputs={str(x.relative_to(ROOT)):sha(x) for x in paths},
                          excluded_morphology=excluded,outputs={table.name:sha(table)}))


if __name__=='__main__':
    main()
