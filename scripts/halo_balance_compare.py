#!/usr/bin/env python3
"""Separate balancing effects from changes of estimand and common-support population."""
from photdis.stage1 import require_compute
from halo_common import *
from sv18_common import cached,index_join
from sv18_populations import match,cells
from halo_balance import contrast


def median(x,w):
    order=np.argsort(x);s=np.cumsum(w[order]);return np.interp(s[-1]/2,s,x[order]) if s[-1]>0 else np.nan


def main():
    require_compute();d=folder('balance_comparison');inputs={};rows=[]
    t=cached(inputs,('TYPE','ls_id','region','m','observed_r','err','EBV','pop','q','specdis_d','spdist_d','gaia_d'))
    z=read(OUT/'balance',('balanced_ids.npz','TYPE','ls_id','reference','lo','weight'),inputs)
    for ref in ('specdis','spdist','gaia'):
        for lo in (2,5):
            use=(z['reference']==ref)&(z['lo']==lo)
            if not use.any():continue
            ids=z['ls_id'][use];j=index_join(ids,t['ls_id']);assert (j>=0).all()
            a={k:v[j] for k,v in t.items()};delta=5*np.log10(a['q'][:,2]/a[ref+'_d'])
            y=(a['pop']==2).astype(int);w=z['weight'][use]
            old,cc,cw=match(cells(a,'original',a[ref+'_d']),a['pop'],delta)
            rows.append(dict(reference=ref,lo=lo,n=len(j),
                old_coarse_cell_median_contrast_same_ids=old,
                coarse_weighted_mean_contrast_same_ids=contrast(delta,y,cw),
                continuous_weighted_mean_contrast=contrast(delta,y,w),
                continuous_weighted_median_contrast=median(delta[y==1],w[y==1])-median(delta[y==0],w[y==0]),
                coarse_common_fraction=float((cw>0).mean()),continuous_common_fraction=float((w>0).mean()),
                common_support_intersection=int(((cw>0)&(w>0)).sum())))
    csv_rows(d/'estimand_population_comparison.csv',rows)
    complete(d,inputs,scope='point comparisons on exactly the same fully role-excluded identities; coarse-cell medians, weighted means and weighted marginal medians are distinct estimands',
        intervals='only the continuous weighted mean has refitted sky-bootstrap intervals in balance/balanced_metallicity.csv')


if __name__=='__main__':main()
