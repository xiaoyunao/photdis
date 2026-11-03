#!/usr/bin/env python3
"""Diagnose frozen-prediction reproduction without altering any predictions."""
from baseline_v1_1_common import *
from photdis.stage1 import require_compute

def main():
    require_compute();result={}
    for name in MODELS:
        for v in ('wide','oracle','natural','balanced'):
            old=load(EXPERT/f'{name}_{v}_evaluation.npz');new=load(OUT/f'{name}_{v}_evaluation.npz');pos=np.searchsorted(old['rows'],new['rows'])
            assert np.array_equal(old['rows'][pos],new['rows'])
            dm=abs(old['mu'][pos]-new['mu']);cdf=np.cumsum(old['pdf'][pos].astype(float),axis=1)-np.cumsum(new['pdf'].astype(float),axis=1)
            worst=np.unravel_index(np.argmax(dm),dm.shape);i,j=worst
            result[name+':'+v]=dict(n=len(pos),mu_diff_quantiles=np.quantile(dm,[.5,.95,.99,1]).tolist(),
                n_mu_over3e5=int(np.sum(dm>3e-5)),max_cdf_difference=float(np.max(abs(cdf))),
                max_pdf_difference=float(np.max(abs(old['pdf'][pos]-new['pdf']))),
                worst=dict(row=int(new['rows'][i]),quantile=int(j),old=old['mu'][pos[i]].tolist(),new=new['mu'][i].tolist()))
            print(name,v,result[name+':'+v],flush=True)
    write_json(OUT/'reproduction_audit.json',result)
if __name__=='__main__':main()
