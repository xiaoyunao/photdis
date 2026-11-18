#!/usr/bin/env python3
"""Recompute the fixed simple baseline at the production 128 Sobol draws."""
from scipy.spatial import cKDTree
from photdis.stage1 import require_compute
from sv18_common import *
from sv18_benchmark import predict, comparison
from v3_9_catalog import from_north

def main():
    require_compute()
    d=out('numerics');inputs={}
    c=CONFIG.with_name('science_validation_20261118_numerical.json')
    if not c.exists():c=ROOT/'configs'/c.name
    inputs['numerical_protocol']=sha(c)
    t=read(DEST/'benchmark',('evaluated.npz','TYPE','ls_id','source_id','RA','DEC','region','m','err','q',
       'simple','bj','pop','was_f15','specdis_d','spdist_d'),inputs)
    use=t['was_f15'].any(axis=1)
    t={key:value[use] for key,value in t.items()}
    q=np.full_like(t['q'],np.nan)
    for side,north in (('south',False),('north',True)):
        b=read(DEST/'benchmark',(side+'_simple_relation.npz','centers','hist','Mg_edges','training_count'),inputs)
        model=(cKDTree(b['centers']),b['hist'],b['Mg_edges'],b['training_count'],b['centers'])
        offset=from_north()[0]['mu_offset'] if north else 0.
        rows=np.flatnonzero((t['region']==3)==north)
        for start in range(0,len(rows),512):
            j=rows[start:start+512]
            q[j]=predict(model,t['m'][j],t['err'][j],t['source_id'][j],offset,draws=128)[0]
    result=[]
    for i,ref in enumerate(('specdis','spdist')):
        keep=t['was_f15'][:,i]
        shift=5*np.log10(q[keep,2]/t['simple'][keep,2])
        a=metric(t['simple'][keep],t[ref+'_d'][keep])
        b=metric(q[keep],t[ref+'_d'][keep])
        result.append(dict(reference=ref,n=int(keep.sum()),median_shift_mu=np.median(shift),
             p95_absolute_shift_mu=np.quantile(abs(shift),.95),nmad32=a['nmad'],nmad128=b['nmad'],
             numerical_nmad_change=b['nmad']-a['nmad']))
        groups=dict(headline=keep,ordinary=keep&(t['pop']==1),metal_poor=keep&(t['pop']==2),
                    north=keep&(t['region']==3),south=keep&(t['region']!=3))
        comparison(t,dict(Photdis=t['q'],Simple=q,BJ=t['bj']),groups,t[ref+'_d'],sky(t),d,ref)
    csv_rows(d/'baseline_integration.csv',result)
    atomic_npz(d/'numerical_predictions.npz',TYPE=t['TYPE'],ls_id=t['ls_id'],simple128=q,simple32=t['simple'])
    complete(d,inputs,scope='same fixed model and labels, numerical quadrature refinement only')

if __name__=='__main__':
    main()
