#!/usr/bin/env python3
"""Explain the archived-parent membership lost in the expanded same-window replay."""
from photdis.stage1 import require_compute
from sv18_common import *
from v3_10_evidence import DEST as EVIDENCE

def main():
    require_compute()
    inputs={};d=out('cohorts')
    t=read(DEST/'benchmark',('evaluated.npz','TYPE','ls_id','region','q','specdis_d','spdist_d'),inputs)
    historical={}
    for region,label in ((2,'south_low'),(3,'north')):
        historical[region]=read(EVIDENCE/label,('references.npz','TYPE','ls_id','row_state','evaluation_row',
             'sky_role','class_label','gaia_state','specdis_quality','spdist_quality'),inputs)
    rows=[]
    for ref in ('specdis','spdist'):
        p=DEST/'benchmark'/(ref+'_historical_parent_delta.npz')
        with np.load(p) as a:
            ids=a['removed']
        inputs[p.name]=sha(p)
        j=index_join(ids,t['ls_id']);assert (j>=0).all()
        reason=np.full(len(j),'other_historical_admission',dtype='U40')
        reg=t['region'][j]
        reason[~np.isin(reg,(2,3))]='outside_original_native_domains'
        for region,old in historical.items():
            use=np.flatnonzero(reg==region)
            ix=index_join(ids[use],old['ls_id'])
            reason[use[ix<0]]='absent_from_reference_carrier'
            ii=use[ix>=0];oo=ix[ix>=0]
            checks=(('historical_strict_photometry',old['row_state'][oo]==7),
                    ('historical_evaluation_eligibility',old['evaluation_row'][oo]),
                    ('historical_sky_role',old['sky_role'][oo]==4),
                    ('historical_dwarf_label',old['class_label'][oo]==0),
                    ('historical_reference_quality',old[ref+'_quality'][oo]),
                    ('historical_weak_geometry',np.isin(old['gaia_state'][oo],(3,4,5))))
            alive=np.ones(len(ii),bool)
            for label,keep in checks:
                reason[ii[alive&~keep]]=label
                alive&=keep
        for r in np.unique(reason):
            for region in np.unique(reg[reason==r]):
                k=(reason==r)&(reg==region);ii=j[k]
                rows.append(dict(reference=ref,first_failing_admission=r,region=int(region),
                           **metric(t['q'][ii],t[ref+'_d'][ii])))
        assert sum(r['n'] for r in rows if r['reference']==ref)==len(ids)
    csv_rows(d/'expanded_parent_differences.csv',rows)
    complete(d,inputs)

if __name__=='__main__':
    main()
