#!/usr/bin/env python3
"""Identity and sky-cell relations of all fit, calibration and evaluation cohorts."""
from photdis.stage1 import require_compute
from sv18_common import *
from final_catalogue_benchmark import load_parent, masks, published
from release_readiness_compare import available
from v3_9_catalog import OUT as V39
from v3_10_selection import DEST as CLASS
from paper_v2_scan import training


def main():
    require_compute()
    d=out('provenance');inputs={}
    t=cached(inputs,('TYPE','ls_id','source_id','RA','DEC','region','heldout','specdis_stage','spdist_stage','state'))
    use=t['heldout']&(t['state']==3)&((t['specdis_stage']==3)|(t['spdist_stage']==3))
    t={k:v[use] for k,v in t.items()}
    s=read_columns(ROOT/'data/v3_0/references.npz',
        ('TYPE','ls_id','source_id','RA','DEC','role','identity','class_label','historical_fit','cell'))
    qsky=(np.floor(s['RA']/10).astype(int)+3*np.floor((s['DEC']+90)/10).astype(int))%5
    role_map=[]
    for historical in np.unique(s['role']):
        for current in range(5):
            chosen=(s['role']==historical)&(qsky==current)
            role_map.append(dict(historical_role=int(historical),northern_qsky_role=current,
                                 n=int(chosen.sum()),historical_sky_cells=len(np.unique(s['cell'][chosen]))))
    csv_rows(d/'historical_qsky_roles.csv',role_map)
    rr=read(ROOT/'data/v3_0/controls',('training.npz','TYPE','rows'),inputs)
    roles=[]
    def add(name,src,ix):
        roles.append((name,{k:src[k][ix] for k in ('ls_id','source_id','RA','DEC')}))
    add('south_qrf_fit',s,rr['rows'])
    for ref in ('specdis','spdist'):
        p=ROOT/f'data/v3_0/calibration_{ref}_inputs.npz'
        q=read_columns(p,('TYPE','identity','rows'))
        assert (q['TYPE']=='PSF').all();np.testing.assert_array_equal(s['identity'][q['rows']],q['identity'])
        inputs[str(p.relative_to(ROOT))]=sha(p)
        add('south_'+ref+'_scale_calibration',s,q['rows'])
    trainrows=rr['rows'][s['class_label'][rr['rows']]>=0]
    add('south_original_classifier_fit',s,trainrows)
    add('south_original_classifier_temperature',s,np.flatnonzero((s['role']==3)&(s['class_label']>=0)))
    add('south_historical_fit',s,np.flatnonzero(s['historical_fit']))
    nr=read(V39/'north/native_model',('roles.npz','TYPE','ls_id','source_id','role'),inputs)
    # Native reference caches provide the original sky coordinates for every role.
    coords=[]
    for p in sorted((V39/'north/reference').glob('[0-9][0-9].json')):
        r=json.loads(p.read_text())
        for name,h in r['outputs'].items():
            if name.endswith('.npz'):
                x=read_columns(verified(p.with_name(name),h),('TYPE','ls_id','RA','DEC'))
                j=index_join(x['ls_id'],nr['ls_id']);k=j>=0
                if k.any(): coords.append({key:v[k] for key,v in x.items()})
    from v3_9_references import join
    x=join(coords);ii=index_join(nr['ls_id'],x['ls_id']);assert (ii>=0).all()
    nr['RA']=x['RA'][ii];nr['DEC']=x['DEC'][ii]
    for role in np.unique(nr['role']):
        add('north_'+role,nr,np.flatnonzero(nr['role']==role))
    allref=cached(inputs,('TYPE','ls_id','source_id','RA','DEC'))
    for side in ('south','north'):
        q=read(CLASS/side,('scores.npz','TYPE','ls_id','training','calibration','held_out'),inputs)
        for role in ('training','calibration','held_out'):
            ids=q['ls_id'][q[role]]
            # All matched references have native coordinates in the v2 census.
            ii=index_join(ids,allref['ls_id']);valid=ii>=0
            src={k:v[ii[valid]] for k,v in allref.items() if k!='TYPE'}
            roles.append((side+'_selection_'+role,src))
            csv_rows(d/(side+'_'+role+'_unlinked.csv'),[dict(expected=len(ids),linked=int(valid.sum()),unlinked=int((~valid).sum()))])
    del allref
    v,vm=load_parent();pub,_,_=published(v,vm['outputs']['ids.npy'])
    cohorts={'expanded_spectral_holdout':t}
    for ref in ('specdis','spdist'):
        keep=masks(v)['weak_gaia_dwarfs']&available(pub['Photdis'])&available(pub['BJ_photogeometric'])
        keep&=(v[ref+'_kpc']>=2)&(v[ref+'_kpc']<5)
        cohorts['F15_'+ref]={k:v[k][keep] for k in ('ls_id','source_id','RA','DEC')}
    rows=[];ledger=[]
    for name,r in roles:
        skycell=(np.floor(r['RA']/10).astype(int)*18+np.minimum(np.floor((r['DEC']+90)/10).astype(int),17))
        for cohort,z in cohorts.items():
            ls=np.isin(z['ls_id'],r['ls_id'])
            ga=(z['source_id']>0)&np.isin(z['source_id'],r['source_id'][r['source_id']>0])
            cell=np.floor(z['RA']/10).astype(int)*18+np.minimum(np.floor((z['DEC']+90)/10).astype(int),17)
            same=np.isin(cell,skycell)
            rows.append(dict(role=name,cohort=cohort,role_n=len(r['ls_id']),cohort_n=len(z['ls_id']),
                native_overlap=int(ls.sum()),gaia_overlap=int(ga.sum()),either_overlap=int((ls|ga).sum()),
                fit_cells=len(np.unique(skycell)),evaluation_cells=len(np.unique(cell)),
                common_cells=len(np.intersect1d(cell,skycell)),evaluated_rows_in_common_cells=int(same.sum())))
        ledger.extend(dict(role=name,cell=int(k),n=int((skycell==k).sum())) for k in np.unique(skycell))
    csv_rows(d/'overlap_matrix.csv',rows);csv_rows(d/'role_sky_cells.csv',ledger)
    from v3_13_clusters import SYSTEMS
    systems=[]
    for name,rr in roles:
        for cluster,(center,_) in SYSTEMS.items():
            dra=(rr['RA']-center[0]+180)%360-180
            offset=np.hypot(dra*np.cos(np.deg2rad(center[1])),rr['DEC']-center[1])
            systems.append(dict(role=name,system=cluster,radius_deg=.8,n=int((offset<.8).sum())))
    csv_rows(d/'known_system_training_regions.csv',systems)
    extra=np.zeros(len(t['ls_id']),bool)
    for name,r in roles:
        if 'scale_calibration' in name or 'temperature' in name or name.endswith('_selection_training') or name.endswith('_selection_calibration'):
            extra|=np.isin(t['ls_id'],r['ls_id'])|((t['source_id']>0)&np.isin(t['source_id'],r['source_id'][r['source_id']>0]))
    atomic_npz(d/'additional_exclusions.npz',TYPE=t['TYPE'],ls_id=t['ls_id'],additional_calibration_or_selection_overlap=extra)
    complete(d,inputs,role_count=len(roles),extra_exclusions=int(extra.sum()))


if __name__=='__main__':
    main()
