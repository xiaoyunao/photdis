#!/usr/bin/env python3
"""Connect fit/calibration identities directly to complete native reference inventories."""
from photdis.stage1 import require_compute
from halo_common import *
from sv18_common import cached,index_join
from v3_9_catalog import OUT as V39
from v3_10_selection import DEST as CLASS


def main():
    require_compute();d=folder('roles');inputs={};roles=[];ledger=[]
    s=read_columns(ROOT/'data/v3_0/references.npz',('TYPE','ls_id','source_id','RA','DEC','role','identity','class_label','historical_fit'))
    assert (s['TYPE']=='PSF').all()
    inputs['south_original_references']=sha(ROOT/'data/v3_0/references.npz')
    def add(name,t,ii):
        assert (t['TYPE'][ii]=='PSF').all()
        roles.append((name,{k:t[k][ii] for k in ('ls_id','source_id','RA','DEC')}))
    a=read(ROOT/'data/v3_0/controls',('training.npz','TYPE','rows'),inputs)
    add('south_qrf_fit',s,a['rows'])
    add('south_original_classifier_fit',s,a['rows'][s['class_label'][a['rows']]>=0])
    add('south_original_classifier_temperature',s,np.flatnonzero((s['role']==3)&(s['class_label']>=0)))
    add('south_historical_fit',s,np.flatnonzero(s['historical_fit']))
    for ref in ('specdis','spdist'):
        p=ROOT/f'data/v3_0/calibration_{ref}_inputs.npz'
        z=read_columns(p,('TYPE','rows','identity'));assert (z['TYPE']=='PSF').all()
        np.testing.assert_array_equal(s['identity'][z['rows']],z['identity'])
        inputs[ref+'_calibration']=sha(p);add('south_'+ref+'_scale_calibration',s,z['rows'])
    # These are the actual native caches used by v3.10 features, including
    # Gaia-only/class-only sources absent from the supported spectral union.
    native={}
    for side in ('south','north'):
        pieces=[]
        for p in sorted((V39/side/'reference').glob('[0-9][0-9].json')):
            r=json.loads(p.read_text());inputs[str(p.relative_to(ROOT))]=sha(p)
            for name,h in r['outputs'].items():
                if name.endswith('.npz'):
                    z=read_columns(verified(p.with_name(name),h),('TYPE','ls_id','source_id','RA','DEC'))
                    assert (z['TYPE']=='PSF').all();pieces.append(z)
        assert pieces,side
        z={k:np.concatenate([x[k] for x in pieces]) for k in pieces[0]}
        assert len(np.unique(z['ls_id']))==len(z['ls_id'])
        native[side]=z
        q=read(CLASS/side,('scores.npz','TYPE','ls_id','training','calibration','held_out'),inputs)
        for role in ('training','calibration','held_out'):
            ids=q['ls_id'][q[role]];ix=index_join(ids,z['ls_id']);valid=ix>=0
            ledger.append(dict(role=side+'_selection_'+role,expected=len(ids),linked=int(valid.sum()),unlinked=int((~valid).sum())))
            assert valid.all(),(side,role,int((~valid).sum()))
            add(side+'_selection_'+role,z,ix)
            if role=='held_out':add(side+'_working_rule_development',z,ix)
    q=read(V39/'north/native_model',('roles.npz','TYPE','ls_id','source_id','role'),inputs)
    ix=index_join(q['ls_id'],native['north']['ls_id']);assert (ix>=0).all()
    for role in np.unique(q['role']):
        use=q['role']==role;add('north_'+role,native['north'],ix[use])
        ledger.append(dict(role='north_'+role,expected=int(use.sum()),linked=int(use.sum()),unlinked=0))
    t=cached(inputs,('TYPE','ls_id','source_id','RA','DEC','gaia_stage','specdis_stage','spdist_stage','heldout'))
    bits=np.zeros(len(t['ls_id']),'u4');exclude=np.zeros(len(bits),bool);counts=[];sky=[]
    gaia_only=(t['gaia_stage']==3)&(t['specdis_stage']!=3)&(t['spdist_stage']!=3)
    for i,(name,r) in enumerate(roles):
        byls=np.isin(t['ls_id'],r['ls_id'])
        byga=(t['source_id']>0)&np.isin(t['source_id'],r['source_id'][r['source_id']>0])
        overlap=byls|byga;bits[overlap]|=1<<i
        # Historical development is disclosed rather than relabelled as blind.
        direct=(('fit' in name and name!='south_historical_fit') or 'scale_calibration' in name or
                'temperature' in name or name.endswith('_selection_training') or
                name.endswith('_selection_calibration') or name=='north_calibration')
        if direct:exclude|=overlap
        counts.append(dict(bit=i,role=name,role_n=len(r['ls_id']),native_overlap=int(byls.sum()),
            gaia_alias_overlap=int(byga.sum()),either_overlap=int(overlap.sum()),
            gaia_only_overlap=int((overlap&gaia_only).sum()),direct_fit_or_calibration=direct))
        cell=np.floor(r['RA']/10).astype(int)*18+np.minimum(np.floor((r['DEC']+90)/10).astype(int),17)
        for c,n in zip(*np.unique(cell,return_counts=True)):sky.append(dict(role=name,cell=c,n=n))
    atomic_npz(d/'reference_roles.npz',TYPE=t['TYPE'],ls_id=t['ls_id'],source_id=t['source_id'],
               role_bits=bits,direct_fit_or_calibration=exclude,gaia_only=gaia_only)
    csv_rows(d/'role_links.csv',ledger);csv_rows(d/'reference_overlap.csv',counts);csv_rows(d/'role_sky.csv',sky)
    complete(d,inputs,reference_rows=len(bits),gaia_only_rows=int(gaia_only.sum()),
        gaia_only_excluded=int((gaia_only&exclude).sum()),excluded=int(exclude.sum()),role_count=len(roles),
        scope='all cached native/Gaia reference identities; direct native source inventory links, including unsupported and Gaia-only classifier roles')


if __name__=='__main__':main()
