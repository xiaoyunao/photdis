import numpy as np
from v3_1_validation_channels import associate_mws,join_vac


def test_same_ID_channels_retain_binary_flags_conflicts_and_unknowns(tmp_path):
    ids=np.arange(9007199254740993,9007199254740996,dtype='i8'); target=np.array([11,22,33],dtype='i8')
    source=dict(identity=ids,source_id=ids,rvs_primary=np.ones(3,bool),rvs_sn_r=np.full(3,50.),rvs_targetid=target,
        rvs_feh=np.full(3,-1.5),rvs_feh_err=np.full(3,.1),rvs_teff=np.full(3,5000.),rvs_logg=np.array([4.6,2.,3.5]))
    np.savez(tmp_path/'sources.npz',**source)
    np.savez(tmp_path/'source_evidence.npz',class_label=np.array([0,1,2],dtype='i1'),rvs_quality_reason=np.zeros(3,'i4'),
        rvs_parameter_quality=np.ones(3,bool),redrock_nonstellar_evidence=np.zeros(3,bool),
        class_identity_conflict=np.array([False,True,False]),target_identity_conflict=np.zeros(3,bool))
    np.savez(tmp_path/'specdis_raw.npz',targetid=target,source_id=ids,dist=np.array([4.,7.,9.]),
        disterr=np.array([.3,.4,1.]),binary_flag=np.array([2,1,0]))
    np.savez(tmp_path/'spdist_raw.npz',targetid=target,source_id=ids,g0=np.full(3,15.),mg_50=np.full(3,3.),
        mg_16=np.full(3,2.9),mg_84=np.full(3,3.1),e_gmag=np.full(3,.1),flag_good=np.ones(3,bool),qflag_input=np.ones(3,bool))
    query=np.r_[ids,np.int64(9007199254740996),np.int64(0)]
    value,audit=associate_mws(dict(identity=query,source_id=query),tmp_path)
    np.testing.assert_array_equal(value['mws_class_label'],[0,-1,2,-1,-1])
    np.testing.assert_array_equal(value['specdis_valid'],[True,False,True,False,False])
    np.testing.assert_array_equal(value['spdist_valid'],[True,False,True,False,False])
    np.testing.assert_array_equal(value['specdis_raw_binary_flag'],[2,1,0,-1,-1])
    assert audit['specdis_qualified_non1_binary_flag']==2 and audit['identity_conflicts']==1


def test_duplicate_VAC_source_refused_unless_exact_pair_resolves_it():
    source=np.array([9007199254740993]*2,dtype='i8')
    raw=dict(source_id=source,targetid=np.array([11,22],dtype='i8'))
    index,fallback=join_vac(np.array([11,0,0]),np.array([source[0],source[0],0]),raw)
    np.testing.assert_array_equal(index,[0,-1,-1]); assert not fallback.any()
