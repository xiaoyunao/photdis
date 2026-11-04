import json
import numpy as np
import baseline_v1_4_assemble as a


def test_independent_labels_and_historical_identity_in_full_assembly(tmp_path,monkeypatch):
    out=tmp_path/'data/baseline_v1_4';out.mkdir(parents=True)
    old=tmp_path/'data/baseline_v0';old.mkdir()
    monkeypatch.setattr(a,'ROOT',tmp_path);monkeypatch.setattr(a,'OUT',out)
    monkeypatch.setattr(a,'require_compute',lambda:None)
    monkeypatch.setattr(a,'parent_coverage',lambda native:dict(n=26000,matched=26000))
    monkeypatch.setattr(a,'galactic',lambda ra,dec:(np.zeros(len(ra)),np.full(len(ra),50)))
    monkeypatch.setattr(a,'spatial_split',lambda ra,dec,bounds:(np.arange(len(ra)),np.arange(len(ra))%3))
    n=6;sid=np.arange(11,17,dtype='i8');target=np.arange(21,27,dtype='i8')
    native=dict(ls_id=np.arange(31,37,dtype='i8'),source_id=sid,RA=np.arange(n,dtype=float),DEC=np.full(n,45.),EBV=np.full(n,.01),RELEASE=np.full(n,11010),
                grz_base=np.ones(n,bool),grz_clean=np.ones(n,bool),w1_clean=np.zeros(n,bool),w2_clean=np.zeros(n,bool),snr_w1=np.ones(n),snr_w2=np.ones(n),
                sigma_g=np.full(n,.1),sigma_r=np.full(n,.1),sigma_z=np.full(n,.1),g0=np.full(n,19.5),r0=np.full(n,19.),z0=np.full(n,18.8),r=np.full(n,19.),i_usable=np.ones(n,bool))
    np.savez(out/'native_photometry.npz',**native)
    (out/'native_photometry.json').write_text(json.dumps(dict(sha256=a.sha(out/'native_photometry.npz'))))
    np.savez(out/'zero_photometry.npz',**{k:v[:0] for k,v in native.items()})
    np.savez(out/'zero_match.npz',identity=np.empty(0,'i8'),candidate_row=np.empty(0,'i8'),unknown_epoch=np.empty(0,bool),separation_arcsec=np.empty(0))
    (out/'zero_match.json').write_text(json.dumps(dict(sha256=a.sha(out/'zero_match.npz'),photometry_sha256=a.sha(out/'zero_photometry.npz'))))
    np.savez(out/'zero_candidates_raw.npz',ls_id=np.empty(0,'i8'))
    (out/'zero_candidates_raw.json').write_text(json.dumps(dict(sha256=a.sha(out/'zero_candidates_raw.npz'),candidate_sha256=a.sha(out/'zero_photometry.npz'))))
    (out/'zero_audit.json').write_text('{}')
    (out/'freeze.json').write_text(json.dumps(dict(hashes={},legacy_bounds=[])))
    source=dict(identity=sid,source_id=sid,original_row=np.arange(n),rvs_primary=np.ones(n,bool),rvs_sn_r=np.full(n,20.),rvs_targetid=target,split=np.arange(n)%3,cell=np.arange(n))
    np.savez(out/'sources.npz',**source)
    evidence=dict(identity=sid,source_id=sid,original_row=np.arange(n),class_label=np.arange(n)%3,
                  class_identity_conflict=np.zeros(n,bool),target_identity_conflict=np.zeros(n,bool),redrock_nonstellar_evidence=np.zeros(n,bool),contaminant_candidate=np.zeros(n,bool),
                  anchor_astrometry=np.zeros(n,bool),astro_quality=np.ones(n,bool),zpt_valid=np.ones(n,bool),p_corr=np.full(n,-.2),gaia_parallax_error=np.full(n,.1))
    np.savez(out/'source_evidence.npz',**evidence)
    np.savez(out/'specdis_raw.npz',targetid=target[:1],source_id=sid[:1],dist=np.array([2.]),disterr=np.array([.3]),binary_flag=np.ones(1))
    np.savez(out/'spdist_raw.npz',targetid=target[:1],source_id=sid[:1],flag_good=np.ones(1,bool),qflag_input=np.ones(1,bool),g0=np.array([19.]),e_gmag=np.array([.03]),mg_16=np.array([3.]),mg_50=np.array([4.]),mg_84=np.array([5.]))
    np.savez(old/'references.npz',identity=sid[:1],ls_id=native['ls_id'][:1],desi_match=np.ones(1,bool),desi_match_ambiguous=np.zeros(1,bool),desi_object_id=target[:1].astype('U'))
    for name,key in [('sources','source_sha256'),('specdis_raw','sha256'),('spdist_raw','sha256'),('source_evidence','output_sha256')]:
        (out/(name+'.json')).write_text(json.dumps({key:a.sha(out/(name+'.npz'))}))
    # Only code-provenance hashing is mocked; all artifact hashes remain real.
    original_sha=a.sha
    monkeypatch.setattr(a,'sha',lambda path:'test-code' if str(path).endswith(('.py',)) else original_sha(path))
    a.main()
    t=a.load(out/'references.npz');report=json.loads((out/'assembly.json').read_text())
    assert all(report['checks'].values())
    assert t['split'][0]==3 and t['historical_identity'][0]
    assert t['class_train_eligible'][1:].all()  # No distance or precise-error requirement.
    assert not t['fit_eligible'].any()  # Distance training retains its own precision cut.
    assert t['parallax_eligible'].all() and (t['p_corr']<0).all()
    assert len(np.unique(t['identity']))==n
