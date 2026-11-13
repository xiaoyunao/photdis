import csv
import json
import numpy as np
import v3_1_structure_report as report
from v2_1_near import sha,save_json,atomic_npz


def test_postfit_report_preserves_missing_membership_and_shared_count_contract(tmp_path,monkeypatch):
    root=tmp_path/'data/v3_1/structure'; posterior=root/'posterior'; matrix=root/'matrices'
    old=tmp_path/'data/v3_0/independent_scale'
    for path in [root,posterior,matrix,old,tmp_path/'configs',tmp_path/'scripts',tmp_path/'src/photdis/v3_1',tmp_path/'docs']:
        path.mkdir(parents=True,exist_ok=True)
    for name in ['scripts/v3_1_structure_report.py','src/photdis/v3_1/structure_draws.py','docs/v3_1_structure_counts_protocol.md']:
        (tmp_path/name).write_text('synthetic input fixture')
    save_json(tmp_path/'configs/v3_0_independent_scale.json',{})
    ids=np.arange(512,dtype='i8')+2**54; types=np.full(512,'PSF'); original_rows=np.arange(511,-1,-1)
    atomic_npz(old/'photometry.npz',identity=ids,TYPE=types)
    available=np.arange(512)<256; probability=np.where(available,np.where(np.arange(512)<128,.95,.05),np.nan)
    atomic_npz(old/'membership.npz',identity=ids,TYPE=types,membership_available=available,
        membership_probability=probability,selected_member=available & (probability>=.9))
    save_json(old/'association.json',dict(input_sha256=sha(old/'photometry.npz'),config_sha256=sha(tmp_path/'configs/v3_0_independent_scale.json')))
    atomic_npz(root/'photometry.npz',identity=ids[original_rows],TYPE=types,original_rows=original_rows)
    save_json(root/'complete.json',dict(n=512,all_PSF=True,outputs={'photometry.npz':sha(root/'photometry.npz')}))
    mu=np.array([14.,15.]); z=np.array([np.log(.3/.7)])
    atomic_npz(matrix/'matrices.npz',identity=ids[original_rows],TYPE=types,radius_deg=np.zeros(512),common_mu=mu,
        logresponse=np.zeros((2,2,32,512,7)))
    save_json(matrix/'complete.json',dict(n=512,states=32,all_PSF=True,outputs={'matrices.npz':sha(matrix/'matrices.npz')}))
    inputs={str(p.relative_to(tmp_path)):sha(p) for p in [matrix/'complete.json',matrix/'matrices.npz']}; markers={}
    covariance=512*.3*.7*np.array([[1.,-1.],[-1.,1.]])
    for branch in ['full','gaussian']:
        folder=posterior/branch; folder.mkdir(); payload=folder/'base_m64_z256.npz'
        atomic_npz(payload,identity=ids[original_rows],TYPE=types,response_ids=np.arange(32),common_mu=mu,zeta=z,
            global_weights=np.full((32,2,1),1/64),mean=np.array([512*.3,512*.7]),covariance=covariance,
            conditional_covariance=covariance,shared_covariance=np.zeros((2,2)),per_object_member_probability=np.full(512,.3))
        marker=payload.with_suffix('.json'); save_json(marker,dict(inputs=inputs,sha256=sha(payload)))
        markers[str(marker.relative_to(posterior))]=sha(marker)
    save_json(posterior/'complete.json',dict(n=512,states=32,all_PSF=True,inputs=inputs,results=markers,
        summary=[dict(branch=b,numerically_passed=b=='full') for b in ['full','gaussian']]))
    monkeypatch.setattr(report,'ROOT',tmp_path); monkeypatch.setattr(report,'require_compute',lambda:None)
    report.main(); out=root/'report'; done=json.loads((out/'complete.json').read_text())
    assert done['membership_available']==256 and done['published_p_ge_0_9']==128
    assert not done['passed'] and all(r['sampling_passed'] for r in done['summary'])
    assert all(r['accepted']==(r['branch']=='full') for r in done['summary'])
    rows=list(csv.DictReader((out/'membership_comparison.csv').open()))
    for branch in ['full','gaussian']:
        missing=next(r for r in rows if r['branch']==branch and r['group']=='membership_missing')
        assert int(missing['n'])==256 and int(missing['membership_available_n'])==0
        assert missing['published_probability_sum']==''
        low=next(r for r in rows if r['branch']==branch and r['group']=='available_p_lt_0.9')
        assert int(low['n'])==128
    stars=list(csv.DictReader((out/'membership.csv').open()))
    assert int(stars[0]['identity'])==int(ids[-1]) and stars[0]['published_p_ge_0_9']==''
