import numpy as np
import baseline_v1_4_experiments as ex
from photdis.baseline_v0 import HistogramQRF


def test_cal_workpoint_excludes_unknown_and_never_uses_test():
    n=120
    t=dict(identity=np.arange(n),split=np.r_[np.ones(60),np.full(60,2)],cell=np.zeros(n),
           class_label=np.r_[np.zeros(50),np.ones(10),np.zeros(30),np.ones(20),np.full(10,-1)],
           class_eligible=np.r_[np.ones(110,bool),np.zeros(10,bool)],r=np.full(n,19.),
           B=np.zeros(n,bool),C=np.zeros(n,bool),optical_error=np.full(n,.02))
    prob=np.tile([.5,.4,.1],(n,1));prob[49,1]=.9
    a=ex.class_scores(t,prob)
    assert a['cal_dwarf_fpr']<=.02
    assert a['groups']['all']['unknown']==10
    changed=prob.copy();changed[60:,1]=.99
    assert ex.class_scores(t,changed)['threshold']==a['threshold']


def test_full_gaia_keeps_negative_and_normalizes_snr_bins():
    edges=np.linspace(-1,9,51);pdf=np.full((3,50),1/50)
    t=dict(identity=np.arange(3),g0=np.array([18.,19.,20.]),p_corr=np.array([-.2,0.,.3]),gaia_parallax_error=np.array([.1,.2,.3]))
    p=ex.gaia_values(t,pdf,edges)
    assert np.isfinite(p['full_log_density']).all()
    assert np.all((p['full_pit']>=0)&(p['full_pit']<=1))
    np.testing.assert_allclose(p['bin_probability'].sum(axis=1),1,atol=1e-10)


def test_resumable_distance_chunk_preserves_ids_and_signed_gaia(tmp_path):
    rng=np.random.default_rng(4);n=160
    mags=rng.normal(17,.2,(n,5));y=rng.normal(5,.5,n)
    model=HistogramQRF(2,n_estimators=3,min_samples_leaf=8).fit(mags,y,np.arange(n)%3)
    t={k:mags[:,i] for i,k in enumerate(ex.BANDS)}
    t.update({k:np.full(n,.02) for k in ex.ERR})
    t.update(identity=np.arange(n,dtype='i8'),p_corr=np.linspace(-.2,.2,n),gaia_parallax_error=np.full(n,.1),
             parallax_eligible=np.ones(n,bool),anchor_astrometry=np.zeros(n,bool),nonstellar_risk=np.zeros(n,bool),
             label_source=np.ones(n),ref_distance=np.full(n,2.),specdis_distance=np.full(n,2.),spdist_distance=np.full(n,np.nan))
    ex._T=t;ex._MODEL=model;ex._MODE='wide';ex._DEST=tmp_path
    path=ex.distance_chunk((0,np.arange(8)));saved=ex.load(path)
    np.testing.assert_array_equal(saved['identity'],np.arange(8))
    assert np.isfinite(saved['full_log_density']).all()
    assert np.all(np.diff(saved['mu'],axis=1)>=0)
    assert ex.distance_chunk((0,np.arange(8)))==path
