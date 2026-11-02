import numpy as np
from photdis.baseline_v0 import HistogramQRF,band_noise,features,distance_quantiles
from photdis.quantile import QuantileForest

def test_compressed_leaf_weights_and_missing_wise():
    rng=np.random.default_rng(41);m=rng.normal(size=(300,5));m[:,3:]=np.nan;y=2*m[:,0]+rng.normal(size=300);lab=np.arange(300)%4-1
    h=HistogramQRF(2,n_estimators=7,min_samples_leaf=8,max_depth=None,seed=19).fit(m,y,lab)
    q=QuantileForest(n_estimators=7,min_samples_leaf=8,random_state=19).fit(features(m,2),y)
    weights=q.predict_weights(features(m[:25],2));pdf,cl=h.distribution(features(m[:25],2))
    expected=np.column_stack([weights[:,(y>=a)&(y<b)].sum(axis=1) for a,b in zip(h.edges[:-1],h.edges[1:])])
    assert np.allclose(pdf,expected,atol=1e-6)
    assert np.allclose(cl.sum(axis=1),1)
    result=h.predict(m[:25],np.zeros((25,5)),np.arange(25))
    assert np.allclose(result['pdf'],pdf,atol=1e-6)
    assert np.max(abs(result['q_Meffective']-q.predict(features(m[:25],2))))<=.051

def test_counter_noise_shared_band_covariance():
    ids=np.arange(50000);a=band_noise(ids,0);b=np.vstack([band_noise(ids[:120],0),band_noise(ids[120:],0)])
    assert np.array_equal(a,b)
    cov=np.cov(np.array([a[:,0],a[:,0]-a[:,1],a[:,1]-a[:,2]]))
    assert abs(cov[0,1]-1)<.04 and abs(cov[1,2]+1)<.04
    assert not np.array_equal(a,band_noise(ids,1))

def test_same_g_shifts_mu_with_correct_sign():
    # A constant luminosity distribution isolates direct g propagation.
    m=np.tile([2.,1.,0.,0.,0.],(40,1));h=HistogramQRF(2,n_estimators=2,min_samples_leaf=8).fit(m,np.zeros(40),np.zeros(40,dtype=int))
    e=np.zeros((1,5));e[:,0]=.2
    ids=np.array([123]);p=h.predict(m[:1],e,ids,draws=1)
    shift=band_noise(ids,0)[0,0]*.2
    mean=(p['pdf'][0]*h.centers).sum()
    assert abs(mean-(h.centers[np.argmax(h.distribution(features(m[:1],2))[0][0])]-shift))<1e-5
    mu,d=distance_quantiles(np.array([20.]),p['q_Meffective']);assert np.all(np.diff(mu)>=0) and np.all(np.diff(d)>=0)
