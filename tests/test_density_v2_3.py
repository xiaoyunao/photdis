import numpy as np
from photdis.density_v2_3 import observation_covariance, inverse3, conditional_latent, ErrorDeconvolvedMixture


def test_correlated_errors():
    s=observation_covariance(np.array([[.1,.2,.3]]),np.array([.4]))[0]
    a=np.array([[1,-1,0,0],[0,1,-1,0],[1,0,0,-1]])
    np.testing.assert_allclose(s,a@np.diag([.01,.04,.09,.16])@a.T)


def test_inverse_and_latent_posterior():
    rng=np.random.default_rng(42);a=rng.normal(size=(30,3,3));s=a@a.transpose(0,2,1)+np.eye(3)*.01
    inv,log=inverse3(s);np.testing.assert_allclose(inv,np.linalg.inv(s),rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(log,np.linalg.slogdet(s)[1])
    y=rng.normal(size=(30,3));v=np.eye(3)*2
    b,B,_=conditional_latent(y,s,np.zeros(3),v)
    expected=np.linalg.inv(np.linalg.inv(s)+np.eye(3)/2)
    np.testing.assert_allclose(B,expected,rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(b,np.einsum('nij,njk,nk->ni',expected,np.linalg.inv(s),y),atol=1e-10)


def test_shared_g_conditional_and_pdf():
    model=ErrorDeconvolvedMixture(1);model.weights=np.ones(1);model.means=np.array([[.5,.2,5.]])
    model.covariances=np.array([[[.08,.01,.12],[.01,.06,.04],[.12,.04,.5]]]);model.class_frequency=np.array([[.5,0,0,.5]])
    m=np.array([[18.,17.4,17.1]]);e=np.array([[.2,.1,.1]])
    w,mean,sigma=model.conditional(m,e)
    v=model.covariances[0];E=observation_covariance(e,[0])[0];cross=v[2,:2]-[.04,0]
    T=v[:2,:2]+E[:2,:2];delta=np.array([.6,.3])-model.means[0,:2]
    np.testing.assert_allclose(mean[0,0],5+cross@np.linalg.solve(T,delta))
    np.testing.assert_allclose(sigma[0,0]**2,v[2,2]+.04-cross@np.linalg.solve(T,cross))
    p=model.predict(m,e,np.linspace(-30,40,701));np.testing.assert_allclose(p['pdf'].sum(),1)
    assert p['weights'][0,3]==.5 and 0<=p['qnear'][0]<=1


def test_deconvolution_recovers_latent_variance():
    rng=np.random.default_rng(9);n=6000;s=np.tile(np.diag([.4,.2,.5]),(n,1,1))
    true=np.diag([.3,.4,.6]);y=rng.multivariate_normal([0,0,0],true+s[0],size=n)
    model=ErrorDeconvolvedMixture(1,max_iter=256,tol=1e-6).fit(y,s,np.full(n,-1))
    assert model.converged
    np.testing.assert_allclose(np.diag(model.covariances[0]),np.diag(true),atol=.04)
    assert model.class_frequency[0,3]==1
    assert np.min(np.diff(model.history))>-1e-7


def test_tiny_right_tail_is_retained():
    from scipy.special import ndtr
    model=ErrorDeconvolvedMixture(1);model.weights=np.ones(1);model.means=np.zeros((1,3));model.covariances=np.array([np.eye(3)]);model.class_frequency=np.array([[0.,0.,0.,1.]])
    # Direct CDF subtraction returns zero here; Gaussian survival CDF does not.
    p=model.predict(np.zeros((1,3)),np.zeros((1,3)),np.arange(-40.,41.))
    expected=ndtr(-10.)-ndtr(-11.)
    assert expected>0
    np.testing.assert_allclose(p['pdf'][0,50],expected,rtol=1e-12)
