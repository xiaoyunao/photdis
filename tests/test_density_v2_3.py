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


def forward_band_posterior(model, observed, errors):
    """Integrate p(m|mu) directly in grz, with flat mu prior.

    Maps (g-r,r-z,Mg) to (Mg,Mr,Mz), then completes the square
    along the common distance-modulus direction. No conditional-color formula.
    """
    from scipy.special import logsumexp
    transform=np.array([[0,0,1],[-1,0,1],[-1,-1,1.]])
    means=[];variances=[];logevidence=[]
    one=np.ones(3)
    for w,mean,cov in zip(model.weights,model.means,model.covariances):
        absolute=transform@mean
        total=transform@cov@transform.T+np.diag(errors**2)
        inv=np.linalg.inv(total);delta=observed-absolute
        precision=one@inv@one;mu=one@inv@delta/precision
        means.append(mu);variances.append(1/precision)
        residual=delta-mu*one
        logevidence.append(np.log(w)-.5*(np.linalg.slogdet(total)[1]+np.log(precision)+residual@inv@residual))
    logevidence=np.array(logevidence)
    return np.exp(logevidence-logsumexp(logevidence)),np.array(means),np.sqrt(variances)


def fixture_mixture():
    model=ErrorDeconvolvedMixture(2);model.weights=np.array([.7,.3])
    model.means=np.array([[.5,.2,5.],[.9,.4,1.]])
    model.covariances=np.array([[[.08,.01,.12],[.01,.06,.04],[.12,.04,.5]],[[.12,-.01,-.1],[-.01,.08,.03],[-.1,.03,.8]]])
    model.class_frequency=np.array([[.5,0,0,.5],[0,.7,0,.3]])
    return model


def test_known_sed_recovers_weighted_three_band_distance():
    model=ErrorDeconvolvedMixture(1);model.weights=np.ones(1)
    model.means=np.array([[.5,.2,5.]])
    model.covariances=np.zeros((1,3,3));model.class_frequency=np.array([[1,0,0,0]])
    m=np.array([[18.2,17.5,17.3],[18.2,17.5,17.3]])
    e=np.array([[.1,.1,.1],[.2,.1,.3]])
    w,mean,sigma=model.conditional(m,e)
    inverse=1/e**2;band_mu=m-np.array([5,4.5,4.3])
    expected=(band_mu*inverse).sum(axis=1)/inverse.sum(axis=1)
    np.testing.assert_allclose(m[:,0]-mean[:,0],expected,atol=1e-12)
    np.testing.assert_allclose(sigma[:,0],1/np.sqrt(inverse.sum(axis=1)),atol=1e-12)
    np.testing.assert_allclose(expected[0],13.0666666666667,atol=1e-12)


def test_general_gaussian_forward_model_and_mixture_pdf():
    from scipy.special import ndtr
    model=fixture_mixture();m=np.array([[18.,17.4,17.1],[19.2,18.5,18.1]])
    e=np.array([[.2,.1,.1],[.05,.2,.13]])
    w,mean,sigma=model.conditional(m,e);edges=np.arange(-30,40.0001,.01)
    p=model.predict(m,e,edges)
    for i in range(len(m)):
        wf,mu,sf=forward_band_posterior(model,m[i],e[i])
        np.testing.assert_allclose(w[i],wf,rtol=1e-12)
        np.testing.assert_allclose(m[i,0]-mean[i],mu,rtol=1e-12)
        np.testing.assert_allclose(sigma[i],sf,rtol=1e-12)
        expected=sum(wj*(ndtr((m[i,0]-edges[:-1]-mj)/sj)-ndtr((m[i,0]-edges[1:]-mj)/sj)) for wj,mj,sj in zip(wf,mu,sf))
        np.testing.assert_allclose(p['pdf'][i],expected,atol=2e-15)
        np.testing.assert_allclose(p['qnear'][i],sum(wf*ndtr((15-mu)/sf)),rtol=1e-12)


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
