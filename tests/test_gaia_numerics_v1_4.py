import numpy as np
from photdis.gaia_numerics_v1_4 import refined_gaia,scalar_log_density,scalar_cdf


def test_strong_parallax_limit_and_negative_observations():
    edges=np.array([0.,1.]);pdf=np.ones((3,1));g=np.full(3,10.)
    p=10**(.37/5);y=np.array([p,p,-.2]);error=np.array([p/10000,.05,.3])
    result=refined_gaia(pdf,edges,g,y,error)
    # Uniform M implies density1/(k*pi); negligible noise at S/N10000.
    assert abs(result['full_log_density'][0]+np.log(np.log(10)/5*p))<1e-5
    assert abs(result['full_pit'][0]-.37)<1e-5
    for i in range(3):
        assert abs(result['full_log_density'][i]-scalar_log_density(pdf[i],edges,g[i],y[i],error[i]))<1e-5
        assert abs(result['full_pit'][i]-scalar_cdf(pdf[i],edges,g[i],y[i],error[i]))<1e-5
    assert result['full_pit'][2]>0 and np.isfinite(result['full_log_density']).all()


def test_multibin_scores_against_adaptive_integrals():
    rng=np.random.default_rng(9);edges=np.linspace(-2,5,15)
    pdf=rng.uniform(size=(12,14));pdf[:,4:7]=0;pdf/=pdf.sum(axis=1)[:,None]
    g=np.full(12,17.);y=np.linspace(-.2,.5,12);error=np.geomspace(.002,.3,12)
    r=refined_gaia(pdf,edges,g,y,error)
    for i in range(12):
        assert abs(r['full_log_density'][i]-scalar_log_density(pdf[i],edges,g[i],y[i],error[i]))<1e-5
        assert abs(r['full_pit'][i]-scalar_cdf(pdf[i],edges,g[i],y[i],error[i]))<1e-5
