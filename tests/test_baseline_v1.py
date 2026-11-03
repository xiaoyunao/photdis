import numpy as np
from scipy.special import ndtr
from photdis.baseline_v0 import HistogramQRF
from photdis.baseline_v1 import quantiles_interpolated,qmc_noise,integrate,PopulationModel


def test_cdf_inverse_with_empty_bins():
    q=quantiles_interpolated(np.array([[0,.25,0,.75]]),np.arange(5.))
    assert np.allclose(q,[[1.1,1.64,3+1/3,3+.59/.75,3+.725/.75]])


def test_sobol_chunk_identity_and_gaussian_tails():
    ids=np.arange(500);z=qmc_noise(ids,9)
    assert np.array_equal(z,np.vstack([qmc_noise(ids[:39],9),qmc_noise(ids[39:],9)]))
    # Shared g and colors use the same perturbation, also across band subsets.
    assert not np.array_equal(z,qmc_noise(ids,9,1))
    draws=np.stack([qmc_noise(ids,d)[:,0] for d in range(128)],axis=1)
    q=np.quantile(draws,[.025,.975],axis=1)
    mass=ndtr(q[1])-ndtr(q[0])
    assert abs(mass.mean()-.95)<.02


def test_integrated_constant_luminosity_and_oracle_isolation():
    rng=np.random.default_rng(4);m=rng.normal(size=(360,5));labels=np.repeat(np.arange(3),120)
    wide=HistogramQRF(2,n_estimators=2,min_samples_leaf=8).fit(m,labels*3.,labels)
    model=PopulationModel(wide).fit(m,labels*3.,labels)
    mags=m[:5];e=np.full((5,5),.02);ids=np.arange(5)
    a=integrate(model,mags,e,ids,draws=4,variants=('natural','balanced','oracle'),oracle_labels=np.zeros(5))
    b=integrate(model,mags,e,ids,draws=4,variants=('natural','balanced','oracle'),oracle_labels=np.ones(5))
    for name in ('natural','balanced'):
        assert np.array_equal(a[name]['pdf'],b[name]['pdf'])
        assert np.allclose(a[name]['pdf'].sum(axis=1),1)
        assert np.allclose(a[name]['weights'].sum(axis=1),1)
        assert np.all(np.diff(a[name]['mu'],axis=1)>=0)
    assert not np.allclose(a['oracle']['mu'],b['oracle']['mu'])
