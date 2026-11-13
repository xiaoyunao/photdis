from types import SimpleNamespace
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm
from v3_1_validation_predict import class_loglike
from photdis.v3_1.diagnostics import control_log_density


def test_collapsed_native_likelihood_retains_original_class_binary_mixture():
    rng=np.random.default_rng(319501)
    factors=rng.normal(size=(4,7,7))*.07
    model=SimpleNamespace(means=rng.uniform(.3,3.,(4,7)),
        covariances=factors@factors.transpose(0,2,1)+np.eye(7)[None]*.03,
        weights=np.array([.1,.2,.3,.4]),classes=np.array([[.8,.1,.1],[.2,.7,.1],[.1,.2,.7],[.3,.3,.4]]),
        binary_brightness_factor=2.,binary_reference_fraction=.17,system='DECam_DR11')
    obs=dict(flux=np.array([2.,1.,-.1,3.,-.3,1.]),flux_error=np.full(6,.15),band_mask=np.ones(6,bool),
        transmission=np.exp(-np.array([.12,.09,.07,.05,.01,.005])),EBV=.025,photometric_system='DECam_DR11')
    grid=np.linspace(-5,30,701); calibration=np.linspace(-.02,.015,6)
    actual=logsumexp(class_loglike(model,obs,grid,calibration)+np.log(model.weights@model.classes)[:,None],axis=0)+norm.logpdf(grid,15,5)
    expected=logsumexp(control_log_density(model,obs,grid,calibration_mag=calibration),axis=0)
    np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=1e-12)
