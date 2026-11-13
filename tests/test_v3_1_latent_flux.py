from types import SimpleNamespace
import numpy as np
import pytest
from scipy.special import ndtr
from photdis.v3.likelihood import normalize_joint
from photdis.v3_1.latent_flux import negative_flux_support


def test_latent_sign_uses_observation_not_component_prior():
    model=SimpleNamespace(weights=np.ones(1),means=np.array([[-1.,1.,1.,1.,1.,1.,0.]]),
        covariances=np.array([np.eye(7)]),binary_brightness_factor=2.)
    o=dict(band_mask=np.array([1,0,0,0,0,0],bool),flux=np.array([2.,np.nan,np.nan,np.nan,np.nan,np.nan]),
        flux_error=np.array([.2,np.nan,np.nan,np.nan,np.nan,np.nan]),transmission=np.ones(6))
    grid=np.array([10.-1e-8,10.,10.+1e-8]); lp=np.full((6,3),-np.inf);lp[0]=0
    p=normalize_joint(grid,lp)
    result=negative_flux_support(model,o,p)
    expected_mean=(-1+2/.2**2)/(1+1/.2**2); expected_sigma=np.sqrt(1/(1+1/.2**2))
    np.testing.assert_allclose(result['probability_by_band'][0],ndtr(-expected_mean/expected_sigma),rtol=1e-6)
    assert result['probability_by_band'][0] < 1e-10 < ndtr(1.)
    np.testing.assert_allclose(result['probability_by_band'][1:],ndtr(-1.))
    assert result['any_negative_lower'] <= result['any_negative_upper'] <= 1
    o['flux'][0]=-2
    assert negative_flux_support(model,o,p)['probability_by_band'][0]>.999
    o['upper_limit']=np.array([1,0,0,0,0,0],bool)
    with pytest.raises(ValueError):negative_flux_support(model,o,p)
