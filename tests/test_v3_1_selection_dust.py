from types import SimpleNamespace
import numpy as np
from photdis.v3.inference import kernel_loglike
from photdis.v3_1.selection_dust import entry_model_with_local_dust


def test_entry_dust_covariance_matches_native_likelihood_in_both_binary_states():
    rng=np.random.default_rng(319301); means=rng.uniform(1,4,(3,7)); factors=rng.normal(size=(3,7,7))*.05
    covariance=factors@factors.transpose(0,2,1)+np.eye(7)[None]*.02
    model=SimpleNamespace(means=means,covariances=covariance,weights=np.ones(3)/3,
        binary_brightness_factor=2.,system='DECam_DR11')
    original=covariance.copy(); trans=np.exp(-np.array([.12,.09,.07,.05,.01,.005]))
    obs=dict(flux=np.array([2.,1.,-.1,3.,-.3,1.]),flux_error=np.full(6,.15),band_mask=np.ones(6,bool),
        transmission=trans,EBV=.025,photometric_system='DECam_DR11')
    grid=np.linspace(5,20,101); cal=np.linspace(-.01,.02,6)
    expected,_=kernel_loglike(model,obs,grid,calibration_mag=cal)
    integrated=entry_model_with_local_dust(model,trans,.025)
    actual,_=kernel_loglike(integrated,dict(obs,dust_sigma_ebv=0.),grid,calibration_mag=cal)
    np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=1e-12)
    np.testing.assert_array_equal(model.covariances,original)
