import numpy as np
from scipy.stats import norm
from photdis.v3.compression import DISTANCE_MU_BOUNDARIES
from photdis.v3.likelihood import trapezoid_weights
from photdis.v3_1.radial_products import bin_integrals,generator_joint_parameters
from photdis.v3_1.radial import RadialPopulation


def test_reporting_bins_and_latent_normalization():
    grid=np.linspace(-15,45,12001)
    density=norm.pdf(grid,13.,.4)
    truth=np.diff(np.r_[0.,norm.cdf(DISTANCE_MU_BOUNDARIES,13.,.4),1.])
    np.testing.assert_allclose(bin_integrals(density,grid),truth,atol=1e-5)
    likelihood=np.ones((2,3,len(grid)));model=RadialPopulation(grid,likelihood,likelihood,[.7,.1,.2],local=True)
    z=np.random.default_rng(9).normal(0,.3,19);posterior=model.joint_density(z)
    np.testing.assert_allclose(np.einsum('ncm,m->n',posterior,trapezoid_weights(grid)),1.,atol=1e-12)
    np.testing.assert_allclose(bin_integrals(posterior,grid).sum(axis=(1,2)),1.,atol=1e-12)


def test_true_population_parameters_are_not_observed_counts():
    catalog=dict(scenario='narrow_4kpc',generated_class_weights=[.75,.05,.20])
    parameter=generator_joint_parameters(catalog)
    np.testing.assert_allclose(parameter.sum(axis=1),[.75,.05,.20],atol=1e-14)
    assert parameter[:,1].sum()>.999999
    correlated=generator_joint_parameters(dict(scenario='class_distance_correlated',generated_class_weights=[.55,.20,.25]))
    independent=generator_joint_parameters(dict(scenario='reference_like',generated_class_weights=[.55,.20,.25]))
    assert correlated[1,3:].sum()>independent[1,3:].sum()
