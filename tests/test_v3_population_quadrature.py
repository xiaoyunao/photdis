import numpy as np
from test_v3_inference import model
from photdis.v3.fast_hierarchy import FastFieldPopulation
from photdis.v3.population_quadrature import integrate_population


def test_uninformative_catalog_recovers_proper_prior_integral_and_moments():
    m=model();rng=np.random.default_rng(17)
    # Likelihood equals selection, hence cancels exactly: posterior is N(0,I).
    r=rng.uniform(.1,.9,(4,2,5));h=FastFieldPopulation(r,r,m,np.array([0,0,1,1]))
    pilot=rng.normal(size=(6000,h.dimension))
    got=integrate_population(h,pilot,draws=8192,seed=12)
    assert got['effective_sample_size']>5000
    assert abs(got['log_integral'])<.03
    mean=got['weights']@got['samples'];variance=got['weights']@(got['samples']**2)-mean**2
    assert abs(mean).max()<.06 and abs(variance-1).max()<.08
