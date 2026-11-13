import numpy as np
from scipy.stats import norm
from photdis.v3_1.luminosity import luminosity_transform
from photdis.v3_1.luminosity_log import log_luminosity_transform


def test_log_luminosity_matches_linear_operation_and_keeps_tiny_tails():
    grid = np.linspace(-10,10,2001); log = np.stack([norm.logpdf(grid,-1,.7),norm.logpdf(grid,2,1.)])
    expected = luminosity_transform(np.exp(log),grid,.237,.42)
    actual = log_luminosity_transform(log,grid,.237,.42)
    np.testing.assert_allclose(np.exp(actual),expected,rtol=1e-12,atol=1e-15)
    tiny = log_luminosity_transform(log-10000,grid,.237,.42)
    np.testing.assert_allclose(tiny+10000,actual,atol=3e-12,rtol=1e-12)
    np.testing.assert_array_equal(log_luminosity_transform(log,grid,0.,0.),log)
