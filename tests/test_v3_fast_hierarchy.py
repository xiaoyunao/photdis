import numpy as np
from test_v3_inference import model
from photdis.v3.hierarchy import FieldPopulation
from photdis.v3.fast_hierarchy import FastFieldPopulation


def test_cached_objective_preserves_likelihood_and_all_gradients():
    rng = np.random.default_rng(39); m = model()
    r = rng.lognormal(0, 2, (71, 2, 5)); a = rng.uniform(.001, 1, r.shape)
    field = np.arange(71)%4; weight = rng.uniform(.1, 5, 71)
    slow = FieldPopulation(r, a, m, field, weight)
    fast = FastFieldPopulation(r, a, m, field, weight)
    for scale in [.01, 1, 5]:
        x = rng.normal(0, scale, slow.dimension)
        sv, sg = slow.objective(x); fv, fg = fast.objective(x)
        np.testing.assert_allclose(sv, fv, atol=1e-11)
        np.testing.assert_allclose(sg, fg, atol=2e-12)
