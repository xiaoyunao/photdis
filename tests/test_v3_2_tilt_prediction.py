import numpy as np
from scipy.special import logsumexp

from photdis.v3.inference import kernel_loglike
from photdis.v3_1.shared_prior import marginalize_native
from photdis.v3_1.validation_scores import score_pdf
from test_v3_inference import model, observation
from v3_2_tilt_predict import entry_transfer_scores


def test_entry_transfer_matches_direct_native_forecasts_for_all_levels():
    m = model()
    grid = np.linspace(-20, 50, 3501)
    logs, _ = kernel_loglike(m, observation(), grid)
    native = logsumexp(logs + np.log([1-m.binary_reference_fraction,
                                      m.binary_reference_fraction])[:, None, None], axis=0)
    rng = np.random.default_rng(73)
    theta = rng.dirichlet(np.ones(15), size=8).reshape(8, 3, 5)
    beta = rng.normal(0, .5, (8, 2))
    weights = np.array([rng.dirichlet(np.ones(8)), np.full(8, 1/8),
                        rng.dirichlet(np.ones(8))])
    evidence = dict(parallax_eligible=True, p_corr=-.05, gaia_parallax_error=.13,
                    class_label=1, distance_available=True, distance_precise=True,
                    distance_kpc=4.7, distance_error_kpc=.45)
    direct = [marginalize_native(m, grid, logs, theta, beta, w) for w in weights]

    def scores(value):
        return score_pdf(grid, value['full_logpdf'], value['posterior'].quantiles(),
                         value['class_distance'], evidence)

    base = scores(direct[1])
    probability, deltas = entry_transfer_scores(m, grid, native, theta, beta, direct[1],
                                                weights, base, evidence)
    np.testing.assert_allclose(probability, [v['class_distance'] for v in direct],
                               rtol=0, atol=1e-12)
    for level in (0, 2):
        expected = scores(direct[level])
        prefix = f'entry_{level}'
        comparisons = (
            ('gaia_score', expected['gaia']['log_score']-base['gaia']['log_score']),
            ('gaia_pit', expected['gaia']['pit']-base['gaia']['pit']),
            ('teacher_score', expected['references']['astronn']['log_score']
             - base['references']['astronn']['log_score']),
        )
        for name, target in comparisons:
            np.testing.assert_allclose(deltas[f'{prefix}_{name}_delta'], target, rtol=0, atol=1e-10)
