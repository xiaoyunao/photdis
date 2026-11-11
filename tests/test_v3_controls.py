import numpy as np
import pytest
from photdis.v3.controls import probe_features, probability_metrics, reliability_rows, calibrate_temperature, apply_temperature


def test_probe_ignores_truth_and_preserves_signed_and_missing_bands():
    t = dict(flux=np.array([[10., 11., -1., 12., -.5, np.nan]]),
             flux_error=np.ones((1, 6)), transmission=np.ones((1, 6)),
             band_mask=np.array([[1, 1, 1, 1, 1, 0]], bool),
             RA=np.array([10.]), DEC=np.array([-40.]), EBV=np.array([.01]))
    before = probe_features(t, 'multiband')
    t.update(class_label=np.array([1]), reference_mu=np.array([20.]), rvs_feh=np.array([-2.]))
    np.testing.assert_equal(before, probe_features(t, 'multiband'))
    assert before[0, -3] < 0 and before[0, -2] < 0 and np.isnan(before[0, -1])


def test_probability_endpoint_accounting_and_calibration():
    p = np.array([[1., 0., 0.], [0., 1., 0.], [.3, .3, .4]])
    y = np.array([0, 1, 2])
    rows = reliability_rows(p, y)
    assert sum(r['n'] for r in rows) == 9
    assert probability_metrics(p, y)['accuracy'] == 1
    temperature = calibrate_temperature(p, y)
    assert probability_metrics(apply_temperature(p, temperature), y)['log_score'] >= probability_metrics(p, y)['log_score']
    with pytest.raises(ValueError): probability_metrics(p, [-1, 1, 2])
