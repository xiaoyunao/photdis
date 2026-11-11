import numpy as np
import pytest
from test_v3_inference import model, observation
from photdis.v3.model import inference_model
from photdis.v3.inference import infer_reference


def test_discrepancy_channel_is_separate_from_class_and_binary():
    base = model(); robust = inference_model(base)
    assert robust.classes.shape == (4, 3)
    assert np.isclose(robust.weights[robust.discrepancy_channel].sum(), .01)
    o = observation(); o['band_mask'][4] = False
    ordinary = infer_reference(robust, o)
    o['flux'][0] *= 2
    discrepant = infer_reference(robust, o)
    p0 = ordinary['component_probability']@robust.discrepancy_channel
    p1 = discrepant['component_probability']@robust.discrepancy_channel
    assert p1 > p0+.5
    assert base.means.shape == (2, 7)
    with pytest.raises(ValueError): inference_model(robust)
