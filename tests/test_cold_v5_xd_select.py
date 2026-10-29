import importlib.util
from pathlib import Path
import sys


def test_configuration_selection_weights_sky_fields_equally():
    path = Path(__file__).resolve().parents[1] / "scripts/cold_v5_xd_select.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("xd_select", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    digest = module.sha(module.CONFIG)
    config = {"components": [16, 32]}
    manifests = {}
    for size, values in ((16, (1., 3.)), (32, (2., 1.))):
        manifests[size] = {"configuration_sha256": digest,
            "tiers": {"A_grz": {"converged": True, "development": {
                "large": {"n": 1000,
                          "mean_negative_conditional_log_density": values[0]},
                "small": {"n": 100,
                          "mean_negative_conditional_log_density": values[1]}}}}}
    chosen, scores = module.choose(config, manifests, ["large", "small"])
    assert chosen == 32
    assert scores[16]["equal_field_mean_nll"] == 2.
    assert scores[32]["equal_field_mean_nll"] == 1.5


def test_unfinished_optimization_cannot_be_selected():
    path = Path(__file__).resolve().parents[1] / "scripts/cold_v5_xd_select.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("xd_select_unfinished", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fit = {"configuration_sha256": module.sha(module.CONFIG),
           "tiers": {"A_grz": {"converged": False,
                   "development": {"field": {"n": 100,
                       "mean_negative_conditional_log_density": 0.}}}}}
    import pytest
    with pytest.raises(ValueError, match="unfinished"):
        module.choose({"components": [16]}, {16: fit}, ["field"])
