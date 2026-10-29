"""The locked XD gate uses only registered whole development fields."""
from __future__ import annotations

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "cold_v5_xd_decide", Path(__file__).resolve().parents[1] /
    "scripts/cold_v5_xd_decide.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def field(nmad=.24, response=.015):
    return {"n_same_selected": 500,
            "qrf": {"n": 500, "center_mag": .04, "nmad_mag": .25,
                    "abs_gt_0p5": .10},
            "xd": {"n": 500, "center_mag": .03, "nmad_mag": nmad,
                   "abs_gt_0p5": .09},
            "perturbation_plus_0p01_mag": {
                band: {"qrf": {"median_abs_mag": .02},
                       "xd": {"median_abs_mag": response}}
                for band in ("g", "r")}}


def test_gate_requires_both_fields_and_both_bands():
    config = {"candidate_gate": {
        "absolute_center_noninferiority_mag": .01,
        "point_nmad_noninferiority_mag": .01,
        "tail_abs_residual_gt_0p5_noninferiority_fraction": .01,
        "median_g_and_r_perturbation_abs_response_reduction_fraction": .25}}
    selection = {"selected_components": 32,
                 "configuration_sha256": "same"}
    comparison = {"components": 32, "configuration_sha256": "same",
                  "fields": {"first": field(), "second": field()}}
    assert module.decide(config, selection, comparison,
                         ("first", "second"))["decision"] == (
                             "lock_candidate_for_independent_validation")
    comparison["fields"]["second"]["perturbation_plus_0p01_mag"]["r"]["xd"][
        "median_abs_mag"] = .016
    result = module.decide(config, selection, comparison, ("first", "second"))
    assert result["decision"] == "retain_frozen_v4"
    assert not result["development_fields"]["second"]["checks"]["response_r"]
