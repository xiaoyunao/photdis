"""Check the preregistered distance-space score on a synthetic RGB source."""
from pathlib import Path
import sys

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]/"scripts")]
from cluster_rgb_ngc5466_score import describe, score_row


def test_score_reports_comparable_distance_density_and_ordered_mu_interval():
    config = {"forward_model":{"posterior_grid_step_mag":.02},
              "whole_cluster_test":{"ngc5466":{"external_mu":16.0}}}
    model = {"support":{"lower_rgb":[[.5,1.5,10,10]]},
             "models":[{"beta":[[.7,.1,0,0]],"intrinsic":[[.02**2]]}]}
    row = np.array([(42,17.,.7,.2,.02,.02,.02)],
                   dtype=[("source_id","i8"),("g0","f8"),("gr0","f8"),
                          ("rz0","f8"),("sigma_g","f8"),("sigma_r","f8"),
                          ("sigma_z","f8")])[0]
    result = score_row(row,"lower_rgb","gr_common",model,config,[-1.7],[1.])
    assert result["mu_lo95"] <= result["mu_pred"] <= result["mu_hi95"]
    assert np.isfinite(result["log_predictive_density"])
    assert np.isfinite(result["log_color_likelihood"])
    summary = describe([result])
    assert summary["mean_log_predictive_density"] == result["log_predictive_density"]
