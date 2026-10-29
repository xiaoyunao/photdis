"""Check the band-covariance terms in the local RGB width diagnostic."""
from pathlib import Path
import sys

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "scripts")]
from cluster_rgb_system_audit import core_loci, local_residuals, sequence


def test_shared_g_error_cancels_for_unit_gr_slope():
    points = [{"mg": 0.0, "color": 0.0}, {"mg": 1.0, "color": 1.0}]
    mg = np.array([0.5])
    res, expected = local_residuals(mg, np.array([0.5]),
                                    np.array([0.1]), np.array([0.1]),
                                    np.array([0.2]), points, True)
    np.testing.assert_allclose(res, 0)
    np.testing.assert_allclose(expected, 0.2)


def test_rz_error_has_no_shared_g_covariance():
    points = [{"mg": 0.0, "color": 0.0}, {"mg": 1.0, "color": 1.0}]
    _, expected = local_residuals(np.array([0.5]), np.array([0.5]),
                                  np.array([0.1]), np.array([0.2]),
                                  np.array([0.3]), points, False)
    np.testing.assert_allclose(expected, np.sqrt(0.14))


def test_locus_bins_use_actual_selected_subset():
    mg = np.array([0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,1.2])
    color = np.arange(len(mg),dtype=float)
    pts = sequence(mg,color)
    assert len(pts) == 1
    assert pts[0]["n"] == 8


def test_core_loci_accepts_fits_byte_stage_names():
    new=np.zeros(10,dtype=[("phase","S20"),("pmember","f4"),
                           ("gr_clean","?"),("g0","f4"),("gr0","f4")])
    old=np.zeros(10,dtype=[("grz_clean","?"),("g0","f4"),
                           ("gr0","f4"),("rz0","f4")])
    new["phase"][:8]=b"lower_rgb"
    new["phase"][8:]=b"upper_rgb"
    new["pmember"]=.99
    new["gr_clean"]=True
    new["g0"]=15.041+np.linspace(1,2,10)
    new["gr0"]=.6
    old["grz_clean"]=True
    old["g0"]=new["g0"]
    old["gr0"]=.6
    old["rz0"]=.3
    result=core_loci("m3",new,old)
    assert result["gr"]["core_n"]==8
    assert result["diagnostic_upper_rgb"]["n"]==2
