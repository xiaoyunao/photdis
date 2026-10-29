"""Published northern PS1 color terms remain a diagnostic transform."""
from pathlib import Path
import sys

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "scripts")]
from cluster_rgb_ps1_north_check import transformed_ps1


def test_transform_uses_ps1_g_minus_i():
    phot=np.array([[18.0,17.5,17.0,16.8]])
    result=transformed_ps1(phot)
    np.testing.assert_allclose(result["g"],18+.00464+.08672-.00668-.00255)
    np.testing.assert_allclose(result["r"],17.5+.00110-.06875+.02480-.00855)
    np.testing.assert_allclose(result["z"],16.8+.03664-.11084+.04477-.01223)
