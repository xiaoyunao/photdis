"""Check that overlapping image-quality causes are counted independently."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "cluster_rgb_quality_causes", ROOT / "scripts/cluster_rgb_quality_causes.py")
quality = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(quality)


def test_bright_saturation_and_fraction_can_overlap():
    fields = [("grz_base", "?"), ("grz_clean", "?"),
              ("grz_fraction_clean", "?"), ("grz_mask_clean", "?"),
              ("maskbits", "i8")]
    for band in "grz":
        fields += [(f"fracmasked_{band}", "f8"),
                   (f"fracin_{band}", "f8"), (f"fracflux_{band}", "f8")]
    tab = np.zeros(2, dtype=fields)
    tab["grz_base"] = True
    tab["grz_fraction_clean"] = [False, True]
    tab["grz_mask_clean"] = [False, True]
    tab["grz_clean"] = [False, True]
    tab["maskbits"] = [1 << 2, 0]
    for band in "grz":
        tab[f"fracin_{band}"] = 1.
    tab["fracmasked_g"] = [.5, 0.]
    result = quality.causes(tab, np.ones(2, bool))
    assert result["both_frac_mask_fail"] == 1
    assert result["maskbit_causes_among_base"]["satur_g"] == 1
    assert result["fraction_causes_among_base"]["g"]["fracmasked_ge_0p4_or_invalid"] == 1
