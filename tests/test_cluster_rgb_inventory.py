"""Focused identity checks for whole-cluster membership processing."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import zipfile

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/cluster_rgb_inventory.py"
SPEC = importlib.util.spec_from_file_location("cluster_rgb_inventory", SCRIPT)
cluster = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cluster)


def test_membership_parser_preserves_gaia_64bit_ids_and_highest_probability(tmp_path):
    path = tmp_path / "members.zip"
    sid = 1454895986561961216
    def row(p):
        return f"{sid} 0 0 0 0 0 0 0 0 0 0 0 16.2 1.1 0 3 {p}\n"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("catalogues/test.txt", "# header\n" + row(.5) + row(.95))
    members = cluster.read_members(path, "catalogues/test.txt")
    assert len(members) == 1
    assert int(members["source_id"][0]) == sid
    assert members["pmember"][0] == .95


def test_separation_is_zero_at_center():
    angle = cluster.separation_arcmin(np.array([205.548]), np.array([28.377]),
                                      (205.548, 28.377))
    assert angle[0] < .001
