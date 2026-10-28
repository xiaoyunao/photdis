#!/usr/bin/env python3
"""Inspect Gaia and Legacy properties of the exact crossover template stars."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".deps")]
import fitsio
import numpy as np

from photdis.followup import identity_hash
from photdis.stage1 import require_compute, write_json
from cold_v4_policy_noise import REGISTRATION, source_templates

BASE = ROOT / "data/cold_v4_crossover"
OUT = BASE / "native_population.json"
ROWS = BASE / "native_population.npz"
NAMES = ("p_corr", "zpt_mas", "snr_corr", "gaia_parallax_error",
         "gaia_phot_g_mean_mag", "gaia_ruwe", "gaia_bp_rp", "ebv",
         "gaia_visibility_periods_used")


def stats(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0}
    median = float(np.median(values))
    return {"n": len(values), "median": median,
            "nmad": float(1.4826 * np.median(abs(values - median))),
            "p05": float(np.quantile(values, .05)),
            "p95": float(np.quantile(values, .95))}


def source_residuals(rows, field):
    names = [f"{field}|{env}|{depth}__native_residual"
             for env in (0, 1) for depth in range(4)]
    stacked = np.stack([rows[name] for name in names])
    first = np.argmax(np.isfinite(stacked), axis=0)
    return stacked[first, np.arange(stacked.shape[1])]


def main():
    require_compute()
    if OUT.exists() or ROWS.exists():
        raise FileExistsError("Native population census already exists")
    registration = json.loads(REGISTRATION.read_text())
    cross = json.loads((BASE / "diagnostic.json").read_text())
    raw = np.load(BASE / "rows.npz")
    result = {"method": "same source IDs and order as frozen crossover; descriptive only",
              "template_source_hashes": cross["template_source_hashes"],
              "fields": {}}
    saved = {}
    for index, (key, offset) in enumerate((
            ("interval_calibration", 101), ("independent_final", 202))):
        field = registration[key]["field_config"]
        template, _, _, source_id, metadata = source_templates(
            field, np.random.default_rng(registration["seed"] + offset))
        if identity_hash(source_id) != cross["template_source_hashes"][index]:
            raise ValueError("Source identities differ from frozen crossover")
        with fitsio.FITS(metadata["path"]) as file:
            anchor = np.flatnonzero(file[1].read(columns=["anchor"])["anchor"])
            extra = file[1].read(columns=["source_id", *NAMES], rows=anchor)
        lookup = {int(value): row for row, value in enumerate(extra["source_id"])}
        positions = np.array([lookup[int(value)] for value in source_id])
        selected = extra[positions]
        residual = source_residuals(raw, index)
        value = {name: np.asarray(selected[name], float) for name in NAMES}
        value["native_residual"] = residual
        value["r0"] = np.asarray(template["r0"], float)
        value["native_gr"] = raw[f"{index}|0|0__native_gr"]
        value["native_rz"] = raw[f"{index}|0|0__native_rz"]
        value["source_ag"] = raw[f"{index}|0|0__source_ext_g"]
        saved.update({f"{index}__{name}": column for name, column in value.items()})
        good = np.isfinite(residual)
        bins = {}
        for name, edges in (("p_corr", [0, .5, 1, 2, np.inf]),
                            ("gaia_bp_rp", [0, 1.5, 2.3, 3, np.inf]),
                            ("native_rz", [-np.inf, .8, 1.4, np.inf])):
            bins[name] = [
                {"range": [float(lo) if np.isfinite(lo) else None,
                           float(hi) if np.isfinite(hi) else None], "residual": stats(
                    residual[good & (value[name] >= lo) & (value[name] < hi)])}
                for lo, hi in zip(edges[:-1], edges[1:])]
        result["fields"][str(index)] = {
            "name": metadata["name"], "source_count": len(source_id),
            "native_residual": stats(residual),
            "properties": {name: stats(column[good]) for name, column in value.items()
                           if name != "native_residual"},
            "residual_by_property": bins}
    BASE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ROWS, **saved)
    write_json(OUT, result)


if __name__ == "__main__":
    main()
