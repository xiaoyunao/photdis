#!/usr/bin/env python3
"""Compare frozen A/B/C forests on identical external final-test stars."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".deps")]
import fitsio
import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from photdis.quantile import metrics
from photdis.stage1 import require_compute, write_json

OUT = Path("data/external_common_v1")
FIG = Path("figures/external_common")
TIERS = ("A_grz", "B_grzW1", "C_grzW1W2")
FIELDS = (("test_ra190_200_dec_p015_p025", "data/field_test_ra190_200_v1/field_diagnostics.fits"),
          ("test_ra320_330_dec_m025_m015", "data/field_test_ra320_330_v1/field_diagnostics.fits"))


def nmad(values):
    center = np.median(values)
    return 1.4826 * np.median(np.abs(values - center))


def model_rows(name, tier):
    path = Path("data/external_validation_v1") / f"{name}_{tier}.fits"
    table = fitsio.read(path)
    order = np.argsort(table["source_id"])
    return table[order]


def field_result(name, catalog):
    arrays = {tier: model_rows(name, tier) for tier in TIERS}
    common = arrays["C_grzW1W2"]["source_id"]
    aligned = {}
    for tier, table in arrays.items():
        where = np.searchsorted(table["source_id"], common)
        if not np.array_equal(table["source_id"][where], common):
            raise ValueError(f"Missing common source in {name} {tier}")
        aligned[tier] = table[where]
        if not np.allclose(aligned[tier]["mg_gaia"], arrays["C_grzW1W2"]["mg_gaia"]):
            raise ValueError("Truth differs across tiers")
    sky = fitsio.read(catalog, columns=["source_id", "ra", "dec"])
    sort = np.argsort(sky["source_id"])
    where = np.searchsorted(sky["source_id"][sort], common)
    if not np.array_equal(sky["source_id"][sort][where], common):
        raise ValueError("Missing sky coordinates")
    cells = hp.ang2pix(64, sky["ra"][sort][where], sky["dec"][sort][where],
                       nest=True, lonlat=True)
    truth = aligned["C_grzW1W2"]["mg_gaia"]
    result = {"common_rows": len(common), "nside64_cells": len(np.unique(cells)), "tiers": {}}
    residual = {}
    for tier, table in aligned.items():
        raw = np.column_stack([table[f"raw_q{q}"] for q in ("025", "160", "500", "840", "975")])
        cal = np.column_stack([table[f"cal_q{q}"] for q in ("025", "160", "500", "840", "975")])
        result["tiers"][tier] = {"raw": metrics(truth, raw), "calibrated": metrics(truth, cal)}
        residual[tier] = raw[:, 2] - truth
    blocks = np.unique(cells)
    indices = {int(cell): np.flatnonzero(cells == cell) for cell in blocks}
    rng = np.random.default_rng(20261030)
    samples = {key: [] for key in ("C_minus_A", "C_minus_B", "B_minus_A")}
    for _ in range(500):
        chosen = rng.choice(blocks, size=len(blocks), replace=True)
        rows = np.concatenate([indices[int(cell)] for cell in chosen])
        scatter = {tier: nmad(value[rows]) for tier, value in residual.items()}
        for key, first, second in (("C_minus_A", "C_grzW1W2", "A_grz"),
                                   ("C_minus_B", "C_grzW1W2", "B_grzW1"),
                                   ("B_minus_A", "B_grzW1", "A_grz")):
            samples[key].append(scatter[first] - scatter[second])
    result["block_bootstrap_NMAD_difference_mag"] = {
        key: {"point": float(nmad(residual[first]) - nmad(residual[second])),
              "p2p5": float(np.percentile(samples[key], 2.5)),
              "p97p5": float(np.percentile(samples[key], 97.5))}
        for key, first, second in (("C_minus_A", "C_grzW1W2", "A_grz"),
                                   ("C_minus_B", "C_grzW1W2", "B_grzW1"),
                                   ("B_minus_A", "B_grzW1", "A_grz"))}
    return result


def main():
    require_compute()
    if (OUT / "summary.json").exists():
        raise FileExistsError(OUT / "summary.json")
    summary = {"comparison": "identical C-available stars per final-test field; frozen full models",
               "block_bootstrap": "500 resamples of occupied nside64 NESTED cells",
               "fields": {}}
    for name, catalog in FIELDS:
        summary["fields"][name] = field_result(name, catalog)
        print(json.dumps({"field": name, **summary["fields"][name]}), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(FIELDS))
    for i, (tier, color) in enumerate(zip(TIERS, ("#356e9b", "#58894e", "#bf6f38"))):
        ax.scatter(x + (i - 1) * .12,
                   [summary["fields"][name]["tiers"][tier]["raw"]["nmad_mag"]
                    for name, _ in FIELDS], label=tier, color=color, s=65)
    ax.set(xticks=x, xticklabels=["north", "south"], ylabel="$M_g$ NMAD on the same stars [mag]")
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "01_same_star_band_gain.png", dpi=170)
    plt.close(fig)
    write_json(OUT / "summary.json", summary)
    write_json("manifests/external_common.json", summary)


if __name__ == "__main__":
    main()
