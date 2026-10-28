"""Frozen-sample checks and interval calibration for the pilot follow-up."""
from __future__ import annotations

import hashlib
import numpy as np


def identity_hash(ids):
    values = np.sort(np.asarray(ids, dtype="<i8"))
    if len(np.unique(values)) != len(values):
        raise ValueError("Duplicate Gaia source identity")
    return hashlib.sha256(values.tobytes()).hexdigest()


def common_nonoptical(table, external):
    """Apply the same astrometric, contamination and branch rules to both groups."""
    mg = table["g0"] - 10 + 5 * np.log10(table["p_corr"])
    nonstar = ((table["desi_known_contaminant"] & ~table["desi_match_ambiguous"]) |
               (table["apogee_known_contaminant"] & ~table["apogee_match_ambiguous"]))
    giant = ((table["desi_quality"] & (table["desi_logg"] < 3.5)) |
             (table["apogee_quality"] & (table["apogee_logg"] < 3.5)))
    flags = {"Gaia_anchor_astrometry": np.asarray(table["anchor_astrometry"], bool),
             "not_Gaia_candidate_contaminant": ~table["contaminant_candidate"],
             "not_spectroscopic_nonstar": ~nonstar,
             "not_quality_spectroscopic_giant": ~giant,
             "not_external_reserved": ~np.asarray(external, bool),
             "Gaia_BP_RP_precise": table["bp_rp_precise"],
             "cold_Gaia_color": (table["bp_rp_checked"] >= 1.3) & (table["bp_rp_checked"] < 3.5),
             "cold_CMD_luminosity": np.isfinite(mg) & (mg > 5),
             "measured_grz": table["grz_base"]}
    accepted = np.logical_and.reduce(list(flags.values()))
    return accepted, flags, mg


def ps1_replacement(table):
    """Replace observed grz together with their dereddened colors and M_g label."""
    g0 = table["ps1_g_decam"] + (table["g0"] - table["g"])
    r0 = table["ps1_r_decam"] + (table["r0"] - table["r"])
    z0 = table["ps1_z_decam"] + (table["z0"] - table["z"])
    colors = np.column_stack((g0 - r0, r0 - z0))
    mg = g0 - 10 + 5 * np.log10(table["p_corr"])
    return colors, mg


def cqr_offset(truth, prediction, low, high, alpha):
    """Split-CQR finite-sample order statistic; never shrink a base interval."""
    truth = np.asarray(truth, float)
    prediction = np.asarray(prediction, float)
    scores = np.maximum(prediction[:, low] - truth, truth - prediction[:, high])
    scores = np.sort(scores[np.isfinite(scores)])
    n = len(scores)
    if n == 0:
        raise ValueError("Calibration sample is empty")
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    if rank > n:
        return float("inf"), n
    return float(max(0, scores[rank - 1])), n


def apply_cqr(prediction, q68, q95):
    adjusted = np.asarray(prediction, float).copy()
    adjusted[:, 0] -= q95
    adjusted[:, 1] -= q68
    adjusted[:, 3] += q68
    adjusted[:, 4] += q95
    return adjusted
