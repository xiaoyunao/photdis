"""Shared five-band flux experiment and final distance transformations."""

from __future__ import annotations

import numpy as np

from photdis.stage1 import PHOT_COLUMNS


BANDS = ("G", "R", "Z", "W1", "W2")


def simulate_shared_flux(anchor, target, rng):
    """Move each anchor to a target r0, using target sky/quality and flux IVAR.

    The same output row is used for A/B/C. The measured anchor flux contains
    template noise; additional Gaussian noise is reduced in quadrature where
    possible. The output IVAR describes the larger of the target variance
    and the scaled template variance, including impossible-to-denoise pairs.
    """
    if len(anchor) != len(target):
        raise ValueError("Anchor and target rows must be paired")
    columns = ["MASKBITS", *PHOT_COLUMNS]
    raw = np.empty(len(anchor), dtype=[(name, target[name.lower()].dtype) for name in columns])
    for name in columns:
        raw[name] = target[name.lower()]
    rflux = np.asarray(target["flux_r"], float)
    rtrans = np.asarray(target["mw_transmission_r"], float)
    if np.any((rflux <= 0) | (rtrans <= 0)):
        raise ValueError("Target r flux and transmission must be positive")
    target_r0 = 22.5 - 2.5 * np.log10(rflux) + 2.5 * np.log10(rtrans)
    shift_mag = target_r0 - np.asarray(anchor["r0"], float)
    scale = 10 ** (-0.4 * shift_mag)
    template_to_target_ratio = np.full((len(anchor), len(BANDS)), np.nan)
    for index, band in enumerate(BANDS):
        key = band.lower()
        source_trans = np.asarray(anchor[f"mw_transmission_{key}"], float)
        target_trans = np.asarray(target[f"mw_transmission_{key}"], float)
        source_ivar = np.asarray(anchor[f"flux_ivar_{key}"], float)
        target_ivar = np.asarray(target[f"flux_ivar_{key}"], float)
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = scale * target_trans / source_trans
        source_flux = np.asarray(anchor[f"flux_{key}"], float) * factor
        with np.errstate(divide="ignore", invalid="ignore"):
            source_sigma = np.abs(factor) / np.sqrt(source_ivar)
            target_sigma = 1 / np.sqrt(target_ivar)
            template_to_target_ratio[:, index] = source_sigma / target_sigma
        valid = (np.isfinite(source_flux) & np.isfinite(source_sigma) &
                 np.isfinite(target_sigma) & (target_sigma > 0))
        observed = np.zeros(len(anchor), dtype=float)
        effective_variance = np.maximum(target_sigma[valid] ** 2,
                                        source_sigma[valid] ** 2)
        extra_variance = effective_variance - source_sigma[valid] ** 2
        observed[valid] = (source_flux[valid] + rng.normal(size=valid.sum()) *
                           np.sqrt(extra_variance))
        raw[f"FLUX_{band}"] = observed
        effective_ivar = np.zeros(len(anchor), dtype=float)
        effective_ivar[valid] = 1 / effective_variance
        raw[f"FLUX_IVAR_{band}"] = effective_ivar
    return raw, shift_mag, template_to_target_ratio


def transformed_distance_quantiles(g0_noisy, absolute_magnitude_quantiles):
    """Transform ordered Mg quantiles to ordered mu and distance quantiles."""
    mg = np.asarray(absolute_magnitude_quantiles, float)
    if mg.ndim != 2 or mg.shape[1] != 5:
        raise ValueError("Expected five ordered Mg quantiles")
    mu = np.asarray(g0_noisy, float)[:, None] - mg[:, ::-1]
    distance_pc = 10 ** ((mu + 5) / 5)
    return mu, distance_pc


def interval_diagnostics(truth, quantiles):
    """Median residual, NMAD, tail misses and interval widths."""
    truth = np.asarray(truth, float)
    q = np.asarray(quantiles, float)
    residual = q[:, 2] - truth
    center = np.median(residual)
    return {"n": int(len(truth)),
            "bias": float(center),
            "nmad": float(1.4826 * np.median(np.abs(residual - center))),
            "coverage_68": float(np.mean((truth >= q[:, 1]) & (truth <= q[:, 3]))),
            "coverage_95": float(np.mean((truth >= q[:, 0]) & (truth <= q[:, 4]))),
            "miss_low_68": float(np.mean(truth < q[:, 1])),
            "miss_high_68": float(np.mean(truth > q[:, 3])),
            "miss_low_95": float(np.mean(truth < q[:, 0])),
            "miss_high_95": float(np.mean(truth > q[:, 4])),
            "median_width_68": float(np.median(q[:, 3] - q[:, 1])),
            "median_width_95": float(np.median(q[:, 4] - q[:, 0]))}
