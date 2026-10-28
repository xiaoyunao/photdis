#!/usr/bin/env python3
"""Independently audit finalized stage-one FITS tables on allocated compute."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import time

import fitsio
import numpy as np


ROLES = ("parent", "anchor", "extended", "spectroscopy", "external_test", "coverage_reference")
FLAGS = ("anchor", "extended", "extended_quality", "spectroscopy_match", "spectro_quality",
         "external_any", "external_cluster_member", "cluster_region", "widebinary",
         "identity_ambiguity_reserved", "duplicate_identity_reserved", "coverage_reference",
         "train_eligible", "known_contaminant", "gaia_candidate_contaminant", "gaia_match",
         "finite_parallax", "astro_quality", "zpt_inrange", "zpt_valid", "anchor_astrometry",
         "grz_base", "grz_clean", "grz_high_quality", "grz_precise", "grz_fraction_clean", "grz_mask_clean")
GAIA = ("source_id", "parallax", "parallax_error", "ruwe", "visibility_periods_used",
        "astrometric_params_solved", "duplicated_source", "phot_g_mean_mag",
        "nu_eff_used_in_astrometry", "pseudocolour", "ecl_lat", "in_qso_candidates",
        "in_galaxy_candidates")
COVERAGE = ("g0", "gr0", "rz0", "zw10", "w1w20", "b", "ebv", "gaia_phot_g_mean_mag",
            "gaia_parallax", "snr_corr", "mg", "desi_feh", "desi_logg", "apogee_feh", "apogee_logg")
COLUMNS = list(dict.fromkeys(
    ["ls_id", "source_id", "partition", "maskbits", "b", "ebv", "g0", "mg", "sigma_mg",
     "zpt_mas", "p_corr", "snr_raw", "snr_corr", "photometry_tier"]
    + list(FLAGS) + ["gaia_" + key for key in GAIA] + list(COVERAGE)
    + [f"{key}_{band}" for band in ("g", "r", "z") for key in
       ("flux", "flux_ivar", "mw_transmission", "nobs", "fracmasked", "fracin", "fracflux", "sigma", "anymask", "rchisq")]
    + [f"{survey}_{key}" for survey in ("desi", "apogee") for key in
       ("object_id", "match", "quality", "match_ambiguous", "known_contaminant")]))


def require_compute():
    if not os.environ.get("SLURM_JOB_ID") or not socket.gethostname().startswith("nid"):
        raise RuntimeError("Final table validation must run inside an allocated NERSC compute step")


def write_json(path, value):
    temporary = Path(path).with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def chunks(path, columns, chunk_rows):
    with fitsio.FITS(str(path)) as hdus:
        missing = set(columns) - set(hdus[1].get_colnames())
        if missing:
            raise ValueError(f"Missing audit columns in {path}: {sorted(missing)}")
        for start in range(0, hdus[1].get_nrows(), chunk_rows):
            rows = np.arange(start, min(start + chunk_rows, hdus[1].get_nrows()))
            yield hdus[1].read(columns=columns, rows=rows)


class Checks:
    def __init__(self):
        self.results = {}

    def add(self, name, valid):
        valid = np.asarray(valid, dtype=bool)
        item = self.results.setdefault(name, {"tested": 0, "violations": 0})
        item["tested"] += valid.size
        item["violations"] += int((~valid).sum())

    @property
    def passed(self):
        return all(item["violations"] == 0 for item in self.results.values())


def identity_hash(ids):
    value = np.asarray(ids, dtype=np.uint64).copy()
    with np.errstate(over="ignore"):
        value += np.uint64(0x9E3779B97F4A7C15)
        value = (value ^ (value >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
        value = (value ^ (value >> 27)) * np.uint64(0x94D049BB133111EB)
    return value ^ (value >> 31)


class Coverage:
    """Exact finite counts/extrema plus a bounded, order-independent sample."""
    def __init__(self, maximum):
        self.maximum = maximum
        self.rows = 0
        self.sample = None
        self.keys = np.zeros(0, dtype="u8")
        self.finite = Counter()
        self.low, self.high = {}, {}

    def update(self, table, selected):
        self.rows += int(selected.sum())
        if not np.any(selected):
            return
        data = np.column_stack([table[name][selected] for name in COVERAGE]).astype("f4")
        for survey in ("desi", "apogee"):
            for quantity in ("feh", "logg"):
                data[~table[f"{survey}_quality"][selected], COVERAGE.index(f"{survey}_{quantity}")] = np.nan
        for column, name in enumerate(COVERAGE):
            finite = data[:, column][np.isfinite(data[:, column])]
            self.finite[name] += len(finite)
            if len(finite):
                self.low[name] = min(self.low.get(name, np.inf), float(finite.min()))
                self.high[name] = max(self.high.get(name, -np.inf), float(finite.max()))
        keys = identity_hash(table["ls_id"][selected])
        if len(keys) > self.maximum:
            take = np.argpartition(keys, self.maximum - 1)[:self.maximum]
            keys, data = keys[take], data[take]
        if self.sample is not None:
            keys, data = np.concatenate((self.keys, keys)), np.concatenate((self.sample, data))
        if len(keys) > self.maximum:
            take = np.argpartition(keys, self.maximum - 1)[:self.maximum]
            keys, data = keys[take], data[take]
        self.keys, self.sample = keys, data

    def result(self):
        fields = {}
        for index, name in enumerate(COVERAGE):
            values = self.sample[:, index] if self.sample is not None else np.zeros(0)
            values = values[np.isfinite(values)]
            fields[name] = {"finite_count": self.finite[name], "min": self.low.get(name),
                            "max": self.high.get(name), "sample_finite_count": len(values),
                            "percentiles_1_5_16_50_84_95_99": np.percentile(values, [1, 5, 16, 50, 84, 95, 99]).tolist() if len(values) else None}
        return {"rows": self.rows, "sample_rows": len(self.keys), "fields": fields}


def role_masks(table):
    return (np.ones(len(table), bool), table["anchor"], table["extended"],
            table["spectroscopy_match"], table["external_any"] | table["identity_ambiguity_reserved"]
            | table["duplicate_identity_reserved"], table["coverage_reference"])


def check_physics(table, config, checks):
    """Recompute scientific predicates from raw columns, without pipeline imports."""
    phot = config.get("photometry", {})
    n = len(table)
    measured, clean, precise = np.ones(n, bool), np.ones(n, bool), np.ones(n, bool)
    for band in ("g", "r", "z"):
        flux = table[f"flux_{band}"].astype("f8")
        ivar = table[f"flux_ivar_{band}"].astype("f8")
        trans = table[f"mw_transmission_{band}"].astype("f8")
        valid = np.isfinite(flux) & (flux > 0) & np.isfinite(ivar) & (ivar > 0)
        valid &= np.isfinite(trans) & (trans > 0) & (table[f"nobs_{band}"] >= phot.get("nobs_min", 1))
        snr = np.full(n, np.nan)
        snr[valid] = (flux[valid] * np.sqrt(ivar[valid])).astype("f4")
        measured &= valid & np.isfinite(snr) & (snr >= phot.get("min_snr", 5.))
        error = np.full(n, np.nan)
        error[valid] = ((2.5 / np.log(10)) / snr[valid]).astype("f4")
        checks.add(f"{band}_error_propagation", np.isclose(table[f"sigma_{band}"], error, rtol=2e-6, atol=1e-8, equal_nan=True))
        precise &= error < phot.get("optical_sigma_max", .03)
        clean &= (np.isfinite(table[f"fracmasked_{band}"]) & np.isfinite(table[f"fracin_{band}"])
                  & np.isfinite(table[f"fracflux_{band}"])
                  & (table[f"fracmasked_{band}"] < phot.get("fracmasked_max", .4))
                  & (table[f"fracin_{band}"] > phot.get("fracin_min", .7))
                  & (table[f"fracflux_{band}"] < phot.get("fracflux_max", .5)))
        if band == "g":
            expected_g = np.full(n, np.nan)
            expected_g[valid] = 22.5 - 2.5 * np.log10(flux[valid] / trans[valid])
            checks.add("dereddened_g", np.isclose(table["g0"], expected_g, atol=3e-6, rtol=0, equal_nan=True))
    rejected_bits = sum(1 << bit for bit in set(phot.get("clean_reject_maskbits", [2, 3, 4, 5, 6, 7, 10])))
    checks.add("grz_fraction_clean_definition", table["grz_fraction_clean"] == clean)
    mask_clean = (table["maskbits"].astype("i8") & rejected_bits) == 0
    checks.add("grz_mask_clean_definition", table["grz_mask_clean"] == mask_clean)
    checks.add("grz_precise_definition", table["grz_precise"] == (measured & precise))
    clean &= mask_clean
    clean &= measured
    checks.add("parent_measured_grz", measured & table["grz_base"])
    checks.add("grz_clean_definition", table["grz_clean"] == clean)
    checks.add("grz_high_quality_definition", table["grz_high_quality"] == (clean & precise))
    sky = config["sky"]
    checks.add("sky_domain", np.isfinite(table["b"]) & (np.abs(table["b"]) >= sky["abs_b_min"] - 3e-6)
               & np.isfinite(table["ebv"]) & (table["ebv"] >= 0) & (table["ebv"] < sky["ebv_max"]))

    match = table["gaia_match"]
    p, error = table["gaia_parallax"].astype("f8"), table["gaia_parallax_error"].astype("f8")
    finite = match & np.isfinite(p) & np.isfinite(error) & (error > 0)
    solved = table["gaia_astrometric_params_solved"]
    solution = (solved == 31) | (solved == 95)
    quality = finite & solution & np.isfinite(table["gaia_ruwe"]) & (table["gaia_ruwe"] > 0)
    quality &= (table["gaia_ruwe"] < 1.4) & (table["gaia_visibility_periods_used"] >= 9) & ~table["gaia_duplicated_source"]
    g, nu, pseudo, lat = (table[f"gaia_{name}"].astype("f8") for name in
                         ("phot_g_mean_mag", "nu_eff_used_in_astrometry", "pseudocolour", "ecl_lat"))
    domain = match & solution & (g > 6) & (g < 21) & np.isfinite(lat) & (np.abs(lat) <= 90)
    domain &= ((solved == 31) & (nu >= 1.1) & (nu <= 1.9)) | ((solved == 95) & (pseudo >= 1.24) & (pseudo <= 1.72))
    zvalid = domain & np.isfinite(table["zpt_mas"])
    checks.add("gaia_matched_identity", ~match | ((table["source_id"] > 0) & (table["source_id"] == table["gaia_source_id"])))
    checks.add("finite_parallax_definition", table["finite_parallax"] == finite)
    checks.add("astro_quality_definition", table["astro_quality"] == quality)
    checks.add("zpt_domain_definition", table["zpt_inrange"] == domain)
    checks.add("zpt_valid_definition", table["zpt_valid"] == zvalid)
    corrected = np.full(n, np.nan)
    correction_ok = zvalid & np.isfinite(p)
    corrected[correction_ok] = p[correction_ok] - table["zpt_mas"][correction_ok]
    checks.add("parallax_correction", np.isclose(table["p_corr"], corrected, rtol=2e-6, atol=1e-7, equal_nan=True))
    snr = np.full(n, np.nan)
    snr[finite & zvalid] = corrected[finite & zvalid] / error[finite & zvalid]
    checks.add("corrected_snr", np.isclose(table["snr_corr"], snr, rtol=3e-6, atol=1e-6, equal_nan=True))
    raw_snr = np.full(n, np.nan)
    raw_snr[finite] = p[finite] / error[finite]
    checks.add("raw_signed_snr", np.isclose(table["snr_raw"], raw_snr, rtol=2e-6, atol=1e-6, equal_nan=True))
    checks.add("extended_preserves_signed_parallaxes", table["extended"] == finite)
    candidate = match & (table["gaia_in_qso_candidates"] | table["gaia_in_galaxy_candidates"])
    known = ((table["desi_known_contaminant"] & ~table["desi_match_ambiguous"])
             | (table["apogee_known_contaminant"] & ~table["apogee_match_ambiguous"]))
    checks.add("candidate_contaminants", table["gaia_candidate_contaminant"] == candidate)
    checks.add("known_contaminants", table["known_contaminant"] == known)
    astrometric_anchor = quality & zvalid & (table["snr_corr"] >= 20)
    anchor = clean & precise & astrometric_anchor & ~known & ~candidate
    checks.add("anchor_astrometry_definition", table["anchor_astrometry"] == astrometric_anchor)
    checks.add("anchor_definition", table["anchor"] == anchor)
    checks.add("extended_quality_definition", table["extended_quality"] == (quality & zvalid & clean & ~known & ~candidate))
    checks.add("spectroscopy_role_definition", table["spectroscopy_match"] == (table["desi_match"] | table["apogee_match"]))
    specq = (table["desi_quality"] | table["apogee_quality"]) & ~known
    checks.add("spectroscopy_quality_definition", table["spectro_quality"] == specq)
    checks.add("anchor_luminosity", np.isfinite(table["mg"][anchor]) & np.isclose(
        table["mg"][anchor], table["g0"][anchor] - 10 + 5 * np.log10(corrected[anchor]), atol=1e-5, rtol=0))
    checks.add("anchor_luminosity_uncertainty", np.isclose(table["sigma_mg"][anchor],
        np.hypot(table["sigma_g"][anchor], 5 / np.log(10) / table["snr_corr"][anchor]), rtol=3e-6, atol=1e-7))
    external = table["external_cluster_member"] | table["cluster_region"] | table["widebinary"]
    checks.add("external_union", table["external_any"] == external)
    reserved = external | table["identity_ambiguity_reserved"] | table["duplicate_identity_reserved"]
    checks.add("reserved_partition", ~reserved | (table["partition"] == 3))
    checks.add("valid_partition", (table["partition"] >= 0) & (table["partition"] <= 3))
    checks.add("training_partition_and_roles", table["train_eligible"] == ((table["partition"] == 0) & (anchor | table["extended_quality"] | specq)))


def check_spectrum_partitions(ids, partitions, checks, survey):
    if not len(ids):
        checks.add(f"{survey}_spectrum_cross_partition", np.ones(0, bool))
        checks.add(f"{survey}_shared_spectrum_reserved", np.ones(0, bool))
        return {"associations": 0, "unique_objects": 0, "shared_objects": 0}
    order = np.argsort(ids, kind="stable")
    ids, partitions = ids[order], partitions[order]
    starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
    sizes = np.diff(np.r_[starts, len(ids)])
    low, high = np.minimum.reduceat(partitions, starts), np.maximum.reduceat(partitions, starts)
    checks.add(f"{survey}_spectrum_cross_partition", low == high)
    checks.add(f"{survey}_shared_spectrum_reserved", (sizes == 1) | ((low == 3) & (high == 3)))
    return {"associations": len(ids), "unique_objects": len(starts), "shared_objects": int((sizes > 1).sum())}


def validate_run(run_dir, chunk_rows=200000, sample_size=50000):
    require_compute()
    if chunk_rows < 1 or sample_size < 1:
        raise ValueError("chunk_rows and sample_size must be positive")
    started = time.time()
    root = Path(run_dir).resolve()
    manifest = json.loads((root / "scan_manifest.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    config, signature = manifest["config"], manifest["signature"]
    checks = Checks()
    checks.add("manifest_summary_signature", summary["signature"] == signature)
    ids, sources, partitions, membership = [], [], [], []
    spectra = {survey: [[], []] for survey in ("desi", "apogee")}
    counts, cutflow = Counter(), Counter()
    expected_partitions = {role: np.zeros(4, dtype="i8") for role in ROLES}
    coverage = {role: Coverage(sample_size) for role in ("parent", "anchor", "extended", "spectroscopy", "external_test")}
    references = Path(config["reference_dir"]) / "holdout_references_v1.npz"
    with np.load(references, allow_pickle=False) as ref:
        members = np.unique(np.concatenate((ref["member_source_id"], ref["binary_source_id"])))
    for table in chunks(root / "tables/parent.fits", COLUMNS, chunk_rows):
        check_physics(table, config, checks)
        role_bits = np.zeros(len(table), dtype="u1")
        for index, (role, mask) in enumerate(zip(ROLES, role_masks(table))):
            role_bits[mask] |= 1 << index
            valid = mask & (table["partition"] >= 0) & (table["partition"] <= 3)
            expected_partitions[role] += np.bincount(table["partition"][valid], minlength=4)
            if role in coverage:
                coverage[role].update(table, mask)
        ids.append(table["ls_id"])
        sources.append(table["source_id"])
        partitions.append(table["partition"])
        membership.append(role_bits)
        positions = np.searchsorted(members, table["source_id"])
        known_member = positions < len(members)
        known_member[known_member] &= members[positions[known_member]] == table["source_id"][known_member]
        checks.add("published_member_ids_reserved", ~known_member | ((table["partition"] == 3) & table["external_any"]))
        counts["published_members_found"] += int(known_member.sum())
        counts["negative_parallax"] += int((table["extended"] & (table["gaia_parallax"] < 0)).sum())
        counts["zero_parallax"] += int((table["extended"] & (table["gaia_parallax"] == 0)).sum())
        counts["negative_in_quality"] += int((table["extended_quality"] & (table["gaia_parallax"] < 0)).sum())
        for name in ("gaia_match", "extended_quality", "spectro_quality", "known_contaminant", "gaia_candidate_contaminant"):
            counts[name] += int(table[name].sum())
        astro = table["anchor_astrometry"]
        phot_precise = astro & table["grz_precise"]
        fractions = phot_precise & table["grz_fraction_clean"]
        image_clean = fractions & table["grz_mask_clean"]
        cutflow["retained_measured_parent"] += len(table)
        cutflow["raw_parallax_snr_ge20"] += int((table["snr_raw"] >= 20).sum())
        cutflow["anchor_astrometry"] += int(astro.sum())
        cutflow["then_grz_precise"] += int(phot_precise.sum())
        cutflow["then_grz_fraction_clean"] += int(fractions.sum())
        cutflow["then_grz_mask_clean"] += int(image_clean.sum())
        cutflow["final_anchor_after_contaminants"] += int(table["anchor"].sum())
        old_anymask, old_rchisq = np.ones(len(table), bool), np.ones(len(table), bool)
        for band in ("g", "r", "z"):
            old_anymask &= table[f"anymask_{band}"] == 0
            old_rchisq &= np.isfinite(table[f"rchisq_{band}"]) & (table[f"rchisq_{band}"] < 10)
        cutflow["astro_anchors_failing_old_anymask"] += int((astro & ~old_anymask).sum())
        cutflow["astro_anchors_failing_old_rchisq"] += int((astro & ~old_rchisq).sum())
        cutflow["astro_anchors_failing_new_grz_mask"] += int((astro & ~table["grz_mask_clean"]).sum())
        for survey in spectra:
            good_anchor = table["anchor"] & table[f"{survey}_quality"]
            counts[f"anchor_{survey}_quality"] += int(good_anchor.sum())
            counts[f"anchor_{survey}_giant_logg_lt3p2"] += int((good_anchor & (table[f"{survey}_logg"] < 3.2)).sum())
            counts[f"anchor_{survey}_metalpoor_feh_ltm1p5"] += int((good_anchor & (table[f"{survey}_feh"] < -1.5)).sum())
            object_ids = np.char.strip(table[f"{survey}_object_id"].astype("U"))
            matched = table[f"{survey}_match"] & (object_ids != "")
            spectra[survey][0].append(object_ids[matched])
            spectra[survey][1].append(table["partition"][matched])
    def concatenate(values, dtype):
        return np.concatenate(values) if values else np.zeros(0, dtype=dtype)
    ids, sources = concatenate(ids, "i8"), concatenate(sources, "i8")
    partitions, membership = concatenate(partitions, "i2"), concatenate(membership, "u1")
    checks.add("positive_legacy_ids", ids > 0)
    checks.add("unique_legacy_ids", len(np.unique(ids)) == len(ids))
    known = sources[sources > 0]
    checks.add("unique_gaia_ids", len(np.unique(known)) == len(known))
    sorting = np.argsort(ids, kind="stable")
    sorted_ids = ids[sorting]
    actual_counts = {}
    for index, role in enumerate(ROLES):
        path = root / "tables" / f"{role}.fits"
        with fitsio.FITS(str(path)) as hdus:
            header = hdus[1].read_header()
            actual_counts[role] = hdus[1].get_nrows()
            checks.add(f"{role}_header_signature", header.get("RUNHASH") == signature[:32])
            checks.add(f"{role}_header_role", header.get("ROLE") == role)
        got, role_partition = [], np.zeros(4, dtype="i8")
        for table in chunks(path, ["ls_id", "source_id", "partition"], chunk_rows):
            got.append(table["ls_id"])
            position = np.searchsorted(sorted_ids, table["ls_id"])
            inside = position < len(ids)
            matched = np.zeros(len(table), bool)
            matched[inside] = sorted_ids[position[inside]] == table["ls_id"][inside]
            checks.add(f"{role}_belongs_to_parent", matched)
            dest = sorting[position[matched]]
            checks.add(f"{role}_same_identity_partition", (table["source_id"][matched] == sources[dest]) & (table["partition"][matched] == partitions[dest]))
            checks.add(f"{role}_membership_condition", (membership[dest] & (1 << index)) != 0)
            valid = (table["partition"] >= 0) & (table["partition"] <= 3)
            checks.add(f"{role}_valid_partition", valid)
            role_partition += np.bincount(table["partition"][valid], minlength=4)
        got = concatenate(got, "i8")
        expected = ids[(membership & (1 << index)) != 0]
        checks.add(f"{role}_exact_identity_set", np.array_equal(np.sort(got), np.sort(expected)))
        checks.add(f"{role}_summary_count", actual_counts[role] == summary["counts"][role])
        checks.add(f"{role}_summary_partition_counts", role_partition == summary["partition_counts"][role])
        checks.add(f"{role}_parent_partition_counts", role_partition == expected_partitions[role])
    for key, label in (("gaia_match", "gaia_matched"), ("negative_parallax", "negative_parallax"),
                       ("zero_parallax", "zero_parallax"), ("extended_quality", "extended_quality"),
                       ("spectro_quality", "spectro_quality")):
        checks.add(f"summary_{label}", counts[key] == summary["counts"][label])
    spectrum_summary = {}
    for survey, (object_ids, part) in spectra.items():
        spectrum_summary[survey] = check_spectrum_partitions(concatenate(object_ids, "U1"), concatenate(part, "i2"), checks, survey)
    result = {"passed": checks.passed, "checked_utc": datetime.now(timezone.utc).isoformat(),
              "run_dir": str(root), "signature": signature, "source_code_commit": manifest["code_commit"],
              "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "host": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
              "elapsed_seconds": time.time() - started, "limited_smoke_run": manifest["limited_smoke_run"],
              "checks": checks.results, "counts": {**actual_counts, **dict(counts)},
              "anchor_cutflow_on_retained_measured_parent": dict(cutflow),
              "spectrum_identity": spectrum_summary,
              "coverage_method": f"All-row finite counts/min/max; quantiles from at most {sample_size} deterministic lowest-hash Legacy IDs per role. Spectral labels use quality-approved spectra only. No model fitting.",
              "coverage": {role: value.result() for role, value in coverage.items()},
              "scope": "Final table consistency and physical predicates, including published member IDs from normalized references. Original source catalogs are not rescanned; angular reference regions and zero-point coefficients are not independently refit."}
    write_json(root / "validation.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--chunk-rows", type=int, default=200000)
    parser.add_argument("--sample-size", type=int, default=50000)
    args = parser.parse_args()
    require_compute()
    try:
        result = validate_run(args.run_dir, args.chunk_rows, args.sample_size)
    except Exception as error:
        result = {"passed": False, "error": f"{type(error).__name__}: {error}",
                  "checked_utc": datetime.now(timezone.utc).isoformat()}
        write_json(Path(args.run_dir) / "validation.json", result)
    print(json.dumps({key: result[key] for key in ("passed", "counts", "anchor_cutflow_on_retained_measured_parent", "spectrum_identity", "elapsed_seconds", "error") if key in result}, indent=2))
    if not result["passed"]:
        if "checks" in result:
            print(json.dumps({key: value for key, value in result["checks"].items() if value["violations"]}, indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
