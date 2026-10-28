"""Build auditable sample roles and locked test partitions; no model fitting."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import socket
import subprocess
import time

import fitsio
import healpy as hp
import numpy as np
from scipy.spatial import cKDTree

from .photometry import derive_photometry
from .gaia import join_gaia, derive_gaia_labels
from .holdouts import load_reference_catalogs, annotate_holdouts


CTX = {}
ROTATOR = hp.Rotator(coord=["C", "G"])
BANDS = ("G", "R", "Z", "W1", "W2")
CORE_COLUMNS = ["RELEASE", "BRICKID", "OBJID", "TYPE", "RA", "DEC", "EBV",
                "MASKBITS", "REF_CAT", "REF_ID", "REF_EPOCH", "PARALLAX",
                "PARALLAX_IVAR", "FITBITS"]
PHOT_COLUMNS = [f"{p}_{b}" for b in BANDS for p in
                ("FLUX", "FLUX_IVAR", "MW_TRANSMISSION", "NOBS", "FRACFLUX")]
PHOT_COLUMNS += [f"{p}_{b}" for b in ("G", "R", "Z") for p in
                 ("FRACMASKED", "FRACIN", "ANYMASK", "RCHISQ")]
PHOT_COLUMNS += ["WISEMASK_W1", "WISEMASK_W2"]
ROLE_NAMES = ("parent", "anchor", "extended", "spectroscopy", "external_test",
              "coverage_reference")


def require_compute():
    if not os.environ.get("SLURM_JOB_ID") or not socket.gethostname().startswith("nid"):
        raise RuntimeError("Full catalog work must run in an allocated NERSC compute step")


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def structured(columns):
    arrays = {}
    length = None
    for name, value in columns.items():
        value = np.asarray(value)
        if value.ndim != 1:
            raise ValueError(f"Only scalar table columns supported: {name}")
        if value.dtype.kind == "U":
            value = value.astype(f"S{value.dtype.itemsize // 4}")
        elif value.dtype.kind in "iu" and value.dtype.itemsize == 1:
            value = value.astype(np.int16)
        if length is not None and len(value) != length:
            raise ValueError(f"Column length mismatch: {name}")
        length = len(value)
        arrays[name] = value
    result = np.empty(length or 0, dtype=[(k, v.dtype) for k, v in arrays.items()])
    for name, value in arrays.items():
        result[name] = value
    return result


def hash_ids(ids):
    # SplitMix64 is deterministic across process counts and input ordering.
    x = np.asarray(ids, dtype=np.uint64).copy()
    with np.errstate(over="ignore"):
        x += np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> 27)) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> 31)


def legacy_ids(data):
    release = data["RELEASE"].astype(np.int64)
    brick = data["BRICKID"].astype(np.int64)
    obj = data["OBJID"].astype(np.int64)
    if np.any((release < 0) | (release >= 2**23) | (brick < 0)
              | (brick >= 2**20) | (obj < 0) | (obj >= 2**20)):
        raise ValueError("Legacy identity outside collision-free packing range")
    return (release << 40) | (brick << 20) | obj


def galactic(ra, dec):
    longitude, latitude = ROTATOR(ra, dec, lonlat=True)
    return np.atleast_1d(np.mod(longitude, 360)), np.atleast_1d(latitude)


def tile_bounds(path):
    match = re.fullmatch(r"sweep-(\d{3})([pm])(\d{3})-(\d{3})([pm])(\d{3})\.fits", Path(path).name)
    if not match:
        raise ValueError(f"Unexpected sweep filename: {path}")
    r0, s0, d0, r1, s1, d1 = match.groups()
    return (float(r0), float(d0) * (1 if s0 == "p" else -1),
            float(r1), float(d1) * (1 if s1 == "p" else -1))


def eligible_tiles(config):
    paths = sorted(Path(config["sweep_root"]).glob("sweep-*.fits"))
    selected = []
    for path in paths:
        r0, d0, r1, d1 = tile_bounds(path)
        width = (r1 - r0) % 360
        if width == 0:
            width = 360
        _, bc = galactic(np.array([(r0 + width / 2) % 360]), np.array([(d0 + d1) / 2]))
        # The sum of half coordinate spans is a conservative spherical bound.
        radius = width / 2 + (d1 - d0) / 2
        if abs(float(bc[0])) + radius > config["sky"]["abs_b_min"]:
            selected.append(path)
    return paths, selected


def unit_vectors(ra, dec):
    ra, dec = np.deg2rad(ra), np.deg2rad(dec)
    c = np.cos(dec)
    return np.column_stack((c * np.cos(ra), c * np.sin(ra), np.sin(dec)))


def load_spectroscopy(run_dir):
    index = json.loads((Path(run_dir) / "spectroscopy/index.json").read_text())
    result = {}
    for survey in ("desi", "apogee"):
        table = fitsio.read(index[survey], ext=1)
        rows = np.flatnonzero(table["source_id"] > 0)
        # Resolve multiple target observations of the same Gaia source by label
        # quality, then S/N. Preserve the full projected pools separately.
        order = np.lexsort((rows, -np.nan_to_num(table["snr"][rows], nan=-1),
                            ~table["quality"][rows], table["source_id"][rows]))
        rows = rows[order]
        first = np.r_[True, np.diff(table["source_id"][rows]) != 0] if len(rows) else np.array([], bool)
        rows = rows[first]
        result[survey] = {"table": table, "ids": table["source_id"][rows], "rows": rows}
    return result, index["audit"]


def match_spectroscopy(source_ids, ra, dec, bounds, catalog, radius_arcsec):
    table, ids, rows = catalog["table"], catalog["ids"], catalog["rows"]
    matched = np.full(len(ra), -1, dtype=np.int64)
    method = np.zeros(len(ra), dtype=np.int16)
    ambiguous = np.zeros(len(ra), dtype=bool)
    if len(ids):
        positions = np.searchsorted(ids, source_ids)
        safe = np.minimum(positions, len(ids) - 1)
        ok = (source_ids > 0) & (positions < len(ids)) & (ids[safe] == source_ids)
        matched[ok], method[ok] = rows[safe[ok]], 1
    missing = np.flatnonzero(matched < 0)
    if not len(missing):
        return matched, method, ambiguous
    r0, d0, r1, d1 = bounds
    margin = radius_arcsec / 3600 * 1.1
    # A generous RA margin near the poles; the final cut uses spherical distance.
    ra_margin = min(180, margin / max(np.cos(np.deg2rad(max(abs(d0), abs(d1)))), 0.001))
    delta = (table["ra"] - r0 + ra_margin) % 360
    local = np.isfinite(table["ra"]) & np.isfinite(table["dec"])
    local &= (table["dec"] >= d0 - margin) & (table["dec"] <= d1 + margin)
    local &= delta <= ((r1 - r0) % 360) + 2 * ra_margin
    candidates = np.flatnonzero(local)
    if not len(candidates):
        return matched, method, ambiguous
    tree = cKDTree(unit_vectors(table["ra"][candidates], table["dec"][candidates]))
    chord = 2 * np.sin(np.deg2rad(radius_arcsec / 3600) / 2)
    distance, nearest = tree.query(unit_vectors(ra[missing], dec[missing]), k=2,
                                   distance_upper_bound=chord, workers=1)
    ok = np.isfinite(distance[:, 0])
    chosen = candidates[np.minimum(nearest[:, 0], len(candidates) - 1)]
    # Do not use proximity to override a disagreement between two known DR3 IDs.
    ok &= (source_ids[missing] <= 0) | (table["source_id"][chosen] <= 0)
    good = missing[ok]
    matched[good], method[good] = chosen[ok], 2
    ambiguous[good] = np.isfinite(distance[ok, 1])
    return matched, method, ambiguous


def spec_columns(catalog, matched, method, ambiguous, prefix):
    table = catalog["table"]
    ok = matched >= 0
    out = {f"{prefix}_match": ok, f"{prefix}_match_method": method,
           f"{prefix}_match_ambiguous": ambiguous}
    for name in ("source_id", "object_id", "snr", "teff", "teff_err", "logg", "logg_err",
                 "feh", "feh_err", "quality", "quality_reason", "is_stellar",
                 "known_contaminant", "spectype", "label_dwarf", "label_giant"):
        dtype = table.dtype[name]
        value = np.empty(len(matched), dtype=dtype)
        value[:] = np.nan if dtype.kind == "f" else ("" if dtype.kind in "SU" else 0)
        value[ok] = table[name][matched[ok]]
        if name in ("quality", "label_dwarf", "label_giant"):
            value &= ~ambiguous
        out[f"{prefix}_{name}"] = value
    return out


def make_partition_map(config):
    nside, seed = config["nside"], config["seed"]
    cells = np.random.default_rng(seed).permutation(hp.nside2npix(nside))
    result = np.zeros(len(cells), dtype=np.int16)
    nv = int(round(len(cells) * config["validation_fraction"]))
    nt = int(round(len(cells) * (1 - config["train_fraction"] - config["validation_fraction"])))
    result[cells[:nt]] = 2
    result[cells[nt:nt + nv]] = 1
    return result


def setup_context(config, run_dir, signature):
    spectroscopy, audit = load_spectroscopy(run_dir)
    CTX.update(config=config, run_dir=Path(run_dir), signature=signature,
               spectroscopy=spectroscopy, spectroscopy_audit=audit,
               references=load_reference_catalogs(config["reference_dir"]),
               partitions=make_partition_map(config["split"]))


def process_tile(path_string):
    start = time.time()
    path = Path(path_string)
    cfg, root, signature = CTX["config"], CTX["run_dir"], CTX["signature"]
    outpath = root / "fragments" / path.name
    metapath = root / "fragments" / (path.stem + ".json")
    if metapath.exists() and outpath.exists():
        old = json.loads(metapath.read_text())
        if old.get("signature") == signature and old.get("source_mtime_ns") == path.stat().st_mtime_ns:
            return old
        raise RuntimeError(f"Stale output requires a new run directory: {outpath}")
    raw = fitsio.read(str(path), ext=1, columns=CORE_COLUMNS + PHOT_COLUMNS)
    counts = Counter(raw_rows_inspected=len(raw))
    psf = np.char.strip(raw["TYPE"].astype("U")) == "PSF"
    counts["psf_rows"] = int(psf.sum())
    raw = raw[psf]
    longitude, latitude = galactic(raw["RA"], raw["DEC"])
    highlat = np.abs(latitude) > cfg["sky"]["abs_b_min"]
    counts["highlat_psf"] = int(highlat.sum())
    domain = highlat & np.isfinite(raw["EBV"]) & (raw["EBV"] >= 0) & (raw["EBV"] < cfg["sky"]["ebv_max"])
    raw, longitude, latitude = raw[domain], longitude[domain], latitude[domain]
    counts["domain_psf"] = len(raw)
    phot = derive_photometry(raw, **cfg["photometry"])
    embedded_sn = raw["PARALLAX"] * np.sqrt(np.maximum(raw["PARALLAX_IVAR"], 0))
    embedded_high = np.isfinite(embedded_sn) & (embedded_sn >= 20)
    counts["domain_embedded_snr20_uncorrected"] = int(embedded_high.sum())
    counts["embedded_snr20_without_measured_grz"] = int((embedded_high & ~phot["grz_base"]).sum())
    old_anymask = np.ones(len(raw), bool)
    old_chi = np.ones(len(raw), bool)
    for b in ("G", "R", "Z"):
        old_anymask &= raw[f"ANYMASK_{b}"] == 0
        old_chi &= raw[f"RCHISQ_{b}"] < 10
    counts["measured_snr20_failing_old_anymask"] = int((embedded_high & phot["grz_base"] & ~old_anymask).sum())
    counts["measured_snr20_failing_old_rchisq"] = int((embedded_high & phot["grz_base"] & ~old_chi).sum())
    base = phot["grz_base"]
    raw, longitude, latitude = raw[base], longitude[base], latitude[base]
    phot = {k: v[base] for k, v in phot.items()}
    counts["grz_base"] = len(raw)
    for key in ("grz_clean", "grz_precise", "grz_high_quality", "grz_w1", "grz_w1_w2", "full_high_quality"):
        counts[key] = int(phot[key].sum())
    source_ids = raw["REF_ID"].astype(np.int64).copy()
    source_ids[(np.char.strip(raw["REF_CAT"].astype("U")) != "G3") | (source_ids <= 0)] = 0
    gaia = join_gaia(source_ids, cfg["gaia_root"])
    counts["legacy_g3_positive_id"] = int((source_ids > 0).sum())
    counts["legacy_g3_unmatched"] = int(((source_ids > 0) & ~gaia["gaia_match"]).sum())
    counts["gaia_file_missing"] = int(gaia["gaia_file_missing"].sum())
    if counts["gaia_file_missing"]:
        raise FileNotFoundError(f"Gaia source tiles missing for {path.name}")
    labels = derive_gaia_labels(gaia)
    sid = np.where(gaia["gaia_match"], gaia["source_id"], source_ids)
    specs = {}
    for survey, catalog in CTX["spectroscopy"].items():
        match, method, ambig = match_spectroscopy(sid, raw["RA"], raw["DEC"], tile_bounds(path),
                                                 catalog, cfg["spectroscopy_match_radius_arcsec"])
        specs.update(spec_columns(catalog, match, method, ambig, survey))
    # Recover missing identities only from unambiguous, mutually consistent
    # spectroscopy associations. Rejoin before labels, memberships and splits.
    desi_id = np.where(specs["desi_match"] & ~specs["desi_match_ambiguous"], specs["desi_source_id"], 0)
    apogee_id = np.where(specs["apogee_match"] & ~specs["apogee_match_ambiguous"], specs["apogee_source_id"], 0)
    identity_conflict = (desi_id > 0) & (apogee_id > 0) & (desi_id != apogee_id)
    recovered_id = np.maximum(desi_id, apogee_id)
    recover = (sid <= 0) & (recovered_id > 0) & ~identity_conflict
    if np.any(recover):
        recovered_gaia = join_gaia(recovered_id[recover], cfg["gaia_root"])
        if np.any(recovered_gaia["gaia_file_missing"]):
            raise FileNotFoundError("Gaia tiles missing for recovered spectroscopy identities")
        for key in gaia:
            gaia[key][recover] = recovered_gaia[key]
        sid[recover] = recovered_id[recover]
        labels = derive_gaia_labels(gaia)
    identity_reserved = identity_conflict | specs["desi_match_ambiguous"] | specs["apogee_match_ambiguous"]
    holds = annotate_holdouts(CTX["references"], sid, raw["RA"], raw["DEC"])
    specmatch = specs["desi_match"] | specs["apogee_match"]
    contaminant = ((specs["desi_known_contaminant"] & ~specs["desi_match_ambiguous"])
                   | (specs["apogee_known_contaminant"] & ~specs["apogee_match_ambiguous"]))
    candidate = labels["contaminant_candidate"]
    stellar_core = ~contaminant & ~candidate
    anchor = phot["grz_high_quality"] & labels["anchor_astrometry"] & stellar_core
    extended = labels["finite_parallax"]
    extended_quality = extended & labels["astro_quality"] & labels["zpt_valid"] & phot["grz_clean"] & stellar_core
    specquality = (specs["desi_quality"] | specs["apogee_quality"]) & ~contaminant
    lsids = legacy_ids(raw)
    reference = hash_ids(lsids) % cfg["coverage_reference_modulus"] == 0
    keep = gaia["gaia_match"] | specmatch | holds["external_any"] | reference
    counts.update(gaia_matched=int(gaia["gaia_match"].sum()), anchor=int(anchor.sum()),
                  extended=int(extended.sum()), extended_quality=int(extended_quality.sum()),
                  spectroscopy=int(specmatch.sum()), spectro_quality=int(specquality.sum()),
                  external_reserved=int(holds["external_any"].sum()),
                  negative_parallax=int((extended & (gaia["parallax"] < 0)).sum()),
                  zero_parallax=int((extended & (gaia["parallax"] == 0)).sum()),
                  gaia_candidate_contaminant=int(candidate.sum()), known_contaminant=int(contaminant.sum()),
                  coverage_reference=int(reference.sum()), retained_parent=int(keep.sum()))
    # Match positions at Gaia epoch for deterministic duplicate-source partitions.
    split_ra = np.where(gaia["gaia_match"], gaia["ra"], raw["RA"])
    split_dec = np.where(gaia["gaia_match"], gaia["dec"], raw["DEC"])
    pixels = hp.ang2pix(cfg["split"]["nside"], split_ra, split_dec, nest=True, lonlat=True)
    partition = CTX["partitions"][pixels].copy()
    partition[holds["external_any"] | identity_reserved] = 3
    mg = np.full(len(raw), np.nan, np.float32)
    sigma_mg = np.full(len(raw), np.nan, np.float32)
    luminosity_ok = labels["anchor_astrometry"] & phot["grz_base"]
    mg[luminosity_ok] = phot["g0"][luminosity_ok] - 10 + 5 * np.log10(labels["p_corr"][luminosity_ok])
    sigma_mg[luminosity_ok] = np.hypot(phot["sigma_g"][luminosity_ok],
                                      5 / np.log(10) / labels["snr_corr"][luminosity_ok])
    columns = {"ls_id": lsids, "source_id": sid, "ra": raw["RA"], "dec": raw["DEC"],
               "l": longitude.astype("f4"), "b": latitude.astype("f4"),
               "anchor": anchor, "extended": extended, "extended_quality": extended_quality,
               "spectroscopy_match": specmatch, "spectro_quality": specquality,
               "known_contaminant": contaminant, "gaia_candidate_contaminant": candidate,
               "coverage_reference": reference, "partition": partition,
               "gaia_identity_recovered": recover, "identity_ambiguity_reserved": identity_reserved,
               "duplicate_identity_reserved": np.zeros(len(raw), bool),
               "split_healpix": pixels.astype("i4"), "mg": mg, "sigma_mg": sigma_mg,
               "train_eligible": (partition == 0) & (anchor | extended_quality | specquality)}
    for name in raw.dtype.names:
        if name.lower() not in columns:
            columns[name.lower()] = raw[name]
    columns.update(phot)
    columns.update({"gaia_" + k: v for k, v in gaia.items()})
    columns.update(labels)
    columns.update(specs)
    columns.update(holds)
    table = structured({k: v[keep] for k, v in columns.items()})
    outpath.parent.mkdir(parents=True, exist_ok=True)
    temp = outpath.with_suffix(".fits.tmp")
    fitsio.write(str(temp), table, extname="STAGE1", clobber=True,
                 header={"RUNHASH": signature[:32], "SOURCE": path.name})
    temp.replace(outpath)
    metadata = {"source": str(path), "source_bytes": path.stat().st_size,
                "source_mtime_ns": path.stat().st_mtime_ns, "signature": signature,
                "counts": dict(counts), "elapsed_seconds": time.time() - start,
                "rows": len(table), "output": str(outpath)}
    write_json(metapath, metadata)
    return metadata


def run_signature(config):
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    for name in ("stage1.py", "photometry.py", "gaia.py", "holdouts.py", "spectroscopy.py"):
        digest.update(Path(__file__).with_name(name).read_bytes())
    reference = Path(config["reference_dir"]) / "holdout_references_v1.npz"
    with reference.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan(config, workers=16, limit=None):
    require_compute()
    run_dir = Path(config["output_dir"])
    signature = run_signature(config)
    all_paths, paths = eligible_tiles(config)
    if limit:
        paths = paths[:limit]
    setup_context(config, run_dir, signature)
    manifest = {"config": config, "signature": signature, "all_source_tiles": len(all_paths),
                "eligible_tiles": len(paths), "limited_smoke_run": bool(limit),
                "source_bytes": sum(p.stat().st_size for p in paths),
                "files": [str(p) for p in paths], "compute_host": socket.gethostname(),
                "slurm_job_id": os.environ["SLURM_JOB_ID"],
                "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "partition_map": CTX["partitions"].tolist()}
    manifest_path = run_dir / "scan_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["signature"] != signature:
            raise RuntimeError("Run signature changed; choose a fresh output directory")
    write_json(manifest_path, manifest)
    print(json.dumps({k: manifest[k] for k in ("eligible_tiles", "source_bytes", "signature")}), flush=True)
    totals = Counter()
    start = time.time()
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as executor:
        tasks = {executor.submit(process_tile, str(path)): path for path in paths}
        for number, task in enumerate(as_completed(tasks), 1):
            result = task.result()
            totals.update(result["counts"])
            if number <= 4 or number % 10 == 0 or number == len(paths):
                print(json.dumps({"completed": number, "total": len(paths),
                                  "last": tasks[task].name, "elapsed_seconds": round(time.time()-start),
                                  "counts": dict(totals)}), flush=True)
    write_json(run_dir / "scan_summary.json", {"counts": dict(totals), "elapsed_seconds": time.time()-start,
                                               "completed_tiles": len(paths), "signature": signature})


def finalize(config):
    require_compute()
    root = Path(config["output_dir"])
    manifest = json.loads((root / "scan_manifest.json").read_text())
    if manifest["signature"] != run_signature(config):
        raise RuntimeError("Source code/config/reference signature changed since scan")
    paths = [root / "fragments" / Path(p).name for p in manifest["files"]]
    if any(not p.exists() for p in paths):
        raise RuntimeError("Not all requested tiles completed")
    totals = Counter()
    identities, legacy, scores, lengths = [], [], [], []
    reservation_flags = []
    spectrum_ids = {survey: [] for survey in ("desi", "apogee")}
    for path in paths:
        meta = json.loads(path.with_suffix(".json").read_text())
        if meta["signature"] != manifest["signature"]:
            raise RuntimeError(f"Mixed fragment signatures: {path}")
        totals.update(meta["counts"])
        small = fitsio.read(str(path), ext=1, columns=["source_id", "ls_id", "grz_clean", "sigma_g", "sigma_r", "sigma_z", "desi_object_id", "apogee_object_id", "external_any", "identity_ambiguity_reserved"])
        identities.append(np.where(small["source_id"] > 0, small["source_id"], -small["ls_id"]))
        legacy.append(small["ls_id"])
        scores.append((~small["grz_clean"]).astype(float) * 100 + small["sigma_g"] + small["sigma_r"] + small["sigma_z"])
        lengths.append(len(small))
        reservation_flags.append(small["external_any"] | small["identity_ambiguity_reserved"])
        for survey in spectrum_ids:
            spectrum_ids[survey].append(small[f"{survey}_object_id"])
    keys, lsids, score = np.concatenate(identities), np.concatenate(legacy), np.concatenate(scores)
    order = np.lexsort((lsids, score, keys))
    first = np.r_[True, keys[order[1:]] != keys[order[:-1]]] if len(order) else np.array([], bool)
    keep = np.zeros(len(keys), bool)
    keep[order[first]] = True
    duplicate_reserved = np.zeros(len(keep), bool)
    if len(order):
        starts = np.flatnonzero(first)
        sizes = np.diff(np.r_[starts, len(order)])
        any_reserved = np.logical_or.reduceat(np.concatenate(reservation_flags)[order], starts)
        duplicate_reserved[order[first]] = any_reserved & (sizes > 1)
    duplicate_rows = int(len(keys) - keep.sum())
    if len(np.unique(lsids[keep])) != int(keep.sum()):
        raise RuntimeError("Conflicting Gaia identities for a repeated Legacy ID")
    # Multiple optical identities sharing one spectrum are conservatively
    # quarantined together. This also covers spectra without a Gaia identity.
    identity_reserved = np.zeros(len(keep), bool)
    for survey, chunks in spectrum_ids.items():
        ids = np.char.strip(np.concatenate(chunks).astype("U"))
        unique, counts = np.unique(ids[keep & (ids != "")], return_counts=True)
        repeated = unique[counts > 1]
        identity_reserved |= np.isin(ids, repeated)
    del identities, legacy, scores, score, order, keys, lsids, spectrum_ids, ids
    table_dir = root / "tables"
    table_dir.mkdir(exist_ok=True)
    if any((table_dir / f"{name}.fits").exists() for name in ROLE_NAMES):
        raise RuntimeError("Final tables already exist; inspect instead of overwriting")
    writers = {name: fitsio.FITS(str(table_dir / f"{name}.fits"), "rw", clobber=False) for name in ROLE_NAMES}
    initialized = set()
    final_counts = Counter()
    partition_counts = {name: [0, 0, 0, 0] for name in ROLE_NAMES}
    coverage = {"anchor_by_photometry_tier": [0, 0, 0], "external_cluster_member": 0,
                "cluster_region": 0, "cluster_region_only": 0, "widebinary": 0, "negative_in_quality": 0,
                "desi_quality": 0, "apogee_quality": 0, "metal_poor_desi": 0,
                "metal_poor_apogee": 0, "desi_giant": 0, "apogee_giant": 0}
    offset = 0
    try:
        for path, length in zip(paths, lengths):
            rows = np.flatnonzero(keep[offset:offset+length])
            reserved = identity_reserved[offset:offset+length][rows]
            duplicate_external = duplicate_reserved[offset:offset+length][rows]
            offset += length
            table = fitsio.read(str(path), ext=1, rows=rows)
            table["identity_ambiguity_reserved"] |= reserved
            table["duplicate_identity_reserved"] = duplicate_external
            quarantine = table["identity_ambiguity_reserved"] | duplicate_external
            table["partition"][quarantine] = 3
            table["train_eligible"][quarantine] = False
            role_masks = {"parent": np.ones(len(table), bool), "anchor": table["anchor"],
                          "extended": table["extended"], "spectroscopy": table["spectroscopy_match"],
                          "external_test": table["external_any"] | table["identity_ambiguity_reserved"] | table["duplicate_identity_reserved"], "coverage_reference": table["coverage_reference"]}
            if np.any(table["external_any"] & (table["partition"] != 3)):
                raise RuntimeError("External group leaked into a nonexternal partition")
            if np.any(table["anchor"] & (~np.isfinite(table["mg"]) | (table["snr_corr"] < 20))):
                raise RuntimeError("Invalid anchor luminosity/precision")
            for name, mask in role_masks.items():
                subset = table[mask]
                if name not in initialized:
                    writers[name].write(subset, extname="STAGE1", header={"RUNHASH": manifest["signature"][:32], "ROLE": name})
                    initialized.add(name)
                elif len(subset):
                    writers[name][1].append(subset)
                final_counts[name] += len(subset)
                bins = np.bincount(subset["partition"], minlength=4)
                partition_counts[name] = (np.asarray(partition_counts[name]) + bins).tolist()
            for key in ("gaia_match", "extended_quality", "spectro_quality", "known_contaminant", "gaia_candidate_contaminant", "identity_ambiguity_reserved", "duplicate_identity_reserved", "gaia_identity_recovered"):
                final_counts[key] += int(table[key].sum())
            final_counts["negative_parallax"] += int((table["extended"] & (table["gaia_parallax"] < 0)).sum())
            final_counts["zero_parallax"] += int((table["extended"] & (table["gaia_parallax"] == 0)).sum())
            for key in ("external_cluster_member", "cluster_region", "widebinary"):
                coverage[key] += int(table[key].sum())
            coverage["cluster_region_only"] += int((table["cluster_region"] & ~table["external_cluster_member"] & ~table["widebinary"]).sum())
            coverage["negative_in_quality"] += int((table["extended_quality"] & (table["gaia_parallax"] < 0)).sum())
            for tier in range(3):
                coverage["anchor_by_photometry_tier"][tier] += int((table["anchor"] & (table["photometry_tier"] == tier)).sum())
            for survey in ("desi", "apogee"):
                quality = table[f"{survey}_quality"]
                coverage[f"{survey}_quality"] += int(quality.sum())
                coverage[f"metal_poor_{survey}"] += int((quality & (table[f"{survey}_feh"] < -1.5)).sum())
                coverage[f"{survey}_giant"] += int((quality & (table[f"{survey}_logg"] < 3.2)).sum())
    finally:
        for writer in writers.values():
            writer.close()
    final_counts["gaia_matched"] = final_counts.pop("gaia_match")
    final_counts["external_reserved"] = final_counts["external_test"]
    # Roles overlap intentionally; never sum anchor + extended as independent stars.
    summary = {"signature": manifest["signature"], "code_commit": manifest["code_commit"],
               "counts": {**dict(totals), **dict(final_counts)}, "pre_dedup_counts": dict(totals),
               "deduplicated_rows": duplicate_rows, "partition_counts": partition_counts,
               "partition_labels": ["train", "validation", "spatial_test", "external_reserved"],
               "coverage": coverage, "eligible_tiles": len(paths),
               "all_source_tiles": manifest["all_source_tiles"], "source_bytes": manifest["source_bytes"],
               "limited_smoke_run": manifest["limited_smoke_run"],
               "coverage_reference_modulus": config["coverage_reference_modulus"],
               "role_overlap_note": "Anchor is a high-quality subset of extended; spectroscopy and external roles overlap. Select by partition before any future fitting.",
               "validation": {"unique_identity_per_parent": True, "unique_legacy_id_per_parent": True,
                              "external_groups_quarantined": True, "no_training_performed": True}}
    write_json(root / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
