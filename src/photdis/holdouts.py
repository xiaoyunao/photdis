"""Published group reservations for leakage control, not distance truth.

Raw reference catalogs and the normalized NPZ stay on NERSC. Run preparation
on allocated compute: ``python -m photdis.holdouts prepare --reference-dir DIR``.
All published non-rejected Hunt/Reffert members, probable globular members,
and both endpoints of El-Badry pairs with R < 0.1 are reserved. Angular regions
quarantine background stars too: they must not be interpreted as membership.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


NORMALIZED_NAME = "holdout_references_v1.npz"
GC_ARCHIVE_ALIASES = {
    "ESO452_SC11": "ESO452-11", "ESO280_SC06": "ESO280-06",
    "Djorg_2_ESO456": "Djorg_2_ESO456-",
}
SOURCES = {
    "vasiliev_baumgardt_2021_clusters.txt":
        "https://raw.githubusercontent.com/GalacticDynamics-Oxford/GaiaTools/master/input.txt",
    "vasiliev_baumgardt_2021_members.zip":
        "https://zenodo.org/records/4891252/files/clusters.zip",
    "hunt_reffert_2024_clusters_extended.tsv":
        "https://vizier.idia.ac.za/viz-bin/asu-tsv?-source=J/A%2BA/686/A42/clusters&-out=ID,Name,Type,RA_ICRS,DE_ICRS,r50,rt,rtot,rJ&-out.max=unlimited",
    "hunt_reffert_2024_members.tsv":
        "https://vizier.idia.ac.za/viz-bin/asu-tsv?-source=J/A%2BA/686/A42/members&-out=ID,GaiaDR3,Prob&-out.max=unlimited",
    "elbadry_2021_hwang_2022_pairs.tsv":
        "https://vizier.idia.ac.za/viz-bin/asu-tsv?-source=J/MNRAS/512/3383/table2&-out=GaiaEDR3-1,GaiaEDR3-2,R&-out.max=unlimited",
}


def _tsv_rows(path):
    """Read the single-table VizieR TSV format without converting IDs to float."""
    with open(path, encoding="utf-8") as stream:
        # A units row can consist entirely of tabs/spaces for dimensionless
        # columns; preserve it while skipping truly empty delimiter-free lines.
        rows = csv.reader((s for s in stream if (s.strip() or "\t" in s) and not s.startswith("#")), delimiter="\t")
        names = [s.strip() for s in next(rows)]
        next(rows)  # units
        separator = next(rows)
        if not all(set(s.strip()) <= {"-"} for s in separator):
            raise ValueError(f"Unexpected VizieR header in {path}")
        for row in rows:
            if len(row) != len(names):
                raise ValueError(f"Malformed VizieR row in {path}")
            yield dict(zip(names, (s.strip() for s in row)))


def _digest(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_member_groups(source_ids, groups):
    order = np.lexsort((groups, source_ids))
    ids = np.asarray(source_ids, dtype=np.int64)[order]
    groups = np.asarray(groups, dtype=np.int64)[order]
    unique = np.r_[True, ids[1:] != ids[:-1]]
    return ids[unique], groups[unique]


def prepare_reference_catalogs(reference_dir):
    """Normalize complete published references; fail closed on truncation."""
    root = Path(reference_dir)
    provenance = {}
    for name, url in SOURCES.items():
        path = root / name
        provenance[name] = {"url": url, "sha256": _digest(path), "bytes": path.stat().st_size,
                            "download_file_mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()}
    globular_zip = root / "vasiliev_baumgardt_2021_members.zip"
    if _digest(globular_zip, "md5") != "3a07093be9a1fcbfb0e3c545ca175c99":
        raise ValueError("Globular membership archive differs from Zenodo's published MD5")

    clusters = list(_tsv_rows(root / "hunt_reffert_2024_clusters_extended.tsv"))
    if len(clusters) != 7167:
        raise ValueError("Incomplete Hunt/Reffert cluster catalog")
    accepted = {int(row["ID"]) for row in clusters if row["Type"] != "r"}
    regions = []
    group_names = {}
    for row in clusters:
        # Published IDs include zero; reserve zero exclusively for unmatched.
        group = int(row["ID"]) + 1
        group_names[group] = {"name": row["Name"], "catalog": "HuntReffert2024", "catalog_id": int(row["ID"]), "type": row["Type"]}
        # Reserve all listed members, including tidal tails. Their total radius
        # rtot is unsuitable as a circular background guard (Hyades: 49 deg).
        # Use the larger published King/Jacobi radius for bound o/g groups.
        if row["Type"] in {"o", "g"}:
            physical = [float(row[key]) for key in ("rt", "rJ") if row[key]]
            physical = [value for value in physical if np.isfinite(value) and value > 0]
            if not physical or max(physical) > 180:
                raise ValueError(f"Invalid cluster radius: {row}")
            radius = max(physical)
            regions.append((float(row["RA_ICRS"]), float(row["DE_ICRS"]), radius, group))

    member_ids, member_groups = [], []
    hunt_rows = 0
    for row in _tsv_rows(root / "hunt_reffert_2024_members.tsv"):
        hunt_rows += 1
        catalog_id = int(row["ID"])
        if catalog_id in accepted:
            # The published list itself defines the conservative reservation;
            # avoid adding a probability cut that could leak its lower-P tail.
            member_ids.append(int(row["GaiaDR3"]))
            member_groups.append(catalog_id + 1)
    if hunt_rows != 1291929:
        raise ValueError(f"Incomplete Hunt/Reffert members: {hunt_rows}")
    hunt_reserved = len(member_ids)

    gc = np.genfromtxt(root / "vasiliev_baumgardt_2021_clusters.txt", names=True, dtype=None, encoding="utf-8")
    if len(gc) != 170:
        raise ValueError("Incomplete Vasiliev/Baumgardt cluster catalog")
    gc_rows = gc_reserved = 0
    with zipfile.ZipFile(globular_zip) as archive:
        expected_names = {f"catalogues/{GC_ARCHIVE_ALIASES.get(str(row['Name']), str(row['Name']))}.txt" for row in gc}
        actual_names = {name for name in archive.namelist() if name.startswith("catalogues/") and name.endswith(".txt")}
        if expected_names != actual_names:
            raise ValueError("Globular cluster centers and membership archive do not map one-to-one")
        for i, row in enumerate(gc):
            name, group = str(row["Name"]), 100000 + i
            group_names[group] = {"name": name, "catalog": "VasilievBaumgardt2021", "type": "g"}
            regions.append((float(row["ra"]), float(row["dec"]), float(row["rmax"]) / 60, group))
            archive_name = GC_ARCHIVE_ALIASES.get(name, name)
            with archive.open(f"catalogues/{archive_name}.txt") as stream:
                for line in io.TextIOWrapper(stream):
                    if not line.strip() or line.startswith("#"):
                        continue
                    cols = line.split()
                    gc_rows += 1
                    # The archive includes foreground/background stars too.
                    if float(cols[-1]) >= 0.5:
                        member_ids.append(int(cols[0]))
                        member_groups.append(group)
                        gc_reserved += 1
    member_ids, member_groups = _unique_member_groups(member_ids, member_groups)
    if np.any(member_groups <= 0):
        raise ValueError("Cluster group IDs must not collide with zero/unmatched")

    first, second, probability = [], [], []
    pair_rows = 0
    for row in _tsv_rows(root / "elbadry_2021_hwang_2022_pairs.tsv"):
        pair_rows += 1
        chance = float(row["R"])
        if np.isfinite(chance) and chance < 0.1:
            first.append(int(row["GaiaEDR3-1"]))
            second.append(int(row["GaiaEDR3-2"]))
            probability.append(chance)
    if pair_rows != 1817594:
        raise ValueError(f"Incomplete wide-binary catalog: {pair_rows}")
    first, second = np.asarray(first, dtype=np.int64), np.asarray(second, dtype=np.int64)
    endpoints = np.unique(np.concatenate((first, second)))
    # Connected components also handle unexpected chains/duplicates. Every
    # endpoint is external regardless of its assigned component label.
    edges = coo_matrix((np.ones(len(first), dtype=np.int8),
                        (np.searchsorted(endpoints, first), np.searchsorted(endpoints, second))),
                       shape=(len(endpoints), len(endpoints))).tocsr()
    components, labels = connected_components(edges, directed=False)
    group_min = np.full(components, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(group_min, labels, endpoints)
    binary_groups = group_min[labels]
    regions = np.array(regions, dtype=[("ra", "f8"), ("dec", "f8"), ("radius", "f8"), ("group", "i8")])
    if np.any(member_ids <= 0) or np.any(endpoints <= 0):
        raise ValueError("Reference catalog contains an invalid Gaia source_id")
    target = root / NORMALIZED_NAME
    np.savez_compressed(target, member_source_id=member_ids, member_group=member_groups,
                        binary_source_id=endpoints, binary_group=binary_groups,
                        region_ra=regions["ra"], region_dec=regions["dec"],
                        region_radius=regions["radius"], region_group=regions["group"])
    np.savez_compressed(root / "wide_binary_pairs_v1.npz", source_id1=first, source_id2=second,
                        chance_alignment=np.asarray(probability), group=group_min[labels[np.searchsorted(endpoints, first)]])
    summary = {
        "prepared_utc": datetime.now(timezone.utc).isoformat(), "schema_version": 1,
        "normalized_sha256": _digest(target), "sources": provenance,
        "counts": {"hunt_clusters": len(clusters), "hunt_members_rows": hunt_rows,
                   "hunt_reserved_members": hunt_reserved, "globular_clusters": len(gc),
                   "globular_catalog_rows": gc_rows, "globular_probable_members": gc_reserved,
                   "unique_cluster_members": len(member_ids), "cluster_regions": len(regions),
                   "wide_binary_candidate_pairs": pair_rows, "wide_binary_reserved_pairs": len(first),
                   "wide_binary_unique_endpoints": len(endpoints), "wide_binary_components": components},
        "policy": {
            "members": "All Hunt/Reffert members of non-rejected o/m/g/d groups; Vasiliev/Baumgardt memberprob>=0.5.",
            "regions": "Hunt/Reffert Type o/g: max(rt,rJ) degrees among finite positive published King/Jacobi radii. Vasiliev/Baumgardt published query rmax arcmin /60. No arbitrary radius cap. Region flag is background quarantine, not membership.",
            "region_limitations": "Full rtot encloses sparse tails in enormous discs (Hyades49deg, Coma36deg); it is not used as background guard. All catalogued members including tails remain reserved by ID. Uncatalogued tail members and missing clusters are not guaranteed excluded.",
            "binary": "El-Badry2021 endpoints and R reproduced in Hwang2022 full 1817594-row table; reserve both endpoints for finite R<0.1; components share minimum endpoint source_id.",
            "distances": "No reference distances used. Gaia-selected membership/near-equal binary distances do not give an independent Gaia absolute calibration.",
            "split": "Every reference match is external, taking precedence over any spatial or random split.",
            "epoch": "Gaia membership IDs are EDR3/DR3; centers are published coordinates. Region radii are broad group radii, not arcsecond crossmatch gates.",
        },
    }
    (root / "holdout_reference_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    (root / "holdout_group_names.json").write_text(json.dumps(group_names, indent=2) + "\n")
    return summary


@dataclass
class HoldoutReferences:
    member_source_id: np.ndarray
    member_group: np.ndarray
    binary_source_id: np.ndarray
    binary_group: np.ndarray
    region_ra: np.ndarray
    region_dec: np.ndarray
    region_radius: np.ndarray
    region_group: np.ndarray


def load_reference_catalogs(reference_dir):
    """Load normalized references, failing if preparation has not completed."""
    with np.load(Path(reference_dir) / NORMALIZED_NAME, allow_pickle=False) as data:
        refs = HoldoutReferences(**{name: data[name] for name in HoldoutReferences.__dataclass_fields__})
    for ids in (refs.member_source_id, refs.binary_source_id):
        if np.any(ids <= 0) or np.any(ids[1:] <= ids[:-1]):
            raise ValueError("Reference Gaia source IDs must be sorted, unique, positive integers")
    for ids, groups in ((refs.member_source_id, refs.member_group),
                        (refs.binary_source_id, refs.binary_group)):
        if ids.shape != groups.shape or np.any(groups <= 0):
            raise ValueError("Every reference source must have a positive group ID")
    return refs


def _id_group(query, ids, groups):
    result = np.zeros(len(query), dtype=np.int64)
    if len(ids):
        indices = np.searchsorted(ids, query)
        indices = np.minimum(indices, len(ids) - 1)
        good = (query > 0) & (ids[indices] == query)
        result[good] = groups[indices[good]]
    return result


def _unit_vectors(ra, dec):
    ra, dec = np.deg2rad(ra), np.deg2rad(dec)
    return np.column_stack((np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)))


def annotate_holdouts(refs, source_id, ra, dec):
    """Return independent masks plus compact IDs; zero group means unmatched.

    ``reference_kind`` is 1=published cluster member, 2=angular region only,
    3=wide binary, 0=unmatched. Distinct masks retain overlaps. Region IDs select
    the smallest containing published region; this is not inferred membership.
    Input source IDs must be integer typed: float64 silently loses Gaia ID bits.
    """
    source_id, ra, dec = np.asarray(source_id), np.asarray(ra), np.asarray(dec)
    if source_id.dtype.kind not in "iu":
        raise TypeError("Gaia source IDs must be integer typed, never float64")
    if not (source_id.ndim == ra.ndim == dec.ndim == 1 and len(source_id) == len(ra) == len(dec)):
        raise ValueError("source_id, ra, dec must be aligned one-dimensional arrays")
    member_group = _id_group(source_id, refs.member_source_id, refs.member_group)
    binary_group = _id_group(source_id, refs.binary_source_id, refs.binary_group)
    region_group = np.zeros(len(source_id), dtype=np.int64)
    good = np.isfinite(ra) & np.isfinite(dec) & (np.abs(dec) <= 90)
    good_indices = np.flatnonzero(good)
    if len(good_indices):
        tree = cKDTree(_unit_vectors(ra[good], dec[good]))
        # Descending radii mean narrower regions deterministically overwrite
        # broader overlapping regions. All these objects remain external.
        for i in np.argsort(-refs.region_radius, kind="stable"):
            chord = 2 * np.sin(np.deg2rad(refs.region_radius[i]) / 2)
            indices = tree.query_ball_point(_unit_vectors([refs.region_ra[i]], [refs.region_dec[i]])[0], chord)
            region_group[good_indices[indices]] = refs.region_group[i]
    member, region, binary = member_group > 0, region_group > 0, binary_group > 0
    group = np.where(member, member_group, np.where(region, region_group, binary_group))
    kind = np.where(member, 1, np.where(region, 2, np.where(binary, 3, 0))).astype(np.int8)
    return {"external_cluster_member": member, "cluster_region": region, "widebinary": binary,
            "external_any": member | region | binary, "cluster_member_group": member_group,
            "cluster_region_group": region_group, "widebinary_group": binary_group,
            "reference_group": group, "reference_kind": kind}


def validate_reference_catalogs(reference_dir):
    """Check every retained pair/member and the full set of angular centers."""
    root = Path(reference_dir)
    refs = load_reference_catalogs(root)
    with np.load(root / "wide_binary_pairs_v1.npz", allow_pickle=False) as pairs:
        first = _id_group(pairs["source_id1"], refs.binary_source_id, refs.binary_group)
        second = _id_group(pairs["source_id2"], refs.binary_source_id, refs.binary_group)
        if not (np.all(first > 0) and np.array_equal(first, second) and np.array_equal(first, pairs["group"])):
            raise ValueError("A wide binary endpoint is unreserved or assigned another component")
        pair_count = len(first)
    centers = annotate_holdouts(refs, np.zeros(len(refs.region_ra), dtype=np.int64), refs.region_ra, refs.region_dec)
    if not np.all(centers["cluster_region"]):
        raise ValueError("A published cluster center is outside the quarantined regions")
    if not np.all(_id_group(refs.member_source_id, refs.member_source_id, refs.member_group) > 0):
        raise ValueError("A published cluster member is unreserved")
    summary = {"all_pairs_same_nonzero_group": True, "all_cluster_members_reserved": True,
               "all_cluster_centers_quarantined": True, "pair_count": pair_count,
               "region_count": len(refs.region_ra), "cluster_member_count": len(refs.member_source_id),
               "normalized_sha256": _digest(root / NORMALIZED_NAME)}
    (root / "holdout_validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "validate"])
    parser.add_argument("--reference-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        summary = prepare_reference_catalogs(args.reference_dir)["counts"]
    else:
        summary = validate_reference_catalogs(args.reference_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
