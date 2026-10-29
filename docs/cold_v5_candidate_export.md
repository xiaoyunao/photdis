# Frozen-v4 candidate route export

The three registered 10°×10° final fields were exported on allocated NERSC
CPU nodes with `scripts/cold_v4_candidate_export.py` and the unchanged
`clean_wise_snr_20` policy. The source is each field's existing diagnostic
FITS table. Catalog products remain under ignored NERSC
`data/cold_v4_candidates/`; only their small manifests are tracked here.
This table describes the original v1 files. A versioned v2 export adds
existing spectroscopic class evidence without changing the frozen model,
route, distances or original files.

| Field | Input rows | Exported | A / B / C | Infrared support conflicts |
|---|---:|---:|---:|---:|
| RA0–10, Dec −35 to −25 | 1,109,197 | 197,165 | 132,585 / 35,242 / 29,338 | 23,590 |
| RA150–160, Dec +20 to +30 | 694,673 | 125,598 | 71,472 / 28,991 / 25,135 | 14,547 |
| RA220–230, Dec +20 to +30 | 336,859 | 72,866 | 45,597 / 15,185 / 12,084 | 9,235 |
| **Total** | **2,140,729** | **395,629** | **249,654 / 79,418 / 66,557** | **47,372** |

`tier` is the selected A/B/C level; `reason_b` and `reason_c` distinguish
missing or invalid WISE, low S/N, failed WISE quality and higher-dimensional
support conflict. `support_ratio_a/b/c` is each available feature vector's
tenth-neighbor radius divided by its frozen tier limit. The reported ratio
can be finite for an unavailable tier, so its reason field decides whether
it can be interpreted. `infrared_fallback_conflict` marks reliable infrared
measurements that lie outside the higher-dimensional cold-dwarf support
region. It does not remove a source or give a giant probability.

`mg_median`, `mu_median` and `distance_pc_median` are frozen v4 estimates
**conditional on the cold-dwarf hypothesis**. `used_tier_high_quality`
records the stricter photometric quality status of the chosen tier.
`depth_state` records measured `r<18`, `18≤r<22` or `r≥22`;
the middle range has stress tests with undercovered intervals, while
the last lies beyond those tests. No deep interval or calibrated stellar
type probability is included. The input magnitudes and model route retain
the existing five-band DR11 `MW_TRANSMISSION` correction and low-dust
validation convention.

The output mix is A/B/C = 63.10%/20.07%/16.82%; strict Gaia comparisons
used predominantly C-layer stars (93.5% in RA205 and 90.3% in RA60).
Consequently, their aggregate scatter, interval coverage and XD improvement
do not measure performance for all 395,629 photometric candidates. Candidate
count is a route count, not the number of confirmed cold dwarfs. The 47,372
infrared conflicts (11.97%) are support warnings, not a giant fraction.

The v2 output fields `class_evidence_available`, `class_evidence`,
`class_evidence_source`, `class_evidence_quality` and
`class_conflicts_cold_dwarf` carry existing DESI evidence. A qualified
4200–5000 K RVS star with `logg−σ_logg>4.2` is a cool main-sequence label;
one with `logg+σ_logg<3.2` is a cool-giant conflict. An unambiguous DESI
Redrock QSO/galaxy type with S/N≥10 and the retained quality bits is a
nonstellar conflict. Other qualified stars remain a separate transitional
or other-star category. No matched qualified evidence is `unknown`, which
does not imply dwarf status. These are diagnostic flags, not a fitted
probability or a requirement for applying photometric distances.

The three jobs completed with exit code zero and empty stderr. For each
product, the FITS metadata check found exactly one binary-table extension
and the manifest row count. The same check read its field names and first
and last row only on the login node; the full input scans, predictions and
output SHA256 computations ran on allocated CPU nodes. The initial invalid
seven-extension RA150 product was archived before these valid exports.

The per-field manifests are:

- `manifests/cold_v4_candidates_cold_v4_final_ra000_010_dec_m035_m025.json`
- `manifests/cold_v4_candidates_cold_v4_final_ra150_160_dec_p020_p030.json`
- `manifests/cold_v4_candidates_cold_v4_final_ra220_230_dec_p020_p030.json`
