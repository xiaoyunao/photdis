# Photdis v2 science and figures

Reference and PDF baseline `87069408`. The exact v1 benchmark from
`1a69ff82` is reproduced without re-estimating any distance.
The 28,917 archive parts reconcile to 72,289,386 owned, native DR11 PSFs,
including 43,347,612 with complete PDFs. All individual source records and
posteriors stay under `data/paper_v2` on NERSC.

## Reference coverage and performance

Each r bin uses observed magnitude. Each reference entry is
**qualified / retained for the new role-4 holdout** within the full-PDF
denominator. The reference catalogues can share stars and external scale.

| Observed (r) | Full PDFs | Gaia geometry | SpecDis | SPDist |
|---|---:|---:|---:|---:|
| All | 43,347,612 | 3,056,182 / 605,119 | 804,896 / 143,506 | 681,281 / 122,191 |
| 20–21 | 8,484,780 | 7,226 / 1,471 | 8,818 / 1,887 | 4,688 / 1,091 |
| 21–22 | 10,238,967 | 1 / 0 | 70 / 8 | 74 / 15 |
| 22–23 | 8,191,551 | 0 / 0 | 0 / 0 | 0 / 0 |

The original owned parent at 22–23 has 212 qualified SpecDis and 2 SPDist
matches. None have a complete PDF. At 21–22, there are 80 SpecDis and
1,867 SPDist recorded matches in the PDF sample before the reference
quality cuts. Direct validation and shared observable support are separate.
Replaying the v1 `>=20 F15 pairs/cell` rule yields 806,611 PDF sources in
common cells.

| Reference | All qualified PDF NMAD | Held-out PDF NMAD | 20–21 held-out NMAD |
|---|---:|---:|---:|
| Gaia geometry | 14.67% | 14.65% | 42.87% |
| SpecDis | 13.91% | 13.80% | 35.22% |
| SPDist | 14.53% | 14.27% | 28.02% |

The 20–21 aggregates condition on available qualified reference stars.
F15 uses its exact earlier same-star pairs and retains the 9.77%/10.38%
Photdis NMAD on 9,355/8,927 stars. A broader comparison in the nominal
18–20 mag, weak-Gaia, spectroscopic dwarf, reference 2–5 kpc window now
has 13,986/13,456 role-4 stars with 10.52%/11.34% NMAD. Differences in
population, geometry and the requirement for valid BJ in the exact F15
frame affect the comparison. Benchmark rows are disjoint from fit IDs by
the same v1 identity checks.

There are 17,004,196 signed-parallax predictions, including 2,719,624
negative corrected parallaxes. 17,003,485 have a numerically usable
conditional CDF. At 21–22 there are 291,076 signed measurements.
Their conditioning and the S/N ≥ 10 geometry selection are reported
separately.

- [Complete cohort definitions](reference/README.md)
- [Native coverage](reference/coverage.csv)
- [Distance metrics, including population, color and radial strata](reference/distance_metrics.csv)
- [Training overlaps](reference/train_test_overlap.csv)
- [Same-star reference intersections](reference/common_references.csv)
- [Frozen F15 replay](reference/benchmark_replay.csv)
- [Signed Gaia diagnostics](reference/signed_gaia.csv)

## Projected sky densities and structure controls

| Distance layer | Number by median distance | Integrated PDF mass |
|---|---:|---:|
| 0–1 kpc | 6,237,172 | 6,697,962.35 |
| 1–10 kpc | 25,229,239 | 24,976,430.15 |
| ≥10 kpc | 11,881,201 | 11,673,219.50 |
| Total | 43,347,612 | 43,347,612.00 |

Galactic Aitoff panels are densities on the original HEALPix support. The
independent native randoms give 9,686.99 effective deg². The displayed mask
retains 9,659.35 deg² at the fixed 10% fractional-area threshold. The hard
map uses sources per effective deg². The separate soft map sums the full
PDF mass within each shell per effective deg². White indicates insufficient
effective coverage. The area and source accounting include both native
systems and combine the overlapping bright/faint south area once.

The prescribed minimum matched-filter pilot did not pass. The true-sky
controls and pre-registered decision are in the [filter record](filter/README.md).
The Aitoff figure contains the two direct density panels. The ≥10 kpc
layer, area masks and native region totals are in the [distance-layer table](spatial/distance_layers.csv).

## Luminosity-class migration

In the role-4 2–5 kpc reference sample, among stars in the applicable working
selection domain, 15,865/23,397 ordinary SpecDis dwarfs pass (67.8%) and
230/2,511 metal-poor dwarfs pass (9.2%). For SPDist the corresponding counts
are 14,624/21,377 (68.4%) and 185/2,281 (8.1%). These rates use this expanded
holdout and the available working selection. Among supported 5–10 kpc
intermediate-gravity sources, 983/1,103 (89.1%) and 988/1,074 (92.0%) go
below 5 kpc in median distance. The table records separately objects without
PDF support, objects outside the applicable working-selection domain,
assessed rejections and predicted destinations.

For each native region, observed r, dereddened gr and rz, extinction and
maximum grz error cell with ≥5 of each dwarf group, subtract the ordinary
dwarf median Δμ from the metal-poor dwarf median Δμ. Weight each difference
by the smaller cohort size. The mean within-cell offsets are +0.377 mag
(50 cells, sum of weights 3,474) for SpecDis and +0.382 mag (49, 3,223) for
SPDist at reference 2–5 kpc. At 5–10 kpc they are +0.203 mag (25, 2,384) and
+0.207 mag (23, 1,543). Positive differences mean the metal-poor distances
are systematically larger relative to their own reference. These estimates
condition on matched observable support and the spectroscopic scale.

The archived strict M30 transfer recovers M68 within 0.8% of its reference
distance and puts M5 11.9% below its reference. The archived GD-1 plate uses
the signed Hess residual against the offset ribbons and 259 spectroscopic
members with distances from the external track. The full stream-distance
gradient remains outside the supported scientific claims. Spectroscopic
selection and the existing LF scenarios do not identify a catalogue-wide
giant fraction. The original science-core rare-class counts, including the
single held-out giant, remain in the [v1 conditional table](../paper_v1_evidence/benchmark/conditional_branch.csv).

- [Migration counts](figures/population_distance_migration.csv)
- [Matched photometry cells](figures/matched_physical_conditions.csv)
- [M30 cluster transfer](spatial/cluster_template_transfer.csv)
- [GD-1 same-star distance residuals](spatial/gd1_hess_member_residuals.csv)

## Figures and reproducibility

All eight plates have PDF, 300 dpi PNG and a numerical sidecar. See the
[figure index](../../figures/paper_v2/README.md) and the
[AAS one-/two-column preview](../../figures/paper_v2/aas_width_preview.png).
Small receipt-bound tables are in this directory. Full join rows, pixel
maps and the 48 scan receipts are archived on NERSC under
`/pscratch/sd/y/yunao/photdis_pipeline/data/paper_v2`. The original release
and original v1 evidence are unchanged. The [delivery receipt](delivery.json)
links the stage hashes.

The Slurm entry point is `scripts/run_paper_v2.sh`. Its `scan` stage accepts
the task index from `SLURM_ARRAY_TASK_ID` over 0–47. Once all scans complete,
run `reference`. The `figures` and `spatial` stages consume that receipt.
The `filter` stage consumes the frozen control design and existing native
structure archives. `verify` checks these four stage receipts, the F15
replay and data closure. The earlier production directories are input only.
