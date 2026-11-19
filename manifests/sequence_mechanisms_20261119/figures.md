# v7 science figure index

All plates use the shared Pal5/v3.8 PAPER settings, colors and exports. PNG300dpi,
single-page PDF and companion CSV are provided under
[`figures/sequence_mechanisms_20261119`](../../figures/sequence_mechanisms_20261119).
All plot text is English. Existing sky figures are unchanged.

| Plate | Scientific reading |
|---|---|
| [Controlled scales and paired scores](../../figures/sequence_mechanisms_20261119/controlled_scales_scores.png) | Top: same v5 five-control common members, raw lower-MS/TO scales, shared 95% CIs and the ±0.10 mag target. Bottom: original primary, adjacent pairs on common support, marginal log-score gain and 95% CIs. N is in the tables. |
| [Native loci and branch mass](../../figures/sequence_mechanisms_20261119/native_loci_branch_mass.png) | All parent native CMD points against A/B/C/E curves; reference luminosity is diagnostic only. E post-MS probability versus observed color, with open points outside E support. |
| [QRF complete-PDF cases](../../figures/sequence_mechanisms_20261119/QRF_complete_PDF.png) | Three worst exact score declines under diagnostic centering and one near-median change, selected on original M13 lower-MS members. Original and shifted PDFs; separate full-CDF panels retain their tails. |
| [Sequence complete-PDF cases](../../figures/sequence_mechanisms_20261119/sequence_complete_PDF.png) | M5 lower-MS and all three systems' TO representatives nearest their phase-median A residual. The same photometry passes through all five controls. |
| [Branch complete-PDF cases](../../figures/sequence_mechanisms_20261119/branch_complete_PDF.png) | Largest D→E post-MS-mass change in each system, on common support. D/E and frozen QRF; complete broad QRF support remains in separate CDF panels. These cases explain potential branch mechanisms, not typical performance. |

In PDF plots, the gray Gaussian reference profile has its height scaled to 80%
of the largest prediction peak for display; its position and width are unchanged.
Prediction density ordinates remain normalized per magnitude. Scores use the
normalized reference kernel, not the displayed height. Dotted curves are
likelihood diagnostics outside the respective color-support gate. Main PDF
windows resolve central quantiles; separate full-CDF panels and supplied parameters retain
all tails. Branch main windows use D/E, while their full CDF panels include full QRF.

The first exports with the unscaled reference height remain in the private
execution archive; final presentation prevents the narrow reference from
flattening the prediction curves. Geometry, fonts, palette and scientific arrays
are unchanged. The actual source IDs and fixed selection rules are saved in
`examples/selected.csv` and `data/cases.json`.

The figures illustrate the aggregate results in `evaluate/summary.csv`,
`paired.csv` and `phase_gaps.csv`; examples do not replace whole-sample inference.
Every A–E age/FeH choice is conditional on the known system, rather than a new
field-star production model. Portable pixel and PDF checks are in `delivery/`.
