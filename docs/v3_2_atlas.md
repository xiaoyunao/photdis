# Photdis v3.2 figure atlas

## Signed Gaia scores and HST distance scales

The left and middle panels show same-object score differences on the
opened development field and the frozen DR20 field. Horizontal bars show
95% sky-block intervals. The horizontal axes have different ranges.
The right panel shows median relative residuals for the full native HST
cohorts. Vertical bars propagate shared reference-scale uncertainty.

![Photdis signed Gaia scores and HST distance residuals](../figures/v3_2/external_validation/gaia_and_hst.png)

[PNG](../figures/v3_2/external_validation/gaia_and_hst.png) ·
[PDF](../figures/v3_2/external_validation/gaia_and_hst.pdf) ·
[Source and output hashes](../manifests/v3_2/final_validation_dr20/figure.json) ·
[External results](v3_2_external_result.md)

## DR20 near-distance and giant-class reliability

Points show matched-gate probability bins on the MINESweeper reference
subset. Area increases with the number of objects in the bin. Vertical
bars show available 95% sky-block intervals. QRF supplies a distance
distribution. Its class probabilities are unavailable for this
comparison. The near panel uses 982 qualified distance centers and the
giant panel uses 676 class labels.

![DR20 near-distance and giant-class reliability](../figures/v3_2/external_validation/dr20_reliability.png)

[PNG](../figures/v3_2/external_validation/dr20_reliability.png) ·
[PDF](../figures/v3_2/external_validation/dr20_reliability.pdf) ·
[Source bins](../manifests/v3_2/final_validation_dr20/scores/reliability.csv)

## Opened-sky population and SED tilt contrasts

The three panels show paired log-score changes for the same signed Gaia
rows, the labeled class sample and the giant subset. The intervals resample
whole sky blocks within each comparison.

[PNG](../figures/v3_2/tilt_factorial/opened_factorial.png) ·
[PDF](../figures/v3_2/tilt_factorial/opened_factorial.pdf) ·
[Source table](../manifests/v3_2/tilt_report/initial/factorial.csv) ·
[Results](v3_2_factorial_result.md)

## Opened new-sky class reliability

[PNG](../figures/v3_2/opened_diagnostics/class_reliability.png) ·
[PDF](../figures/v3_2/opened_diagnostics/class_reliability.pdf) ·
[Probability bins](../manifests/v3_2/opened_diagnostics/reliability.csv) ·
[Opened comparison](v3_2_opened_result.md)

## Opened class probe and count audit

Left panel compares the frozen multiband class log score with three
historical anchors on the same 157 labeled objects from eight sky blocks.
The giant subset contains 142 of these objects. Bars show paired 95%
sky-block intervals. Right panel shows signed changes at the largest
count-shift bin within each of 34 correction states. The 33 failed states
are below −0.5 objects. Selection reweighting supplies the full shift
at those bins in this particular broad-envelope entry-level comparison.

![Opened class probe and count transfer sensitivity](../figures/v3_2/opened_audits/probe_and_count_audit.png)

[PNG](../figures/v3_2/opened_audits/probe_and_count_audit.png) ·
[PDF](../figures/v3_2/opened_audits/probe_and_count_audit.pdf) ·
[Plotted records](../manifests/v3_2/opened_audits_figure/plotted.csv) ·
[Source hashes](../manifests/v3_2/opened_audits_figure/complete.json) ·
[Audit result](v3_2_opened_probe_and_count_result.md)

## NGC288 fixed-distance spatial odds quadrature

Absolute changes between successive spatial-odds integrations on the
64-node distance grid. The dashed lines are the registered tolerances.
The first refinement exceeds the membership and mean-count thresholds
for both likelihoods. The second refinement meets all four thresholds
for both likelihoods. Full distance-axis integration remains pending.

![NGC288 fixed-distance odds quadrature](../figures/v3_2/structure_odds_audit/fixed_distance_odds.png)

[PNG](../figures/v3_2/structure_odds_audit/fixed_distance_odds.png) ·
[PDF](../figures/v3_2/structure_odds_audit/fixed_distance_odds.pdf) ·
[Check rows and source hashes](../manifests/v3_2/structure_odds_audit/)

## Narrow-structure mock membership after selection

The four fixed-truth panels contain 256 mock catalogs. The upper-right
panel contains 32 separate prior-predictive catalogs. Each catalog has
128 admitted objects. At a generating fraction of 60%, the median
selected member fraction for empirical SEDs with a deep background is
70.3–70.7% across the two widths, compared with 59.8% with a near
background. The plotted fractions come from the simulated membership
record before fitting.

![Realized member fractions in the 288 frozen mock catalogs](../figures/v3_2/radial_design/selected_membership.png)

[PNG](../figures/v3_2/radial_design/selected_membership.png) ·
[PDF](../figures/v3_2/radial_design/selected_membership.pdf) ·
[Catalog records](../manifests/v3_2/radial_design/design.csv) ·
[Source and output hashes](../manifests/v3_2/radial_design/figure.json)

## Narrow-structure prior shrinkage tables

The recovery report records the distance from each simulated truth to
the value at the center of the latent prior, minus the corresponding
distance from the posterior median. A positive value means the median is
closer to that anchor. Location, width and the pre-entry member fraction
use their own units. Summaries keep the registered catalog denominators
for each fixed-truth design cell and the separate prior-predictive cohort.

In the first empirical pilot, the width median of 0.339 mag is
0.081 mag closer to the 0.250 mag prior anchor than the 0.080 mag
truth is. Its 95% interval still misses the simulated width.
[Pilot recovery records](../manifests/v3_2/radial_recovery/catalog_128/recovery.csv)

The radial report also compares empirical and model SED recovery within
each fixed-truth cell on the same context repeats. It uses each mock's
own realized truth for member counts and joint class-distance counts.
The paired differences describe the change in absolute error, coverage
and interval length under the two generators. Each comparison records
the number of paired fits and the number of registered repeats.
