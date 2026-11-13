# Complete radial source/field uncertainty summaries

This reporting supplement implements the source/field clustered uncertainty
already required by `v3_1_radial_protocol.md`. It changes no generating design,
fit, numerical schedule, acceptance gate or saved outcome. All176catalogs,
352arms and22scenarios are required before execution. The existing pilot and
partial outcomes have already been inspected; this is not a new blind design.

Use the authenticated complete `radial_report`, full native-PSF photometry,
truth template identities and catalog field assignments. Report each scenario's
eight paired catalogs, never treat18class-distance bins or128simulated stars
as independent recovery experiments. Preserve all outcomes and separately
show accepted-only denominators; paired contrasts require both arms accepted.

Primary error contrast is the existing local-minus-broad fraction L1. For each
class-distance bin retain mean-minus-truth and95%interval coverage, separately
for selected realized counts and pre-entry population fractions. Structural
zero truth is included in the all-bin report and explicitly excluded only in
the separately named nonzero-truth report. Empty groups stay empty. Local
location/width summaries exist only where the generating design defines truth.

Compute equal-catalog means and2048whole-block bootstrap draws(seed321201)
under three explicit partitions: recorded field; connected components of
catalogs sharing an actually used template identity; and the transitive union
of source overlap or field sharing. Compute partitions on the complete fixed
eight-catalog design, before the numerical-acceptance filter. In each bootstrap
draw take complete blocks with all their catalogs and preserve their original
equal-catalog weighting; a singleton block has no interval, not a zero-width
confidence interval. No empirical cluster interval is reported for empty data.

These are conditional repeated-design intervals and dependency diagnostics,
not uncertainty over independently regenerated teacher/source libraries.
Disconnected used-template components do not prove independent reference
scales; they still share their empirical bank, model and selection assumptions.
Keep field-only intervals explicitly conditional on those shared ingredients,
and the joint partition separately visible even when it collapses to one block.
Never claim measured independent library uncertainty from these resamples.
The original entry-table omission of local dust covariance and its direct
audit limitation remain; this supplement does not repair or hide that mismatch.

Tables and future figures retain all14oldscenarios and eight new position/width
designs. There is no score-selected scenario exclusion, retuning, physical
density inference or real-population calibration claim.
