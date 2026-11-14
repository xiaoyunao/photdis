# Potential fresh field references for v3.2

The v3.1 field roles (170 new-sky and 356 new-source/seen-sky objects)
and common-scale role (3032 objects in one system) were already opened.
The 3250 v3.0 identities were also opened and yielded no qualifying
distance evidence under that protocol. Preserve each role's identity
and footprint exclusions. See the v3.1 validation role receipt and the
v3.0 reservation availability report.

Start with the SDSS DR19 Astra 0.6.0 APOGEE AstroNNDist and ASPCAP
summary products. The archive responded with HTTP 200 and lengths
534769970 and 1171102556 bytes on 2026-11-14. Download and verify gzip
integrity on NERSC. Use the APOGEE extension and join the summaries on
consistent positive Gaia DR3 and SDSS IDs. Exclude previously exposed
IDs, historical imaging IDs, used sky blocks and systems, and require
native DECam DR11 TYPE=PSF under the locked photometric selection.
This acquisition alone assigns no final role and reads no outcomes.

The archive includes combined SDSS-V spectra as well as legacy DR17
spectra. Exclude DR17 APOGEE IDs and Gaia identities from the proposed
fresh DR19-only channel and require the APO 2.5m SDSS-V 1.3
reduction and at least one good visit in both products.
The archive is published in DR19. The source-row release column uses
`sdss5` for SDSS-V provenance and `dr17` for the legacy spectra.
The first identity census used the archive's release name as a row
value and returned zero requests. Its unchanged receipt records the
actual provenance labels. The corrected census restores the intended
SDSS-V filter with all other eligibility rules preserved.
[SDSS DR19 infrared data](https://www.sdss.org/dr19/data_access/get_data/)
Use the Astra ASPCAP flags for independent stellar-parameter eligibility.
AstroNNDist uses spectra, 2MASS Ks and extinction and was trained using
DR17 plus Gaia EDR3. Use it as a teacher-distance channel. Keep
signed Gaia, whole-system distances and candidate-parent contamination
as separately qualified channels. [Astra summary schemas](https://www.sdss.org/dr19/mwm/astra/pipelines-in-astra/) · [AstroNNDist method](https://www.sdss.org/dr19/mwm/astra/pipelines-in-astra/astronn/)

DR20 copies the DR19 APOGEE spectra and analysis. Treat duplicate releases
as the same source lineage. The NERSC public DESI DR2 tree contains a
placeholder README and its `vac/dr2` directory had no listed files at this
check. DESI DR1 SpecDis and SPDist were used historically. A new sky and
identity audit is required for any additional DR1 reference use. The BHB
VAC has spectroscopic classification and a photometric absolute-magnitude
relation. Any distant BHB use would be a separate tracer channel.
[SDSS DR20 release](https://sdss.org/dr20/) · [DESI DR1 SpecDis](https://data.desi.lbl.gov/doc/releases/dr1/vac/mws-specdis/) · [DESI DR1 BHB](https://data.desi.lbl.gov/doc/releases/dr1/vac/mws-bhb/)

The index reads only provenance, positions and identity columns.
The allocated native matching job applies exact G3, primary, release,
morphology, historical imaging identity and photometric criteria. Its
reference index contains row pointers paired with the same PSF identities.
Archive checksums and both files' explicit column selection precede any
reference-value access.

Acquire and index sources without model residuals. Freeze the candidate,
reference eligibility, outcome-access rule and actual source roles before
evaluating any new validation outcomes.

## Outcome access and teacher channel

The freeze requires completed opened-domain evaluation and a forecast
made from the admitted photometry for candidate, frozen grz QRF, nominal
reference and old target. Preserve all object identities and numerical
statuses at every forecast view. Bind the scoring program, admission,
both archives, metadata index and source recipes to the same receipt.
Validate available histograms, both continuous grids, class-distance
normalization and ordered quantiles before writing the freeze receipt.
The initial evidence access records the receipt and Gaia file inventory.
An interrupted join can resume against the same receipt and files.

Take the AstroNNDist `nn_dist` channel, whose uncertainties combine the
reported predictive and model components in quadrature. Require positive
finite distance and predictive error, nonnegative finite model error
and no distance warnings or bad flags. Precision is relative uncertainty
below 20%. The ASPCAP cool-star labels require S/N at least 50, no bad
fit flag and a one-standard-error gravity bound above 4.2 for dwarfs
or below 3.2 for giants. Report [Fe/H] eligibility separately using the
abundance flag and error. Keep uncertain classifications as unknown.
Report counts in 0–2, 2–5, 5–10, 10–20, 20–50 and beyond 50 kpc, with
giants and qualified [Fe/H] below −1.5 tracked explicitly.
[Astra distance columns](https://data.sdss.org/datamodel/files/MWM_ASTRA/V_ASTRA/summary/astraAllStarAstroNNdist.html) · [ASPCAP flags](https://www.sdss.org/dr19/mwm/astra/pipelines-in-astra/aspcap/)

The scorer retains each view's full continuous density or native QRF
histogram. Compare signed Gaia proper scores and PIT, teacher scores and
interval agreement, and class probability where available. Retain raw and
numerically accepted comparisons. Report near-probability error, Brier,
and the high-q selection in each teacher-qualified distance group. The
teacher-far fraction among high-q objects has a view-specific denominator
and describes the selected reference sample. Scientific application counts
and independent parent-population contamination require separate evidence.

The comparator job uses fixed nominal SED under the reference prior,
response12/calibration12 with the old shared target chain and transfer
prior, and the frozen grz QRF with Sobol128. Historical training overlap
in QRF is retained as unavailable; QRF extrapolation status is explicit.
The joint views keep two native distance grids and class-distance tables.
Combine these files with the chosen candidate forecast under the same
admitted photometry hash and completed opened-development decision.

Before opening any outcomes, record the historical candidate gate on
each native-admitted object using observed r < 20, maximum grz magnitude
uncertainty < 0.1 and the frozen QRF support. Authenticate the exact
identity-aligned mask and its count with the four-view forecasts.
Report both the full native cohort and the matched candidate gate on
the same outcome qualifications. Export gate membership per object.
Targeted spectroscopy has unknown inclusion probabilities within the
candidate parent. Its high-q teacher-far fraction applies to the
qualified DR19 reference subset.

After assembling the complete outcome-closed forecast, submit
`sbatch scripts/run_v3_2_reference_final.sh PATH_TO_EVALUATED_DECISION`
from the canonical checkout. The allocated job verifies and writes the
freeze, opens the bound evidence, then runs the registered scorer.
In each admission scope, retain low/high signed Gaia PIT tail rates,
qnear and high-q fractions, and the reference-qualified distance
strata through and beyond 50 kpc.

Use ten equal-width bins for class and near-probability reliability,
with occupancy and sky-block intervals on the observed fraction.
Sum predicted probability and reference indicators on the same eligible
rows for each view, separately for dwarf, giant and near. Report their
count difference and the sky-block interval for the mean per-object
difference. These are counts of agreement with available reference
labels. Joint predictive count intervals require the shared-state
forecasts and are evaluated separately in the injection experiments.
For class confusion, report reliable dwarf and giant labels against
all three posterior argmax classes. Include missing forecasts and
numerical exclusions in the reference-row denominator ledger.
The high-q audit counts every selected reference object, including
missing or numerically flagged distances. It reports available and
precise teacher labels separately, with the missing-label bounds under
each qualification, and retains the model's expected far count in its
own column. These bounds apply to the targeted reference cohort and
hold each qualified teacher center fixed.

## Common-system forecast controls

The common-system native census admits 77 members of 47 Tuc and 26
members of NGC362. Job 61264223 runs the nominal, old-target and QRF
forecasts on these exact native PSFs with `--common`. Use the same photometric
candidate gate and predictor definitions as the field comparison.
The published member IDs determine the validation sample. The
predictors receive the photometry and the sky transfer treatment.
Use `--common` in the forecast assembly to bind those comparator
outputs and the finalized common-system candidate. Freeze the assembled
predictions before evaluating the HST scales. The final job
`run_v3_2_common_scale_final.sh` accepts the completed mechanism
decision and candidate forecast receipt. Each HST scale uses 4,096
positive draws, scaling normal offsets by the published minus/plus
uncertainties. Every member in a system shares the same distance in
each draw. Report median bias, severe error, and 68% and 95% interval
agreement. Candidate-minus-control contrasts use the same members and
scale draws. The intervals quantify reference-scale uncertainty for
these members.
