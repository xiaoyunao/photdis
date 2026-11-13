# v3.1 final external evaluation and outcome freeze

Registered before reading prediction/reference residuals on either frozen new
role. Availability and label/precision metadata were used at role acquisition;
this is not label-blind acquisition. No new source, sky or quality selection may
be chosen from final residuals. The frozen roles remain526field and3032common
native DR11 PSF objects, including objects without qualifying evidence.

## Required prediction identities and closure

Require full field AND common inventories for the four reference-prior controls
(nominal, response12, corrected_point, corrected_bootstrap), original response12
with old target population/SED tilts, and the four final candidate-target views
(zero, point, bootstrap_evidence, bootstrap_cut). Main candidate is the3540cohort
response12/calibration12 beta=0 bootstrap_evidence population, exactly as in
`v3_1_target_ensemble_protocol.md`. No main identity is selected by final scores.
All34conditional states must exist; missing/failed states cannot be renormalized
away. Optional broad-cohort transfer is a separately named secondary experiment.
Primary validation also requires the identical3540candidate photometric admission
specified in `v3_1_validation_admission_protocol.md`. All frozen objects remain
predicted/scored; outside-envelope summaries are explicitly diagnostic. Final
highq comparisons require this same admission before the qnear>.95 cut.

`v3_1_final_freeze.py` authenticates whole inventories, all stored attempts,
selected PDFs, shared populations, roles/reference pointers, method protocols
and scoring/association code before creating a write-once freeze. It refuses a
pilot, missing identity, mismatched hash, altered PSF status or incomplete code.
Numerical flags remain frozen and exported; a terminal flag does not delete its
object. Scientific pass/fail is not a prerequisite for recording a complete
failed experiment. A missing main population is not a complete experiment.
No final evidence association is allowed until the full freeze exists and its
hash closure still verifies. Evidence association is idempotent against that
same freeze, writes a started marker before opening values, and records all
inputs. An interrupted run resumes the same frozen experiment, never refits it.

## Independently qualified channels

Field: exact join to the frozen AstroNN/APOGEE native reference rows; preserve
raw source fields and identity/TYPE. AstroNN `dist/dist_error` in kpc, not
`weighted_dist`; distance availability and precision flags remain unchanged.
Finite positive distance/error qualifies the distance observation channel;
error/distance<.2 defines the already registered precision subset. Convert to
mu=10+5log10(d/kpc), sigma_mu=(5/ln10)*error/d for the explicitly approximate
Gaussian-mu score. Report all-qualified and precise separately; this error model
and Gaia-trained teacher ancestry prohibit an independent absolute-scale claim.
Reliable APOGEE dwarf/giant quality labels remain0/1; neither label means unknown
(-1), never other-class2. FeH requires its own finite value/error and source
quality; missing FeH/class never excludes a distance or Gaia observation.

Same positive native Gaia DR3 IDs receive a fresh exact join to the existing
local DR3 healpix catalog, only after freeze. Preserve signed parallax and the
full joined quality payload. Use existing `derive_gaia_labels` astrometric
quality, zero-point validity and finite-parallax flags; no parallax sign or SNR
cut. Nonstellar/duplicate/quality flags stay explicit. Negative or weak
parallaxes are never inverted. Missing tiles/matches remain unvalidated rows.
The external AstroNN and Gaia channels are correlated evidence, not independent
replications. Existing MWS/VAC availability on these same frozen identities is a
separate channel audit, not permission to change the sample or primary teacher.

Common: exact strict-member identities/probabilities and original scales.json.
For NGC1261 use HST16.775±.872kpc (already includes1.7%); uncertainty is ONE
shared system distance, no sqrt(N) reduction. Use4096normal-distance draws with
seed320501; reject nonpositive distance draws without replacing by a floor.
Repeat shared-scale summaries for multiplicative.97/1/1.03 method scenarios;
the±3% scenario is a sensitivity, not another fitted correction or uncertainty
term. NGC1851/7492 empty admissions remain explicit. Individual common-scale
marginal log scores integrate an additive
Gaussian distance likelihood in kpc against the full photometric mu PDF; their
units are inverse kpc, and they must not be multiplied into a system likelihood.
The shared-scale draws above govern aggregate reference-agreement uncertainty.
Published membership uses astrometry; Gaia scores here are membership-conditioned. System agreement is
not a field contamination estimate or3032independent scale measurements.

## Metrics, grouping and numerical acceptance

Each named view reports full-PDF source-error-convolved log score; signed Gaia
log score/PIT and PIT<.025 or>.975; median mu/relative-distance residual,
|delta_mu|>1 severe fraction, central68/95 interval widths and reference-center
containment. Call the latter reference agreement, not physical coverage.
Class labels supply marginal class probability/log score and confusion at
argmax; unknown/other cannot be silently relabeled dwarf. Preserve all channels
and denominators instead of choosing a reference that favors the model.

Compare finalcandidate with each frozen control on exactly the same qualifying
identities. Field uncertainty:1024resamples of whole frozen sky cells, seed320511,
with paired identities/views, no per-star bootstrap. Report n/cells; one-cell
comparisons have no empirical sky interval. Report new_sky_field and
new_source_seen_sky_field separately; do not pool into independent validation.
Prespecified strata: reference distance0–2/2–5/5–10/10–20/20–50/>50kpc;
reference class dwarf/giant/unknown; FeH<−1 versus>=−1 or unavailable;
r0<18/18–20/20–22/>=22, and available W1 signal/noise<5 versus>=5 or unavailable.
Sparse or zero cells are reported, not merged until a desired answer appears.

At fixed finalcandidate qnear>.95 record the same-identity predicted far mass,
class-distance probabilities and qualified-reference disagreement. Requiring a
reference selects another population: its disagreement is not fullapplication
contamination. Unqualified identities remain in the model-based denominator.
The common-scale membership sample supplies only a system-conditioned near-error
test. No independently measured fullcandidate contamination claim is justified.

Re-evaluate proper scores on the last two saved numerical levels (first reference
sidecar versus original.02 where necessary): logscore difference<.02 and Gaia
PIT difference<.003, in addition to prior probability/quantile/MC gates. Preserve
and report score failures; any subsequent numerical-only refinement must retain
the frozen model/sample and initial results. Never drop flagged identities or
retune a scientific parameter after opening. Report accepted-only numbers with
the full flagged denominator and all-object descriptive numbers alongside.

No result in this protocol establishes representative fullcatalog calibration,
an independently measured density, a universal scale correction or permission
to replace deployed v1. Scientific acceptance remains a separate report.
