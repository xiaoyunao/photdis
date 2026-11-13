# v3.1 — conditional calibration and scientific use domains

Authorized 2026-11-13 by the user from the latest Pro message in
`docs/pro_review_31f1277.md`, especially section six. Baseline: `31f1277`.
This protocol registers the research sequence, not successful calibration.
The active chat goal includes all four work packages and final delivery.

## Preserved products and scope

- All v3.0 data/models/predictions/failed and refined numerics remain immutable.
  Deployed v1 remains unchanged. New outputs use `data/v3_1`, small records use
  `manifests/v3_1`; no full-candidate rerun before representative acceptance.
- Native Legacy DR11 TYPE=PSF is required at admission and export. Missing or
  unverifiable TYPE fails; report both missing-TYPE and non-PSF counts.
- Large inputs/products stay on NERSC. Scientific scans, fits and simulations
  run on allocated compute. No timers, subagents or deployment are authorized.
- Negative observed flux and signed Gaia remain measurements. No weak-parallax
  inversion; no forced positive observations or artificially flattened response
  weights. Figures use English exclusively.

## Roles before modification

The v3 source-calibration, hyperparameter and seen-development roles remain
distinct. v3 new_sky, NGC288 and the opened 3250 reservation are now seen
diagnostic/regression evidence, never a new blind final test. Initial mechanism
diagnostics reuse seen data only; no new validation outcomes are accessed.

Before fitting a v3.1 correction, inventory historical identity, sky, source,
teacher and common-scale usage. Select new test blocks by available reference
coverage, without model residuals. Prefer whole unused reference sky/source
blocks, or freeze photometry-first predictions before opening matched evidence.
Availability may establish distance/FeH coverage but does not license selecting
on prediction errors. Preserve the selection history and all excluded blocks.
Freeze the actual IDs/block assignments and reference definitions before fitting.
If historical use leaves no independent block, acquire new evidence or mark the
domain unvalidated; do not rename a seen sample. Do not exclude every historical
reference identity and then assume the remaining photometric IDs have labels.

Required evidence grid: distance 0–2/2–5/5–10/10–20/20–50/>50 kpc;
reference dwarf/evolved/other/unknown; FeH below -1.5, -1.5 to -.5, above -.5,
and missing; r below18/18–20/20–22/22+; band availability/quality and sky.
Report overlapping marginal strata plus supported cross-strata, true denominators,
unique stars/sky blocks/scales, teacher ancestry and empty/unverified cells.

## WP1 — mechanism before correction

1. Freeze a deterministic, stratified seen-development diagnostic cohort with
   selection probabilities and fixed sky blocks; include all available rare
   evolved/poor-metal strata up to registered caps. Also retain the full frozen
   evaluation denominator for baseline tables. Pilot timing changes compute
   size only; never select a cohort based on residual success.
2. Same-star contrasts: nominal/reference; response12/reference;
   response12/target; reliable class conditioned; measured FeH conditioned with
   uncertainty; class+FeH; grz/griz/multiband; single/binary attribution;
   necessary individual-band error/omission controls. Truth-assisted views are
   diagnostics and never production observables.
3. FeH conditioning must update the correlated flux distribution as well as
   mixture weights. Source distance scores convolve reference errors; shared
   teacher errors and common scales are not independent per-star evidence.
4. Diagnose branch location versus weight, signed-Gaia predictive PIT/score,
   reference-point interval agreement versus calibrated coverage, color/FeH/
   brightness/source/sky effects, rare component effective support, latent
   negative-flux probability and mode class/binary/component responsibility.
5. Based on those results, record one targeted low-dimensional constrained
   correction design before fitting it on allowed calibration roles. Candidate
   possibilities are conditional luminosity scale, band/model discrepancy,
   rare-branch support with known sampling weights or binary model. Do not make
   all changes automatically or search model families without evidence.

## WP2 — radial flexibility and response support

Fixed SED/noise/selection comparison: frozen five wide bases versus one regularized
multiscale radial family. Its exact basis, penalty/prior and inference tolerances
are frozen in a versioned amendment before examining recovery outcomes. Preserve
all 14 old scenarios and add a crossed grid of narrow structure positions and
widths, including but not restricted to 4 kpc/.08 mag. Use independent source
templates, shared repeats and source/sky uncertainty. Separate latent realized
counts, population parameters and Poisson/design uncertainty; report all failures.

Compare all26k design objects, their final-candidate-admission subset, and held-out
fields under selection normalization matching each sample. Compare class/radial
weights and external predictive scores, including high-q rare-class errors.
Regenerate an independent response set and decompose evidence by observable/source/
sky strata; report response ESS and distance/count sensitivity. If needed introduce
diagnostically supported continuous systematic parameters with a registered prior.
More discrete draws or longer MCMC alone are not a scientific repair.
Physical density requires effective area/complete selection, outside current claims.

## WP3 — new validation with actual evidence

Before correction training, freeze roles and available independent evidence for
2–5/5–10 kpc field stars, class labels, poor-metal giants and multiple common scales.
Far claims require 10–20/20–50 kpc evidence. Open each reserved final outcome only
after method/prediction freeze; no retuning on that outcome. Retain negative Gaia,
source-specific likelihoods and sky/system correlated errors. Report missing
coverage as unvalidated, not passed. Common-scale uncertainty never shrinks as
1/sqrt(member count). Existing v3 validation can supply regression comparisons.

## WP4 — product identity and scientific use

Every product records input IDs, SED/calibration hashes, prior/selection definitions,
shared draw IDs/weights, role and numerical state. Main evaluation must match the
final proposed product; compare nominal/reference, response12/reference and
response12/target explicitly. New candidate performance is a separate column.

Deliver joint class×distance counts in high-qnear samples, model-expected leakage
separate from independent observed contamination, mode class/binary/component
attribution, and an evidence-backed usable/extrapolated/unvalidated domain map.
Retain exact failed flags and reference labels (evolved is not all physical giants).
Audit prior replacement support, tail mass, ESS and shared parameter alignment.

One registered structure example must compare a fair point/baseline analysis with
full-posterior marginalization for members, distance and background. Withdraw the
old prior when introducing a structure prior, propagate shared calibration/template
draws, apply selection and test recovery. Do not double-count a target prior via
posterior stacking; distinguish a seen-system demonstration from independent proof.

## Acceptance and delivery

- Mathematical/identity tests and actual-array validation are separate from
  scientific performance gates. Start with syntax and focused tests.
- Cal-only correction selection must consider bias, tails, uncertainty, signed
  Gaia, minority-class high-confidence errors and fair matched baseline; freeze
  quantitative gates after baseline characterization and before candidate fits.
  Final-test failure cannot change those gates or remove difficult objects.
- A domain is usable only when independent relevant evidence supports the exact
  product/observable selection. Sparse domains carry uncertainty/insufficient
  evidence; unsupported domains remain unvalidated. Engineering tests do not
  establish physical truth or representative application contamination.
- Required outputs: methods, result, domain map, English figure atlas (PNG/PDF
  with source CSV/JSON), requirement-by-requirement evidence and limitation ledger,
  source/model/selection provenance, job records, numerical/visual validation.
- Maintain README, AGENTS, reverse-chronological WORKLOG and current PLAN;
  commit milestones and safely synchronize local/GitHub/NERSC. Goal completes
  only after the authorized implementation/evidence work is delivered, including
  substantiated unavailable evidence and explicit limits, not after planning.

## Initial execution order

1. Archive review, recover three-endpoint state, register this protocol/ledger.
2. Implement and test truth-assisted diagnostics and mode attribution separately
   from frozen production; run a bounded allocated pilot then registered cohort.
3. Inventory prospective evidence blocks before correction fitting; freeze roles.
4. Execute paired mechanism/identity/response and domain-transfer audits.
5. Register and fit the supported correction and radial comparison; evaluate all
   old and new scenarios, inspect systematics, freeze final candidate.
6. Open registered final evidence once; build domain map and structure example;
   complete all four packages, figures, artifact acceptance and synchronization.
