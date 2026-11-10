# v2.1: near-selection and joint-reference probability audit

Authorization: corrected review chat `6aeebe6f-8abc-83e9-aeed-c0107222fa1a`;
complete latest review archived in `pro_review_602727c_near_probability.md`.
Frozen baseline: 602727c. This is development, not independent final validation.

## Execution and acceptance matrix

| ID | Requirement | Deliverable / acceptance |
|---|---|---|
| A1 | Verified native DR11 PSF; independent class/distance/parallax admission | Per-role missing/non-PSF exclusions, PSF identity exports; preserve historical inputs |
| A2 | Heliocentric bins 0–2,2–5,5–10,10–20,20–50,>=50 kpc | Full-sample class counts, bias, NMAD, severe tails, coverage, widths |
| A3 | Reference-near versus actual selected-near | Migration, completeness and far contamination, class/brightness/errors/WISE/sky/metallicity strata |
| A4 | Reference boundary uncertainty | Separate 1-sigma mu intervals crossing15, unknown uncertainty; do not treat these as certain labels |
| A5 | Application scope | Existing weighted parent predictions; actual baseline_v1 route; no full DR11 population claim |
| B1 | Exact frozen training identities and trees | Rebuild class and label-source joint leaf contributions; verify both PDF and class marginals |
| B2 | Full PDF reproduction | Bounded checkpoints, Sobol128, shared g error; no Gaussian reconstruction from quantiles |
| B3 | Per-object near/far and branch attribution | Joint class-distance and source-distance masses, nominal/noise contrast, representative decompositions |
| C1 | Independent colors and full covariance | Local cross-class neighbors/mixing/Mg span, brightness/error/WISE/sky strata |
| C2 | Measurement degradation | Fixed good templates and actual application errors/missingness/routes; signed weak flux retained |
| C3 | One fixed classifier framework | No-color prior, QRF contribution, fixed forest; same training labels, evaluation IDs, bands, calibration |
| C4 | Matched band contrasts | grz/griz, grz/grzW1, grzW1/grzW1W2; report coverage and identity differences |
| C5 | Full probability scores | Teacher-space log score and interval score, coverage/width; observed Gaia parallax score incl negative; clear observation model |
| C6 | Near-event probability | Reliability, Brier, risk-retention at0.5/0.8/0.95; cal chooses rule, development test evaluates |
| C7 | Weight sensitivity | Evolved contribution multipliers0.5,1,2, explicitly stress tests, not calibrated priors/count intervals |
| C8 | Remote/faint/metallicity/quality/sky | Far10–20/20–50, distance×magnitude×error×WISE; zero-point/dust sensitivity |
| C9 | Nonstellar and WD evidence | Reliable PSF nonstellar controls;49 candidates never promoted to confirmed labels |
| C10 | New final validation reservation | Identity/sky/system/replica leakage audit before candidate evaluation; existing test remains development |
| D | Evidence-dependent decision | One targeted candidate only if justified, or bounded scientific validation/evidence gaps; no family search/deployment |

Science uses full qualified samples, never the historical display-only3sigma cut.
Thresholds use strict inequalities d<10 and q>threshold. Distance bins are
left-closed/right-open, last bin>=50. Unknown class is missing reliable evidence.
Boundary crossing means mu_ref-sigma_mu<=15<=mu_ref+sigma_mu (diagnostic only).

Frozen legacy training reconstruction is a baseline audit exception: no dropping
historical members while claiming original-PDF reproduction. Every new probe,
comparison or application sample requires verified PSF; unverified fails.
Old trees may encode legacy selection: disclose this when comparing a PSF-only
probe against a frozen-tree audit. Identical labels alone do not erase this.

Joint channels are reference contributions, not a calibrated physical posterior.
Never multiply independent class and distance marginals for joint events.
Empirical-QRF probabilities are not likelihoods for multiplying another Galactic
prior. Probability sums are model-conditional expectations. Reference-selection
contamination is not application-population contamination. Teacher predictive
scores, reference-likelihood scores and independent distance accuracy are separate.

No newly independent final set can be made by relabeling inspected test rows or
by sampling noise replicas. Quarantined identities/geometry are not a holdout.
Reserve only demonstrably unused objects/sky, with system-member exclusions;
if unavailable, record acquisition requirements and block final-validation claims.

Keep outputs isolated in data/v2_1, compact exports in manifests/v2_1, figures
in figures/v2_1. Six scientific figure groups only; all labels English, existing
Pal5-derived typography. Freeze input/code hashes per checkpoint, refuse stale
reuse. Full computation on Slurm. No monitoring, agents or catalog replacement.
