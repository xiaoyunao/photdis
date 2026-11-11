# v2.5: selection-normalized population inference prototype

Frozen before new outcomes. Source review: pro_review_de4904b.md. Existing QRF,
XD, v2.4 products and deployment remain unchanged. All scientific computation
runs on allocated NERSC nodes; admission/export requires verified native PSF.

## Empirical diagnostics

1. Match the full fixed 711-object historical population (not its 147 low-PIT
subset) to non-fitting observable application identities using gr, rz, r,
log optical error and log Gaia uncertainty. Fixed scales (.2,.2,1,.5,.5),
nearest control with replacement and maximum normalized distance 2; record
all failures and reuse. Select without PIT, parallax value or distance outcome.
Report red rz>=.8 and blue strata, raw versus historical selection-conditional
PIT separately; no claim of selection equivalence. Field/control-cluster
intervals, sampling weights, signed parallax and risk flags remain explicit.
2. Original 2055 poor/ordinary pairs: all clean eligible signed Gaia, no SNR
threshold or inversion. Full frozen QRF/XD predictive parallax scores, source,
fit overlap, negative/weak counts, repeated-control and sky-pair intervals.
Both-side and one-side availability have separate denominators. Report common
SpecDis/SPDist pairs separately; no global metallicity correction.
3. Same-ID band gains by class with class-specific sky uncertainty, retaining
the matched-sample bright/faint support limit.

## Limited population model

Use an empirical forward response in discretized observed gr/rz/r, NOT a
division of QRF probabilities by a global class prior. Four physical template
groups: ordinary dwarf, poor dwarf, evolved, other. Five fixed distance nodes
1,3.5,7.5,15,30 kpc span near bins and far contamination. Only four parameters:
three group log tilts relative to ordinary dwarf and one log-distance tilt.
Factorization/group-independent distance tilt is an explicit limitation.
Condition on observed sample size; never infer absolute Galactic density.

Calibration templates come from split=cal, excluding QRF fit and all 224
v2.4 test-template identities. Up to64/64/48/48 labeled templates plus32
distance-qualified missing-label templates. Missing labels are marginalized
over four groups with fixed color/Mg kernel memberships inferred from labeled
cal templates; never a fifth physical group. This is an assumption to stress,
not a calibrated physical classification. Include a labeled-only sensitivity.
All source translations/noise replicas stay in their identity role.

Noise is the parent median per-band uncertainty by r bin, with .005mag floor;
shared native g/r/z draws induce the correct color covariance. Known empirical
SEDs remain teacher-scaled. Response uses independent random noise realizations
and template-specific measurement errors. Observations are binned using fixed
gr/rz/r edges including overflow. A fixed small uniform contamination floor
prevents zero likelihood; its mass and unsupported cells must be reported.
Selection is 14<=observed dereddened r<20 and max assigned grz error<.1. Additional actual
noise-domain support is audited; this is not full detection/PSF/saturation
selection. Separate supported selection and its complement as stress regimes.
Each response column includes its own alpha_j; conditional likelihood uses
sum eta_j L_j / sum eta_j alpha_j. Keep >10kpc components in every fit.

Fixed bounded uniform log-tilt prior [-3,3]^4; deterministic Sobol quadrature
and doubled-grid convergence, posterior correlations, boundary mass and ESS
are diagnostics, not permission to conceal unidentifiable components.

## Recovery and acceptance

Keep four v2.4 mixtures and add cal-reference composition, only giant change,
only poor change, only distance tilt. Test templates are the untouched224
empirical SEDs. Generate independent observational catalogs with independent
identity draws and fresh noise, 12 catalogs of128 selected objects per design
and regime. Same catalogs compare QRF point, QRF PDF and population posterior.
Report conditional expected group/distance fractions and realized counts;
distinguish population-parameter credible intervals from predictive count
intervals. Near bins0–2/2–5/5–10, far10–20/20–50 retained. No nominal coverage
claim from12replicates without binomial uncertainty. Quantify observation,
finite-cal-template bootstrap, parameter posterior and coherent teacher
scale +/-1sigma separately; include noise-support and label-imputation stress.
No automatic adoption: biases/coverage/identifiability may reject the prototype.

## Prospective3250 identities

Keep outcomes closed during development. Intended estimand is equal-field
mixture of the previously unused, native-PSF observable population in the13
existing fields,250 per field; each row weight1/3250. This is NOT the full parent
area/number-weighted population. Historical-reference exclusion is part of
selection and may depend on Gaia SNR. Freeze QRF and correctedXD comparison,
full signed-Gaia log score/PIT, distance-source-convolved metrics where available,
color/r/error/W1/field and0–2/2–5/5–10/>10distance strata before any opening.
Do not claim unconditional PIT uniformity absent a quantified historical
exclusion likelihood. New population prototype has only simulation acceptance;
prospective opening is deferred until model/selection suitability is established.
No blind score is reported in this stage. Preserve reservation hash and audit.

## Delivery

New code, focused independent mathematical tests, compute provenance/checks,
compact result tables, English Pal5-style figures and checked report. Maintain
WORKLOG/PLAN and synchronize git safely. No timers, agents or full-catalog run.

Implementation freeze: response2048fresh draws/component;270photometry cells; contamination floor1e-4 preserving each alpha;2^14 Sobol points compared to2^13,16identity bootstrap responses and fixed +/-teacher-sigma plus labeled-only sensitivity. Calibration templates retain their own minimum measurement uncertainty.
