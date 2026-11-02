# Actual K-giant teacher audit and conditional distance experiment

Registered 2026-11-02 after Pro review of 1689ec2, before inspecting new paired
teacher residuals or distance scores. Source: [full review](pro_review_1689ec2.md).

## Scope and endpoint

Keep all existing models, classification thresholds and test identities frozen.
No classifier or sky-field search. Application features are photometry only.
This round finishes with teacher evidence, bounded conditional predictions and
an explicit external-calibration decision, including a negative feasibility
finding if existing independent references do not cover this population.

## 1. Evidence table and teacher comparison

- Start with all existing clean high-latitude low-dust DECam reliable K giants,
  4200<=Teff<5000, identity-representative/nonconflicting objects. Retain optical
  and W1-qualified counts separately. The paired A/B experiment uses every
  W1-qualified reliable K giant with a qualified teacher; never only the 190
  B-wide selected successes. Reproduce the frozen 190 selection for evaluation.
- Join the two existing audited VAC tables on BOTH Gaia source_id and DESI
  target identity; ambiguous matches are excluded and counted, not guessed.
- Retain each teacher's match/quality flags, errors and invalidity causes.
  Retrieve SPDist percentiles, Gaia-G error and QFLAG_COMMENTS from its actual
  public table for the paired identities. Reproduce existing valid masks.
- Tabulate selected population, all eligible K population, common-valid
  intersection, and SP-valid versus remaining objects. Save row evidence and
  mask/field/source provenance; compare delta_mu=5log10(dSP/dSPEC) by fixed
  Teff [4200,4600,4800,5000], logg [0,1.8,2.2,2.6,3.2], calibrated FeH
  [-5,-1.5,-1,-.5,1], SN [0,20,60,100,infinity] bins with counts.
- SPDist mu width combines half16–84 span, Gaia-G uncertainty and .167mag
  method floor once. SPECDIS errors are measured on these actual stars.
  Quote fractional-error quantiles and fractions >.2/.5; no automatic promotion
  of all valid labels to precise truth. Do not inverse-variance combine teachers.

Official definitions checked on 2026-11-02:
[SPDist](https://data.desi.lbl.gov/doc/releases/dr1/vac/mws-spdist/) and
[SPECDIS](https://data.desi.lbl.gov/doc/releases/dr1/vac/mws-specdis/).
Both share Gaia and spectroscopic information; cross-catalog disagreement is
not an independent zero-point measurement.

## 2. Independent references

Crossmatch all eligible K identities to the prepared cluster membership lists,
inspect exact-source astrometric anchors and the previously observed cluster
resources. Report system, native camera, member/phase evidence, luminosity,
abundance range and bright-end losses. Cluster/system members are excluded from
teacher training. Independent phase stays unknown unless separately supplied.
Existing M3/M53/NGC7006/NGC5466/NGC4147 are not automatically appropriate merely
because they have distances; NGC2419 has different camera/dust and poor-metal
support. A usable whole-system check needs matching native input and a supported
stellar population, with its common distance error kept common. If absent,
deliver the measured gap and precise next reference requirement, not a calibrated
absolute-distance claim. Teacher development proceeds regardless.

## 3. Finite teacher-scale distribution models

Use a Gaussian kernel mixture with one component per eligible training star.
Run SPDist and SPECDIS separately, never pooling their labels. Each teacher has
three fixed variants: no-color training Mg mixture, A=(g-r,r-z),
B=(g-r,r-z,z-W1). Shared training/test IDs and weights within each teacher.
Uniform star weights; fixed color bandwidth .12mag per color and Mg smoothing
.15mag. No tuning against held-out results, no parametric phase assignment.

For each held-out field, train on all other eligible nonmember fields. Require
>=20 training stars and >=2 training fields, otherwise explicitly skip. Exclude
all shared identity groups. Evaluate every eligible test source, additionally
report the frozen B-wide1% selected subset; a color-support flag is descriptive,
not a residual-based rejection. These reused fields are development validation.

Training components have covariance from the SAME g/r/z/W1 measurements in
(colors,Mg): Cov(g-r,Mg)=sigma_g^2, neighboring-color covariances are negative.
Add teacher mu variance to Mg variance once. Query measurement covariance is
included jointly: set Z=Mg+epsilon_g, condition (colors_observed,Z), then
mu=g_observed-Z. This retains Cov(colors,epsilon_g), rather than independently
perturbing g after conditioning. No-color uses the same label widths and g error.
This is an error-broadened teacher empirical mixture, NOT a deconvolved intrinsic
population model. Unknown teacher–Legacy cross-survey covariance is not known;
shared within-Legacy terms are exact under independent band errors.

SPECDIS reports d and sigma_d, not a full posterior. Main sensitivity convention
is a median-preserving lognormal width sigma_mu=(5/ln10)*sqrt(log(1+(sigma_d/d)^2));
also report the fixed linearized proxy (5/ln10)*sigma_d/d as a sensitivity run.
Neither is claimed to reconstruct a catalog posterior. SPDist uses its documented
mu-width approximation above, preserving the .167 term. Model output is a full
mixture of Gaussian mu components, equivalently lognormal distance components,
including possible multiple modes; store mixtures, not only quantiles.

## 4. Evaluation and validation

Report counts, delta_mu=pred-reference median/NMAD, relative-distance residuals,
|delta_mu|>.5 and |relative|>.5 tails, raw68/95 intervals/widths and coverage of
the noisy point reference, observation-error-convolved log score and PIT. Show
both all-test and selected-test, each field, r16–18/18–20, temperature, FeH and
teacher-error strata. Paired field bootstrap of A/B/prior score and scatter
differences is descriptive because training folds overlap. Report effective
mixture size and extrapolation flags without deleting difficult objects.
No claim of calibrated true-distance posteriors without independent evidence.

Unit checks: shared-g covariance/sign against direct joint conditioning, no-color
normalization, positive conditional variance, mixture quantiles and row isolation.
Real checks: exact 190 selection, unique joins, teacher error accounting, identical
A/B IDs, no field/identity leakage, reproducible source hashes, summary row sums.
Save selected190 evidence table, teacher scatter/residual/coverage figures,
held-field A/B/prior distance and distribution plots, support/decision report.
All catalog scans and experiments run in allocated NERSC compute. Finish with
WORKLOG/PLAN/README and synchronized code/small artifacts; large mixtures stay
under data/kgiant_distance_v1. Do not prolong this round with more model families.
