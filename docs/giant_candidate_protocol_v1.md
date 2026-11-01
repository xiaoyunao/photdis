# DECam giant candidate application protocol v1

Registered before new scoring, following the complete [Pro review](pro_review_8e26f78.md).
Parent checkpoint 8e26f78 and every original result/model remain frozen.
No new fields, release migration, photometric cuts, RGB fits or model-family search.

## Data and native system

Reuse `data/giant_information_v1/labels.npz` and original frozen models: all
clean, high-latitude, low-dust, unambiguous spectral matches were retained there,
including uncertain/nonstellar evidence. Apply representative/no-conflict identity
rules and release=11010; require finite usable measured bands. There is no Teff,
logg, abundance or distance cut in application. Spectral reference categories
are diagnostic only. Record missing-band counts and the original full photometric
WISE availability separately. A spectral-reference composition is not catalog purity.

DR11 includes north products (11011=Bok g/r, Mosaic z; no i) and south
(11010=DECam). North uses observations already in DR9, not necessarily an
unchanged DR9 catalog. Sources checked 2026-11-01:
- https://www.legacysurvey.org/dr11/description/
- https://www.legacysurvey.org/release/
- https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr11/north/sweep/

Keep natural camera systems distinct. Names such as `native_north_final_ra130`
describe sky geography, not BASS/MzLS: the existing RELEASE census identifies
that field as DECam. Northern promotion is deferred: at least three adequately
populated native-system fields for two train/one test, not a release search.

## Fixed comparisons

1. Apply existing optical A and common5 A/B/C models to each entire held-out
   spectral field with the appropriate usable bands. Retain their original K
   test scores by identity, avoiding a change from Monte Carlo row ordering.
   Missing original models remain explicit unscored cases. Categorize cold/K/warm
   giants and dwarfs, intermediate gravity, hot qualified stars, invalid/unknown,
   and raw Redrock nonstellar risk. General class validity remains Teff<7000 as
   in class-v3; hotter objects are not newly assigned physical class. Report
   counts and selected fractions at .5, support box in/out, brightness,
   W2 quality, spectral parameters of selected reliable giants.
2. Train only new A/B models on W1>=5 clean objects, without any W2 requirement.
   K training labels and every original hyperparameter/16 shared-band antithetic
   draws stay unchanged. Same objects for A and B. Outer whole-field validation;
   new predictions include broader reference objects. Split results W2 pass/fail,
   r16–18,18–20 and other brightness/temperature/support strata.
3. Low-FPR descriptive ROC from outer scores is feasibility-only (target 1%,.1%).
   Also register train-only thresholds for W1 A/B: partition the outer training
   fields into three deterministic groups; hold each group out, fit on remaining
   K labels and score all usable broad-reference dwarfs in that group. No outer
   object enters any inner fit, scaler, support or threshold. Choose the least
   conservative threshold allowing at most floor(alpha*N) inner dwarf false
   positives; score ties excluded with nextafter. Outside the inner training
   color box is rejected. N<1/alpha has inadequate empirical resolution, retained
   as a flag, not a guaranteed target FPR. Apply frozen threshold plus outer
   training box to the broad outer field. Report all counts/Wilson intervals,
   field variation and purity as function of assumed giant base rate. Low-count
   zero false positives do not prove zero leakage. No fixed score is a probability.
4. Mechanism only: same old optical/common5/common20 giant training/test identities,
   q0 uses x=log10(T/5000)/.1, x^2, logg; q1 adds the original A/B/C colors.
   Fixed Gaussian linear conditional family, ridge/floor as before. The x^2 term
   admits the form of the known temperature correction without a new search.
   Observed spectral T/logg are conditioned on, not uncertain latent inputs;
   original corrected-label errors (including T propagation) retained identically
   on both sides. Color covariance propagated; spectral-feature covariance set
   zero conditional on observations. This is a diagnostic of added predictive
   value beyond measured T/logg, not a causal isolation or deconvolved physical
   FeH calibration. Label variance added once in fitting and once in predictive
   score convolution, not twice within either likelihood. Whole-field mean log
   score gains, coverage and field bootstrap; informative fields N>=10,
   >=3 fields, 95% lower>0 and >=75% positive for a bounded stable-gain claim.

## Output and stopping rule

Deliver per-star scores/identities/support/thresholds on NERSC; aggregate JSON,
ROC and population/coverage/abundance figures, reproducible report in git.
Primary evaluation r16–18; 18–20 secondary, r>=20 evidence limits explicit.
A usable candidate prescription uses only release, sky/dust/clean photometry,
W1 validity, training-color support and train-selected score threshold. It is
not automatically adopted as a high-purity catalog: determine from real outer
results whether low leakage retains useful recall. Report failures without tuning.
Any next luminosity calibration must match actual selected phase/FeH evidence;
low logg is not lower RGB, and q(FeH|colors) is not an independent distance prior.
Complete these fixed comparisons once, validate isolation and figures, synchronize
programs/small results, and close the goal even if no publishable selection emerges.
