# v3.1 constrained luminosity correction — registered before fitting

This single candidate addresses the remaining **within-class** scale and interval
failures. It does not assume that the dominant wrong-class branch failure is a
uniform luminosity offset. The paired diagnostic found that reliable class
conditioning changes evolved-star errors far more than FeH conditioning, while
class-conditioned 68% reference-point agreement remains about41% for evolved
stars and59% for dwarfs. Small/mixed band-error changes and the negative latent
flux pilot do not motivate changing those channels. These are seen-development
findings, not calibration targets or independent scale evidence.

Keep response12 SED/calibration, binary fraction, discrepancy component and
within-class kernel proportions fixed. For each of three reference classes,
introduce an achromatic luminosity brightening delta and Gaussian extra
luminosity dispersion sigma, in distance-modulus magnitudes:

`L_corrected,c(mu) = integral L_original,c(mu-delta-t) Normal(t;0,sigma) dt`.

Positive delta means intrinsically brighter and a larger inferred distance.
This convolution acts on the **likelihood before any distance prior**. It is
not convolution of an already normalized posterior. Selection-entry curves
must receive exactly the same transformation in any population fit. Correlated
flux scaling preserves colors and flux/FeH structure. No observed negative
flux is discarded; no reference label or metallicity enters application inference.

Use one independent correction per class, delta in[-1,1]mag with truncated
Normal(0,.3) prior and sigma in[0,1]mag with truncated half-Normal(.3) prior.
Six parameters total; no color/FeH terms or additional model search. Priors are
regularizers in a conditional calibration objective, not a claim of measured
absolute-scale uncertainty. The zero correction is included at the boundary.
Report boundary fits, source conflicts and all candidate failures.

Only original v3 **hyperparameter role3** stars may enter this correction;
role2 remains source-consistency calibration and role4 remains seen development.
Deterministic sky-cell hashing (salt318101, parity) separates correction fitting
and calibration guards. Per class and partition, retain at most512 eligible
identities by stable rank salt318102, with exact inclusion probabilities.
Require native TYPE=PSF and original stellar eligibility; missing classes do not
enter fitting but stay in downstream photometry-only evaluations. No new frozen
v3.1 validation object enters any fit or guard.

Choose one available reference likelihood per star: qualified signed Gaia first
(no S/N cut, negative retained), then SpecDis, then SPDist with the frozen
source-calibration offset/scatter/broad fraction. Never multiply shared teachers.
Optimize the design-weighted mean conditional predictive log score within each
class, rescaled to the retained count, plus the stated regularizer. Conditional
posteriors use the original Normal(15,5) reference prior. Report source-specific
scores separately, including all overlapping secondary channels at evaluation.
Fit from zero and one fixed interior start (.0,.3); retain both optimizer records.
Use32 common sky Bayesian-bootstrap refits, seed318103, to propagate shared
calibration variation; these are bootstrap sensitivity draws, not exact Bayesian
posterior draws. All classes share each draw's sky weights.

Native likelihood grid[-20,50]mag with.02spacing; Gaussian convolution truncation
at8sigma. Audit against.01native curves on a fixed32-object fit/guard subset,
selected before outcomes, and require joint probability delta<.003 and quantile
delta<.03mag. Preserve all original and refined artifacts. Full inference reuses
the same likelihood operation and retains numerical failures.

Before final-test opening, guards compare the unchanged and candidate
photometry-only predictions on the separate calibration sky cells: no class's
mean predictive log score may fall by more than.02nat; severe distance tails
and signed-Gaia PIT tails may not rise by more than.01absolute; reference-point
68/95% agreement must not decrease by more than.01. Denominators and sky-block
intervals remain explicit; fewer than20 relevant references is insufficient
evidence, not a pass. These gates select adoption, not permission to hide a
candidate's final-test results. Freeze both candidate and baseline predictions
before once-only external residual scoring. A rejected candidate remains a
negative research result and does not replace v3.0 or deployed v1.
