# Fixed-kernel target transfer — registered before these fits

The luminosity-correction protocol freezes within-class SED proportions, binary
fraction and discrepancy components. Accordingly these new target fits use
class x five-Gaussian radial mixtures with the original global .7 / field .35
log-ratio shrinkage and reference class-frequency center. They do not introduce
or fit within-class FeH/color SED tilts. The original v3.0 target with fitted
tilts remains a separately named control, not an identical candidate product.

Removing the old hierarchy's two tilt coordinates with beta_sigma=0 integrates
two independent normalized standard-normal nuisance priors to one. The retained
objective/gradients and field weights must reproduce the original hierarchy;
an independent finite-difference gradient test is also required. This is the
population model implied by the already registered fixed-kernel correction.

Fit the same response12/calibration12 separately to: all26000 application
sample rows; the existing3540 final-candidate rows; and training fields after
holding out three complete named fields. Select those fields by stable SHA256
rank of `319801:<field_id>`, lowest three. Use no reference residual in selection.
Audit both broad and candidate subsets in held-out fields; new-field theta is
drawn from the fitted global hypermean with the original .35field dispersion.
Record all identities, TYPE, admission/exclusion counts and exact design weights.
Rescale case weights to the actual fitted sample size, never millions of parent
objects. This remains a design-weighted generalized likelihood and conditional
observed-catalog inference; no effective-area/physical-density claim follows.

Controls: zero correction and the point correction, each with the same fixed
kernel/radial family. Main uncertainty ensemble:32 pre-existing shared sky
bootstrap luminosity corrections. Each correction gets its own conditional
population fit and bridge evidence with a proper normalized latent prior.
Retain reference-only equal-bootstrap cut sensitivity separately from
population-evidence-weighted results. Do not flatten weights if ESS collapses.
This finite bootstrap sensitivity library is not exact Bayesian calibration or
an independently measured absolute-scale prior. No new test labels enter.

Use the Gaussian-convolution identity for BOTH native likelihood and selection
integrals. Native likelihood uses all kernels; no compressed-tail pruning is
silently accepted. A preparation pilot uses the128 lowest SHA256-ranked
application identities with salt319802 plus the three fixed entry-context IDs.
Recompute all native class log likelihoods on original adaptive grids united
with uniform [-20,50]mag/.02, and .01 grids for every pilot. Production numerical
audits additionally include128 identity-ranked objects with salt319803.
Compare class/basis log integrals within30nat of the per-star maximum to.02nat,
and fixed-reference class/basis probabilities to.003. Retain all failures;
actual target-posterior and joint-count coarse/fine changes still gate final use.
Keep one common per-object log scale across correction draws for valid evidence.

Entry integration requires the joint-QMC pilot to pass. Prefer direct measured
contexts; any context compression/interpolation requires an additional explicit
photometry-only design, observed-context hull checks and independent direct
comparisons before it can be used. No hard-cut deletion or dust omission is
permitted in the new fits. Retain tiny-entry numerical errors and their effects
on fitted counts, rather than claiming relative precision from absolute bounds.

Sampling:4chains,512warmup,512retained with original HMC target_acceptance=.9.
Use numerical-only2048 then8192 extensions with fixed seeds if required; preserve
all shorter attempts. Require Rhat<1.05, minimumESS>=100 and zero divergences.
Bridge integration uses2048 proposal draws; fixed-point convergence, proposal
ESS>=100 and half-log-integral difference<.2 are required before ensemble weights.
Counts require shared global indices and conditional covariance, not products
of marginal class and distance probabilities. Posterior fine checks require
joint-probability delta<.003 and count-mean delta<.5, with all original flags kept.

The separate all176radial recovery tests remain required evidence about radial
flexibility and narrow structures. Their empirical/noise/library limitations and
original dust-entry approximation remain explicit. A five-basis target fit does
not claim to solve narrow-structure recovery or pass real far-distance validation.
Final external outcomes remain unopened until model/calibration/prior/selection
identities and all baseline/candidate predictions are frozen.
