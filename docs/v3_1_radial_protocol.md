# v3.1 radial amendment — frozen before recovery outcomes

Registered 2026-11-13. This is the WP2 radial mechanism comparison, not a new
SED calibration or permission to open prospective real validation outcomes.
Frozen v3.0 products remain unchanged.

## Matched models

Use the same native photometry, error/dust/mask contexts, fixed nominal SED,
binary fraction and conditional grz entry likelihood in both arms. Freeze the
within-class SED tilt at zero in both arms to isolate radial flexibility.
The old adaptive-tilt recovery remains a separately named historical comparison.

Baseline: the five original Gaussian radial bases, with class×basis weights
theta. For single-field catalogs the additive log ratios have the original
reference center and standard deviation sqrt(.7²+.35²), which marginalizes the
original global/field Gaussian prior exactly. No redundant global/local latent
coordinates are needed for this single-field comparison.

Alternative: for class c, replace each basis density phi_d by
`(1-eta_c)*phi_d + eta_c*Normal(mu; location,width)`.
One shared location/width describes a local structure; three class amplitudes
allow different class composition. This is one smooth-background plus local
structure model, not a family search. Proper independent standard-normal latent
coordinates specify:

- `eta_c = sigmoid(-2 + 1.5*z_c)`;
- `location = 8 + 12*sigmoid(z_location)` mag;
- `width = .04 + 1.96*sigmoid(logit((.25-.04)/1.96)+z_width)` mag.

The location support spans approximately0.4–100kpc; the width floor is .04mag.
Do not change these priors after recovery scores. The alternative has19 latent
coordinates versus14 for the baseline. Neither arm uses truth class/FeH/distance.

## Likelihood, selection and numerics

Recompute class-conditional native likelihood curves from the fixed SED, using
all kernel/binary support, rather than assuming compressed broad-basis integrals
resolve a narrow local component. Integrate both likelihood and conditional
entry curves against the *same* radial density. The per-object denominator is
the predicted grz S/N entry probability given its recorded error/transmission;
sky, morphology and image masks remain conditioned-on covariates, not an area
or detection-completeness correction. Native TYPE=PSF at source/context admission
and output remains mandatory, including simulated samples' source provenance.

Use a mu grid covering[-15,45]mag with at most.02mag spacing where local structure
can have mass. Check finer.01mag quadrature on pilot and flagged catalogs;
require probability changes<.003 and count means<.5star before acceptance.
Selection profile interpolation must retain hull flags and compare against
direct Gaussian entry integration on a fixed pilot subset. Do not normalize
away missing tail support. Retain original and refined numerical artifacts.

MAP initialization is fixed before outcomes: baseline reference center;
alternative location9/11/13/15/17/19mag crossed with width.08/.30mag, plus
the prior center. Preserve every local optimum and objective. Use4 HMC chains,
initial512warmup/512retained, target acceptance.9; initialize across distinct
competitive optima (objective within10 of best) when present. Rhat<1.05,
minimum ESS>=100, zero retained divergences, and count marginal half-delta<.005
are numerical gates. An unresolved multimodal chain is flagged, not averaged
into a falsely calibrated interval. Numerical-only extensions retain seeds and
original results; no scientific-prior tuning based on these diagnostics.

## Catalogs and estimands

Replay all112 existing catalogs (14scenarios×8repeats) on identical identities.
Add a crossed narrow grid at2/4/8/16kpc and widths.08/.25mag,8repeats each,
128selected objects/catalog, using an independent empirical-source realization
seed311501 and context/noise seeds311600+catalog. Repeats use the same underlying
source bank and do not create independent real reference stars. Model/same-library
controls retain their intentionally optimistic source ancestry. Preserve signed
Gaia diagnostic draws; they never enter photometry-only inference.

Report paired class×distance realized counts and intervals/covariance, radial
location/width where the simulated generating design defines them, parameter
versus realized-count recovery separately, distance/PDF/Gaia scores, numerical
failures and source/field clustered uncertainty. A local Gaussian fit to a broad
or class-correlated scenario is not assigned a fictitious true location.
Far/poor-metal/rare classes with absent source support remain explicitly limited.

No scientific acceptance threshold is set from recovery outcomes. Improvement
must coexist with numerical acceptance, old-scenario regression results and real
validation of the eventual product. This experiment alone cannot establish an
absolute distance scale or authorize full-catalog replacement.
