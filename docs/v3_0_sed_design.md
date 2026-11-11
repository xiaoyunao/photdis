# v3.0 empirical SED implementation design

This is development design, not the final prospective freeze. Fixed implementation
family: a local Gaussian native-flux/metallicity mixture, generalized projected
heteroscedastic EM with reference distance quadrature. The mathematical projection
follows [Bovy, Hogg & Roweis (2011)](https://arxiv.org/abs/0905.2979): unobserved
coordinates are marginalized and individual measurement covariances enter the
conditional latent moments. Here distance nodes additionally couple all fluxes.
No choice among competing model families is performed.

Coordinates are intrinsic G,R,I,Z,W1,W2 flux at 1 kpc plus corrected RVS FeH.
Corrections use the existing author-correction implementation and validity domain;
uncorrectable/missing FeH is marginalized, while raw evidence remains in the
reference artifact. Intermediate metallicity is not discarded. Three reference
class labels are auxiliary categorical likelihoods; missing labels have likelihood
one, not a fourth physical class. The application never supplies a spectroscopic
label, metallicity or Gaia parallax to the photometric likelihood.

Initial centers use grz color and reference absolute-r coordinates to retain
luminosity branches across the large flux dynamic range. Optional coordinates
have prior initialization only: their missing observations are projected out of
EM. Initial sample covariance subtracts expected measurement covariance; subsequent
conditional-moment EM propagates errors directly. Reference distance uncertainty
is integrated over source-dependent quadrature, not learned as intrinsic scatter
and then added again. The absolute/native flux change-of-variable Jacobian is
included at each distance node. EDSD L=1.35kpc is the explicit Gaia-only reference
prior; signed weak/negative likelihoods are retained. VAC nodes use a flat-mu
measure with calibrated offset/scatter/broad reference mixture.

A constrained equal-luminosity unresolved-binary proxy (flux factor2, reference
fraction0.1) is a latent branch in fitting and inference, rather than being appended
after the SED has absorbed all binary luminosity. It does not estimate mass ratios.
Target-population binary-prior sensitivity and shared calibration/template finite-
sample uncertainty still require production implementation/evaluation.

128 components and at most32768 uniform identity-ranked real training sources
are fixed compression settings; retain the full eligible inventory and selection
probability. This increases independent source support over the old2048/224
libraries without pretending repeated noise realizations are new stars. The first
1024-source/two-iteration run is solely a performance benchmark, not a scientifically
accepted fit. One full fit follows measured cost and error checks. Likelihood and
convergence history, model hashes, source roles and sampling IDs are exported.

The local Gaussian flux approximation has potentially unphysical negative latent
support, unlike legal negative *observed* flux. Per-component negative probability
is explicitly exported and must be evaluated with posterior weights; unsupported
or inadequate domains cannot be called validated. Continuous-distance quadrature,
full versus compressed secondary-mode comparison, source transfer and independent
real validation are required before interpreting an application result.

The source-calibration ensemble uses common sky-block Bayesian-bootstrap weights
for SpecDis and SPDist. This retains shared-source dependence in the uncertainty
approximation; it does not turn Gaia-trained VACs into independent truth. Both
sources' likelihoods are never multiplied for a training star. Joint score/coverage
and sensitivity to calibration remain required. The original models stay unchanged.
