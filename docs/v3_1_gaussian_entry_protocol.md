# Joint Gaussian-basis entry quadrature

The registered correction freezes within-class kernel proportions. For each
class, Gaussian luminosity convolution can be moved into a Gaussian radial
basis exactly: its mean becomes m-delta and variance becomes s²+sigma².
The same identity applies to native likelihood and conditional entry BEFORE
normalization. It does not permit averaging normalized per-draw posteriors or
applying this correction to a different SED response without calibration.

Numerical sidecar: one scrambled five-dimensional Sobol set samples the
class-conditional kernel index, effective distance modulus and three sequential
truncated Gaussian flux coordinates. Integrate the two binary states exactly.
Keep full covariance, measured-error thresholds, local dust covariance and
original QRF support; use only certified exact support-query shortcuts. Return
the Gaussian survival weight times the exact support indicator. Skipping queries
with survival weight<=1e-14 has a separately recorded upper bound. Integration
weight ESS is not response-posterior ESS or independent sample size.

Before production, audit all three originally frozen error contexts under zero,
point and bootstrap0/15/31 luminosity corrections, with matched local dust.
Also compare zero/point no-local-dust calculations to the preserved median8192
kernel/grid curves. That old calculation retains its failed per-kernel flag;
agreement is an independent-method cross-check, not promotion of the old table.

Fixed levels8192/32768/131072/524288 and seeds319701/319702. Require both successive-
level and independent-seed max absolute changes<.0005 and changes divided by
max(alpha,.001)<.01. The latter is a stabilized error bound, not a relative-error
claim below alpha=.001. Require old-grid class-integral differences<.003.
Store all attempts and stop at the first passing level, with unresolved failures
retained at the fixed bound. No prior, sample, support or scientific tolerance
is selected from recovery outcomes. Downstream actual-posterior/count changes
must still be checked before accepting population fits, especially tiny entries.

This integrates the fixed Gaussian bases, including the reference N(15,5),
without a distance grid. It does not certify arbitrary local-structure priors,
all observation contexts, SED tilts or future response libraries. No new science
fit or final validation outcome is opened by this numerical audit.
