# Matched local-dust selection sidecar

Native `kernel_loglike` adds the mean-linearized covariance from local
sigma_EBV = 0.1 EBV. The old selection tables omit this term. Preserve those
original approximations and all their numerical flags. A temporary integration
model adds the identical rank-one covariance to the six native flux dimensions;
an analytic test reproduces native likelihoods including binary states and
nonzero global calibration. This temporary model is never a new fitted SED.

Audit the change on the fixed 15 radial and three application contexts before
interpreting its scientific impact. Old radial controls remain identifiable;
small impact cannot establish exact equality or excuse a failed fit.

For hard candidate support, use all three original min/median/max error context
identities, including the max context even if its slow original run times out.
Reconstruct its context from the frozen application rows, without reference
outcomes. Retain exact original brightness/error cuts and QRF support. Use the
validated exact query backend and bounded pruning. Start a separate matched
512-point baseline, then 2048, 8192, 32768 with seeds317101/317102 and .05mag grid.
The old no-dust8192 failure remains unresolved and is not silently replaced.
Both previous-level and independent-seed maximum component/basis differences
must be <.003, followed by a .025mag same-seed check <.003. Stop at the first
passing level or retain a failure at32768. No tolerance or support cut changes.
Per-kernel checkpoints may resume exact inputs across allocations.

These response12 curves initially omit the newly fitted luminosity correction.
Before corrected population fitting, apply the same class-conditional luminosity
operator to likelihood AND entry curves, including shared calibration draws.
An accepted three-context pilot alone does not validate full-cohort interpolation.
No full candidate population or final validation claim follows from this audit.
