# v3.0 target-population transfer: exact scope

The 26,000-object stratified target sample supplies the selection-normalized,
design-rescaled population likelihood. Its observed-star joint probabilities and
latent counts integrate the same fitted shared parameters. This is the primary
target-population inference experiment. The case weights are normalized to the
actual sample size; they do not manufacture millions of independent observations.

The qualified candidate catalog and prospective validation predictions additionally
receive a **conditional transfer product** from that learned population. Its joint
distribution is explicitly

\[
q(\eta\mid Y_{\rm sample})\prod_i
p(z_i\mid y_i,\eta),
\]

where the same response/calibration/population draw \(\eta\) is used for all
objects and chunks. This retains shared covariance and latent membership
uncertainty. It is a modular conditional distribution: it **does not refit the
population on the remaining catalog**, and must not be described as the full
generative posterior conditioned on all those additional observations. Its count
intervals require the same recovery and external calibration qualification as its
per-star predictions. They are neither intrinsic Galactic density intervals nor
proof of independently calibrated response uncertainty.

An actual additional-catalog Bayesian update instead multiplies the common draw
weights by each new object's selection-normalized marginal likelihood. The
independently enumerated tests in `test_v3_catalog_conditioning.py` verify this
operation, its shared count covariance, and exclusion of already fitted identities
from a second likelihood contribution. The implementation exposes importance-weight
collapse. Existence of this operation is not evidence that a large-catalog update
has numerically converged. In particular, the stricter candidate cuts also require
their own selection normalization before such a full refit could be claimed.

For previously unseen sky fields, conditional transfer draws field deviations from
the frozen hierarchical hypermean and field dispersion. The offsets are common to
all objects of that field. A new field is not silently assigned an existing field's
fitted population. Native reference and transferred products preserve every SED
kernel, signed optional flux, full distance tails and the actual shared calibration
draw. Prior sensitivity is exported continuously as changes in near probability and
median modulus. The `prior_sensitive` flag uses absolute near-probability change
greater than 0.05; it is descriptive and is not a validated quality cut.

Release tables must keep three things separate: observed-sample joint counts,
sampling-design-expanded parent estimates, and conditional transferred candidate
counts. None of their uncertainty conventions may be substituted for another.
Finite response-prior concentration must be reported even when its numerical
integration passes. This transfer limitation is part of the v3.0 research product,
not evidence for deployment or a claim of universal distance calibration.
