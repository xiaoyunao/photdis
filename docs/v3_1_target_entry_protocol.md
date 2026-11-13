# Direct observed-context target entry integration

The three-context shared Gaussian proposal pilot passed all34 correction states
and the independent Gaussian-basis crosschecks. Production evaluates every
observed application context directly; there is no context interpolation.
Use the frozen26000 native PSF sample and the3540 candidate identities from the
target-native pilot roles. Compute grzSNR5 admission for all26000 and the exact
brightness/error/QRF candidate admission additionally for3540. Both include
native local-dust covariance and response12/calibration12, with the same fixed
class-conditional kernels, binary fraction and34 luminosity corrections.

Reuse the validated shared N(15,6) numerical proposal, which cancels in each
Gaussian integral. Evaluate five radial bases, seeds319711/319712 and levels
131072/524288/2097152. At every context and envelope require both successive
and independent-seed differences below.0005 and below1% relative to
max(alpha,.001). Retain per-cell difference arrays and pruning bounds for
subsequent population/count sensitivity checks. Require finite selection
probabilities in[0,1]; no clipping, omission of failed objects or weight flattening.
The tiny-entry floor is a numerical comparison convention, not a guarantee
of1% relative accuracy for tiny entries. Posterior/count sensitivity is still
required before accepting a fitted product.

Save every level/seed and failed terminal attempt. Per-object markers permit
exact resume. Split rows deterministically into16 disjoint resource shards,
at most4nodes concurrently and32workers/node; these are resource partitions,
not new scientific samples. All held-out-field comparisons reuse these same
direct entry arrays. No external reference values enter preparation.
