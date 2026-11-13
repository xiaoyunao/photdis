# Propagate numerical matrix differences through the target posterior

Matrix-level tests alone do not establish stable inferred counts. Save both
independent Sobol-seed entry matrices and the immediately previous resolution
for every object; retain raw values and all original failures. Numerical sidecars
do not change scientific priors, models, data weights or sample membership.

For each conditional population posterior, evaluate the exact selected-data
log-likelihood ratio for replacing the accepted entry matrix with each actual
independent-seed/previous-resolution matrix. The response term is unchanged;
the ratio is sum_i w_i[log A_i(theta)-log A_alt_i(theta)]. No clipping of entry
probabilities or arbitrary flattening of importance weights is allowed.
Require importance ESS>half the stored draws and split-half log-normalizer
change<.05 before using this numerical transfer. Otherwise preserve the flag
and explicitly refit the unchanged population at refined numerical matrices.

The native-response fine audit replaces responses only for the registered
pilot+128production audit identities. Its likelihood ratio uses
sum_audit w_i[log R_fine_i(theta)-log R_coarse_i(theta)]. This is a sampled
numerical sensitivity probe, not proof that the full26k fine likelihood has
been evaluated. State the exact audited denominator. In audited per-object
predictions use actual fine joint-distance integrals, retaining the true-mu
mapping of the count protocol.

For each accepted transfer compare normalized joint class/distance per-object
probabilities and shared count means under the same identities/draw indices.
Require maximum probability change<.003 and maximum count-mean change<.5.
All fine-grid/seed failures survive in sidecars and the final domain flags;
matrix assembly or a converged sampler alone does not satisfy these gates.
