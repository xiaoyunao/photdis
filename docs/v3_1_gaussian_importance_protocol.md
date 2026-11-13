# Shared numerical proposal for Gaussian-basis entry

The joint-QMC pilot passed all17tasks but evaluates each correction and basis
separately. This sidecar reuses one proper N(15,6) numerical distance proposal
for all34 zero/point/bootstrap corrections and all Gaussian bases. The proposal
is not a scientific prior: its density divides out separately in every integral.
All scientific means/widths, covariance, selection cuts and class-kernel weights
remain fixed. Binary states remain exactly summed.

For each class generate the same kernel/radial/truncated-flux Sobol variables
once; retain exact QRF support and Gaussian survival weights. Multiply by each
transformed target Gaussian density divided by N(15,6). The log ratio is a
concave quadratic because every registered target width is<6. Record its exact
global maximum and multiply the skipped-query bound by that maximum. This
certifies only the prior ratio/pruning operation, not QMC integration accuracy.
Integration-weight ESS is a quadrature diagnostic, never response-posterior ESS.

Before production, compare all three original contexts and all34 correction
states at131072/524288/2097152 nodes, seeds319711/319712. Require successive and
independent max changes<.0005 and differences/max(alpha,.001)<.01. Stop at first
passing level; preserve all failures. Cross-check the five shared states of each
context against the accepted original joint-QMC pilot at<.001, the sum of the two
.0005 numerical budgets. Original failed per-kernel and successful class-level
integrations remain separately identified. No error-context interpolation is
introduced, and no population or external-reference outcome is read.
