# Matched response development numerical sidecars

The original64-response protocol, predictions, entry integrals and flags are
immutable. Response0 has three observed development numerical failures. This
amendment changes numerical integration only, never the model, calibration,
reference likelihood, prior, identities, ensemble weights or admission rules.

For each response retain all256development objects. Originally passing objects
keep their original coarse/fine pair. For failures evaluate the same native
likelihood and N(15,5) reference prior on the union of the original fine grid and
global grids over[-20,50] at0.005,0.0025,0.00125mag. Compare the previous actual
fine evaluation with the new level. Stop at the first pair passing the original
joint probability0.003, quantile0.03mag, log evidence0.02, source/Gaia logscore
0.02 and Gaia PIT0.003 gates. Preserve every attempted level and original flags;
bounded failures remain in the denominator and invalidate affected diagnostics.

All64sidecar inventories are required before the revised response report runs.
The report authenticates the selected pair and uses it consistently for the
per-response and mixture development comparisons. Original and unresolved flags
are exported separately. Target evidence, entry integrals and weights continue
to use their original authenticated products; this amendment does not authorize
target refinement, flattening weights or new validation outcome access.
