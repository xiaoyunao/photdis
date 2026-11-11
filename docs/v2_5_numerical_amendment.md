# v2.5 posterior quadrature refinement

Initial scientific computation61127456completed; validator61127009passed1958
checks. Initial192catalogs have minimum Sobol-weight ESS5.94 and7catalogs whose
maximum component mean changes>0.01between2^13and2^14points (max0.01470).
This is insufficient for a precise interval claim.

Retain all generated catalogs, QRF predictions, templates, response settings,
parameter prior/bounds, selections and scenario definitions. Archive the initial
population tables/posterior matrices/validation. Re-evaluate the SAME likelihood
with2^18points, compare to2^17; sensitivity runs use2^14instead of2^12.
No refit, selection tuning or outcome-driven model-family choice occurs.

The coarse photometric entrance is not equivalent to the entire empirical
parent support:12248/12288selected synthetic rows have a parent color/magnitude
neighbor within1scaled unit, versus8034/12288in the complementary stress sample.
Keep this distinction and the sparse14–15noise support in the final report.
True interval undercoverage cannot be repaired by merely calling the numerical
refinement a new scientific model; retain any remaining failure.
