# Size-matched broad population control

The four-arm opened experiment leaves a signed Gaia loss of 0.2194
nat when moving from the full broad fit to the candidate fit at zero
tilt. That contrast combines a change of sample size with the change
in admissible population and its corresponding selection. The
candidate remains below the nominal reference after the separate
branch adjustment. The Pro review requests a size control in this case.

Draw one simple random sample without replacement of 3,540 rows from
the authenticated 26,000-PSF broad roster, in its existing row order.
Use NumPy PCG64 seed 320617 and retain the selected keys, fields and
original design weights. Every row has equal inclusion probability.
Retain all broad fields and apply the existing broad selection
normalization, priors, response12/calibration12, zero luminosity
correction and refined quadrature. Fit both zero and free SED tilt on
the same sampled rows. Reuse the initial experiment's code path
and sampler gates.

On the same opened objects and admission masks, compare the broad
subsample to full broad and compare candidate to the broad subsample,
separately for each tilt mode. Use whole-sky-block paired intervals
for the registered Gaia, class, qnear and conditional giant diagnostics.
Keep the direct original contrasts, sampler status and forecast
numerical flags. This single sample quantifies sensitivity to the
population size and this draw. It leaves candidate versus broad
admission and selection normalization coupled. No final-validation
outcome enters this control or gets opened again.
