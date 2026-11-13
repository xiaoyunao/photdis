# True-distance counts under the fixed-kernel correction

The native likelihood coordinate is v=mu-delta-t, with Gaussian t of width
sigma_corr. After multiplying by one scientific radial Gaussian N(m,s),
the effective native Gaussian has mean m-delta and variance s^2+sigma_corr^2.
Conditional on v, true mu has mean
m+s^2/(s^2+sigma_corr^2)*(v-(m-delta)) and variance
s^2*sigma_corr^2/(s^2+sigma_corr^2). Integrate its true-distance bin mass
against the native likelihood and effective Gaussian. Do not bin v as if it
were true mu, and do not multiply marginal class and distance probabilities.

For zero correction scatter, true mu=v+delta; split the piecewise-linear
trapezoidal density exactly at shifted bin boundaries. All bin weights are
nonnegative and sum to the original integration weights. For positive scatter,
normal-bin probabilities are evaluated in log space, including extreme tails.
Use the existing0–2/2–5/5–10/10–20/20–50/>50kpc bins; qnear sums the first
three bins and means P(d<10kpc). Every correction state retains one shared
population/field posterior draw index for all objects.

The numerical integral must sum back to the unbinned class/basis response.
Fine-grid checks remain required for final probabilities and count means.
Analytic Gaussian-likelihood tests check both signs of the luminosity shift,
zero/nonzero scatter, log likelihoods near-20000, tail bins and nonuniform
grid mass conservation. These are mathematical checks, not calibration evidence.

Final counts are conditional on the observed sample and stated prior/selection.
Preserve conditional categorical variance separately from shared population,
luminosity and response uncertainty. Evidence-weighted finite bootstrap results
and equal-bootstrap cut sensitivity remain distinct. No effective-area or
independently measured physical-population interpretation is introduced.
