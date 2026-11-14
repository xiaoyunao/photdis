# v3.2 narrow-structure recovery protocol

This experiment is independent of the population tilt and model choice.
Fixed truth crosses pre-entry local fraction 0.1/0.3/0.6/0.9, width 0.08/0.25
mag, near/deep five-basis background and SED source model/empirical.
Eight replications per cell give 256 catalogs of 128 admitted objects at
a true structure location of 4 kpc. A further 32 model-matched catalogs
draw the latent vector from the original standard-normal prior. These
prior-predictive catalogs answer a calibration question with random truth.
Conditional fixed-truth coverage answers a separate question.

All measurement contexts and empirical templates originate in authenticated
native DR11 PSF tables. Eligible contexts have grz and a measured-band
mask supported in each empirical class. Both source arms use the same
context pool and the same 128 contexts by repeat. Repeated context or
empirical source identities are not new real-world validation objects.
Generation holds contexts and band masks fixed, resampling latent values
and noise per context until the observed grz S/N cut passes. The realized
selected membership fraction can differ from the generating fraction.

The model arm generates exactly the inference model's kernel mixture,
binary brightening and mean-linearized local dust covariance at fixed
context. The empirical arm uses the frozen teacher-scaled SED library,
with its inherited empirical SED limitations and no additional dust draw.
The selection denominator for recovery uses the same nominal inference
model and grz threshold. Direct integration in each context must include
the same local dust covariance as its likelihood. Check two entry scrambles
and coarse/fine distance grids before interpreting recovery.

For the structure arm retain every MAP and chain status. Compare position,
width, pre-entry fraction and realized selected count with their true
values. Retain interval lengths and separate binomial intervals for
coverage estimates across simulation repeats. Also report joint class by
distance counts. Preserve fit and numerical failures with denominators.
Use the same registered prior as v3.1 to test shrinkage. Keep the original
v3.1 failed runs and the NGC288 numerical diagnostics separate.
