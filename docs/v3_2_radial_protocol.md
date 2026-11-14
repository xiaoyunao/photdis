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

Recovery uses identical broad and local arms at the 0.01-mag likelihood grid.
Entry is calculated directly under the local covariance for each context,
using 256 draws on 0.05-mag nodes and 512 on 0.025-mag nodes with independent
scrambles. Retain both and test at the same fitted draws. Record an absolute
objective check of 0.05 separately from the relative objective span check
of 0.05. Entry-curve maximum difference must be below 0.003 and maximum
expected count change below 0.5. A report's conditional accepted subset
also requires Rhat < 1.05, minimum ESS >= 100 and zero divergent transitions.
Report the full fitted subset alongside it and show registered denominators.
The first three recovery runs use catalogs 0, 128 and 256 to exercise fixed
model, fixed empirical and prior-predictive paths before the wider array.

New curve workers save each completed native PSF row to a checkpoint
bound to the simulation inputs and entry integration identity. Reuse
only rows with matching object ID and output digest. A missing receipt
leaves the row pending. The repeat cache and direct/catalog likelihood
paths each use their own checkpoint directories. If an interrupted fit
already created a catalog output directory, preserve that attempt before
running again; verified curve checkpoints remain available.
