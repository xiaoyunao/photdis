# Pre-run luminosity trial details

This freezes the numerical implementation of protocol D before its production.
It is a VAC-conditioned luminosity-scale calibration, not a QRF retraining.
Evaluate the original v2.5 paired identities, retaining each reused ordinary
control and all its poor-star partners as one evaluation group. Calibration
requires original split=1 and excludes the entire union of evaluation identities,
QRF fitting identities, non-PSF/missing TYPE, identity conflicts and nonstellar
risks. Retain all signed Gaia measurements with eligible astrometric quality.
For each VAC separately use valid positive reference errors and at most 512
identity-hash-ranked cal objects per branch (ordinary/poor); no Gaia outcome rank.

mu_true = mu_VAC - source_offset - poor_indicator * poor_offset + Normal(0,s),
with the VAC Gaussian mu error included in quadrature. Ordinary s=0; poor s is
additional dispersion. Source_offset and poor_offset each have uniform priors
on [-1,1] mag, 41 grid points; s has 13 uniform points on [0,.6] mag. Positive
offset makes the absolute magnitude fainter and the inferred distance smaller.
The 41x41x13 joint posterior uses the complete signed Gaia likelihood and 32-node
Gauss-Hermite integration of lognormal predicted parallax. Where positive Gaia
likelihood is narrower than the lognormal, integrate in the observed-Gaussian
coordinate instead; this changes quadrature, not admission or the likelihood.
Independent adaptive-convolution tests cover this numerical switch. No parallax inversion.
Ordinary objects constrain a nuisance source offset, rather than being presumed
to define an absolute scale. Source-conditioned posterior predictive evaluations
are distinct from photometric measurements; VAC/Gaia dependence and conditional
selection are limitations, not corrected by this simple model.

Pure grz demonstration: use the same frozen 256cal SEDs, replacing their teacher
mu with the particular VAC where qualified, and retain chosen teachers elsewhere.
Fixed Gaussian kernels have color standard deviation .08 mag and absolute-Mg
standard deviation sqrt(.2^2+teacher_sigma^2). Native shared-g covariance uses
the corrected v2.3.1 formula. Template priors use v2.5 eligible group fractions;
missing classes remain soft. This fixed kernel is a deliberately simple transfer
demonstration, not a new searched family or a replacement for QRF/XD. Draw the
latent template AND class (no evaluation FeH at inference), then propagate the
joint source/poor/scatter posterior only for VAC-supported ordinary/poor branches.
Other/evolved and unsupported teachers stay unchanged. Save per-object four-group
by six-distance-bin mass, with far tails, predictive PIT/log score and 68/95%
intervals. 4096 draws per object, native observations held identical before/after;
ordinary nuisance and poor correction have shared posterior draws. Report both
source-conditioned and pure-photometric results against the same reference-
qualified evaluation identities, with reused-control bootstrap, plus the frozen
QRF metrics. Do not infer a universal metallicity correction from this trial.
