# Four-system external scales and shared distance — registered experiment

Registered against b18518b before new scores. Full request: pro_review_b18518b.md.
Configuration: configs/cluster_rgb_scales_v1.json. Frozen model hashes, source
identities, nominal luminosity bins, quality, and all faint/outlying stars stay fixed.
M3/M53 are development diagnostics; NGC5466/4147 are already exposed diagnostics.

## External constraints and finite scenarios

Legacy EBV is raw SFD and its grz coefficients (3.214,2.165,1.211) already
incorporate the SF11 calibration. No additional 0.86 factor is applied to
MW_TRANSMISSION. The old numeric EBV substitutions are coordinate sensitivities.

| Cluster | RR distance conditions | Error treatment and derivative |
|---|---|---|
| M3 | H/Ks PLZ; E=.013, CCM Rv=3.23; AH=.009, AK=.005; FeH=-1.34 CG97 | .017 stat, .036 sys; sys already contains zero/slope/FeH. E response bracket .385–.692, midpoint used, bracket propagated as scenario range. |
| M53 | J/H/Ks PLZ; E=.02, CCM Rv=3.1; A=.018,.011,.007; FeH=-2.06 Carretta09 | .024 stat, .033 sys; same components already included. E response bracket .35–.90, unweighted mean proxy .60. |
| NGC5466 | V Fourier absolute magnitudes, E=0, LMC zero-point anchor | published .09 distance error retained as total; dmu/dE=-3.1; no explicit FeH derivative at fixed Fourier coefficients. |
| NGC4147 | V RRc Fourier distance; adopted E=.01, direct color .008±.002, map .022 | published 19.05±.46 kpc; dmu/dE=-3.1, no explicit FeH derivative. Spectroscopic mean -1.84±.02 uses additional assumed .10 dex scale floor. |

The M3 CG97 abundance is not numerically interchangeable with the RGB UVES
abundance. The .1-dex PLZ displacement experiment assumes the same physical
abundance offset on each scale; it is not a conversion between these scales.
PLZ dmu/dFeH=-.18 is a local representative coefficient (Marconi table 6,
mode/band differences about .01); do not add its variance to published total
distance error a second time. Exact combined RR weights/covariances are not
published here; response brackets remain explicit.

Use five finite scenarios, with no ranking by distance residual:

1. Legacy baseline: original SFD coefficients and published distance.
2. Paper CCM: physical reddening adopted in each distance paper, using
   photon-weighted DECam curves and CCM89, 4500-K Planck SED. RR distance stays
   at its paper value. This is a declared spectral/law approximation, not a
   claim to reproduce exact stellar passband extinctions.
3. SF11-proxy CCM: E=.86 times median raw SFD for each cluster; integrate the
   same law; change RR modulus by minus its band response times E-Epaper.
   The .86 map equivalence is approximate, not an exact universal E conversion.
4. Same as 3 with +.1 dex to RGB abundance and the corresponding PLZ distance
   response for M3/M53. Optical Fourier reference distances have no added
   abundance response. This is a coupled scale sensitivity, not a new estimate.
5. Paper CCM with NGC4147 spectroscopic abundance -1.84 and sqrt(.02²+.10²).

Use the measured individual raw SFD extinctions to undo the original correction,
then the explicitly integrated cluster extinction. Store actual Ag/Ar/Az and
4000/4500/5000-K coefficients. Retain fixed nominal membership/bins even when
shifted references leave support. Report outside-support counts, not clipped scores.
Compute gr/grz center, NMAD, tails, 68/95 coverage and width, log density and bins.
Compute two-color residuals whitened by intrinsic narrow covariance plus full
shared-band measurement covariance. Project onto the whitened local RGB tangent
and its perpendicular; these are diagnostics, not calibrated Gaussian p-values.

For training reference correlation, propagate rho=0,0.5,1 between the *existing*
M3/M53 systematic distance budgets; keep their quoted statistical terms independent.
Report common-mean and difference variances, and analogous abundance/photometric
scale cases. This is a reference-budget diagnostic, not a replacement for the
frozen ensemble's independent-training-scale assumption. A new fit is conditional
on a uniquely justified external revision; conflicting external definitions alone
do not mandate one. The stopping outcome can be remaining scale ambiguity.

## Shared-distance prototype

Use all fixed members, plus nested first 10 and 20 by source ID. Broad uniform
mu prior [10,22], identical for all systems; no RR distance enters the likelihood
or prior. Integrate the piecewise-linear frozen forward likelihood exactly in
latent Mg over [.5,3], with the full covariance of (g,g-r,r-z). Include the fixed
robust mixture. For each shared QMC nuisance draw, multiply member likelihoods
FIRST; then marginalize FeH, physical dust, band zero points and model realization.
The target scale is the baseline catalog scale; local physical-dust derivative is
CCM4500, sigma(E)=.005, band zero sigma=.01. These uncertainty sizes are assumptions.
Use 512 scrambled Sobol draws and compare the nested first 256; report nuisance
ESS and shifts in quantiles. Low ESS or large differences mark numerical resolution
limits and motivate a bounded extra draw run, not changed scientific priors.

As controls compute the deliberately incorrect product of separately marginalized
members, and the proper shared-nuisance *no-color* likelihood. Report posterior
widths vs N and a synthetic exactly soluble common-offset test. Reference-distance
selection constrains apparent magnitude endpoints: the no-color control can be
very informative. This prototype is conditional on that selected luminosity range;
it is not independent recovery or a field-star estimator. Freeze these choices
before running, inspect numerical convergence, and retain unresolved limitations.

## Sources

- [Legacy catalog conventions](https://www.legacysurvey.org/dr10/catalogs/)
- [M3 RR PLZ](https://arxiv.org/abs/2008.11745), [M53 RR PLZ](https://arxiv.org/abs/2012.13495)
- [NGC5466 Fourier distance](https://arxiv.org/abs/0711.4027), [NGC4147 Fourier distance](https://arxiv.org/abs/1803.05313)
- [NGC4147 spectroscopy](https://arxiv.org/abs/1605.03408)
- [Marconi PLZ coefficients](https://arxiv.org/abs/1505.02531)
- [CCM89 law](https://articles.adsabs.harvard.edu/pdf/1989ApJ...345..245C)
- [DECam throughput provenance](https://speclite.readthedocs.io/en/v0.15/filters.html)

Deliver a four-cluster scale table, fixed response/geometry figure, shared-distance
and selection control figure, machine-readable results, tests and final decision.
Compute in NERSC allocation; synchronize only code/docs/small manifests/figures.

## Numerical integration addendum (after the 512-draw diagnostic)

The initial primary integrals have nuisance ESS 2.6–17.6; M53's median changes
by .082 mag between 256 and 512 draws. Invoke the registered bounded refinement
to 2048 nested draws, retaining separate 512 files. This is numerical resolution,
not a revision of the prior, sample or model.

Also cross-check the integral using an exact Gaussian collapse and posterior
sampling. Let the common band correction be b and abundance displacement dz.
The equivalent Gaussian variables are dg=b_g and
d=(b_g-b_r,b_r-b_z)-beta_Z dz. Write theta=mu-dg. The likelihood becomes
L(theta,d,k) for frozen realization k. Conditional dg|d,k is Gaussian; integrate
it analytically, including the [10,22] mu-prior bounds. The remaining target is
the member-product likelihood times N(d|0,Scc,k), the conditional prior-bound
mass and the uniform discrete realization prior. This identity is checked
against the original full-nuisance likelihood before use.

Use four independent Metropolis chains, 3000 discarded warmup steps with proposal
covariance adaptation, then 12000 steps per chain with fixed proposals. Continuous
proposals are symmetric; discrete model proposals are uniform with a reversible,
unit-Jacobian color translation at Mg=1.75. The same target priors and all 48
frozen realizations remain. Starting distance uses median(g)-1.75, not RR distance.
Marginal mu densities average the exact conditional Gaussian dg distributions;
the common grey uncertainty is never reduced by member count.

Check rank-normalized and folded split Rhat (<1.01), bulk/tail ESS (>400) for
theta, both color shifts, conditional mu mean and log target; report model
occupancies and per-chain distance summaries as additional diagnostics. If a
run fails, extend only its numerical sampling, preserving the failed diagnostic.
The interpretation of the selected luminosity support and the no-color control
stays the same. [Diagnostic method](https://arxiv.org/abs/1903.08008).
