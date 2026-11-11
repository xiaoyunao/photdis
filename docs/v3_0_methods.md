# Photdis v3.0: implemented method and interpretation

Status: the physical method is frozen; final production and delivery validation
are still in progress. This document describes the implementation, not scientific
acceptance. The authoritative request is [the full Pro review](pro_review_b59e18b.md).

## Evidence, identities and selection

The native reference inventory contains 3,565,499 rows. Exact Legacy identifiers,
Gaia identifiers, duplicate/conflict flags and verified native morphology are
preserved. Admission requires TYPE=PSF, clean native grz with S/N>=5, |b|>30 deg,
0<=E(B-V)<0.03 and the DECam DR11 system. The resulting inventory has 612,114
objects. Exclusion diagnostics include 148,378 non-PSF rows, 2,930,893 photometric
failures and 90 identity conflicts; these diagnostic masks overlap and must not
be added as sequential exclusions. Missing/unverified morphology never passes.

Primary distance evidence is assigned in advance: qualified SpecDis, otherwise
SPDist, otherwise signed Gaia. Counts are 402,468 / 66,644 / 39,019, respectively;
103,983 inventory objects have no qualified primary distance channel. Alternative
references remain available for consistency checks and are not multiplied as
independent likelihoods. SpecDis BINARY_FLAG is retained without a flag=1 cut:
10,298 qualifying SpecDis objects have a different flag. This removes the old
Gaia-residual-dependent truncation but does not establish independence of teachers.

Training, source calibration, hyperparameter selection, seen development and
prospective validation have separate exact-identity roles. The nominal SED uses
32,768 uniformly identity-ranked sources from 277,897 eligible training sources.
It is a probability sample, not all 612,114 inventory rows. Of those fitted
sources, 5,653 lack reliable class labels and 12,270 lack usable FeH. Missing
labels are marginalized. They do not define an additional physical class.

The fixed QRF comparison uses the same 32,768 training identities and primary
reference posterior-mean labels, with historical fixed QRF hyperparameters. It is
a matched research control, not a claim about rerunning the deployed v1 catalog.
The fixed classifier probes use the 27,115 known-class subset and compare on the
same 42,950 development identities. No reference absolute magnitude, target
spectroscopic class, FeH or Gaia parallax is a photometric prediction feature.

## Reference calibration and empirical SED

For each VAC, 8,192 calibration objects constrain an offset, excess scatter and
broad-tail fraction using signed Gaia observations. The correction convention is
mu_corrected = mu_VAC + offset. Nominal offsets are approximately -0.02191 mag
(SpecDis) and -0.00523 mag (SPDist); excess scatters are 0.08201 and 0.20066 mag.
The broad reference component has sigma=hypot(reported_sigma,2 mag). These are
source-consistency calibrations for Gaia-dependent teachers, not absolute scale
proofs. Thirty-two shared sky-block Bayesian-bootstrap/calibration realizations
propagate finite-reference and common-source uncertainty.

The empirical model is a heteroscedastic Gaussian mixture in intrinsic G,R,I,Z,
W1,W2 flux at 1 kpc and corrected FeH. It has 128 fitted components. Generalized
projected EM integrates source-distance quadrature and propagates measurement
covariance rather than treating derived absolute magnitudes as exact. Missing
coordinates are projected out. Auxiliary class likelihoods support conservative
dwarf/evolved/other labels; missing class likelihood is one. The nominal fit
converged after 43 iterations. The original influence approximation was rejected
for uncertainty propagation; all 32 bootstrap models were fully refitted.

Gaia-only reference quadrature uses the stated EDSD L=1.35 kpc prior and signed
likelihood. VAC quadrature uses a flat-mu measure and the calibrated reference
mixture. These assumptions remain part of the inferred SED scale. A fixed 10%
equal-luminosity binary proxy has flux factor two in both training and inference;
it does not infer mass ratios. Inference adds a fixed 1% broad discrepancy channel,
which is a model-inadequacy component, not a fourth stellar population.

Gaussian latent-flux kernels can have unphysical negative intrinsic support. This
is distinct from legal negative observed flux. Exported posterior-weighted
negative-support diagnostics are model-support proxies, not proof that the latent
physical flux has been constrained positive.

## Native photometry and continuous joint posteriors

The canonical observation order is G,R,I,Z,W1,W2. Legal weak/negative fluxes enter
the native Gaussian likelihood; missing observations contribute no likelihood;
true upper limits use the corresponding censoring probability. Artifact/quality
flags are distinct from low signal-to-noise. Each observed band enters once.
Raw photometry is compared to forward extinction and common calibration/dust
perturbations. There is no second dereddening in the likelihood.

Continuous adaptive distance-modulus quadrature retains near/far and multiple
luminosity branches. The 0–2, 2–5, 5–10, 10–20, 20–50 and >50 kpc bins are output
integrals, not latent distance nodes. Native full log-density grids are preserved
for scores; compressed CDFs alone cannot certify rare conflicting-observation
log scores. Integration, CDF/joint mass, mode omission and Monte Carlo errors are
separate diagnostics. Failed early compression products remain archived; final
native products evaluate all SED kernels with no omitted stellar branch.

The fixed reference joint prior is explicit, including mu~N(15,5^2), component
weights, binary branch and class probabilities. Reweightable joint support,
likelihood/normalization information and model hashes remain on NERSC. An
empirical SED posterior is never labeled a prior-free likelihood.

## Target population and correlated counts

The 26,000-object, 13-field stratified sample supplies a selection-normalized,
design-rescaled target likelihood. Weight normalization preserves the actual
sample's information scale rather than creating millions of independent data.
Three class branches have five overlapping continuous Gaussian radial bases
(mu means 8,11.5,14,16.5,20.5; sigmas 3,1.5,1.4,1.8,3.5 mag), with constrained
within-class SED tilts and field shrinkage. The five bases do not discretize stellar
distance. Global/field/SED-tilt prior scales are 0.7/0.35/0.3 in model coordinates.

Per-object responsibilities are normalized separately at each shared draw before
marginalization. Counts sum class×distance joint probabilities, not products of
marginals. Exact int64 latent realizations and covariance retain conditional
membership noise, population-parameter uncertainty and between-response variation.
All chunks share global parameter indices; latent random draws are independent
across fields. Source-response weights use normalized evidence; tiny weights remain
in log space when evaluating mixture densities.

The candidate catalog uses the explicitly **conditional transferred distribution**
described in [the transfer scope](v3_0_transfer_scope.md). Population parameters are
learned from the stratified target sample, not refitted to all remaining objects.
It is not a full-catalog generative posterior or an intrinsic Galactic density.
New sky receives draws from the frozen hierarchical field distribution. Numerical
compression of negligible global response weights, if used, records total-mass,
count and covariance bounds; those bounds do not by themselves bound log scores.
A concentrated finite response posterior is an unresolved support limitation,
even if integration within that finite prior passes.

Full qualified-catalog counts use unit weights and require no sampling expansion.
The original sampled-parent Horvitz–Thompson diagnostic is kept separate: its
plug-in finite-population design covariance is conditional on fitted marginal
probabilities, not a replacement for full shared posterior covariance or a final
validation sample size.

## Prospective freeze and actual scope

The freeze manifest SHA256 is
`14d3ff1d27e044864970237022a16ac834a195d15d079d3bb7677c7af6309528`.
It locks the physical model, priors, core implementation, metrics and numerical
criteria before opening prospective reference outcomes. Subsequent longer chains
and finer quadrature preserve physical assumptions and failed original products.

The 3,250 original reserved objects are conditional on 13 previously used fields
and historical reference exclusions. They are not an all-sky random validation
sample. New-sky samples contain 2,048 photometrically selected objects in two
previously unused sky blocks. The independent NGC 288 layer has 8,192 photometric
objects selected before membership association. Published Gaia-conditioned
membership p>=0.9 selects 1,442 objects; the independently specified dynamical
scale is 9.03 kpc with -0.56/+0.48 kpc errors. This is one shared system-scale
constraint, not 1,442 independent absolute-distance calibrators.

The candidate envelope has 1,183,990 objects in the existing DECam fields:
native admission plus observed r<20, maximum grz magnitude error<0.1 and fixed reference
color support. There is no Gaia, class, truth-distance or qnear admission cut.
The full 9,390,518-row historical parent retains explicit statuses; the 491,184
northern objects remain unmodeled. This is neither full-DR11 prediction nor a
measured all-DR11 selection fraction. Effective area and a same-selection DR11
denominator remain unavailable. Model-expected far leakage and independently
measured contamination must remain separate.
