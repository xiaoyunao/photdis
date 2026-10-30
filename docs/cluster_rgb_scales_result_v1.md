# Four-cluster external scales and shared-member distance

Status: completed bounded audit and shared-distance prototype; numerical limits retained below.
Protocol: [registered experiment](cluster_rgb_scales_protocol_v1.md).

## What changes and what remains a frozen reference

The previous NGC4147 EBV replacements kept the Legacy coefficients and RR distance
fixed. Their numerical results remain in the historical manifest; the report now
calls them raw-SFD coordinate sensitivities. Legacy's SFD field and a paper's
physical E(B-V) cannot be substituted numerically without specifying the band law.
For the NGC4147 median, 0.86 × 0.0257446 = 0.0221403, already close to the cited
SF11 map value .022. The factor is an approximate convention comparison, not an
extra correction to the catalog transmission. [Legacy conventions](https://www.legacysurvey.org/dr10/catalogs/)

New comparisons use 422 M3, 250 M53, 93 NGC5466 and 42 NGC4147 members.
The original model hashes and source identities remain fixed, as do nominal
Mg bins and quality cuts. M3/M53 are development diagnostics; both other systems
have already been exposed. No large-residual star is removed.

The physical scenarios integrate CCM89 through DECam curves for a 4500-K Planck
spectrum, with 4000/5000-K coefficient checks. This is an explicit low-complexity
SED approximation. Neither the Planck approximation nor .86 E_SFD supplies a
unique measured extinction for each RGB star. The manifest records actual Ag/Ar/Az
and the adopted RR reference for each case. [CCM89](https://articles.adsabs.harvard.edu/pdf/1989ApJ...345..245C),
[throughput provenance](https://speclite.readthedocs.io/en/v0.15/filters.html)

## External reference dependencies

M3's distance combines H/Ks PLZ information, using a CG97 abundance of -1.34;
the RGB model uses a separate UVES reference. Treating these two numerical
abundances as the same scale would introduce a spurious correction. The paper's
H/K extinctions give a local reddening-response bracket; its combined band weights
are not reconstructed here. The quoted .036-mag systematic budget already
includes abundance and calibration contributions. [M3 study](https://arxiv.org/abs/2008.11745)

M53 uses J/H/Ks PLZ and FeH=-2.06 on the cited Carretta scale. Its .033-mag
systematic error already contains zero-point, slope and abundance terms. Our
coupled +.1-dex sensitivity shifts both RGB FeH and PLZ reference, using a
representative local derivative -0.18 mag/dex; it does not add another independent
abundance variance to the published total distance budget. Band/mode coefficients
vary slightly, and the response bracket remains explicit. [M53 study](https://arxiv.org/abs/2012.13495),
[PLZ coefficients, table 6](https://arxiv.org/abs/1505.02531)

NGC5466 and NGC4147 use Fourier V-band absolute-magnitude calibrations. Holding
the Fourier measurements fixed, a physical reddening change alters the reference
modulus by -3.1 delta E; there is no explicit FeH term in those adopted M_V
formulae. FeH-dependent bolometric corrections used elsewhere in the papers do
not introduce a FeH derivative into this distance formula. Their published distance
errors are kept as total errors because a full covariance decomposition is unavailable.
[NGC5466](https://arxiv.org/abs/0711.4027), [NGC4147](https://arxiv.org/abs/1803.05313)

The NGC4147 spectroscopy alternative is -1.84 with .02 dex mean error plus an
explicitly assumed .10-dex cross-scale floor. This floor is not a measured error
from that paper. [Spectroscopic study](https://arxiv.org/abs/1605.03408)

## Shared versus differential reference budgets

The table changes only the correlation of the existing M3/M53 systematic budgets,
retaining each quoted marginal variance and independent statistical errors:

| Systematic correlation rho | Sigma of mean modulus | Sigma of modulus difference |
|---|---:|---:|
| 0 | .02850 | .05701 |
| .5 | .03331 | .04541 |
| 1 | .03750 | .02956 |

Units are mag. A common calibration error increases the absolute mean uncertainty
while reducing the differential error. The manifest also records finite common
FeH-scale and per-band zero-point cases. These are reference-budget sensitivities;
they are not newly calibrated RGB posterior uncertainties. In particular, the
frozen 48-model ensemble still uses its original independent training-scale draws.
There is no unique externally mandated replacement of both training references,
so this round does not refit or reweight that ensemble using exposed residuals.

## Common-distance likelihood and its selection control

The likelihood integrates each member's latent M_g over [.5,3] analytically along
the three straight segments of the frozen monotone relation. It uses the full
covariance of (g,g-r,r-z), including the shared g and r measurement terms. The
95/5 robust component is marginalized separately for each member. The 5% weight
continues to be a residual-model assumption, not a measured class fraction.

At each common draw of FeH, dust, band zero points and model realization, member
likelihoods are multiplied. Only then are these shared nuisance parameters
integrated. No target RR distance or RR distance error enters the likelihood or
the common uniform mu prior [10,22]. They enter the comparison after inference.

The no-color model has exactly the same latent luminosity interval and shared
g calibration uncertainty. Because these members were selected using an external
distance and stage information, that interval itself supplies substantial distance
information through the apparent-magnitude endpoints. Close agreement with the RR
distance must therefore be compared to this control before attributing it to colors.
These calculations do not solve a distance-blind membership or stage-selection problem.

The deliberately incorrect comparison marginalizes shared nuisance separately
for each member and then multiplies. An exact synthetic Gaussian check verifies
the distinction: with independent sigma=.1 mag and shared tau=.2 mag, the correct
variance is tau²+sigma²/N, while the incorrect order gives (tau²+sigma²)/N. At
N=20 their standard deviations are .2012 versus .0500 mag; the correct N→infinity
limit is .2 mag. This analytic demonstration is separate from the selected RGB
samples' N-dependent posterior widths.

## Measured fixed-model responses

All centers below are median predicted minus reference distance modulus (mag).
Coverage counts inclusion of each scenario's reference point.
The paper-CCM cases use uniform cluster reddening; the Legacy baseline retains
its per-object map pattern. Thus these are fully stated reference scenarios,
not pure zero-point translations. No scenario is selected by these scores.

| Cluster | N | Legacy center | Paper CCM center | SF11-proxy coupled center | Proxy +.1 dex center |
|---|---:|---:|---:|---:|---:|
| M3 | 422 | +0.0392 | +0.0242 | +0.0571 | +0.0186 |
| M53 | 250 | -0.0323 | -0.0437 | -0.0033 | -0.0440 |
| NGC5466 | 93 | +0.0435 | +0.3664 | +0.1194 | +0.0648 |
| NGC4147 | 42 | -0.2051 | +0.1018 | -0.1239 | -0.1818 |

The M53 proxy cases move one fixed member's reference beyond the trained [.5,3]
range. That member stays in the counts, tails and coverage; the all-star log score
is undefined for those cases. Interior-only scores and all fixed-bin metrics are
explicitly separate in the manifest.

The actual extinction and coupled distance values are:

| Cluster | Scenario | Ag | Ar | Az | RR modulus |
|---|---|---:|---:|---:|---:|
| M3 | legacy | 0.04323 | 0.02912 | 0.01629 | 15.04100 |
| M3 | paper_ccm | 0.04816 | 0.03513 | 0.01998 | 15.04100 |
| M3 | sf11_proxy_ccm | 0.04285 | 0.03126 | 0.01778 | 15.04177 |
| M53 | legacy | 0.06675 | 0.04497 | 0.02515 | 16.40300 |
| M53 | paper_ccm | 0.07146 | 0.05166 | 0.02902 | 16.40300 |
| M53 | sf11_proxy_ccm | 0.06382 | 0.04614 | 0.02592 | 16.40428 |
| NGC5466 | legacy | 0.05499 | 0.03704 | 0.02072 | 16.02000 |
| NGC5466 | paper_ccm | -0.00000 | -0.00000 | -0.00000 | 16.02000 |
| NGC5466 | sf11_proxy_ccm | 0.05258 | 0.03801 | 0.02135 | 15.97438 |
| NGC4147 | legacy | 0.08274 | 0.05574 | 0.03118 | 16.39947 |
| NGC4147 | paper_ccm | 0.03574 | 0.02583 | 0.01451 | 16.39947 |
| NGC4147 | sf11_proxy_ccm | 0.07910 | 0.05719 | 0.03213 | 16.36184 |

M3's paper and map proxy produce only small reference-distance changes because
its RR distance uses H/Ks. The NGC5466 proxy changes the V reference by -0.0456
mag relative to its published zero-reddening assumption; NGC4147 changes by
-0.0376 mag relative to E=.01. This distinction was missing from the old
coordinate-only comparison. Band-weight brackets, SED responses and FeH offsets
are retained in the result manifest, rather than folded into a falsely precise
single conversion.

### NGC4147: same 42 members

| Scenario | gr center | grz center | grz NMAD | grz tail >.5 | grz 68/95 cover | grz W68 | grz mean log density |
|---|---:|---:|---:|---:|---:|---:|---:|
| legacy | -0.1405 | -0.2051 | 0.1874 | 19.0% | 69.0%/88.1% | 0.66 | -0.5456 |
| paper_ccm | +0.1404 | +0.1018 | 0.1352 | 9.5% | 81.0%/92.9% | 0.66 | -0.3312 |
| sf11_proxy_ccm | -0.0192 | -0.1239 | 0.1808 | 14.3% | 73.8%/92.9% | 0.66 | -0.3942 |
| sf11_proxy_ccm_feh_plus_0p1 | -0.0877 | -0.1818 | 0.1919 | 19.0% | 69.0%/88.1% | 0.68 | -0.4494 |
| paper_ccm_spectroscopic_4147 | +0.2534 | +0.1916 | 0.1361 | 9.5% | 66.7%/92.9% | 0.61 | -0.4108 |

The paper-reddening case changes the center's sign. Its better exposed-sample
NMAD does not establish its physical correctness. The spectroscopic-abundance
alternative changes that center again to +.192 mag. The fixed faint-minus-bright
contrast changes from .330 mag (Legacy) to .216 (paper CCM) or .310 (map proxy):
reference changes do not erase the luminosity dependence.

### Two-color geometry

The median signed perpendicular projections (narrow-model plus measurement
covariance units) are:

| Cluster | Legacy | Paper CCM | SF11 proxy | Proxy +.1 dex |
|---|---:|---:|---:|---:|
| M3 | +0.020 | +0.196 | +0.166 | +0.180 |
| M53 | -0.030 | +0.212 | +0.161 | +0.197 |
| NGC5466 | -0.089 | -0.251 | +0.068 | +0.098 |
| NGC4147 | +0.604 | +0.540 | +0.837 | +0.877 |

For NGC4147, the map-proxy case improves the one-dimensional center but increases
the perpendicular median from .604 to .837. Its median color residual is
approximately (+.0006,-.0157) mag: a favorable distance change does not simultaneously
align both colors. The paper and spectroscopy variants give perpendicular
medians .540 and .504. These standardized diagnostics are not Gaussian significance
estimates for the whole cluster, and no sqrt(N) reduction of reference errors is used.

The existing external information therefore does **not identify a unique joint
revision** of both training systems and the targets. We retain the frozen model,
report the scenario dependence, and end the scale audit here. No residual-selected
reddening, common offset, new quality cut or new regression is adopted.

## Shared-distance measured results

The registered prior-draw integration is difficult for a whole cluster: the
512-draw primary grz nuisance ESS is only 2.6–17.6. A nested 2048-draw check
raises full-sample ESS to 13.6–59.6 but still shifts some medians and widths.
Those products remain available as a numerical diagnostic.

The primary integral therefore also uses an exactly equivalent Gaussian
collapse, followed by posterior-directed sampling. The common grey shift is
integrated analytically after conditioning on the color shifts, so its variance
persists as member count grows. Four independent chains use discarded warmup
and fixed production proposals, with rank/folded Rhat and bulk/tail ESS checks.
The first runs expose slow distance/color-ridge mixing; a reversible
unit-Jacobian proposal and a bounded 48000-step refinement address the failed
subsets without changing the scientific target. [Diagnostic method](https://arxiv.org/abs/1903.08008)

The no-color control is numerically stable: its 256-to-512 median changes are
below .00005 mag for all four systems. The gr control has .0018–.0061-mag
changes and nuisance ESS 45–140; it remains a quadrature comparison, separate
from the primary grz convergence assessment.

### Full-member common-distance prototype

All moduli and widths are in mag. Intervals are conditional model intervals,
not externally calibrated coverage statements. RR values enter only this comparison.
M3/M53 are in-sample development controls; NGC5466/4147 are exposed external diagnostics.

| Cluster | N | Shared grz median | 16–84% interval | Offset from RR | No-color W68 | Shared grz W68 | Wrong-independent W68 |
|---|---:|---:|---|---:|---:|---:|---:|
| M3 | 422 | 15.255 | 15.131–15.334 | +0.214 | 0.338 | 0.203 | 0.042 |
| M53 | 250 | 16.432 | 16.403–16.470 | +0.029 | 0.087 | 0.068 | 0.034 |
| NGC5466 | 93 | 15.986 | 15.911–16.124 | -0.034 | 0.281 | 0.213 | 0.086 |
| NGC4147 | 42 | 16.414 | 16.333–16.517 | +0.015 | 0.197 | 0.184 | 0.038 |

For NGC4147, using shared parameters gives W68=.184 mag; incorrectly
independently marginalizing each star gives .038 mag, a factor of 4.8 narrower.
Its no-color control already has W68=.197 mag. This close control is direct
evidence that the reference-selected luminosity range supplies much of the
common-distance constraint. NGC5466 narrows from .281 to .213 mag with colors.
The pooled M3 center remains +.214 mag from its reference: pooling members
does not establish a corrected external distance scale. Its reference lies
inside the model 95% interval, but outside its 68% interval.

M3/M53 reuse training members and the original calibration ensemble. The
ensemble and the target reference priors are not a reconstructed joint
hierarchy of training and target scales; these two rows are numerical and
development diagnostics, not independent validation of their uncertainties.

### Numerical endpoint

| Cluster (all members) | Max parameter Rhat | Log-target Rhat | Minimum bulk/tail ESS, all checks | Chain-median range | Full registered check |
|---|---:|---:|---:|---:|---|
| M3 | 1.0044 | 1.0029 | 534 | 0.0149 | pass |
| M53 | 1.0029 | 1.0107 | 447 | 0.0025 | not passed |
| NGC5466 | 1.0011 | 1.0122 | 434 | 0.0082 | not passed |
| NGC4147 | 1.0004 | 1.0009 | 2513 | 0.0049 | pass |

All 10/20-member subsets pass after the registered numerical refinement.
The full M53 and NGC5466 distance/color parameters pass, but the log-target
Rhat remains 1.0107 and 1.0122, above the registered 1.01 threshold. Their
full joint distributions therefore remain numerically provisional. We retain
the failed flags, raw chains and occupancy checks, and end the bounded run
after this refinement; we do not relax the threshold or report complete convergence.
Across full samples the four chain-median ranges are .0025–.0149 mag.

The member-count experiment uses nested source-ID subsets, so different N
also means a different apparent-luminosity spread. It is not a pure N-scaling
law. The exact Gaussian benchmark demonstrates the asymptotic shared-error
floor; in the real model the residual conditional common-g sigma remains
.0117–.0205 mag across sampled calibration realizations. This analytic shared
component remains in every posterior and is never divided by sqrt(N).

### Decision and deliverables

The bounded external-scale audit is complete. Existing sources do not select
a unique coherent change of both training and target references. Keep the
frozen RGB relation and the explicit scenario spread. The shared-distance
algorithm and its negative controls are implemented; its results remain a
selection-conditional prototype with the numerical flags above. No new
field-star product, reference offset, membership cut or regression is adopted.

Before treating a pooled cluster distance as a calibration result, the next
scientific requirement is a specified membership/stage/selection likelihood
and externally constrained reference covariance. More members alone cannot
remove the measured selection and shared-scale dependence. This round ends
here for review rather than expanding the sample or tuning the exposed clusters.

- [Coupled external-scale response](../figures/cluster_rgb_scales_v1/01_scale_response.png)
- [Two-color residuals](../figures/cluster_rgb_scales_v1/02_color_residuals.png)
- [Shared distance and controls](../figures/cluster_rgb_scales_v1/03_shared_distance.png)
- [Member count and common uncertainty](../figures/cluster_rgb_scales_v1/04_member_count.png)
- [Full small result manifest](../manifests/cluster_rgb_scales_v1.json)

Validation: 29 focused tests pass, including exact likelihood integration,
Gaussian collapse, covariance propagation, shared-vs-independent error floors,
negative support control, sampler recovery and reversible proposal Jacobian.
Four figures visually inspected. Science runs used allocated NERSC nodes.
Final numerical jobs 60538646/648/772/787 and summary 60538545 all completed
with exit code 0; large arrays and chains stay in NERSC data/cluster_rgb_scales_v1.
Initial and refined numerical results remain in the aggregate with input/model
identity hashes and DECam throughput provenance.
