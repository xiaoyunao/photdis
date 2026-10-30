# Four-cluster external scales and shared-member distance

Status: allocated calculations pending; this file records the interpretation
framework and will receive the measured results before the milestone closes.
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

## Measured results and decision

Pending allocated response and four shared-distance calculations.
