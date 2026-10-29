# Bounded metal-poor main-sequence forward-parallax experiment

This is a branch feasibility test using qualified DESI spectra, Legacy DR11
dereddened `grz` photometry, and Gaia DR3 parallaxes. The existing cold-v4
model, its route, and earlier final-test identities are unchanged.

## Sample and likelihood

The two versioned same-star inventories contribute 11,619 unique usable
Gaia objects after a source-ID deduplication check. Five development fields
contain 6,672 stars and five reserved fields contain 4,947. The sample is
restricted to clean, low-dust, conservatively classified main-sequence
stars at `4200≤T_eff<5000 K` with a continuous calibrated DESI `[Fe/H]` and
formal uncertainty. One object with independently discordant DR11 `r`
photometry is retained in the inventory but removed from this trusted core.
The sample retains 81 negative corrected parallaxes in development and 52
in reserved validation. There is no observed-parallax S/N cut in the fit.

We compare two low-degree relations, both with intrinsic `M_g` scatter:

\[
M_g = a + b[(g-r)_0-1] + c[(r-z)_0-0.5]
    + d[(g-r)_0-1]^2 + \epsilon
\]

and the same relation plus `β([Fe/H]+1)`. For each star, the inferred
positive latent parallax is convolved with its Gaussian observed-parallax
error. The `g,r,z` magnitude uncertainties enter through the shared-band
derivatives of `M_g-g_0`; the iron uncertainty and an additional iron-error
floor enter the abundance model. The likelihood is integrated over the
latent luminosity scatter with posterior-centered quadrature, retaining
weak and negative observed parallaxes. All fit choices use development
fields only; reserved fields are scored once with frozen parameters.

## First measured result

| Subset | Stars | No iron: mean log density | One iron term | Gain per star |
|---|---:|---:|---:|---:|
| Development, all | 6,672 | 0.2632 | 0.3771 | +0.1139 |
| Reserved, all | 4,947 | 0.3257 | 0.4396 | +0.1139 |
| Reserved, `[Fe/H]<−1.5` | 144 | −1.0153 | −0.2435 | +0.7718 |
| Reserved, blue and poor | 86 | −0.4006 | −0.1867 | +0.2139 |

All five reserved fields show a positive mean log-density gain. The fitted
iron coefficient is `β=−0.6655 mag/dex`: at fixed optical colors, lower
iron abundance predicts a fainter `M_g` in this restricted spectral sample.
The fitted intrinsic scatter decreases from 0.2925 to 0.2000 mag. Tight and
broad reasonable coefficient priors, and a larger iron-error floor, return
coefficients near −0.665. The poorest blue subset in the development fields
does *not* improve (`−0.318` to `−0.381` mean log density), so the branch
must still be assessed by field, color, error and calibration domain.

The reserved all-star one-iron predictive PIT has 6.39% below 0.05 and
4.06% above 0.95. In the 144 poor-metallicity reserved objects, both tails
are about 11.1%. These are parallax-space diagnostics of the spectroscopic
sample, not a calibrated distance interval for purely photometric targets.

The development iron abundance has `R²=0.358` against the four color-basis
columns: color and iron are related but the abundance retains variation at
fixed fitted color. Leaving each development field out in turn gives
`−0.679≤β≤−0.654 mag/dex` and reserved mean log density of 0.4373–0.4396
for the one-iron model, versus 0.3257 for the no-iron fit. These checks
support identifiability in this bounded spectroscopic sample without making
the coefficient a universal metal-poor main-sequence calibration.

For the entire selected sample, the one-iron predictive distribution expects
77.6 negative parallaxes in development and 46.5 in reserved validation;
81 and 52 are observed. The differences are 0.41 and 0.85 expected standard
deviations, respectively. Comparing only the negative-observation subset's
PIT to an unconditioned distribution would misread the selection itself.

This experiment answers the restricted calibration question: one continuous
abundance coefficient improves the observed-parallax prediction in the
reserved spectroscopic fields. A photo-only application requires a
target-population metallicity distribution; the spectroscopic `[Fe/H]` used
here is unavailable for most intended applications. No new all-sky distance
catalog or nominal interval is adopted.

## Data and code

- [Experiment manifest](../manifests/metal_poor_forward_v1.json)
- [Field, color and negative-parallax diagnostics](../manifests/metal_poor_forward_diagnostics_v1.json)
- [Held-out field and subset comparison](../figures/metal_poor_forward/01_reserved_parallax_logdensity.png)
- [Forward likelihood](../src/photdis/metal_poor_forward.py)
- [Sample and scoring](../scripts/metal_poor_forward_experiment.py)
