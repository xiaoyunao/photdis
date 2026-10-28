# Frozen cold-v4 foreground-extinction sensitivity

## Setup

This experiment used the three already scored native final fields, their
original high-quality Gaia cold-anchor identities, the frozen forests and
the `grz_clean` plus WISE S/N≥20 route. Slurm job 60462368 ran on an
allocated Perlmutter compute node. For every source, it kept the measured
five-band fluxes and Gaia parallax fixed, then subtracted 100%, 75%, 50% or
0% of the DR11 catalog full-column foreground extinction. The five
dereddened magnitudes, four model colors, support decision, tier and final
distance modulus were recomputed. The zero-percent case is an extreme
sensitivity bound, not a proposed physical correction.

The catalog `A_g=-2.5 log10(MW_TRANSMISSION_G)` agrees with
`3.214 E(B−V)` to at most `4.3×10⁻⁸` mag in these anchor tables. Median
catalog `A_g` is 0.058, 0.050 and 0.080 mag in the two southern and one
northern field. The Gaia reference distance modulus remains fixed, so all
differences below are prediction differences. The main extinction
implementation and its limitations are documented in
[the extinction audit](extinction_audit.md).

## Same-field response

Residual is `Δμ = μ̂ − μ_Gaia`; a positive value puts a source too far away.
The table gives the median residual among sources accepted under both the
catalog and perturbed corrections. Counts change slightly with the route.

| Fraction subtracted | South RA330, mag | South RA20, mag | North RA130, mag | North minus RA330/RA20, mag |
|---:|---:|---:|---:|---:|
| 1.00 | −0.0178 | −0.0520 | +0.0687 | +0.0865 / +0.1207 |
| 0.75 | −0.0410 | −0.0747 | +0.0426 | +0.0837 / +0.1174 |
| 0.50 | −0.0685 | −0.1015 | +0.0099 | +0.0785 / +0.1115 |
| 0.00 | −0.1223 | −0.1547 | −0.0564 | +0.0659 / +0.0983 |

At 75% correction, the paired *median* prediction shifts on the same
accepted sources are −0.0007, +0.0002 and −0.0084 mag. Their nside64
sky-block 95% intervals are [−0.0024,+0.0008],
[−0.0008,+0.0014] and [−0.0112,−0.0059] mag. The change of a residual
median between two distributions need not equal the median of paired
changes: the same stars are reordered by the nonlinear color model.

The 75% case retains at least 99.7% of each baseline selected set and
changes tier for 21, 26 and 9 common sources. Even at zero correction,
the corresponding retention is 98.9%, 99.0% and 99.0%; thus the main
effect is on the predicted distance rather than membership.

Among corrected-parallax ≥2 mas stars, catalog-correction medians are
−0.0221, −0.0662 and +0.1121 mag. At 75% they are −0.0530, −0.0957 and
+0.0732; at zero correction they are −0.1597, −0.1938 and −0.0543.
The northern-minus-southern sign contrast persists under every tested
fraction, including the zero-correction bound. Figure
[`10_dust_sensitivity.png`](../figures/cold_v4/10_dust_sensitivity.png)
shows the overall and nearby-star responses.

## Interpretation and next test

The current pipeline **does perform foreground extinction correction**.
Reducing the applied correction shifts residuals negative in all three
fields. Applying the same fraction in all fields does not remove the
north–south contrast: at 75%, its overall amplitude changes by only
0.003–0.004 mag. This calculation does not determine each star's true
three-dimensional dust column or exclude a spatially varying dust-map
error. It establishes that a uniform fractional reduction of the existing
full-column correction is not a useful center correction for this model.

Keep the frozen model and WISE20 route. A causal dust test would need
independent per-source reddening information or a direct comparison of
photometric systems in newly registered sky regions; do not tune a dust
fraction on these already inspected final fields. The observed native
center difference remains a field-dependent limitation of the current
cold-dwarf conditional distance.

Numerical result: [`cold_v4_dust_sensitivity_final.json`](../manifests/cold_v4_dust_sensitivity_final.json).
