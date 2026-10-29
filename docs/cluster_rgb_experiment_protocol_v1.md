# Registered minimal RGB transfer experiment

Registered after reading only M3, M53 and NGC 2419 development loci and
NGC 7006 Gaia CMD/quality counts. No NGC 7006 Legacy absolute magnitude,
color sequence or residual has been opened. The experiment is a diagnostic
of cross-cluster transfer, not an adopted field-star distance model.

## Fixed input and selection

- M3: `Pmember>0.9`, `grz_clean`, Gaia-CMD `lower_rgb` only.
- M53: the same probability/quality criteria, `upper_rgb` or `lower_rgb`.
- NGC 2419: the same criteria, `upper_rgb` only; its lower region visibly
  overlaps red HB. Its `E(B−V)≈0.061` makes it the dust-sensitive layer.
- NGC 7006: **entire reserved system**, scored on the 19 clean
  `upper_rgb` candidates. The 20 clean lower candidates form a separately
  reported sensitivity layer and do not determine the primary result.
- Selection is fixed by published membership and independent Gaia CMD;
  Legacy colors, model residuals and optical distance estimates do not
  decide phase membership. No per-star DESI/SPDist label is required.

## Two fixed, low-degree diagnostics

Model A predicts `M_g` from a quadratic in `(g−r)₀`; model B uses a
quadratic surface in `(g−r)₀,(r−z)₀`. Each cluster contributes the same
total least-squares weight, divided equally among its selected members.
Features are standardized on the development sample, and the non-intercept
coefficients receive fixed ridge penalty `0.001`. No coefficient, degree,
penalty, cluster weighting or model choice will be tuned on NGC 7006.
Model A is the registered primary diagnostic; model B measures the
same-star effect of the second optical color.

The supported color rectangle is the per-feature 1st–99th percentile of
the fitted development data. Both full-sample and in-rectangle holdout
results are reported. A per-cluster leave-one-system-out development
calculation reports transport behavior without selecting a different
model. The holdout is scored once after the model file and this protocol
are committed.

For each cluster and model, report selected count, median
`Δμ=μ_pred−μ_external`, NMAD, and fraction `|Δμ|>0.5 mag`. On the
holdout, give a source-resampled 95% interval for the median, then list
the external modulus uncertainty separately; members do not supply
independent distance zero points. The same selected stars give the WISE
W1/W2 clean acceptance count, but the low counts do not support a fitted
IR comparison.

## Shared-scale sensitivity

Refit once with each development cluster's modulus displaced by its
quoted combined uncertainty and, separately, with dust shifts of
`±0.005` M3, `±0.010` M53 and `±0.020` NGC 2419 in `E(B−V)`.
Use `R_g=3.214`, `R_r=2.165`, `R_z=1.211` and change the same star's
`g₀`, `(g−r)₀`, `(r−z)₀` together. Report these as correlated scale
sensitivities. On NGC 7006, a `±0.010` dust-shift sensitivity and its
quoted external modulus uncertainty are separate from the source-level
NMAD. The fixed NGC 2419 dust shift includes the approximate difference
between its Legacy map `E(B−V)` and the cited RR study's value.

Failure to transfer between clusters, limited color support, or sensitivity
to shared scales is a measured result. Three development systems are not a
calibrated prior over the field-star metallicity or luminosity function;
the fitted curve is not to be exported as a general giant distance catalog.
