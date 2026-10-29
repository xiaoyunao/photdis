# Controlled cold-dwarf distance-model comparison

Registered before reading new candidate results: `configs/cold_v5_xd_experiment.json`.
Parent checkpoint: `a18ef9b`. The frozen v4 forests, their fit identities,
optical color box, feature-matched support radii, `grz_clean` requirement,
WISE used-band S/N≥20 rule, low-extinction validation range and previous
test identities remain unchanged. The new method is a candidate distance
component evaluated on the same accepted objects.

## Model and error propagation

For A/B/C respectively, the latent vector is
`z=(M_g,g-r,r-z[,z-W1[,W1-W2]])` with 3/4/5 dimensions. A full-covariance
Gaussian mixture models the empirical selected-anchor distribution `p(z)`.
Its likelihood convolves each component with that anchor's own error
covariance. The measured band errors are provisionally independent in their
native magnitudes; multiplication by the exact color-difference matrix
introduces the shared-band covariance. For the training `M_g`, the Gaia
corrected-parallax contribution is `(5/ln(10)/SNR_corr)^2` on the first
diagonal. Thus `Cov(M_g,g-r)=σ_g²` and `Cov(g-r,r-z)=−σ_r²` before other
calibration terms. The latter are not available as a per-source matrix in
the current catalog and remain outside this first low-extinction experiment.

Prediction observes `y=(g0,colors)`. For a fixed distance modulus `μ`, the
Gaussian likelihood is `y|μ,k ~ N(mean_k + μ e0, V_k+S_y)`, where `S_y` has
the same shared-band propagation but no Gaia parallax term. Each component
yields a one-dimensional Gaussian likelihood in `μ`; their mixture is
integrated under a uniform prior on `0≤μ≤20` and normalized. This is a fixed
comparison prior, equivalent to `p(distance)∝1/distance` inside the stated
bounds. Output quantiles transform monotonically to distance. The mixture
weights describe the selected v4 training anchors, not general Galactic
population fractions.

This uses a high-S/N magnitude-error approximation. The Legacy catalog's
fluxes and inverse variances support a flux likelihood for eventual faint
sources. The [Legacy DR11 file description](https://d.legacysurvey.org/dr11/files/)
documents those flux/IVAR fields. The heteroscedastic mixture fitting follows
the framework of [Bovy, Hogg & Roweis](https://arxiv.org/abs/0905.2979),
implemented here for complete dimensions with a covariance eigenvalue floor.

## Bounded selection and comparison

Only 16, 32 and 64 full-covariance components are fitted. Each A/B/C fit
must reproduce the corresponding frozen-v4 source-ID hash and count
(7,782/7,351/7,341). The configuration score is the mean held-out
conditional negative log density of observed `M_g` given the observed colors
and heteroscedastic covariance. Two whole development fields, RA205 and
RA60, are scored separately, then weighted equally. RA235 is a small
diagnostic field and does not choose the component count. No random-star
split or new final field is used for configuration selection.

The selected mixture is compared with each frozen QRF on the exact same
strict Gaia cold-anchor identities accepted by the locked WISE20 route.
Report by field and tier: residual center, NMAD, |Δμ|>0.5 mag tail,
raw 68%/95% quantile coverage, each 95% tail, width, error tertile, optical color,
infrared conflict status and fixed-tier +0.01 mag g/r response. The
predeclared decision gate requires noninferiority in center, NMAD and tail
in each informative development field, plus a 25% reduction in the median
absolute response to both small band perturbations. A smooth curve alone
is not a scientific win; inspect color-bin residuals for suppressed genuine
color–luminosity slope. If this gate fails, retain v4 and close the bounded
comparison. If it passes, lock one candidate before independent interval
calibration and one newly registered final-field evaluation.

`scripts/cold_v5_xd_decide.py` applies these inequalities to both registered
configuration-development fields and records every check. It uses the
forest's raw weighted quantiles and the mixture's raw posterior quantiles for
the side-by-side interval diagnostic. The v4 calibrated deep intervals remain
withheld; no nominal deep coverage is inferred from these raw values.

Deep v4 nominal 68%/95% intervals remain withheld. The new mixture's raw
intervals are development diagnostics until separately calibrated and
tested. A giant probability, a north/south constant offset, and a simple
`M_r` relabeling are outside this comparison.

## Per-candidate frozen route output

`scripts/cold_v4_candidate_export.py` writes ignored FITS catalogs on NERSC.
Each accepted row carries the frozen tier, reasons B/C were not used,
actual A/B/C tenth-neighbor radius ratios, an infrared fallback-conflict
marker, used-tier high-quality state, the cold-dwarf-hypothesis median
distance and a measured-r depth state. `support_ratio_*` may be finite even
when that tier's photometry is unavailable; the corresponding `reason_*`
controls interpretation. The conflict marker indicates reliable infrared
measurements that prevented a higher-tier choice through the frozen support
test. It neither deletes the row nor assigns a stellar-class probability.
`r18_22_stress_tested` identifies the range of the existing undercovered
deep stress test; `r_ge_22_unvalidated` lies beyond it. No nominal deep
interval is written to this product.
