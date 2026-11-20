# Independent finite-distance extinction transfer

## Registration and reference

The fixed baseline is `6c5cc2c7049cf840ad05b4a61743e8b11cd647d0`. The full next-stage review is message `ca01b642-1435-46f9-949a-535f58529cb7` in conversation `6afd1691-5264-83e9-8e2e-a4c4f1c91ec8`. The phase is registered in `configs/environment_transfer_20261120.json` and inherits all input, quality, zero-point, QRF, working-sample and iteration settings from `configs/environment_20261120.json`. No fitting is required.

The existing sky coverage arrays were searched using Galactic positions, geometric coverage, SFD and depth only. L032 and L224 have disjoint 1.25° apertures, distinct longitudes and different latitude signs. Their field centers at +11° and −11.5° extend the available low-latitude geometry just outside the old (|b|<10°) design. M249 provides a mid-latitude control with similar imaging and part of the color and magnitude range in common. L248 was the only pre-listed reserve. Eligibility was determined from Gaia geometry, coverage and the original photometric support, before new distance predictions. The freeze SHA256 is `64320c97b1f1b3cac7d5749c948b284c33b78eda8eb81e6257c32b945b351846`. New apertures do not intersect the previous six. Training LS/Gaia identities, old aperture identities, and identities in the historical model-development inventory are excluded from the primary roster. There were 515 and 110 historical matches in the L224 and M249 photo inventories, respectively. The exact 500-star inputs of the three registered dust arms are saved in the freeze. Gaia positions, uniqueness, astrometric quality, zero point and reference uncertainty are inherited from the same pilot. Median SFD, recorded (r) depth and geometric randoms area describe coverage and do not calibrate stellar completeness.

## PDF comparisons and thresholds

A is the original SFD full-range QRF PDF. B clips and renormalizes it to each star's contiguous Bayestar19 reliable-distance support. C uses Bayestar19 medians at trial distances, frozen coefficients, the original support tests, 0.5 damping, 0.02 mag convergence tolerance and a maximum of nine steps. Gaia distance never enters C. B and the C output share the same map interval. A is also saved and shown for the same identities as the B/C comparison. Clipping A identically yields B. A/B/C have identical reference measurements and object IDs. All primary scores require the reference center to be within the interval, at least half the original SFD mass within it, and successful C convergence. The pre-cap population, the 1,800 per-provider reference forecast cap, the 500-row dust cap, all convergence states and final denominators are distinct. D supplies the Gaia distance to Bayestar solely for a paired scale diagnostic on its own valid B/D subset. D has no predictive score.

Bias is the median residual in distance modulus, NMAD is (1.4826) times the median absolute deviation about that median, and the severe threshold is 1 mag. Interval containing rates use the reference center. Widths are arithmetic means for central 68% and 95% intervals. CRPS uses the analytic step-CDF integral against the reference center. Log score uses the convolution of the piecewise-uniform modulus PDF with the same per-object Gaussian reference error. The nominal paired bootstrap draws 0.30° spatial blocks 600 times. A separate scale sensitivity draws one additional 0.01 mas field-wide offset plus a Gaia measurement error per star for each bootstrap replicate, shared by the arms and repeated occurrences of the star. The zero-point scale is an assumed sensitivity, with no extra reference draws added to the primary log score. The three fixed (0,pm0.01) mas checks are also published. Signed changes in scores are C minus B.

For each main field the decision requires at least 200 common stars and eight blocks, lower absolute bias and a C point bias at most 0.10 mag, and a paired 95% upper CRPS change below zero. Guard failures are a 95% upper log-score change below −0.10 nat, a 95% lower increase in severe rate above 0.02, or a 95% lower increase in absolute nominal coverage error above 0.05 for either interval. The gates apply the prespecified tolerances to the full field comparisons. Sparse subgroup results remain descriptive. The study-level decision requires two new low-latitude passes.

## L219 and faint reference inventory

The archived L219 states and converged distributions are replayed with exactly the same roster, initialization and tolerances, storing per-step proposed and input modulus, color excess, retained PDF mass and endpoint quantiles. Saturation near the map edge, large updates and reversals are diagnostic flags. Gaia CMD magnitudes shown are uncorrected for extinction and are combined with the unchanged conservative zero/full-extinction proxy. Photometric quality and nearest-other detections are exact ID matches in the two archived sweep files. Neighbors include all nearby catalog detections in these files and serve as a finite-footprint crowding proxy. No per-field adjustments were fitted.

The faint census re-scanned only the originally intersecting sweeps of the three chosen and six historical apertures. It keeps (20le r<23) native PSFs through the ownership/domain stage and checks that precision membership exactly matches the forecast roster at the original quality cut. Gaia signed usability requires the inherited astrometric and zero-point cuts. The RAW, usable (grz), precision, QRF support and QRF-plus-history-clean stages have separate denominators. The 200-star and 8-block target is a feasibility screen, not a forward-scoring result. SpecDis and SPDist are reported separately with fit and historical overlap. The APOGEE/astroNN inventory reuses receipt-bound columns. Known-system coverage uses Vasiliev–Baumgardt globular-cluster centers and recorded member radii; cluster types outside that archive were not surveyed.

## Execution and outputs

Use the repository checkout at `/pscratch/sd/y/yunao/photdis_pipeline` from a NERSC CPU allocation, never directly from the login node. Python is the repository's DESI conda runtime with `PYTHONPATH=scripts:src:.deps:.deps/v39_dust` and one linear-algebra thread per process. The release includes the commands and preflight specification. Stages:

```sh
python scripts/environment_transfer.py census
python scripts/environment_transfer.py preflight
python scripts/environment_transfer.py forecast
python scripts/environment_transfer.py dust
python scripts/environment_transfer_support.py
python scripts/environment_transfer_l219.py
python scripts/environment_transfer_stats.py
python scripts/environment_transfer_figures.py data/environment_transfer_20261120/results data/environment_transfer_20261120/figures --baseline-regression
python scripts/environment_transfer_verify.py
```

The preflight freeze is immutable. Start a scratch checkout or a new data root for a full rerun. Figures alone can be regenerated by giving the renderer the public `manifests/environment_transfer_20261120/results` directory. Source-bound receipts accompanying the public results identify the on-NERSC binary products. The main paper, production model/catalog, older science products and RGB displays are outside this experiment.
