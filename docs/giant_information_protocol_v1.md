# Existing-field giant selection and abundance information v1

Registered before reading the new label counts or fitting. Source requirements:
[complete Pro review](pro_review_a7b5c5d.md). Frozen configuration:
`configs/giant_information_v1.json`. This is a finite test with existing 14 fields,
not a new distance calibration. Prior RGB curves, samples and products stay frozen.

## Mother sample, identity and camera

Read existing `field_diagnostics.fits` from the three registries (7 + 4 + 3 fields).
These products already select PSF morphology. Recheck |b|>30 and
0<=catalog EBV<0.03; count base and grz_clean separately. Optical selection uses
the existing NOBS, positive flux/ivar/transmission, S/N>=5, masks and fraction
criteria. There is no extra optical error cut or cold-dwarf color box. The
observable working domain is all finite clean grz colors in these fields;
training-color extrapolation is flagged separately, not silently removed from
evaluation. No distance, inferred Mg, Gaia quality or parallax cut is applied.
The stored diagnostic parent omits rows with neither base photometry nor any
Gaia/spectral match: counts are conditional on that stored parent, not all sky
sources. All base and clean objects are retained by that parent definition.

Recover RELEASE exactly from the collision-free LS_ID high bits. Mapping
11010=DECam, 11011=BASS/MzLS is supported by the existing raw-sweep system audit.
Unknown releases stop extraction. Models never pool the two native systems.
All magnitudes, including forced WISE, use Legacy nanomaggies on the AB system;
PS1/Vega selection cuts are not imported.

Count field records and unique photometric LS identities separately. For the
labeled sample, connect equal positive Gaia source IDs, unambiguous DESI object
IDs and LS IDs across fields; assign an entire connected identity to one field
(first registered field), choose highest spectral S/N within it. Flag conflicting
conservative labels, exclude them from supervised fits, and retain them in the
audit. Retain cross-field identity provenance and assert no identity overlaps in
any train/test split. No pooling of repeated measurements as independent stars.

## Class evidence and finite comparison

Reuse `desi_class_evidence_v3`: existing qualified RVS parameters and
logg+error<3.2 for evolved/giant evidence, logg-error>4.2 for dwarfs.
Missing FeH cannot suppress otherwise good class evidence. Intermediate gravity
and uncertain parameters remain unknown; raw Redrock QSO/GALAXY is an unassessed
risk flag, not reliable nonstellar truth. Training/evaluation focuses on
4200<=Teff<5000 K, split at 4500 K. Teff/logg/FeH/Gaia never enter class predictors.
This classification cannot distinguish RGB, RC or HB phases or establish age.

Fixed models A=(g-r,r-z), B=(A,z-W1), C=(B,W1-W2): quadratic color logistic,
L2 C=1, threshold score>=0.5, equal total giant/dwarf training weights. Polynomial
features and scaling are fitted only on training stars. There is no threshold,
regularization or feature search. Scores are not calibrated catalog probabilities.
Sixteen antithetic Gaussian draws in the original independent band magnitudes
carry shared-band covariance into colors; the same draws are used across A/B/C.
Training uses weighted measurement-noise augmentation, evaluation averages scores
over draws. This is a fixed robustness baseline, not latent-flux deconvolution;
the first-order magnitude-error approximation is a limitation at S/N near 5.

Three prespecified panels: A on all clean labeled grz; A/B/C on the same
WISE-clean W1,W2 S/N>=5 training and evaluation identities; A/B/C on the same
high-quality subset with both S/N>=20. The S/N>=5 common panel is also evaluated
separately at 5<=min(W1,W2 S/N)<20. WISE clean retains current contamination and
mask requirements. Availability of W1 and W1W2 at 5 and 20 is counted on the
entire optical mother by measured r, g-r and maximum optical error, regardless
of spectral labeling. Non-WISE objects remain in the optical mother.

All 14 fields are reused in a new leave-one-whole-field-out experiment, explicitly
cross-validation rather than a claim of untouched new final fields. Original
roles remain provenance. Train only other fields of the same camera and exclude
all linked identities. Require >=20 training giants, >=50 dwarfs and >=2 training
fields; otherwise record insufficient data without transferring another camera's
model. No field is added to cure an insufficient cell. Report test TPR/FPR with
Wilson 95% intervals, counts, confusion matrix and temperature/r/error strata.
Record train-color bounding-box extrapolation. Evaluate extrapolated test stars
in the main table, with a separate flag/stratum. Per-field paired model deltas
and resampling of whole fields quantify transfer variation; star intervals within
one field are only conditional counting uncertainty. No DESI-sample precision is
called photometric purity. Plot pi*TPR/[pi*TPR+(1-pi)*FPR] for fixed pi values,
explicitly assuming only giants/dwarfs and transported rates, excluding unknown
stages/temperatures/nonstars from that simplified calculation.

## Hidden-FeH prediction

Within the same qualified K-giant labels, require the existing abundance
calibration domain and 0<reported FeH error<=0.3 dex. Use author-corrected FeH,
propagate the temperature derivative of that correction, and add a fixed 0.1-dex
systematic floor as a sensitivity budget, not a measured independent error.
Compare intercept-only Gaussian marginal distribution to linear-color Gaussian
conditional distributions A/B/C, each with fitted residual width >=0.05 dex and
fixed slope ridge=1. This restricted family tests stable usable information; a
negative result does not rule out all nonlinear abundance information. Fit the
observed-label likelihood with variance intrinsic^2 + label_error^2 + b'Cov(c)b,
retaining negative shared-color covariances. Prediction q(Z|colors) excludes the
held-out FeH error; scoring against its noisy spectrum adds that error once.
The fitted residual width is not intrinsic astrophysical abundance scatter.
Coefficient-estimation uncertainty is not included in this simple baseline;
held-out coverage and field variation explicitly assess that finite-sample limit.

Use the same whole-field splits and same stars across models within each panel;
minimum training N=20 giants in >=2 other fields. The all-grz panel compares
marginal/A, common WISE panels compare marginal/A/B/C. Report convolved mean log
predictive density, 68/95% observed-label coverage, latent and convolved interval
widths, and paired score changes by field, r and optical error. The 0.1-dex floor
contains potentially shared calibration uncertainty, so field resampling is the
primary transfer uncertainty, not sqrt(N) precision. Scores with the floor removed
are a fixed diagnostic using the same frozen predictive distributions, not refits.
No correlation-only conclusion and no selection on a held-out FeH residual.

## End conditions and deliverables

An improvement is called stable only with at least three informative held-out
fields (>=10 test giants each), positive mean field-level score improvement with
a whole-field bootstrap 95% interval above zero, and positive signs in >=75% of
those fields. Report failures and insufficient data without further tuning.
For class discrimination report paired TPR and FPR changes rather than declaring
a universal winner from one aggregate number. Missing sufficient fields or labels
is a valid inconclusive endpoint. WISE advantages must be stated together with
their observed coverage limits. Even successful giant selection does not authorize
applying the lower-RGB distance curve to unknown-phase field stars. If FeH gains
are not stable, retain external population-prior dependence. Never multiply
q(FeH|colors) as an independent prior by a distance likelihood using those colors.

Deliver source/config/tests, per-field cutflows, dedup identity audit, class and
FeH score tables, availability/purity/transfer figures, a concise science report
and explicit target-population/calibration gaps. Large catalogs and model scores
stay on NERSC; small results/code sync to GitHub/local. Full extraction/training
runs only within Slurm allocations. Stop after these panels and diagnostics.
Submission from the canonical checkout after synchronization and authentication:

```bash
sbatch scripts/run_giant_information.sh extract
# After successful extraction (or with an afterok dependency):
sbatch scripts/run_giant_information.sh fit
# After successful fitting:
sbatch scripts/run_giant_information.sh summarize
```

No new fields/clusters, distance-VAC regressors, required PS1/APOGEE, broader
RGB support, new chains or color-cut optimization are in scope.

Context: [Conroy et al. 2018](https://arxiv.org/abs/1805.05954) motivates testing
optical+WISE K-giant selection. [Xue et al. 2014](https://arxiv.org/abs/1211.0549)
uses spectroscopic metallicity and empirical giant luminosity calibration; its
distance precision is not a benchmark for unlabeled pure-photometric field stars.
